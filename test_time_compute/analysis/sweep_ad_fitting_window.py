"""
Fitting window length sweep for advection-diffusion.
Varies the number of input frames L used for one-step fitting at test time.
All evaluations predict from the same point onwards, ensuring identical targets.

We generate trajectories with 32 input frames. For each L in {2, 4, 8, 16, 32},
we use the last L frames of the 32-frame input for fitting, then predict from
frame 33 onwards (same ground truth for all L values).

Addresses reviewer SLt5 question on fitting window sensitivity.
"""
import torch
import argparse
import os
import json
import time
import numpy as np
from datetime import datetime
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from ttc_utils import DEVICE, get_relative_l2_error
from ttc_methods import beam_search_operator_selection_batch
from train.train import DISCOLitModule, TemporalBatchDatasetFly


def load_model_from_checkpoint(checkpoint_path):
    lit_model = DISCOLitModule.load_from_checkpoint(checkpoint_path, map_location=DEVICE)
    lit_model.eval()
    model = lit_model.model.to(DEVICE)
    model.eval()
    print(f"Model loaded from {checkpoint_path}")
    return model


def main():
    parser = argparse.ArgumentParser(description='Fitting window length sweep')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./test-time-compute/results/fitting_window_sweep')
    parser.add_argument('--experiment', type=str, default='E_AD_ALL',
                        choices=['E_AD_ALL', 'E_AD_v', 'E_AD_D'])
    parser.add_argument('--window_sizes', type=int, nargs='+', default=[2, 4, 8, 16, 32],
                        help='Fitting window sizes L to sweep')
    parser.add_argument('--num_samples', type=int, default=512)
    parser.add_argument('--beam_width', type=int, default=4)
    parser.add_argument('--beam_batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    EXPERIMENT_CONFIGS = {
        'E_AD_ALL': {'v_range': (0.01, 1.0), 'D_range': (0.01, 1.0)},
        'E_AD_v': {'v_range': (1.0, 3.0), 'D_range': (0.0, 0.0)},
        'E_AD_D': {'v_range': (0.0, 0.0), 'D_range': (1.0, 3.0)},
    }

    MAX_INPUT_FRAMES = max(args.window_sizes)  # e.g. 32
    N_OUTPUT_FRAMES = 34  # prediction horizon (same as paper)

    os.makedirs(args.output_dir, exist_ok=True)
    model = load_model_from_checkpoint(args.model_path)

    exp_cfg = EXPERIMENT_CONFIGS[args.experiment]

    # Encode dictionary (always use 16-frame windows for encoding, same as training)
    print("Encoding operators from training data...")
    train_dataset = TemporalBatchDatasetFly(
        n_batches=4, batch_size=64,
        sub_x=1, sub_t=1, split='train',
        input_frames=16, output_frames=2,
        L=16.0, nx=256, nt=100, T=10.0,
        v_range=(0.01, 1.0), D_range=(0.001, 1.0),
        fractal_degree=256, fractal_power_range=3, seed=args.seed,
    )

    theta_latent_operators = []
    operator_metadata = []
    cpt = 0
    for batch in train_dataset:
        inp = batch['input'].to(DEVICE)
        state_labels = torch.tensor([0], device=DEVICE)
        with torch.no_grad():
            theta_latent, _ = model.encode_theta_latent(inp, state_labels)
        theta_latent_operators.append(theta_latent)
        for idx in range(len(inp)):
            operator_metadata.append({
                'operator_id': cpt,
                'advection_speed': batch['advection_speed'][idx],
                'diffusion': batch['diffusion'][idx],
            })
            cpt += 1
    theta_latent_operators = torch.cat(theta_latent_operators)
    print(f"Encoded {theta_latent_operators.shape[0]} operators")

    # Create test dataset with MAX_INPUT_FRAMES input frames
    # The full trajectory is MAX_INPUT_FRAMES + N_OUTPUT_FRAMES frames long
    # For each L, we take the last L frames of input and predict the same N_OUTPUT_FRAMES
    print(f"\nGenerating test data with {MAX_INPUT_FRAMES} input frames...")
    n_test_batches = max(1, args.num_samples // 64)
    test_dataset = TemporalBatchDatasetFly(
        n_batches=n_test_batches, batch_size=64,
        sub_x=1, sub_t=1, split='test',
        input_frames=MAX_INPUT_FRAMES, output_frames=N_OUTPUT_FRAMES,
        L=16.0, nx=256, nt=100, T=10.0,
        v_range=exp_cfg['v_range'], D_range=exp_cfg['D_range'],
        fractal_degree=256, fractal_power_range=3, seed=124,
    )

    # Pre-collect test batches
    print("Collecting test data...")
    test_batches = []
    for batch in test_dataset:
        test_batches.append(batch)
    total_samples = sum(b['input'].shape[0] for b in test_batches)
    print(f"Collected {total_samples} test samples")

    # Run sweep
    results = {
        'experiment': args.experiment,
        'beam_width': args.beam_width,
        'max_input_frames': MAX_INPUT_FRAMES,
        'n_output_frames': N_OUTPUT_FRAMES,
        'timestamp': datetime.now().isoformat(),
        'sweep': [],
    }

    for L in sorted(args.window_sizes):
        print(f"\n{'='*50}")
        print(f"Fitting window L={L}")

        all_errors = []
        start_time = time.time()

        for batch in test_batches:
            # Take last L frames of the MAX_INPUT_FRAMES input frames
            batch_input_full = batch['input'].to(DEVICE)  # (B, MAX_INPUT_FRAMES, C, H)
            batch_input = batch_input_full[:, -L:]  # (B, L, C, H)
            batch_target = batch['target'].to(DEVICE)  # (B, N_OUTPUT_FRAMES, C, H)

            for i in range(batch_input.size(0)):
                composition, error, pred = beam_search_operator_selection_batch(
                    model, theta_latent_operators,
                    batch_input[i:i+1], batch_target[i:i+1],
                    beam_width=args.beam_width,
                    max_operators=5,
                    dt=10.0/100,
                    batch_size=args.beam_batch_size,
                )
                all_errors.append(error)

        elapsed = time.time() - start_time
        avg_error = np.mean(all_errors)
        std_error = np.std(all_errors)

        print(f"L={L}: avg NRMSE = {avg_error:.6f} +/- {std_error:.6f} ({elapsed:.1f}s)")

        results['sweep'].append({
            'window_size': L,
            'avg_nrmse': float(avg_error),
            'std_nrmse': float(std_error),
            'num_samples': len(all_errors),
            'time_seconds': elapsed,
        })

    # Summary
    print(f"\n{'='*50}")
    print("FITTING WINDOW SWEEP SUMMARY")
    print(f"{'L':>6} | {'NRMSE':>10} | {'Std':>10}")
    print("-" * 35)
    for r in results['sweep']:
        print(f"{r['window_size']:>6} | {r['avg_nrmse']:>10.6f} | {r['std_nrmse']:>10.6f}")

    output_file = os.path.join(args.output_dir,
        f"fitting_window_{args.experiment}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()

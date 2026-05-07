"""
Dictionary subsampling sensitivity sweep for advection-diffusion.
Encodes a full dictionary of 256 operators, then subsamples to different sizes
and runs beam search on each. Reports NRMSE vs dictionary size.

Addresses reviewer RRZc and rJBq questions on dictionary subsampling sensitivity.
"""
import torch
import argparse
import os
import json
import time
import numpy as np
from datetime import datetime


from test_time_compute.ttc_utils import DEVICE, get_relative_l2_error
from test_time_compute.ttc_methods import beam_search_operator_selection_batch
from train.train import DISCOLitModule, TemporalBatchDatasetFly


def load_model_from_checkpoint(checkpoint_path):
    lit_model = DISCOLitModule.load_from_checkpoint(checkpoint_path, map_location=DEVICE)
    lit_model.eval()
    model = lit_model.model.to(DEVICE)
    model.eval()
    print(f"Model loaded from {checkpoint_path}")
    return model


def main():
    parser = argparse.ArgumentParser(description='Dictionary subsampling sweep')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./test_time_compute/results/subsampling_sweep')
    parser.add_argument('--experiment', type=str, default='E_AD_ALL',
                        choices=['E_AD_ALL', 'E_AD_v', 'E_AD_D'])
    parser.add_argument('--dict_sizes', type=int, nargs='+', default=[16, 32, 64, 128, 256],
                        help='Dictionary sizes to sweep')
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

    N_INPUT_FRAMES = 16
    N_OUTPUT_FRAMES = 34

    os.makedirs(args.output_dir, exist_ok=True)
    model = load_model_from_checkpoint(args.model_path)

    exp_cfg = EXPERIMENT_CONFIGS[args.experiment]

    # Encode full dictionary (256 operators from training distribution)
    max_dict_size = max(args.dict_sizes)
    n_batches_needed = max(1, max_dict_size // 64)
    train_dataset = TemporalBatchDatasetFly(
        n_batches=n_batches_needed,
        batch_size=64,
        sub_x=1, sub_t=1, split='train',
        input_frames=N_INPUT_FRAMES, output_frames=N_OUTPUT_FRAMES,
        L=16.0, nx=256, nt=100, T=10.0,
        v_range=(0.01, 1.0), D_range=(0.001, 1.0),
        fractal_degree=256, fractal_power_range=3, seed=args.seed,
    )

    print("Encoding operators from training data...")
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

    # Create test dataset (fixed seed for reproducibility across sweeps)
    n_test_batches = max(1, args.num_samples // 64)
    test_dataset = TemporalBatchDatasetFly(
        n_batches=n_test_batches,
        batch_size=64,
        sub_x=1, sub_t=1, split='test',
        input_frames=N_INPUT_FRAMES, output_frames=N_OUTPUT_FRAMES,
        L=16.0, nx=256, nt=100, T=10.0,
        v_range=exp_cfg['v_range'], D_range=exp_cfg['D_range'],
        fractal_degree=256, fractal_power_range=3, seed=124,
    )

    # Pre-collect all test batches so we reuse exactly the same data for each dict size
    print("Collecting test data...")
    test_batches = []
    for batch in test_dataset:
        test_batches.append(batch)

    # Run sweep
    results = {
        'experiment': args.experiment,
        'beam_width': args.beam_width,
        'timestamp': datetime.now().isoformat(),
        'sweep': [],
    }

    rng = np.random.default_rng(args.seed)

    for N in sorted(args.dict_sizes):
        print(f"\n{'='*50}")
        print(f"Dictionary size N={N}")

        # Subsample dictionary
        if N >= theta_latent_operators.shape[0]:
            indices = list(range(theta_latent_operators.shape[0]))
        else:
            indices = sorted(rng.choice(theta_latent_operators.shape[0], size=N, replace=False).tolist())
        theta_subset = theta_latent_operators[indices]
        meta_subset = [operator_metadata[i] for i in indices]

        # Run beam search on all test samples
        all_errors = []
        start_time = time.time()

        for batch in test_batches:
            batch_input = batch['input'].to(DEVICE)
            batch_target = batch['target'].to(DEVICE)

            for i in range(batch_input.size(0)):
                composition, error, pred = beam_search_operator_selection_batch(
                    model, theta_subset,
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

        print(f"N={N}: avg NRMSE = {avg_error:.6f} +/- {std_error:.6f} ({elapsed:.1f}s)")

        results['sweep'].append({
            'dict_size': N,
            'avg_nrmse': float(avg_error),
            'std_nrmse': float(std_error),
            'num_samples': len(all_errors),
            'time_seconds': elapsed,
        })

    # Summary
    print(f"\n{'='*50}")
    print("SUBSAMPLING SWEEP SUMMARY")
    print(f"{'N':>6} | {'NRMSE':>10} | {'Std':>10}")
    print("-" * 35)
    for r in results['sweep']:
        print(f"{r['dict_size']:>6} | {r['avg_nrmse']:>10.6f} | {r['std_nrmse']:>10.6f}")

    output_file = os.path.join(args.output_dir,
        f"subsampling_{args.experiment}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()

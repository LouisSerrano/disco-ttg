"""
Evaluate test-time generalization on perturbation data (Burgers nonlinear term).
Loads pre-generated .npz files and runs beam search + direct prediction.
Reports NRMSE vs epsilon for graceful degradation analysis.

Addresses reviewers rJBq W2, SLt5 W2, UJ7L on imperfect decomposability.
"""
import torch
import argparse
import os
import json
import time
import numpy as np
from datetime import datetime


from test_time_compute.ttc_utils import DEVICE
from test_time_compute.ttc_methods import (
    test_direct_prediction,
    beam_search_operator_selection_batch,
)
from train.train import DISCOLitModule, TemporalBatchDatasetFly


def load_model_from_checkpoint(checkpoint_path):
    lit_model = DISCOLitModule.load_from_checkpoint(checkpoint_path, map_location=DEVICE)
    lit_model.eval()
    model = lit_model.model.to(DEVICE)
    model.eval()
    print(f"Model loaded from {checkpoint_path}")
    return model, lit_model


def main():
    parser = argparse.ArgumentParser(description='Evaluate on perturbation data')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory with perturbation_eps*.npz files')
    parser.add_argument('--output_dir', type=str, default='./test_time_compute/results/rebuttal/perturbation_eval')
    parser.add_argument('--beam_width', type=int, default=4)
    parser.add_argument('--beam_batch_size', type=int, default=32)
    parser.add_argument('--max_operators', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epsilons', type=float, nargs='+', default=None,
                        help='Only evaluate these epsilon values (default: all found in data_dir)')
    args = parser.parse_args()

    N_INPUT_FRAMES = 16
    DT = 10.0 / 100  # dt for operator splitting (T=10, nt=100)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model, lit_model = load_model_from_checkpoint(args.model_path)

    # Encode dictionary from pure physics training data (same as paper)
    print("Encoding operators from training data...")
    train_dataset = TemporalBatchDatasetFly(
        n_batches=4, batch_size=64,
        sub_x=1, sub_t=1, split='train',
        input_frames=N_INPUT_FRAMES, output_frames=2,
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

    # Discover available epsilon files
    npz_files = sorted([f for f in os.listdir(args.data_dir) if f.startswith('perturbation_eps') and f.endswith('.npz')])
    if args.epsilons is not None:
        # Filter to only requested epsilon values
        eps_set = set(args.epsilons)
        npz_files = [f for f in npz_files if float(f.replace('perturbation_eps', '').replace('.npz', '')) in eps_set]
    if not npz_files:
        print(f"No perturbation data found in {args.data_dir}")
        return

    print(f"\nFound {len(npz_files)} epsilon values: {npz_files}")

    # Load config
    config_path = os.path.join(args.data_dir, 'config.json')
    if os.path.exists(config_path):
        with open(config_path) as f:
            data_config = json.load(f)
        print(f"Data config: v={data_config['v']}, D={data_config['D']}, T={data_config['T']}")

    results = {
        'model_path': args.model_path,
        'data_dir': args.data_dir,
        'beam_width': args.beam_width,
        'max_operators': args.max_operators,
        'timestamp': datetime.now().isoformat(),
        'sweep': [],
    }

    for npz_file in npz_files:
        data = np.load(os.path.join(args.data_dir, npz_file))
        trajectories = data['trajectories']  # (n_samples, nt, nx)
        epsilon = float(data['epsilon'])
        n_samples = trajectories.shape[0]

        print(f"\n{'='*60}")
        print(f"Evaluating epsilon={epsilon} ({n_samples} samples)")
        print(f"Trajectory shape: {trajectories.shape}")

        # Split into input (first N_INPUT_FRAMES) and target (rest)
        # trajectories: (n_samples, nt, nx) -> need to add channel dim
        n_output = trajectories.shape[1] - N_INPUT_FRAMES
        inp_np = trajectories[:, :N_INPUT_FRAMES, :]  # (n, 16, 256)
        tgt_np = trajectories[:, N_INPUT_FRAMES:, :]  # (n, 84, 256)

        # Convert to torch tensors with channel dimension: (n, t, 1, x)
        inp_tensor = torch.from_numpy(inp_np).float().unsqueeze(2).to(DEVICE)
        tgt_tensor = torch.from_numpy(tgt_np).float().unsqueeze(2).to(DEVICE)

        # === Direct prediction ===
        print("  Running direct prediction...")
        direct_errors = []
        start_time = time.time()
        with torch.no_grad():
            for i in range(0, n_samples, 64):
                batch_end = min(i + 64, n_samples)
                sample_inp = inp_tensor[i:batch_end]
                sample_tgt = tgt_tensor[i:batch_end]

                state_labels = torch.tensor([0], device=DEVICE)
                pred, _ = model(sample_inp, state_labels, n_future_steps=sample_tgt.shape[1])

                # Per-sample NRMSE
                for j in range(sample_inp.shape[0]):
                    error = torch.norm(pred[j] - sample_tgt[j]) / (torch.norm(sample_tgt[j]) + 1e-8)
                    direct_errors.append(error.item())

        direct_time = time.time() - start_time
        avg_direct = np.mean(direct_errors)
        std_direct = np.std(direct_errors)
        print(f"  Direct: NRMSE = {avg_direct:.6f} +/- {std_direct:.6f} ({direct_time:.1f}s)")

        # === Beam search ===
        print("  Running beam search...")
        beam_errors = []
        beam_compositions = []
        start_time = time.time()

        for i in range(n_samples):
            composition, error, pred = beam_search_operator_selection_batch(
                model, theta_latent_operators,
                inp_tensor[i:i+1], tgt_tensor[i:i+1],
                beam_width=args.beam_width,
                max_operators=args.max_operators,
                dt=DT,
                batch_size=args.beam_batch_size,
            )
            beam_errors.append(error)
            beam_compositions.append(composition)

            if i % 32 == 0:
                print(f"    Sample {i}/{n_samples}: error={error:.6f}, composition={composition}")

        beam_time = time.time() - start_time
        avg_beam = np.mean(beam_errors)
        std_beam = np.std(beam_errors)
        print(f"  Beam: NRMSE = {avg_beam:.6f} +/- {std_beam:.6f} ({beam_time:.1f}s)")

        # Composed parameters from beam search
        composed_params = []
        for comp in beam_compositions:
            v_sum = sum(float(operator_metadata[op_id]['advection_speed']) for op_id in comp)
            D_sum = sum(float(operator_metadata[op_id]['diffusion']) for op_id in comp)
            composed_params.append({'v': v_sum, 'D': D_sum})

        avg_v = np.mean([p['v'] for p in composed_params])
        avg_D = np.mean([p['D'] for p in composed_params])
        print(f"  Avg composed params: v={avg_v:.3f} (true=0.5), D={avg_D:.3f} (true=0.3)")

        results['sweep'].append({
            'epsilon': epsilon,
            'n_samples': n_samples,
            'direct_nrmse': float(avg_direct),
            'direct_std': float(std_direct),
            'beam_nrmse': float(avg_beam),
            'beam_std': float(std_beam),
            'direct_time': direct_time,
            'beam_time': beam_time,
            'avg_composed_v': float(avg_v),
            'avg_composed_D': float(avg_D),
        })

    # Summary table
    print(f"\n{'='*60}")
    print("PERTURBATION EVALUATION SUMMARY")
    print(f"{'eps':>8} | {'Direct':>10} | {'Beam':>10} | {'v_est':>8} | {'D_est':>8}")
    print("-" * 55)
    for r in results['sweep']:
        print(f"{r['epsilon']:>8.3f} | {r['direct_nrmse']:>10.6f} | {r['beam_nrmse']:>10.6f} | "
              f"{r['avg_composed_v']:>8.3f} | {r['avg_composed_D']:>8.3f}")

    # Save results
    output_file = os.path.join(args.output_dir,
        f"perturbation_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()

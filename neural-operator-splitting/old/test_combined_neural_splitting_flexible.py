"""
Flexible Combined Neural Operator Splitting Test

This version allows passing custom checkpoint paths and test dataset directly.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import h5py
import argparse

# Add paths
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from neural_ode_operators import create_neural_operators
from neural_splitting_methods import NeuralOperatorSplitting


def load_test_data_from_file(hdf5_file: str, num_samples: int = 5):
    """
    Load test data from a custom HDF5 file.
    
    Args:
        hdf5_file: Path to HDF5 file
        num_samples: Number of samples to load
        
    Returns:
        Tuple of (trajectories, beta, gamma, time_points)
    """
    if not os.path.exists(hdf5_file):
        raise FileNotFoundError(f"Test data file not found: {hdf5_file}")
    
    with h5py.File(hdf5_file, 'r') as f:
        print(f"Available groups in {hdf5_file}: {list(f.keys())}")
        
        # Try different group names
        group = None
        for group_name in ['test', 'valid', 'train']:
            if group_name in f:
                group = f[group_name]
                print(f"Using group '{group_name}'")
                break
        
        if group is None:
            raise ValueError(f"No valid group found in {hdf5_file}")
        
        if 'pde_250-256' in group:
            trajectories = group['pde_250-256'][:num_samples]
            alpha = group['alpha'][:num_samples] if 'alpha' in group else np.zeros(num_samples)
            beta = group['beta'][:num_samples] if 'beta' in group else np.zeros(num_samples)
            gamma = group['gamma'][:num_samples] if 'gamma' in group else np.zeros(num_samples)
        else:
            raise ValueError(f"Expected 'pde_250-256' not found in group")
    
    # Create time points
    nt = trajectories.shape[1]
    time_points = np.linspace(0, 4.0, nt)
    
    print(f"Loaded {trajectories.shape[0]} trajectories with shape {trajectories.shape}")
    print(f"Parameter ranges - alpha: [{alpha.min():.3f}, {alpha.max():.3f}], "
          f"beta: [{beta.min():.3f}, {beta.max():.3f}], gamma: [{gamma.min():.3f}, {gamma.max():.3f}]")
    
    return trajectories, alpha, beta, gamma, time_points


def test_lie_splitting(model1, model2, u0, dt, nt_refined, nt_coarse, method='rk4'):
    """Test Lie splitting with refinement."""
    splitter = NeuralOperatorSplitting(model1, model2, method=method)
    
    # Number of steps per coarse interval
    steps_per_interval = nt_refined // nt_coarse
    
    u_current = u0
    trajectory = [u0.squeeze().numpy()]
    
    for i in range(nt_coarse):
        # Apply splitting with refined time steps
        for j in range(steps_per_interval):
            u_current = splitter.lie_splitting_step(u_current, dt)
        
        trajectory.append(u_current.squeeze().numpy())
    
    return np.array(trajectory)


def test_strang_splitting(model1, model2, u0, dt, nt_refined, nt_coarse, method='rk4'):
    """Test Strang splitting with refinement."""
    splitter = NeuralOperatorSplitting(model1, model2, method=method)
    
    # Number of steps per coarse interval
    steps_per_interval = nt_refined // nt_coarse
    
    u_current = u0
    trajectory = [u0.squeeze().numpy()]
    
    for i in range(nt_coarse):
        # Apply splitting with refined time steps
        for j in range(steps_per_interval):
            u_current = splitter.strang_splitting_step(u_current, dt)
        
        trajectory.append(u_current.squeeze().numpy())
    
    return np.array(trajectory)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Flexible Test Neural Operator Splitting')
    
    # Checkpoint paths
    parser.add_argument('--ckpt1', type=str, required=True,
                        help='Path to first operator checkpoint')
    
    parser.add_argument('--ckpt2', type=str, required=True,
                        help='Path to second operator checkpoint')
    
    # Test dataset
    parser.add_argument('--test_dataset', type=str, required=True,
                        help='Path to test dataset HDF5 file')
    
    # Model architecture parameters
    parser.add_argument('--nx', type=int, default=256,
                        help='Number of spatial grid points (default: 256)')
    
    parser.add_argument('--L', type=float, default=16.0,
                        help='Domain length (default: 16.0)')
    
    parser.add_argument('--hidden_dim', type=int, default=32,
                        help='Hidden dimension (default: 32)')
    
    parser.add_argument('--n_layers', type=int, default=2,
                        help='Number of layers (default: 2)')
    
    # Testing parameters
    parser.add_argument('--num_samples', type=int, default=32,
                        help='Number of test samples (default: 32)')
    
    parser.add_argument('--method', type=str, default='rk4',
                        choices=['rk4', 'dopri5', 'euler'],
                        help='ODE solver method (default: rk4)')
    
    parser.add_argument('--splitting_method', type=str, default='lie',
                        choices=['lie', 'strang'],
                        help='Splitting method to use (default: lie)')
    
    parser.add_argument('--refinement_factor', type=int, default=1,
                        help='Temporal refinement factor for operator splitting (default: 1)')
    
    parser.add_argument('--results_dir', type=str, default=None,
                        help='Custom results directory (default: auto-generated)')
    
    # Operator names for visualization
    parser.add_argument('--op1_name', type=str, default='Operator1',
                        help='Name of first operator for visualization')
    
    parser.add_argument('--op2_name', type=str, default='Operator2',
                        help='Name of second operator for visualization')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("="*60)
    print("Flexible Combined Neural Operator Splitting Test")
    print("="*60)
    
    # Configuration
    print(f"Checkpoint 1: {args.ckpt1}")
    print(f"Checkpoint 2: {args.ckpt2}")
    print(f"Test dataset: {args.test_dataset}")
    print(f"Model architecture: nx={args.nx}, L={args.L}, hidden_dim={args.hidden_dim}, n_layers={args.n_layers}")
    print(f"Testing: {args.num_samples} samples, method={args.method}, splitting={args.splitting_method}")
    
    # Load test data
    print("\nLoading test data...")
    trajectories, alpha, beta, gamma, time_points = load_test_data_from_file(
        args.test_dataset, 
        args.num_samples
    )
    
    # Create model architectures
    print("\nCreating model architectures...")
    operators = create_neural_operators(
        args.nx, 
        args.L, 
        hidden_dim=args.hidden_dim, 
        n_layers=args.n_layers, 
        padding_mode="circular"
    )
    
    # Load checkpoints
    print(f"\nLoading checkpoint 1 from {args.ckpt1}...")
    state_dict1 = torch.load(args.ckpt1, map_location='cpu')
    # Try to match the state dict to available models
    for name, model in operators.items():
        try:
            model.load_state_dict(state_dict1)
            model1 = model
            print(f"  Loaded into {name} model")
            break
        except:
            continue
    
    print(f"\nLoading checkpoint 2 from {args.ckpt2}...")
    state_dict2 = torch.load(args.ckpt2, map_location='cpu')
    # Try to match the state dict to available models
    for name, model in operators.items():
        if model is not model1:  # Skip the already loaded model
            try:
                model.load_state_dict(state_dict2)
                model2 = model
                print(f"  Loaded into {name} model")
                break
            except:
                continue
    
    # Set to eval mode
    model1.eval()
    model2.eval()
    
    # Time stepping setup
    dt = time_points[1] - time_points[0]
    nt = len(time_points) - 1
    
    # Refinement for operator splitting
    small_dt = dt / args.refinement_factor
    small_nt = nt * args.refinement_factor
    
    print(f"\nTime stepping:")
    print(f"  Original: dt={dt:.4f}, nt={nt}")
    print(f"  Refined: small_dt={small_dt:.4f}, small_nt={small_nt}, refinement_factor={args.refinement_factor}")
    
    # Create results directory
    if args.results_dir:
        results_dir = args.results_dir
    else:
        dataset_basename = os.path.splitext(os.path.basename(args.test_dataset))[0]
        results_dir = f"test_results_{dataset_basename}_{args.splitting_method}_r{args.refinement_factor}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Test on samples
    for idx in range(min(args.num_samples, len(trajectories))):
        u0 = trajectories[idx, 0, :]
        u0 = torch.tensor(u0, dtype=torch.float32).unsqueeze(0)
        
        print(f"\nRunning {args.splitting_method} splitting for sample {idx+1}/{args.num_samples}...")
        
        # Choose splitting method
        if args.splitting_method == 'lie':
            result = test_lie_splitting(model1, model2, u0, small_dt, small_nt, nt, args.method)
        else:
            result = test_strang_splitting(model1, model2, u0, small_dt, small_nt, nt, args.method)
        
        print(f"✓ {args.splitting_method.capitalize()} splitting completed. Result shape: {result.shape}")
        
        # Get ground truth
        ground_truth = trajectories[idx, :len(result), :]
        
        # Visualization
        plt.figure(figsize=(12, 10))
        
        # Sample every 10 time steps
        time_indices = np.arange(0, len(result), 10)
        colors = plt.cm.viridis(np.linspace(0, 1, len(time_indices)))
        
        # Ground Truth
        plt.subplot(2, 2, 1)
        for i, t_idx in enumerate(time_indices):
            plt.plot(ground_truth[t_idx, :], '-', color=colors[i], linewidth=1.5, alpha=0.8)
        plt.title("Ground Truth Evolution")
        plt.xlabel("Space")
        plt.ylabel("u")
        plt.grid(True, alpha=0.3)
        
        # Neural Splitting
        plt.subplot(2, 2, 2)
        for i, t_idx in enumerate(time_indices):
            plt.plot(result[t_idx, :], '--', color=colors[i], linewidth=1.5, alpha=0.8)
        plt.title(f"Neural {args.splitting_method.capitalize()} Splitting")
        plt.xlabel("Space")
        plt.ylabel("u")
        plt.grid(True, alpha=0.3)
        
        # Error Evolution
        plt.subplot(2, 2, 3)
        errors = []
        for t in range(len(result)):
            error = np.linalg.norm(result[t] - ground_truth[t]) / np.linalg.norm(ground_truth[t])
            errors.append(error)
        plt.semilogy(errors, 'b-', linewidth=2)
        plt.title("Relative Error Evolution")
        plt.xlabel("Time Step")
        plt.ylabel("Relative L2 Error")
        plt.grid(True, alpha=0.3)
        
        # Final Comparison
        plt.subplot(2, 2, 4)
        plt.plot(ground_truth[-1, :], 'k-', label='Ground Truth', linewidth=2)
        plt.plot(result[-1, :], 'r--', label=f'Neural {args.splitting_method.capitalize()}', linewidth=2)
        plt.title("Final Time Comparison")
        plt.xlabel("Space")
        plt.ylabel("u")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Add parameters info
        param_str = f"α={alpha[idx]:.2f}, β={beta[idx]:.2f}, γ={gamma[idx]:.2f}"
        plt.suptitle(f"{args.op1_name} + {args.op2_name} - {args.splitting_method.capitalize()} (r{args.refinement_factor}) - Sample {idx+1}\n{param_str}", y=1.02)
        
        # Save figure
        plt.savefig(os.path.join(results_dir, f'sample_{idx+1}_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save numerical results
        np.savez(os.path.join(results_dir, f'sample_{idx+1}_results.npz'),
                 ground_truth=ground_truth,
                 neural_result=result,
                 errors=errors,
                 alpha=alpha[idx],
                 beta=beta[idx],
                 gamma=gamma[idx])
    
    print(f"\n✓ Testing completed! Results saved in: {results_dir}")
    print(f"Final relative errors: min={min(errors):.2e}, max={max(errors):.2e}, mean={np.mean(errors):.2e}")


if __name__ == "__main__":
    main()
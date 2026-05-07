"""
Simple test script for neural ODE operators.
Tests operator splitting (Lie/Strang) with two trained models.
"""

import torch
import numpy as np
import h5py
import os
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from einops import rearrange

from neural_ode_operators import AdvectionNeuralODE


def load_test_data(test_file, num_samples=None):
    """Load test data from HDF5 file."""
    with h5py.File(test_file, 'r') as f:
        # Find the correct group
        if 'test' in f:
            group = f['test']
        else:
            # Use first available group
            group = f[list(f.keys())[0]]
        
        # Load data - use same key as training
        trajectories = group['pde_250-256'][:]
        # Generate time points like in training
        time_points = np.linspace(0, 4.0, trajectories.shape[1])
        
        # Load parameters if available
        if 'alpha' in group:
            alphas = group['alpha'][:]
            betas = group['beta'][:]
            gammas = group['gamma'][:]
        else:
            alphas = np.ones(len(trajectories))
            betas = np.zeros(len(trajectories))
            gammas = np.zeros(len(trajectories))
    
    # Limit samples if requested
    if num_samples is not None:
        trajectories = trajectories[:num_samples]
        alphas = alphas[:num_samples]
        betas = betas[:num_samples]
        gammas = gammas[:num_samples]
    
    print(f"Loaded {len(trajectories)} test samples")
    print(f"Trajectory shape: {trajectories.shape}")
    print(f"Parameter ranges - alpha: [{alphas.min():.3f}, {alphas.max():.3f}], "
          f"beta: [{betas.min():.3f}, {betas.max():.3f}], "
          f"gamma: [{gammas.min():.3f}, {gammas.max():.3f}]")
    
    return trajectories, time_points, alphas, betas, gammas


def test_operator_splitting(model1, model2, trajectories, time_points, 
                          splitting_method='strang', refinement_factor=1):
    """Test operator splitting with two models."""
    model1.eval()
    model2.eval()
    
    dt = time_points[1] - time_points[0]
    nt = len(time_points) - 1
    small_dt = dt / refinement_factor
    
    errors = []
    predictions = []
    
    print(f"\nTesting with {splitting_method} splitting, refinement={refinement_factor}")
    print(f"Time step: dt={dt:.4f}, refined dt={small_dt:.4f}")
    
    # Initial condition
    u0 = torch.tensor(trajectories[:, 0], dtype=torch.float32).unsqueeze(1)
    u_pred = u0.clone()
    
    # Store trajectory - initialize with initial condition
    trajectory_pred = [u_pred.squeeze().detach().numpy()]
    
    # Time stepping
    with torch.no_grad():
        for t_idx in range(nt):
            for _ in range(refinement_factor):
                if splitting_method == 'strang':
                    # Strang splitting: half step op1, full step op2, half step op1
                    t_span_half = torch.tensor([0, small_dt/2], device=u_pred.device)
                    t_span_full = torch.tensor([0, small_dt], device=u_pred.device)
                    _, solution = model1(u_pred, t_span=t_span_half)
                    u_pred = solution[-1]
                    _, solution = model2(u_pred, t_span=t_span_full)
                    u_pred = solution[-1]
                    _, solution = model1(u_pred, t_span=t_span_half)
                    u_pred = solution[-1]
                else:  # lie
                    # Lie splitting: full step op1, full step op2
                    t_span_full = torch.tensor([0, small_dt], device=u_pred.device)
                    _, solution = model1(u_pred, t_span=t_span_full)
                    u_pred = solution[-1]
                    _, solution = model2(u_pred, t_span=t_span_full)
                    u_pred = solution[-1]
            
        # Store prediction at each time step
            trajectory_pred.append(u_pred.squeeze().detach().numpy())
        
    # Convert trajectory to numpy array
    trajectory_pred = np.array(trajectory_pred)
    x = rearrange(trajectory_pred, 't b h -> b (t h)')
    y = rearrange(trajectories, 'b t h -> b (t h)')

    rel_error = np.linalg.norm(x - y, axis=1) / np.linalg.norm(y, axis=1)
    
    return rel_error, rearrange(trajectory_pred, 't b h -> b t h')


def plot_results(trajectories, predictions, errors, save_path, num_plots=5):
    """Plot prediction results with enhanced 4-panel visualization."""
    num_plots = min(num_plots, len(trajectories))
    
    # Reshape data if needed - trajectories should be (batch, time, space)
    if len(trajectories.shape) == 2:
        # Reshape from (batch, time*space) back to (batch, time, space)
        nt_steps = predictions.shape[1] // trajectories.shape[-1] + 1  # +1 for initial condition
        nx = trajectories.shape[-1] // nt_steps
        trajectories = trajectories.reshape(trajectories.shape[0], nt_steps, nx)
    
    # Reshape predictions similarly if needed
    if len(predictions.shape) == 2:
        predictions = predictions.reshape(predictions.shape[0], -1, predictions.shape[-1] // trajectories.shape[-1])
    
    for idx in range(num_plots):
        # Get ground truth and prediction for this sample
        ground_truth = trajectories[idx]
        result = predictions[idx]
        
        # Enhanced visualization with 4 panels
        plt.figure(figsize=(12, 10))
        
        # Sample every 10 time steps for visualization
        time_indices = np.arange(0, len(result), max(1, len(result)//10))
        colors = plt.cm.viridis(np.linspace(0, 1, len(time_indices)))
        
        # 1. Ground Truth trajectories every 10 time steps
        plt.subplot(2, 2, 1)
        for i, t_idx in enumerate(time_indices):
            if t_idx < len(ground_truth):
                plt.plot(ground_truth[t_idx, :], '-', color=colors[i], 
                        linewidth=1.5, alpha=0.8)
        
        plt.title("Ground Truth Evolution\\n(Sampled Time Steps)")
        plt.xlabel("Space")
        plt.ylabel("u")
        plt.grid(True, alpha=0.3)
        # Create a colorbar for the time evolution
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=len(time_indices)-1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca(), label='Time Index', shrink=0.8)
        
        # 2. Neural predictions every 10 time steps  
        plt.subplot(2, 2, 2)
        for i, t_idx in enumerate(time_indices):
            if t_idx < len(result):
                plt.plot(result[t_idx], '-', color=colors[i], 
                        linewidth=1.5, alpha=0.8)
        
        plt.title("Neural Predictions\\n(Sampled Time Steps)")
        plt.xlabel("Space")
        plt.ylabel("u")
        plt.grid(True, alpha=0.3)
        # Create a colorbar for the time evolution
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=len(time_indices)-1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca(), label='Time Index', shrink=0.8)
        
        # 3. Spatiotemporal evolution comparison
        plt.subplot(2, 2, 3)
        plt.imshow(ground_truth.T, aspect='auto', origin='lower', cmap='plasma')
        plt.title("Ground Truth\\nSpatiotemporal")
        plt.xlabel("Time")
        plt.ylabel("Space")
        plt.colorbar(shrink=0.8)
        
        # 4. Error evolution and final comparison
        plt.subplot(2, 2, 4)
        # Calculate error at each time step
        time_errors = []
        min_len = min(len(ground_truth), len(result))
        for i in range(min_len):
            error = np.linalg.norm(ground_truth[i, :] - result[i]) / (np.linalg.norm(ground_truth[i, :]) + 1e-8)
            time_errors.append(error)
        
        time_errors = np.array(time_errors)
        
        # Plot error on primary axis
        ax1 = plt.gca()
        line1 = ax1.semilogy(time_errors, 'r-', linewidth=2, label='Relative L2 Error')
        ax1.set_xlabel("Time Step")
        ax1.set_ylabel("Relative L2 Error", color='r')
        ax1.tick_params(axis='y', labelcolor='r')
        ax1.grid(True, alpha=0.3)
        
        # Add final comparison as secondary axis
        ax2 = ax1.twinx()
        ax2.plot(ground_truth[-1, :], 'b-', label='GT Final', linewidth=2, alpha=0.7)
        ax2.plot(result[-1], 'r--', label='Pred Final', linewidth=2, alpha=0.7)
        ax2.set_ylabel("Final State", color='b')
        ax2.tick_params(axis='y', labelcolor='b')
        
        ax1.set_title("Error Evolution &\\nFinal Comparison")
        
        # Add legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
        
        plt.tight_layout()
        plt.suptitle(f"Neural Operator Splitting Results - Sample {idx+1}\\nRelative L2 Error: {errors[idx]:.6f}", y=1.02)
        
        # Save individual figure
        sample_save_path = save_path.replace('.png', f'_sample_{idx+1:03d}.png')
        plt.savefig(sample_save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Print error statistics for this sample
        final_error = np.linalg.norm(ground_truth[-1, :] - result[-1]) / (np.linalg.norm(ground_truth[-1, :]) + 1e-8)
        print(f"Sample {idx+1} - Final Relative L2 Error: {final_error:.6f}")
        print(f"Sample {idx+1} - Max relative error over time: {time_errors.max():.6f}")
        print(f"Sample {idx+1} - Mean relative error over time: {time_errors.mean():.6f}")
    
    print(f"Enhanced plots saved with pattern: {save_path.replace('.png', '_sample_*.png')}")


def main():
    parser = argparse.ArgumentParser(description='Test Neural ODE Operator Splitting')
    
    # Model arguments
    parser.add_argument('--ckpt1', type=str, required=True,
                        help='Path to first model checkpoint')
    parser.add_argument('--ckpt2', type=str, required=True,
                        help='Path to second model checkpoint')
    
    # Data arguments
    parser.add_argument('--test_dataset', type=str, required=True,
                        help='Path to test HDF5 file')
    parser.add_argument('--num_samples', type=int, default=32,
                        help='Number of test samples')
    
    # Splitting settings
    parser.add_argument('--method', type=str, default='rk4',
                        help='ODE solver method (not used in simple forward)')
    parser.add_argument('--splitting_method', type=str, default='strang',
                        choices=['lie', 'strang'],
                        help='Operator splitting method')
    parser.add_argument('--refinement_factor', type=int, default=1,
                        help='Time refinement factor')
    
    # Names for display
    parser.add_argument('--op1_name', type=str, default='Operator1',
                        help='Name of first operator')
    parser.add_argument('--op2_name', type=str, default='Operator2',
                        help='Name of second operator')
    
    # Output settings
    parser.add_argument('--save_dir', type=str, default='./test_results',
                        help='Directory to save results')
    parser.add_argument('--num_plots', type=int, default=5,
                        help='Number of samples to plot')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load test data
    trajectories, time_points, alphas, betas, gammas = load_test_data(
        args.test_dataset, args.num_samples
    )
    
    # Create models
    nx = trajectories.shape[2]
    model1 = AdvectionNeuralODE(nx=nx, L=16.0, hidden_dim=32, n_layers=2)
    model2 = AdvectionNeuralODE(nx=nx, L=16.0, hidden_dim=32, n_layers=2)
    
    # Load checkpoints
    print(f"\nLoading checkpoint 1: {args.ckpt1}")
    model1.load_state_dict(torch.load(args.ckpt1, map_location='cpu'))
    
    print(f"Loading checkpoint 2: {args.ckpt2}")
    model2.load_state_dict(torch.load(args.ckpt2, map_location='cpu'))
    
    # Run test
    errors, predictions = test_operator_splitting(
        model1, model2, trajectories, time_points,
        args.splitting_method, args.refinement_factor
    )
    
    # Print statistics
    print("\n" + "="*60)
    print(f"Summary for {args.op1_name} + {args.op2_name} with {args.splitting_method} splitting")
    print("="*60)
    print(f"Mean relative error: {np.mean(errors):.6f}")
    print(f"Std relative error: {np.std(errors):.6f}")
    print(f"Max relative error: {np.max(errors):.6f}")
    print(f"Min relative error: {np.min(errors):.6f}")
    print("="*60)
    
    # Save results
    results_file = os.path.join(args.save_dir, 'test_results.npz')
    np.savez(results_file,
             errors=errors,
             predictions=predictions,  # Now contains full trajectories
             ground_truth=trajectories,  # Save full ground truth trajectories
             time_points=time_points,
             alphas=alphas,
             betas=betas,
             gammas=gammas,
             splitting_method=args.splitting_method,
             refinement_factor=args.refinement_factor)
    print(f"\nResults saved to: {results_file}")
    
    # Plot results
    plot_path = os.path.join(args.save_dir, 
                            f'{args.splitting_method}_r{args.refinement_factor}_predictions.png')
    plot_results(trajectories, predictions, errors, plot_path, args.num_plots)


if __name__ == "__main__":
    main()
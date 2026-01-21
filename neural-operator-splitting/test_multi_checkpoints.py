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
import json
import glob

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


def load_checkpoints_from_dir(model_dir):
    """Load all checkpoint files from a model directory."""
    checkpoint_files = glob.glob(os.path.join(model_dir, "checkpoint_epoch_*.pth"))
    checkpoint_files.sort(key=lambda x: int(x.split('_epoch_')[1].split('.pth')[0]))
    
    checkpoint_info_path = os.path.join(model_dir, "checkpoint_info.json")
    checkpoint_info = []
    
    if os.path.exists(checkpoint_info_path):
        with open(checkpoint_info_path, 'r') as f:
            checkpoint_info = json.load(f)
    
    print(f"Found {len(checkpoint_files)} checkpoints in {model_dir}")
    return checkpoint_files, checkpoint_info


def load_model_from_checkpoint(checkpoint_path, nx=256):
    """Load model from checkpoint file."""
    model = AdvectionNeuralODE(nx=nx, L=16.0, hidden_dim=32, n_layers=2)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 0)
        train_loss = checkpoint.get('train_loss', 0.0)
        val_loss = checkpoint.get('val_loss', 0.0)
    else:
        # Old format - just state dict
        model.load_state_dict(checkpoint)
        epoch = 0
        train_loss = 0.0
        val_loss = 0.0
    
    return model, epoch, train_loss, val_loss


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

    # Calculate rollout error (full trajectory)
    rollout_error = np.linalg.norm(x - y, axis=1) / np.linalg.norm(y, axis=1)
    
    # Calculate next step prediction error (first time step only)
    pred_first_step = trajectory_pred[1]  # First prediction at t=1
    true_first_step = trajectories[:, 1]  # True value at t=1
    next_step_error = np.linalg.norm(pred_first_step - true_first_step, axis=1) / np.linalg.norm(true_first_step, axis=1)
    
    return rollout_error, next_step_error, rearrange(trajectory_pred, 't b h -> b t h')


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


def test_checkpoint_evolution(model1_dir, model2_dir, trajectories, time_points, 
                            splitting_method='strang', refinement_factor=1):
    """Test operator splitting with models from same epochs only."""
    
    # Load checkpoints from both directories
    ckpts1, info1 = load_checkpoints_from_dir(model1_dir)
    ckpts2, info2 = load_checkpoints_from_dir(model2_dir)
    
    # Extract epochs for matching
    def get_epoch_from_path(path):
        return int(path.split('_epoch_')[1].split('.pth')[0])
    
    epochs1 = {get_epoch_from_path(ckpt): ckpt for ckpt in ckpts1}
    epochs2 = {get_epoch_from_path(ckpt): ckpt for ckpt in ckpts2}
    
    # Find common epochs
    common_epochs = sorted(set(epochs1.keys()) & set(epochs2.keys()))
    
    results = []
    nx = trajectories.shape[2]
    
    print(f"\nTesting checkpoint evolution with {len(common_epochs)} same-epoch pairs")
    print(f"Common epochs: {common_epochs}")
    
    for i, epoch in enumerate(common_epochs):
        ckpt1 = epochs1[epoch]
        ckpt2 = epochs2[epoch]
        
        print(f"\nTesting checkpoint pair {i+1}/{len(common_epochs)} (epoch {epoch})")
        
        # Load models
        model1, epoch1, train_loss1, val_loss1 = load_model_from_checkpoint(ckpt1, nx)
        model2, epoch2, train_loss2, val_loss2 = load_model_from_checkpoint(ckpt2, nx)
        
        # Test operator splitting
        rollout_errors, next_step_errors, predictions = test_operator_splitting(
            model1, model2, trajectories, time_points,
            splitting_method, refinement_factor
        )
        
        # Store results
        result = {
            'model1_epoch': epoch1,
            'model2_epoch': epoch2,
            'model1_checkpoint': ckpt1,
            'model2_checkpoint': ckpt2,
            'model1_train_loss': train_loss1,
            'model1_val_loss': val_loss1,
            'model2_train_loss': train_loss2,
            'model2_val_loss': val_loss2,
            'rollout_errors': rollout_errors,
            'next_step_errors': next_step_errors,
            'mean_rollout_error': np.mean(rollout_errors),
            'std_rollout_error': np.std(rollout_errors),
            'max_rollout_error': np.max(rollout_errors),
            'min_rollout_error': np.min(rollout_errors),
            'mean_next_step_error': np.mean(next_step_errors),
            'std_next_step_error': np.std(next_step_errors),
            'max_next_step_error': np.max(next_step_errors),
            'min_next_step_error': np.min(next_step_errors),
            # Keep legacy fields for backward compatibility
            'test_errors': rollout_errors,
            'mean_test_error': np.mean(rollout_errors),
            'std_test_error': np.std(rollout_errors),
            'max_test_error': np.max(rollout_errors),
            'min_test_error': np.min(rollout_errors)
        }
        
        results.append(result)
        
        print(f"Epoch: {epoch1}, Rollout Error: {np.mean(rollout_errors):.6f} ± {np.std(rollout_errors):.6f}, "
              f"Next Step Error: {np.mean(next_step_errors):.6f} ± {np.std(next_step_errors):.6f}")
    
    return results


def plot_checkpoint_evolution_results(results, save_dir):
    """Plot test error evolution across different checkpoint combinations."""
    
    # Extract data for plotting (simplified for same-epoch pairs)
    epochs = sorted([r['model1_epoch'] for r in results])
    rollout_errors = [r['mean_rollout_error'] for r in results]
    rollout_std_errors = [r['std_rollout_error'] for r in results]
    next_step_errors = [r['mean_next_step_error'] for r in results]
    next_step_std_errors = [r['std_next_step_error'] for r in results]
    train_losses1 = [r['model1_train_loss'] for r in results]
    train_losses2 = [r['model2_train_loss'] for r in results]
    
    # Create expanded plot for both error types
    plt.figure(figsize=(18, 12))
    
    # Rollout error evolution plot
    plt.subplot(3, 3, 1)
    plt.errorbar(epochs, rollout_errors, yerr=rollout_std_errors, marker='o', linewidth=2, markersize=8, capsize=5, label='Rollout Error', color='blue')
    plt.xlabel('Training Epoch')
    plt.ylabel('Rollout Error')
    plt.title('Rollout Error vs Training Progress\n(Full Trajectory)')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.legend()

    # Next step error evolution plot  
    plt.subplot(3, 3, 2)
    plt.errorbar(epochs, next_step_errors, yerr=next_step_std_errors, marker='s', linewidth=2, markersize=8, capsize=5, label='Next Step Error', color='red')
    plt.xlabel('Training Epoch')
    plt.ylabel('Next Step Error')
    plt.title('Next Step Error vs Training Progress\n(First Prediction Only)')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.legend()

    # Combined error comparison
    plt.subplot(3, 3, 3)
    plt.errorbar(epochs, rollout_errors, yerr=rollout_std_errors, marker='o', linewidth=2, markersize=6, capsize=3, label='Rollout Error', color='blue', alpha=0.7)
    plt.errorbar(epochs, next_step_errors, yerr=next_step_std_errors, marker='s', linewidth=2, markersize=6, capsize=3, label='Next Step Error', color='red', alpha=0.7)
    plt.xlabel('Training Epoch')
    plt.ylabel('Error')
    plt.title('Error Comparison\n(Both Error Types)')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.legend()
    
    # Training loss comparison
    plt.subplot(3, 3, 4)
    plt.plot(epochs, train_losses1, 'o-', label='Model 1 (Advection)', linewidth=2)
    plt.plot(epochs, train_losses2, 's-', label='Model 2 (Diffusion)', linewidth=2)
    plt.xlabel('Training Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Rollout error distribution
    plt.subplot(3, 3, 5)
    plt.hist(rollout_errors, bins=min(20, len(rollout_errors)), alpha=0.7, edgecolor='black', color='blue', label='Rollout')
    plt.xlabel('Rollout Error')
    plt.ylabel('Frequency')
    plt.title('Rollout Error Distribution')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Next step error distribution
    plt.subplot(3, 3, 6)
    plt.hist(next_step_errors, bins=min(20, len(next_step_errors)), alpha=0.7, edgecolor='black', color='red', label='Next Step')
    plt.xlabel('Next Step Error')
    plt.ylabel('Frequency')
    plt.title('Next Step Error Distribution')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Error vs training loss correlation
    plt.subplot(3, 3, 7)
    avg_train_loss = [(l1 + l2) / 2 for l1, l2 in zip(train_losses1, train_losses2)]
    plt.scatter(avg_train_loss, rollout_errors, alpha=0.7, s=60, color='blue', label='Rollout')
    plt.scatter(avg_train_loss, next_step_errors, alpha=0.7, s=60, color='red', marker='s', label='Next Step')
    plt.xlabel('Average Training Loss')
    plt.ylabel('Error')
    plt.title('Error vs Training Loss')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    
    # Detailed rollout error statistics
    plt.subplot(3, 3, 8)
    min_rollout_errors = [r['min_rollout_error'] for r in results]
    max_rollout_errors = [r['max_rollout_error'] for r in results]
    
    plt.fill_between(epochs, min_rollout_errors, max_rollout_errors, alpha=0.3, label='Min-Max Range', color='blue')
    plt.plot(epochs, rollout_errors, 'o-', label='Mean Rollout Error', linewidth=2, markersize=6, color='blue')
    plt.xlabel('Training Epoch')
    plt.ylabel('Rollout Error')
    plt.title('Rollout Error Statistics Range')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Summary statistics text
    plt.subplot(3, 3, 9)
    plt.axis('off')
    stats_text = f"""
    Checkpoint Evolution Summary
    ════════════════════════════
    Total epochs tested: {len(epochs)}
    Epoch range: {min(epochs)} - {max(epochs)}
    
    ROLLOUT ERROR:
    Best: {min(rollout_errors):.6f}
    Worst: {max(rollout_errors):.6f}
    Mean: {np.mean(rollout_errors):.6f}
    Std: {np.std(rollout_errors):.6f}
    Best epoch: {epochs[np.argmin(rollout_errors)]}
    
    NEXT STEP ERROR:
    Best: {min(next_step_errors):.6f}
    Worst: {max(next_step_errors):.6f}
    Mean: {np.mean(next_step_errors):.6f}
    Std: {np.std(next_step_errors):.6f}
    Best epoch: {epochs[np.argmin(next_step_errors)]}
    """
    plt.text(0.1, 0.5, stats_text, fontsize=11, fontfamily='monospace', 
             verticalalignment='center', transform=plt.gca().transAxes)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(save_dir, 'checkpoint_evolution_analysis.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Checkpoint evolution analysis saved to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description='Test Neural ODE Operator Splitting Across Checkpoints')
    
    # Model arguments - now supporting directories for checkpoint evolution
    parser.add_argument('--model1_dir', type=str, 
                        help='Directory containing first model checkpoints')
    parser.add_argument('--model2_dir', type=str,
                        help='Directory containing second model checkpoints')
    
    # Legacy support for single checkpoints
    parser.add_argument('--ckpt1', type=str,
                        help='Path to first model checkpoint (legacy)')
    parser.add_argument('--ckpt2', type=str,
                        help='Path to second model checkpoint (legacy)')
    
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
    
    # Check if we're doing checkpoint evolution analysis or single checkpoint test
    if args.model1_dir and args.model2_dir:
        print("Running checkpoint evolution analysis...")
        
        # Run checkpoint evolution test
        results = test_checkpoint_evolution(
            args.model1_dir, args.model2_dir, trajectories, time_points,
            args.splitting_method, args.refinement_factor
        )
        
        # Save detailed results
        results_file = os.path.join(args.save_dir, 'checkpoint_evolution_results.json')
        # Convert numpy arrays to lists for JSON serialization
        for result in results:
            result['test_errors'] = result['test_errors'].tolist()  # Legacy field
            result['rollout_errors'] = result['rollout_errors'].tolist()
            result['next_step_errors'] = result['next_step_errors'].tolist()
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to: {results_file}")
        
        # Plot evolution analysis
        plot_checkpoint_evolution_results(results, args.save_dir)
        
        # Print summary statistics
        print("\n" + "="*80)
        print("CHECKPOINT EVOLUTION ANALYSIS SUMMARY")
        print("="*80)
        all_rollout_errors = [r['mean_rollout_error'] for r in results]
        all_next_step_errors = [r['mean_next_step_error'] for r in results]
        
        print("ROLLOUT ERROR (full trajectory):")
        print(f"  Best combination error: {min(all_rollout_errors):.6f}")
        print(f"  Worst combination error: {max(all_rollout_errors):.6f}")
        print(f"  Average across all combinations: {np.mean(all_rollout_errors):.6f} ± {np.std(all_rollout_errors):.6f}")
        
        print("\nNEXT STEP ERROR (first prediction):")
        print(f"  Best combination error: {min(all_next_step_errors):.6f}")
        print(f"  Worst combination error: {max(all_next_step_errors):.6f}")
        print(f"  Average across all combinations: {np.mean(all_next_step_errors):.6f} ± {np.std(all_next_step_errors):.6f}")
        
        # Find best and worst combinations for both error types
        best_rollout_idx = np.argmin(all_rollout_errors)
        worst_rollout_idx = np.argmax(all_rollout_errors)
        best_next_step_idx = np.argmin(all_next_step_errors)
        worst_next_step_idx = np.argmax(all_next_step_errors)
        
        print(f"\nBest rollout combination: Model1 Epoch {results[best_rollout_idx]['model1_epoch']}, "
              f"Model2 Epoch {results[best_rollout_idx]['model2_epoch']} -> Rollout Error: {results[best_rollout_idx]['mean_rollout_error']:.6f}")
        print(f"Best next step combination: Model1 Epoch {results[best_next_step_idx]['model1_epoch']}, "
              f"Model2 Epoch {results[best_next_step_idx]['model2_epoch']} -> Next Step Error: {results[best_next_step_idx]['mean_next_step_error']:.6f}")
        print("="*80)
        
    elif args.ckpt1 and args.ckpt2:
        print("Running single checkpoint test (legacy mode)...")
        
        # Legacy single checkpoint testing
        nx = trajectories.shape[2]
        model1 = AdvectionNeuralODE(nx=nx, L=16.0, hidden_dim=32, n_layers=2)
        model2 = AdvectionNeuralODE(nx=nx, L=16.0, hidden_dim=32, n_layers=2)
        
        # Load checkpoints
        print(f"\nLoading checkpoint 1: {args.ckpt1}")
        model1.load_state_dict(torch.load(args.ckpt1, map_location='cpu')["model_state_dict"])
        
        print(f"Loading checkpoint 2: {args.ckpt2}")
        model2.load_state_dict(torch.load(args.ckpt2, map_location='cpu')["model_state_dict"])
        
        # Run test
        rollout_errors, next_step_errors, predictions = test_operator_splitting(
            model1, model2, trajectories, time_points,
            args.splitting_method, args.refinement_factor
        )
        
        # Print statistics
        print("\n" + "="*60)
        print(f"Summary for {args.op1_name} + {args.op2_name} with {args.splitting_method} splitting")
        print("="*60)
        print("ROLLOUT ERROR (full trajectory):")
        print(f"  Mean relative error: {np.mean(rollout_errors):.6f}")
        print(f"  Std relative error: {np.std(rollout_errors):.6f}")
        print(f"  Max relative error: {np.max(rollout_errors):.6f}")
        print(f"  Min relative error: {np.min(rollout_errors):.6f}")
        print("")
        print("NEXT STEP ERROR (first prediction):")
        print(f"  Mean relative error: {np.mean(next_step_errors):.6f}")
        print(f"  Std relative error: {np.std(next_step_errors):.6f}")
        print(f"  Max relative error: {np.max(next_step_errors):.6f}")
        print(f"  Min relative error: {np.min(next_step_errors):.6f}")
        print("="*60)
        
        # Save results
        results_file = os.path.join(args.save_dir, 'test_results.npz')
        np.savez(results_file,
                 rollout_errors=rollout_errors,
                 next_step_errors=next_step_errors,
                 errors=rollout_errors,  # Keep legacy field for backward compatibility
                 predictions=predictions,
                 ground_truth=trajectories,
                 time_points=time_points,
                 alphas=alphas,
                 betas=betas,
                 gammas=gammas,
                 splitting_method=args.splitting_method,
                 refinement_factor=args.refinement_factor)
        print(f"\nResults saved to: {results_file}")
        
        # Plot results (using rollout errors for compatibility)
        plot_path = os.path.join(args.save_dir, 
                                f'{args.splitting_method}_r{args.refinement_factor}_predictions.png')
        plot_results(trajectories, predictions, rollout_errors, plot_path, args.num_plots)
        
    else:
        print("Error: Must specify either --model1_dir and --model2_dir for checkpoint evolution,")
        print("       or --ckpt1 and --ckpt2 for single checkpoint testing.")


if __name__ == "__main__":
    main()
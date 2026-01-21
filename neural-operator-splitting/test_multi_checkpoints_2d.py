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

from neural_ode_operators2d import WrapperNeuralODE

# Set default device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def load_test_data(test_file, num_samples=None, n_pred=None):
    """Load test data from HDF5 file."""
    with h5py.File(test_file, 'r') as f:
        # Find the correct group and key for 2D data
        trajectories = f['navier_stokes'][:]
        
        # Generate time points like in training
        #time_points = np.linspace(0, 2.0, trajectories.shape[1])
        time_points = np.linspace(0, 4.0, trajectories.shape[1])
    
    # Limit prediction horizon if requested
    if n_pred is not None:
        trajectories = trajectories[:, :n_pred]
        time_points = time_points[:n_pred]
    
    # Limit samples if requested
    if num_samples is not None:
        trajectories = trajectories[:num_samples]
    
    print(f"Loaded {len(trajectories)} test samples")
    print(f"Trajectory shape: {trajectories.shape}")
    
    return trajectories, time_points


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


def load_model_from_checkpoint(checkpoint_path, input_shape=(128, 128)):
    """Load model from checkpoint file."""
    model = WrapperNeuralODE(input_shape=input_shape, L=16.0, hidden_dim=128, n_layers=4, padding_mode='circular')
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
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
    
    # Move model to device
    model = model.to(device)
    
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
    
    # Initial condition - for 2D data, no need to unsqueeze channel dimension
    u0 = torch.tensor(trajectories[:, 0], dtype=torch.float32, device=device)
    u_pred = u0.clone()
    
    # Store trajectory - initialize with initial condition
    trajectory_pred = [u_pred.detach().cpu().numpy()]
    
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
                elif splitting_method == 'lie':  # lie
                    # Lie splitting: full step op1, full step op2
                    t_span_full = torch.tensor([0, small_dt], device=u_pred.device)
                    _, solution = model1(u_pred, t_span=t_span_full)
                    u_pred = solution[-1]
                    _, solution = model2(u_pred, t_span=t_span_full)
                    u_pred = solution[-1]
                elif splitting_method == 'model1_only':
                    # Only use model1 for the full step
                    t_span_full = torch.tensor([0, small_dt], device=u_pred.device)
                    _, solution = model1(u_pred, t_span=t_span_full)
                    u_pred = solution[-1]
                elif splitting_method == 'model2_only':
                    # Only use model2 for the full step
                    t_span_full = torch.tensor([0, small_dt], device=u_pred.device)
                    _, solution = model2(u_pred, t_span=t_span_full)
                    u_pred = solution[-1]
            
        # Store prediction at each time step
            trajectory_pred.append(u_pred.detach().cpu().numpy())
        
    # Convert trajectory to numpy array
    trajectory_pred = np.array(trajectory_pred)
    x = rearrange(trajectory_pred, 't b h w -> b (t h w)')
    y = rearrange(trajectories, 'b t h w -> b (t h w)')

    # Calculate rollout error (full trajectory)
    rollout_error = np.linalg.norm(x - y, axis=1) / np.linalg.norm(y, axis=1)
    
    # Calculate next step prediction error (first time step only)
    pred_first_step = trajectory_pred[1]  # First prediction at t=1
    true_first_step = trajectories[:, 1]  # True value at t=1
    next_step_error = np.linalg.norm((pred_first_step - true_first_step).reshape(pred_first_step.shape[0], -1), axis=1) / np.linalg.norm(true_first_step.reshape(true_first_step.shape[0], -1), axis=1)
    
    return rollout_error, next_step_error, rearrange(trajectory_pred, 't b h w -> b t h w')


def plot_results(trajectories, predictions, errors, save_path, num_plots=5, num_snapshots=8):
    """Plot prediction results with clean layout: separate plots for pred, gt, errors, and time series."""
    num_plots = min(num_plots, len(trajectories))
    
    print(f"Trajectories shape: {trajectories.shape}")
    print(f"Predictions shape: {predictions.shape}")
    
    for idx in range(num_plots):
        ground_truth = trajectories[idx]  # Shape: (time, height, width)
        result = predictions[idx]         # Shape: (time, height, width)
        
        # Create 4 separate figures for cleaner visualization
        time_indices = np.linspace(0, min(len(result)-1, 99), num_snapshots, dtype=int)
        
        # Calculate time errors
        time_errors = []
        min_len = min(len(ground_truth), len(result))
        for i in range(min_len):
            error = np.linalg.norm(ground_truth[i] - result[i]) / (np.linalg.norm(ground_truth[i]) + 1e-8)
            time_errors.append(error)
        time_errors = np.array(time_errors)
        
        # Main comparison figure with GT, Predictions, and Errors
        fig = plt.figure(figsize=(20, 12))
        
        # Calculate global min/max for consistent GT and prediction colorbars
        gt_vmin, gt_vmax = ground_truth[time_indices].min(), ground_truth[time_indices].max()
        pred_vmin, pred_vmax = result[time_indices].min(), result[time_indices].max()
        data_vmin, data_vmax = min(gt_vmin, pred_vmin), max(gt_vmax, pred_vmax)
        
        # Calculate error maps and their range
        error_maps = [np.abs(ground_truth[t_idx] - result[t_idx]) for t_idx in time_indices]
        error_vmin, error_vmax = 0, max([em.max() for em in error_maps])
        
        # Row 1: Ground Truth
        gt_ims = []
        for i, t_idx in enumerate(time_indices):
            ax = plt.subplot(3, num_snapshots, i + 1)
            im = ax.imshow(ground_truth[t_idx], cmap='viridis', origin='lower', 
                          vmin=data_vmin, vmax=data_vmax)
            ax.set_title(f"GT t={t_idx}", fontsize=11, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_ylabel('Ground Truth', fontsize=12, fontweight='bold')
            gt_ims.append(im)
        
        # Row 2: Predictions
        pred_ims = []
        for i, t_idx in enumerate(time_indices):
            ax = plt.subplot(3, num_snapshots, num_snapshots + i + 1)
            im = ax.imshow(result[t_idx], cmap='viridis', origin='lower',
                          vmin=data_vmin, vmax=data_vmax)
            ax.set_title(f"Pred t={t_idx}", fontsize=11, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_ylabel('Predictions', fontsize=12, fontweight='bold')
            pred_ims.append(im)
        
        # Row 3: Error Maps
        error_ims = []
        for i, t_idx in enumerate(time_indices):
            ax = plt.subplot(3, num_snapshots, 2*num_snapshots + i + 1)
            error_map = np.abs(ground_truth[t_idx] - result[t_idx])
            im = ax.imshow(error_map, cmap='hot', origin='lower',
                          vmin=error_vmin, vmax=error_vmax)
            ax.set_title(f"Error t={t_idx}\nL2: {time_errors[t_idx]:.4f}", fontsize=10, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_ylabel('Absolute Error', fontsize=12, fontweight='bold')
            error_ims.append(im)
        
        # Add colorbars
        # Colorbar for GT and Predictions (shared)
        cbar_ax1 = fig.add_axes([0.92, 0.55, 0.02, 0.35])  # [left, bottom, width, height]
        cbar1 = fig.colorbar(gt_ims[0], cax=cbar_ax1)
        cbar1.set_label('Field Value', fontsize=12, fontweight='bold')
        
        # Colorbar for Errors
        cbar_ax2 = fig.add_axes([0.92, 0.1, 0.02, 0.35])  # [left, bottom, width, height]
        cbar2 = fig.colorbar(error_ims[0], cax=cbar_ax2)
        cbar2.set_label('Absolute Error', fontsize=12, fontweight='bold')
        
        plt.suptitle(f"Neural Operator Splitting Comparison - Sample {idx+1}\nRollout Error: {errors[idx]:.6f}", 
                    fontsize=16, fontweight='bold', y=0.95)
        
        plt.tight_layout()
        plt.subplots_adjust(right=0.9)  # Make room for colorbars
        
        # Save the comparison figure
        comparison_save_path = save_path.replace('.png', f'_sample_{idx+1:03d}_comparison.png')
        plt.savefig(comparison_save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # 4. Time Series and Summary Figure
        plt.figure(figsize=(16, 8))
        
        # Error evolution plot
        plt.subplot(2, 1, 1)
        plt.semilogy(time_errors, 'r-', linewidth=3, label='Relative L2 Error', alpha=0.8)
        plt.fill_between(range(len(time_errors)), time_errors, alpha=0.3, color='red')
        for i, t_idx in enumerate(time_indices):
            plt.axvline(x=t_idx, color='blue', linestyle='--', alpha=0.7, linewidth=1)
            plt.plot(t_idx, time_errors[t_idx], 'bo', markersize=6, alpha=0.8)
        plt.xlabel("Time Step", fontsize=12)
        plt.ylabel("Relative L2 Error", fontsize=12)
        plt.title("Error Evolution Through Time", fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        
        # Statistics panel
        plt.subplot(2, 1, 2)
        plt.axis('off')
        stats_text = f"""Sample {idx+1} Statistics
============================
Final L2 Error: {time_errors[-1]:.6f}
Max Error: {time_errors.max():.6f}
Mean Error: {time_errors.mean():.6f}
Min Error: {time_errors.min():.6f}
Rollout Error: {errors[idx]:.6f}

Snapshots: {num_snapshots}
Data Shape: {ground_truth.shape}
Time Steps: {len(ground_truth)}
Time Indices: {time_indices.tolist()}
"""
        plt.text(0.1, 0.5, stats_text, fontsize=12, fontfamily='monospace', 
                 verticalalignment='center', transform=plt.gca().transAxes)
        
        plt.suptitle(f"Error Analysis and Summary - Sample {idx+1}", fontsize=16, fontweight='bold')
        summary_save_path = save_path.replace('.png', f'_sample_{idx+1:03d}_summary.png')
        plt.savefig(summary_save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Print error statistics for this sample
        final_error = np.linalg.norm(ground_truth[-1] - result[-1]) / (np.linalg.norm(ground_truth[-1]) + 1e-8)
        print(f"Sample {idx+1}: Final error = {final_error:.6f}, Rollout error = {errors[idx]:.6f}")
        
    print(f"All plots saved with prefix: {save_path}")
    print(f"Error statistics summary:")
    print(f"  Mean rollout error: {np.mean(errors):.6f}")
    print(f"  Std rollout error: {np.std(errors):.6f}")
    print(f"  Max rollout error: {np.max(errors):.6f}")
    print(f"  Min rollout error: {np.min(errors):.6f}")


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
    input_shape = trajectories.shape[2:]  # (height, width)
    
    print(f"\nTesting checkpoint evolution with {len(common_epochs)} same-epoch pairs")
    print(f"Common epochs: {common_epochs}")
    
    for i, epoch in enumerate(common_epochs):
        ckpt1 = epochs1[epoch]
        ckpt2 = epochs2[epoch]
        
        print(f"\nTesting checkpoint pair {i+1}/{len(common_epochs)} (epoch {epoch})")
        
        # Load models
        model1, epoch1, train_loss1, val_loss1 = load_model_from_checkpoint(ckpt1, input_shape)
        model2, epoch2, train_loss2, val_loss2 = load_model_from_checkpoint(ckpt2, input_shape)
        
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
    parser.add_argument('--n_pred', type=int, default=50,
                        help='Number of time steps to predict (limits prediction horizon)')
    
    # Splitting settings
    parser.add_argument('--method', type=str, default='rk4',
                        help='ODE solver method (not used in simple forward)')
    parser.add_argument('--splitting_method', type=str, default='strang',
                        choices=['lie', 'strang', 'model1_only', 'model2_only'],
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
    parser.add_argument('--num_snapshots', type=int, default=6,
                        help='Number of time snapshots to show in plots')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load test data
    trajectories, time_points = load_test_data(
        args.test_dataset, args.num_samples, args.n_pred
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
        # Convert numpy arrays and scalars to native Python types for JSON serialization
        for result in results:
            result['test_errors'] = result['test_errors'].tolist()  # Legacy field
            result['rollout_errors'] = result['rollout_errors'].tolist()
            result['next_step_errors'] = result['next_step_errors'].tolist()
            # Convert numpy scalars to Python native types
            for key in ['model1_train_loss', 'model1_val_loss', 'model2_train_loss', 'model2_val_loss',
                       'mean_rollout_error', 'std_rollout_error', 'max_rollout_error', 'min_rollout_error',
                       'mean_next_step_error', 'std_next_step_error', 'max_next_step_error', 'min_next_step_error',
                       'mean_test_error', 'std_test_error', 'max_test_error', 'min_test_error']:
                if key in result and hasattr(result[key], 'item'):
                    result[key] = result[key].item()
        
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
        input_shape = trajectories.shape[2:]  # (height, width)
        model1 = WrapperNeuralODE(input_shape=input_shape, L=16.0, hidden_dim=128, n_layers=4, padding_mode='circular')
        model2 = WrapperNeuralODE(input_shape=input_shape, L=16.0, hidden_dim=128, n_layers=4, padding_mode='circular')
        
        # Load checkpoints
        print(f"\nLoading checkpoint 1: {args.ckpt1}")
        model1.load_state_dict(torch.load(args.ckpt1, map_location=device)["model_state_dict"])
        model1 = model1.to(device)
        
        print(f"Loading checkpoint 2: {args.ckpt2}")
        model2.load_state_dict(torch.load(args.ckpt2, map_location=device)["model_state_dict"])
        model2 = model2.to(device)
        
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
                 splitting_method=args.splitting_method,
                 refinement_factor=args.refinement_factor)
        print(f"\nResults saved to: {results_file}")
        
        # Save individual sample predictions and ground truth
        print("\nSaving individual sample predictions and ground truth...")
        for i in range(len(trajectories)):
            sample_dir = os.path.join(args.save_dir, f'sample_{i+1:03d}')
            os.makedirs(sample_dir, exist_ok=True)
            
            # Save prediction and ground truth for this sample
            sample_file = os.path.join(sample_dir, f'{args.splitting_method}_r{args.refinement_factor}_sample_{i+1:03d}.npz')
            np.savez(sample_file,
                     prediction=predictions[i],
                     ground_truth=trajectories[i],
                     time_points=time_points,
                     rollout_error=rollout_errors[i],
                     next_step_error=next_step_errors[i],
                     splitting_method=args.splitting_method,
                     refinement_factor=args.refinement_factor,
                     method_name=f'{args.splitting_method}_r{args.refinement_factor}')
        
        print(f"Individual sample data saved to: {args.save_dir}/sample_XXX/")
        
        # Plot results (using rollout errors for compatibility)
        plot_path = os.path.join(args.save_dir, 
                                f'{args.splitting_method}_r{args.refinement_factor}_predictions.png')
        plot_results(trajectories, predictions, rollout_errors, plot_path, args.num_plots, args.num_snapshots)
        
    else:
        print("Error: Must specify either --model1_dir and --model2_dir for checkpoint evolution,")
        print("       or --ckpt1 and --ckpt2 for single checkpoint testing.")


if __name__ == "__main__":
    main()
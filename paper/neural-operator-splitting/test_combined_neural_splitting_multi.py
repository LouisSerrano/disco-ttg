"""
Multi-Sample Combined Neural Operator Splitting Test

Loads pre-trained models and tests Lie splitting on multiple samples.
Creates individual visualizations for each sample and a summary report.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import h5py
from typing import List, Tuple, Dict

# Add paths
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from neural_ode_operators import create_neural_operators
from neural_splitting_methods import NeuralOperatorSplitting


def load_trained_models(heat_model_path: str, dispersion_model_path: str, nx: int = 256, L: float = 16.0):
    """Load pre-trained heat and dispersion models."""
    # Create model architectures
    operators = create_neural_operators(nx, L, hidden_dim=32, n_layers=2, padding_mode="zeros")
    
    # Load trained weights
    operators['diffusion'].load_state_dict(torch.load(heat_model_path, map_location='cpu'))
    operators['advection'].load_state_dict(torch.load(dispersion_model_path, map_location='cpu'))
    
    # Set to eval mode
    operators['diffusion'].eval()
    operators['advection'].eval()
    
    return operators['advection'], operators['diffusion']


def load_test_data(hdf5_file: str, num_samples: int = 10):
    """Load test data from HDF5 file."""
    with h5py.File(hdf5_file, 'r') as f:
        if 'valid' in f:
            group = f['valid']
            trajectories = group['pde_250-256'][:num_samples]
            beta = group['beta'][:num_samples]
            gamma = group['gamma'][:num_samples]
        else:
            raise ValueError("No validation data found")
    
    time_points = np.linspace(0, 4.0, trajectories.shape[1])
    return trajectories, beta, gamma, time_points


def test_lie_splitting(advection_model, diffusion_model, u0, dt, nt, method='rk4'):
    """Test Lie splitting with loaded models."""
    # Create neural splitting
    neural_splitting = NeuralOperatorSplitting(advection_model, diffusion_model)
    
    # Run Lie splitting
    result = neural_splitting.lie_splitting(u0, dt, nt, method, save_intermediate=True)
    
    # Convert torch tensors to numpy arrays
    if isinstance(result[0], torch.Tensor):
        result = [r.cpu().numpy() for r in result]
    
    return np.array(result)


def process_single_sample(advection_model, diffusion_model, trajectories, beta, gamma, 
                         sample_idx, dt, nt, results_dir) -> Dict:
    """Process a single sample and create visualization."""
    print(f"Processing sample {sample_idx}...")
    
    # Test on specified sample
    u0 = trajectories[sample_idx, 0, :]  # Initial condition
    u0 = torch.tensor(u0, dtype=torch.float32).unsqueeze(0)  # Convert to tensor and add batch dim
    
    # Run neural splitting
    result = test_lie_splitting(advection_model, diffusion_model, u0, dt, nt)
    
    # Get ground truth from the loaded data
    ground_truth = trajectories[sample_idx, :len(result), :]  # Same sample, all time steps
    u0_numpy = trajectories[sample_idx, 0, :]  # Initial condition as numpy array
    
    # Get parameters for this sample
    beta_val = beta[sample_idx]
    gamma_val = gamma[sample_idx]
    
    # Create visualization
    plt.figure(figsize=(16, 8))
    
    # Sample every 10 time steps for visualization
    time_indices = np.arange(0, len(result), 10)
    colors = plt.cm.viridis(np.linspace(0, 1, len(time_indices)))
    
    # 1. Ground Truth trajectories every 10 time steps
    plt.subplot(1, 4, 1)
    for i, t_idx in enumerate(time_indices):
        plt.plot(ground_truth[t_idx, :], '-', color=colors[i], 
                linewidth=1.5, alpha=0.8)
    
    plt.title(f"Ground Truth Evolution\n(β={beta_val:.3f}, γ={gamma_val:.3f})")
    plt.xlabel("Space")
    plt.ylabel("u")
    plt.grid(True, alpha=0.3)
    # Create a colorbar for the time evolution
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=len(time_indices)-1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), label='Time Index', shrink=0.8)
    
    # 2. Neural predictions every 10 time steps  
    plt.subplot(1, 4, 2)
    for i, t_idx in enumerate(time_indices):
        plt.plot(result[t_idx], '-', color=colors[i], 
                linewidth=1.5, alpha=0.8)
    
    plt.title("Neural Predictions\n(Every 10 Steps)")
    plt.xlabel("Space")
    plt.ylabel("u")
    plt.grid(True, alpha=0.3)
    # Create a colorbar for the time evolution
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=len(time_indices)-1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), label='Time Index', shrink=0.8)
    
    # 3. Spatiotemporal evolution comparison
    plt.subplot(1, 4, 3)
    plt.imshow(ground_truth.T, aspect='auto', origin='lower', cmap='plasma')
    plt.title("Ground Truth\nSpatiotemporal")
    plt.xlabel("Time")
    plt.ylabel("Space")
    plt.colorbar(shrink=0.8)
    
    # 4. Error evolution and final comparison
    plt.subplot(1, 4, 4)
    errors = np.array([np.mean((ground_truth[i, :] - result[i])**2) 
                      for i in range(len(result))])
    
    # Plot error on top
    ax1 = plt.gca()
    line1 = ax1.semilogy(errors, 'r-', linewidth=2, label='MSE Error')
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("MSE", color='r')
    ax1.tick_params(axis='y', labelcolor='r')
    ax1.grid(True, alpha=0.3)
    
    # Add final comparison as inset
    ax2 = ax1.twinx()
    ax2.plot(ground_truth[-1, :], 'b-', label='GT Final', linewidth=2, alpha=0.7)
    ax2.plot(result[-1], 'r--', label='Pred Final', linewidth=2, alpha=0.7)
    ax2.set_ylabel("Final State", color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    
    ax1.set_title("Error Evolution &\nFinal Comparison")
    
    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
    
    plt.tight_layout()
    
    # Save figure
    filename = f"sample_{sample_idx:02d}_beta_{beta_val:.3f}_gamma_{gamma_val:.3f}.png"
    filepath = os.path.join(results_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Calculate error statistics
    final_error = np.mean((ground_truth[-1, :] - result[-1])**2)
    max_error = errors.max()
    mean_error = errors.mean()
    
    print(f"  Sample {sample_idx}: Final MSE = {final_error:.6f}, Mean MSE = {mean_error:.6f}")
    
    return {
        'sample_idx': sample_idx,
        'beta': beta_val,
        'gamma': gamma_val,
        'final_error': final_error,
        'max_error': max_error,
        'mean_error': mean_error,
        'filename': filename
    }


def create_summary_report(results: List[Dict], results_dir: str):
    """Create summary visualization and report."""
    
    # Extract data for summary
    sample_indices = [r['sample_idx'] for r in results]
    betas = [r['beta'] for r in results]
    gammas = [r['gamma'] for r in results]
    final_errors = [r['final_error'] for r in results]
    mean_errors = [r['mean_error'] for r in results]
    
    # Create summary plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Error vs Beta
    axes[0, 0].scatter(betas, final_errors, c='red', alpha=0.7, s=50)
    axes[0, 0].set_xlabel('β (Advection)')
    axes[0, 0].set_ylabel('Final MSE')
    axes[0, 0].set_title('Final Error vs Advection Parameter')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    # 2. Error vs Gamma
    axes[0, 1].scatter(gammas, final_errors, c='blue', alpha=0.7, s=50)
    axes[0, 1].set_xlabel('γ (Diffusion)')
    axes[0, 1].set_ylabel('Final MSE')
    axes[0, 1].set_title('Final Error vs Diffusion Parameter')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale('log')
    
    # 3. Parameter space with error coloring
    scatter = axes[1, 0].scatter(betas, gammas, c=final_errors, cmap='hot', s=60)
    axes[1, 0].set_xlabel('β (Advection)')
    axes[1, 0].set_ylabel('γ (Diffusion)')
    axes[1, 0].set_title('Error in Parameter Space')
    plt.colorbar(scatter, ax=axes[1, 0], label='Final MSE')
    
    # 4. Error distribution
    axes[1, 1].hist(np.log10(final_errors), bins=10, alpha=0.7, color='green')
    axes[1, 1].set_xlabel('log₁₀(Final MSE)')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Error Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save summary plot
    summary_plot_path = os.path.join(results_dir, "summary_analysis.png")
    plt.savefig(summary_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create text report
    report_path = os.path.join(results_dir, "summary_report.txt")
    with open(report_path, 'w') as f:
        f.write("Neural Operator Splitting Test Summary\n")
        f.write("=" * 40 + "\n\n")
        
        f.write(f"Total samples processed: {len(results)}\n\n")
        
        f.write("Error Statistics:\n")
        f.write(f"  Mean final MSE: {np.mean(final_errors):.6f}\n")
        f.write(f"  Std final MSE: {np.std(final_errors):.6f}\n")
        f.write(f"  Min final MSE: {np.min(final_errors):.6f}\n")
        f.write(f"  Max final MSE: {np.max(final_errors):.6f}\n\n")
        
        f.write("Parameter ranges:\n")
        f.write(f"  β (Advection): {np.min(betas):.3f} to {np.max(betas):.3f}\n")
        f.write(f"  γ (Diffusion): {np.min(gammas):.3f} to {np.max(gammas):.3f}\n\n")
        
        f.write("Individual Sample Results:\n")
        f.write("Sample  β      γ      Final MSE  Mean MSE   Filename\n")
        f.write("-" * 60 + "\n")
        for r in results:
            f.write(f"{r['sample_idx']:2d}     {r['beta']:.3f} {r['gamma']:.3f} "
                   f"{r['final_error']:.6f} {r['mean_error']:.6f} {r['filename']}\n")
    
    print(f"Summary report saved to {report_path}")
    print(f"Summary plots saved to {summary_plot_path}")


def main():
    print("Multi-Sample Combined Neural Operator Splitting Test")
    print("=" * 60)
    
    # Configuration
    num_samples = 10  # Number of samples to test
    
    # Model paths - use the parameter grid models
    heat_model_path = "test_results/parameter_grid/models/adv_1.0_nu_0.1_steps_1/diffusion_model.pth"
    dispersion_model_path = "test_results/parameter_grid/models/adv_1.0_nu_0.1_steps_1/advection_model.pth"
    
    # Test data path - same as single sample script
    test_data_path = "./datasets/lpsda/OP_HEAT_valid.h5"
    
    # Create results directory
    results_dir = "test_results_multi_sample"
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {results_dir}")
    
    # Check if models exist
    if not os.path.exists(heat_model_path):
        print(f"Heat model not found at {heat_model_path}")
        print("Please train models first or provide correct path")
        return
    
    if not os.path.exists(dispersion_model_path):
        print(f"Dispersion model not found at {dispersion_model_path}")
        print("Please train models first or provide correct path")
        return
    
    print("Loading pre-trained models...")
    advection_model, diffusion_model = load_trained_models(heat_model_path, dispersion_model_path)
    print("✓ Models loaded")
    
    print(f"Loading test data (first {num_samples} samples)...")
    trajectories, beta, gamma, time_points = load_test_data(test_data_path, num_samples=num_samples)
    print(f"✓ Loaded {len(trajectories)} test samples")
    
    # Test parameters
    dt = time_points[1] - time_points[0]
    nt = len(time_points) - 1
    
    print(f"Testing Lie splitting with dt={dt:.4f}, nt={nt}")
    print()
    
    # Process all samples
    results = []
    for sample_idx in range(len(trajectories)):
        result = process_single_sample(
            advection_model, diffusion_model, 
            trajectories, beta, gamma,
            sample_idx, dt, nt, results_dir
        )
        results.append(result)
    
    # Create summary report
    print("\nCreating summary report...")
    create_summary_report(results, results_dir)
    
    print(f"\n✓ All tests completed! Results saved in {results_dir}/")
    print(f"  - {len(results)} individual sample visualizations")
    print(f"  - summary_analysis.png: Parameter space analysis")
    print(f"  - summary_report.txt: Detailed numerical results")


if __name__ == "__main__":
    main()
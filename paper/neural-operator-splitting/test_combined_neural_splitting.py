"""
Simple Combined Neural Operator Splitting Test

Loads 2 pre-trained models (heat and dispersion) and tests Lie splitting.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import h5py

# Add paths
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from neural_ode_operators import create_neural_operators
from neural_splitting_methods import NeuralOperatorSplitting

def test_strang_splitting(advection_model, diffusion_model, u0, small_dt, small_nt, original_nt, method='rk4'):
    """Test Strang splitting with loaded models using refinement."""
    # Create neural splitting
    neural_splitting = NeuralOperatorSplitting(advection_model, diffusion_model)
    
    # Run Strang splitting with option 2: save at original spacing
    result = neural_splitting.strang_splitting(u0, small_dt, small_nt, method, 
                                              save_intermediate=True, 
                                              num_save_steps=original_nt)
    
    # Convert torch tensors to numpy arrays
    result = [r.cpu().numpy() if isinstance(r, torch.Tensor) else r for r in result]
    
    return np.array(result)


def load_trained_models(heat_model_path: str, dispersion_model_path: str, nx: int = 256, L: float = 16.0):
    """Load pre-trained heat and dispersion models."""
    # Create model architectures
    operators = create_neural_operators(nx, L, hidden_dim=32, n_layers=2, padding_mode="circular")
    
    # Load trained weights
    operators['diffusion'].load_state_dict(torch.load(heat_model_path, map_location='cpu'))
    operators['advection'].load_state_dict(torch.load(dispersion_model_path, map_location='cpu'))
    
    # Set to eval mode
    operators['diffusion'].eval()
    operators['advection'].eval()
    
    return operators['advection'], operators['diffusion']


def load_test_data(hdf5_file: str, num_samples: int = 5):
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


def main():
    print("Simple Combined Neural Operator Splitting Test")
    print("=" * 50)
    
    # Model paths (you'll need to provide actual paths)
    heat_model_path = "./models/heat_model.pth"
    dispersion_model_path = "./models/dispersion_model.pth" 
    
    # Test data path
    test_data_path = "./datasets/lpsda/OP_BG_valid.h5"
    
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
    
    print("Loading test data...")
    trajectories, beta, gamma, time_points = load_test_data(test_data_path, num_samples=32)
    print(f"✓ Loaded {len(trajectories)} test samples")
    
    # Test parameters
    dt = time_points[1] - time_points[0]
    nt = len(time_points) - 1
    
    print(f"Testing Lie splitting with dt={dt:.4f}, nt={nt}")
    
    for idx in range(8):
        # Test on first sample
        u0 = trajectories[idx, 0, :]  # Initial condition
        u0 = torch.tensor(u0, dtype=torch.float32).unsqueeze(0)  # Convert to tensor and add batch dim
        
        print("Running Lie splitting...")
        #result = test_lie_splitting(advection_model, diffusion_model, u0, dt, nt)
        result = test_strang_splitting(advection_model, diffusion_model, u0, dt, nt, nt, 'rk4')
        
        print(f"✓ Lie splitting completed. Result shape: {result.shape}")
        
        # Get ground truth from the loaded data
        ground_truth = trajectories[idx, :len(result), :]  # Same sample, all time steps
        u0_numpy = trajectories[idx, 0, :]  # Initial condition as numpy array
        
        # Enhanced visualization
        plt.figure(figsize=(8, 8))
        
        # Sample every 10 time steps for visualization
        time_indices = np.arange(0, len(result), 10)
        colors = plt.cm.viridis(np.linspace(0, 1, len(time_indices)))
        
        # 1. Ground Truth trajectories every 10 time steps
        plt.subplot(2, 2, 1)
        for i, t_idx in enumerate(time_indices):
            plt.plot(ground_truth[t_idx, :], '-', color=colors[i], 
                    label=f't={t_idx}', linewidth=1.5, alpha=0.8)
        
        plt.title("Ground Truth Evolution\n(Every 10 Steps)")
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
            plt.plot(result[t_idx], '-', color=colors[i], 
                    label=f't={t_idx}', linewidth=1.5, alpha=0.8)
        
        plt.title("Neural Predictions\n(Every 10 Steps)")
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
        plt.title("Ground Truth\nSpatiotemporal")
        plt.xlabel("Time")
        plt.ylabel("Space")
        plt.colorbar(shrink=0.8)
        
        # 4. Error evolution and final comparison
        plt.subplot(2, 2, 4)
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
        plt.savefig(f"comprehensive_lie_splitting_result_{idx}.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        # Print error statistics
        final_error = np.mean((ground_truth[-1, :] - result[-1])**2)
        print(f"Final MSE: {final_error:.6f}")
        print(f"Max error over time: {errors.max():.6f}")
        print(f"Mean error over time: {errors.mean():.6f}")
        


if __name__ == "__main__":
    main()
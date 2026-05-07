"""
Modular Combined Neural Operator Splitting Test

Loads 2 pre-trained models and tests Lie splitting on different scenarios.
Supports testing burgers+heat on OP_HE, dispersion+heat on OP_BG, etc.
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


def load_trained_models(scenario: str, model_dir: str = "./models", nx: int = 256, L: float = 16.0):
    """
    Load pre-trained models based on scenario.
    
    Args:
        scenario: Training scenario ('burgers_heat', 'dispersion_heat', etc.)
        model_dir: Directory containing trained models
        nx: Number of spatial grid points
        L: Domain length
        
    Returns:
        Tuple of (first_operator_model, second_operator_model)
    """
    # Define scenario configurations
    scenario_configs = {
        'burgers_heat': {
            'op1_name': 'burgers',
            'op2_name': 'heat',
            'op1_model_key': 'advection',
            'op2_model_key': 'diffusion'
        },
        'dispersion_heat': {
            'op1_name': 'dispersion',
            'op2_name': 'heat',
            'op1_model_key': 'advection',
            'op2_model_key': 'diffusion'
        },
        'burgers_disp': {
            'op1_name': 'burgers',
            'op2_name': 'dispersion',
            'op1_model_key': 'advection',
            'op2_model_key': 'diffusion'
        }
    }
    
    if scenario not in scenario_configs:
        raise ValueError(f"Unknown scenario '{scenario}'. Available: {list(scenario_configs.keys())}")
    
    config = scenario_configs[scenario]
    
    # Model paths
    op1_model_path = os.path.join(model_dir, f"{config['op1_name']}_model.pth")
    op2_model_path = os.path.join(model_dir, f"{config['op2_name']}_model.pth")
    
    # Check if models exist
    if not os.path.exists(op1_model_path):
        raise FileNotFoundError(f"{config['op1_name'].capitalize()} model not found at {op1_model_path}")
    
    if not os.path.exists(op2_model_path):
        raise FileNotFoundError(f"{config['op2_name'].capitalize()} model not found at {op2_model_path}")
    
    # Create model architectures
    operators = create_neural_operators(nx, L, hidden_dim=32, n_layers=2, padding_mode="circular")
    
    # Load trained weights
    operators[config['op2_model_key']].load_state_dict(torch.load(op2_model_path, map_location='cpu'))
    operators[config['op1_model_key']].load_state_dict(torch.load(op1_model_path, map_location='cpu'))
    
    # Set to eval mode
    operators[config['op2_model_key']].eval()
    operators[config['op1_model_key']].eval()
    
    return operators[config['op1_model_key']], operators[config['op2_model_key']]


def load_test_data(test_scenario: str, num_samples: int = 5):
    """
    Load test data from HDF5 file based on test scenario.
    
    Args:
        test_scenario: Test dataset ('OP_HE', 'OP_BG', etc.)
        num_samples: Number of samples to load
        
    Returns:
        Tuple of (trajectories, beta, gamma, time_points)
    """
    # Define test data file paths
    test_data_files = {
        'OP_HE': './datasets/lpsda/OP_HE_valid.h5',
        'OP_BG': './datasets/lpsda/OP_BG_valid.h5',
        'OP_ED': './datasets/lpsda/OP_ED_valid.h5'
    }
    
    if test_scenario not in test_data_files:
        raise ValueError(f"Unknown test scenario '{test_scenario}'. Available: {list(test_data_files.keys())}")
    
    hdf5_file = test_data_files[test_scenario]
    
    if not os.path.exists(hdf5_file):
        raise FileNotFoundError(f"Test data file not found: {hdf5_file}")
    
    with h5py.File(hdf5_file, 'r') as f:
        if 'valid' in f:
            group = f['valid']
            trajectories = group['pde_250-256'][:num_samples]
            beta = group['beta'][:num_samples]
            gamma = group['gamma'][:num_samples]
        else:
            raise ValueError(f"No validation data found in {hdf5_file}")
    
    time_points = np.linspace(0, 4.0, trajectories.shape[1])
    return trajectories, beta, gamma, time_points


def test_lie_splitting(advection_model, diffusion_model, u0, small_dt, small_nt, original_nt, method='rk4'):
    """Test Lie splitting with loaded models using refinement."""
    # Create neural splitting
    neural_splitting = NeuralOperatorSplitting(advection_model, diffusion_model)
    
    # Run Lie splitting with option 2: save at original spacing
    result = neural_splitting.lie_splitting(u0, small_dt, small_nt, method, 
                                           save_intermediate=True, 
                                           num_save_steps=original_nt)
    
    # Convert torch tensors to numpy arrays
    result = [r.cpu().numpy() if isinstance(r, torch.Tensor) else r for r in result]
    
    return np.array(result)


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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test Neural Operator Splitting')
    
    parser.add_argument('--training_scenario', type=str, default='burgers_heat',
                        choices=['burgers_heat', 'dispersion_heat', 'burgers_disp'],
                        help='Which trained models to load (default: burgers_heat)')
    
    parser.add_argument('--test_scenario', type=str, default='OP_HE',
                        choices=['OP_HE', 'OP_BG', 'OP_ED'],
                        help='Which test dataset to use (default: OP_HE)')
    
    parser.add_argument('--model_dir', type=str, default='./models',
                        help='Directory containing trained models (default: ./models)')
    
    parser.add_argument('--num_samples', type=int, default=32,
                        help='Number of test samples (default: 32)')
    
    parser.add_argument('--method', type=str, default='rk4',
                        choices=['rk4', 'dopri5', 'euler'],
                        help='ODE solver method (default: rk4)')
    
    parser.add_argument('--results_dir', type=str, default=None,
                        help='Custom results directory (default: auto-generated)')
    
    parser.add_argument('--refinement_factor', type=int, default=1,
                        help='Temporal refinement factor for operator splitting (default: 1)')
    
    parser.add_argument('--splitting_method', type=str, default='lie',
                        choices=['lie', 'strang'],
                        help='Splitting method to use (default: lie)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("Modular Combined Neural Operator Splitting Test")
    print("=" * 50)
    print(f"Configuration: {vars(args)}")
    
    # Use command line arguments
    config = {
        'training_scenario': args.training_scenario,
        'test_scenario': args.test_scenario,
        'model_dir': args.model_dir,
        'num_samples': args.num_samples
    }
    
    print(f"Training scenario: {config['training_scenario']}")
    print(f"Test scenario: {config['test_scenario']}")
    print(f"Model directory: {config['model_dir']}")
    
    # Load pre-trained models
    try:
        print("Loading pre-trained models...")
        advection_model, diffusion_model = load_trained_models(
            config['training_scenario'], 
            config['model_dir']
        )
        print("✓ Models loaded")
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ Error loading models: {e}")
        print("Please train models first using the modular training script")
        return
    
    # Load test data
    try:
        print("Loading test data...")
        trajectories, beta, gamma, time_points = load_test_data(
            config['test_scenario'], 
            config['num_samples']
        )
        print(f"✓ Loaded {len(trajectories)} test samples")
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ Error loading test data: {e}")
        return
    
    # Test parameters
    dt = time_points[1] - time_points[0]
    nt = len(time_points) - 1
    
    # Refinement parameters
    small_dt = dt / args.refinement_factor
    small_nt = nt * args.refinement_factor
    
    print(f"Testing {args.splitting_method} splitting:")
    print(f"  Original: dt={dt:.4f}, nt={nt}")
    print(f"  Refined: small_dt={small_dt:.4f}, small_nt={small_nt}, refinement_factor={args.refinement_factor}")
    
    # Create results directory
    if args.results_dir:
        results_dir = args.results_dir
    else:
        results_dir = f"test_results_{config['training_scenario']}_{config['test_scenario']}_{args.splitting_method}_r{args.refinement_factor}"
    os.makedirs(results_dir, exist_ok=True)
    
    for idx in range(min(config['num_samples'], len(trajectories))):
        # Test on sample
        u0 = trajectories[idx, 0, :]  # Initial condition
        u0 = torch.tensor(u0, dtype=torch.float32).unsqueeze(0)  # Convert to tensor and add batch dim
        
        print(f"Running {args.splitting_method} splitting for sample {idx+1}/{config['num_samples']}...")
        
        # Choose splitting method
        if args.splitting_method == 'lie':
            result = test_lie_splitting(advection_model, diffusion_model, u0, small_dt, small_nt, nt, args.method)
        elif args.splitting_method == 'strang':
            result = test_strang_splitting(advection_model, diffusion_model, u0, small_dt, small_nt, nt, args.method)
        else:
            raise ValueError(f"Unknown splitting method: {args.splitting_method}")
        
        print(f"✓ {args.splitting_method.capitalize()} splitting completed. Result shape: {result.shape}")
        
        # Get ground truth from the loaded data
        ground_truth = trajectories[idx, :len(result), :]  # Same sample, all time steps
        
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
        
        plt.title(f"{args.splitting_method.capitalize()} Predictions\n(Every 10 Steps)")
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
        errors = np.array([np.linalg.norm(ground_truth[i, :] - result[i]) / (np.linalg.norm(ground_truth[i, :]) + 1e-8)
                        for i in range(len(result))])
        
        # Plot error on top
        ax1 = plt.gca()
        line1 = ax1.semilogy(errors, 'r-', linewidth=2, label='Relative L2 Error')
        ax1.set_xlabel("Time Step")
        ax1.set_ylabel("Relative L2 Error", color='r')
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
        plt.suptitle(f"{config['training_scenario']} on {config['test_scenario']} - {args.splitting_method.capitalize()} (r{args.refinement_factor}) - Sample {idx+1}", y=1.02)
        
        # Save figure
        save_path = os.path.join(results_dir, f"{args.splitting_method}_splitting_result_sample_{idx+1:03d}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        # Print error statistics
        final_error = np.linalg.norm(ground_truth[-1, :] - result[-1]) / (np.linalg.norm(ground_truth[-1, :]) + 1e-8)
        print(f"Sample {idx+1} - Final Relative L2 Error: {final_error:.6f}")
        print(f"Sample {idx+1} - Max relative error over time: {errors.max():.6f}")
        print(f"Sample {idx+1} - Mean relative error over time: {errors.mean():.6f}")
    
    print(f"\nTest completed! Results saved in {results_dir}/")
    print(f"Configuration used:")
    print(f"  - Training scenario: {config['training_scenario']}")
    print(f"  - Test scenario: {config['test_scenario']}")
    print(f"  - Model directory: {config['model_dir']}")


if __name__ == "__main__":
    main()

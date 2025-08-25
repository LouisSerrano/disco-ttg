"""
Comprehensive Testing for Neural Operator Splitting

This module provides comprehensive tests for neural operator splitting methods,
including training, testing different solvers, dt convergence studies, and 
comparison with ground truth FFT solutions.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
from typing import Dict, List, Optional
import time
import json
import argparse

# Add the operator-splitting directory to the path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from data_generation import generate_training_data, TrajectoryGenerator
from neural_ode_operators import create_neural_operators
from training import train_neural_operators
from neural_splitting_methods import NeuralOperatorSplitting, ComparisonFramework

# Import from operator splitting
try:
    from operator_splitting.operator_splitting_1d import AdvectionDiffusion1DSolver
except ImportError:
    # Fallback: import directly from the actual directory
    operator_splitting_path = os.path.join(parent_dir, 'operator-splitting')
    sys.path.insert(0, operator_splitting_path)
    from operator_splitting_1d import AdvectionDiffusion1DSolver


# Parameter grid for comprehensive testing
PARAMETER_GRID = {
    'advection_speeds': [5.0],
    'viscosities': [0.1],
    'num_steps': [1, 10],
    #'advection_speeds': [1.0, 2.0, 4.0],
    #'viscosities': [0.05, 0.1],
    #'num_steps': [1, 2, 4, 8, 16, 32]
    #'advection_speeds': [1.0],
    #'viscosities': [0.05, 0.1],
    #'num_steps': [1, 2, 4, 8, 16, 32]
}


class NeuralSplittingTestSuite:
    """Comprehensive test suite for neural operator splitting."""
    
    def __init__(self, 
                 nx: int = 128,
                 L: float = 2*np.pi,
                 beta: float = 1.0,
                 nu: float = 0.1,
                 results_dir: str = './results'):
        """
        Initialize test suite.
        
        Args:
            nx: Number of spatial grid points
            L: Domain length
            beta: Advection speed coefficient
            nu: Diffusion viscosity coefficient
            results_dir: Directory to save results
        """
        self.nx = nx
        self.L = L
        self.beta = beta
        self.nu = nu
        self.results_dir = results_dir
        
        # Create results directory and subdirectories
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'predictions_csv'), exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'predictions_png'), exist_ok=True)
        
        # Store test configurations
        self.test_configs = {}
    
    def save_predictions_png(self, predictions: Dict, case_name: str, dt: float, nt: int, idx: int = None):
        """Save predictions as PNG visualizations."""
        png_dir = os.path.join(self.results_dir, 'predictions_png', case_name)
        os.makedirs(png_dir, exist_ok=True)
        print(f"Created PNG directory: {png_dir}")
        
        x = np.linspace(0, self.L, self.nx, endpoint=False)
        time_steps = np.arange(0, nt * dt, dt)
        
        # Create heatmap plots for each method
        print(f"Predictions keys: {list(predictions.keys())}")
        for method, data in predictions.items():
            #print(f"Method: {method}, Data type: {type(data)}, Shape: {getattr(data, 'shape', 'N/A')}")
            # Convert data to numpy array if needed
            if isinstance(data, list):
                data = np.array(data)
            elif isinstance(data, dict):
                print(f"Skipping {method}: dictionary data not supported for visualization")
                continue
            
            # If idx is specified, show only that index
            #print('predata', data.shape)
            if idx is not None:
                data = data[:, idx]  # Keep as 2D array with single time step
            
            print('data', data.shape, "x", x.shape)
            
            if isinstance(data, np.ndarray) and len(data.shape) >= 2:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
                
                # Handle 3D data by taking first few batch examples
                if len(data.shape) == 3:
                    plot_data = data[:, 0, :]  # Take first batch example
                else:
                    plot_data = data
                
                # Heatmap
                im1 = ax1.imshow(plot_data[:min(len(time_steps), plot_data.shape[0])].T, 
                               aspect='auto', origin='lower', 
                               extent=[0, min(len(time_steps), plot_data.shape[0]) * dt, 0, self.L])
                ax1.set_xlabel('Time')
                ax1.set_ylabel('Space (x)')
                ax1.set_title(f'{method.replace("_", " ").title()} - Spatiotemporal Evolution')
                plt.colorbar(im1, ax=ax1)
                
                # Line plots at different times
                step = max(1, plot_data.shape[0] // 5)
                time_indices = np.arange(0, plot_data.shape[0], step)
                
                for i, t_idx in enumerate(time_indices):
                    print('t_idx', t_idx)
                    if t_idx < plot_data.shape[0]:
                        alpha = 0.4 + 0.6 * i / max(1, len(time_indices) - 1)
                        # Ensure we're plotting 1D spatial data
                        spatial_data = plot_data[t_idx]
                        if len(spatial_data.shape) > 1:
                            spatial_data = spatial_data.flatten()
                        # Match dimensions - take subset if needed
                        if len(spatial_data) != len(x):
                            if len(spatial_data) > len(x):
                                spatial_data = spatial_data[:len(x)]
                            else:
                                # Interpolate to match x dimension
                                x_data = np.linspace(0, self.L, len(spatial_data), endpoint=False)
                                spatial_data_interp = np.interp(x, x_data, spatial_data)
                                spatial_data = spatial_data_interp
                        ax2.plot(x, spatial_data, alpha=alpha, 
                               label=f't = {t_idx}')
                
                ax2.set_xlabel('Space (x)')
                ax2.set_ylabel('u(x,t)')
                ax2.set_title(f'{method.replace("_", " ").title()} - Spatial Profiles')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                # Save PNG
                png_file = os.path.join(png_dir, f'{method}_predictions.png')
                plt.savefig(png_file, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Saved {method} visualization to {png_file}")
            else:
                print(f"Skipping {method}: data is not a 2D+ numpy array")
    
    def save_predictions_csv(self, predictions: Dict, case_name: str, dt: float, nt: int, idx: int = None):
        """Save all predictions as CSV files."""
        csv_dir = os.path.join(self.results_dir, 'predictions_csv', case_name)
        os.makedirs(csv_dir, exist_ok=True)
        print(f"Created CSV directory: {csv_dir}")
        
        # Save all predictions in a single CSV file
        all_data = {}
        for method, data in predictions.items():
            # Convert data to numpy array if needed
            if isinstance(data, list):
                data = np.array(data)
            elif isinstance(data, dict):
                print(f"Skipping {method}: dictionary data not supported for CSV export")
                continue
            
            # If idx is specified, show only that index
            if idx is not None and isinstance(data, np.ndarray) and len(data.shape) >= 2:
                if idx >= data.shape[0]:
                    print(f"Skipping {method}: idx {idx} >= data shape {data.shape[0]}")
                    continue
                data = data[idx]  # Single time step
            
            if isinstance(data, np.ndarray):
                all_data[method] = data
        
        if all_data:
            # Create a combined CSV with all methods
            csv_file = os.path.join(csv_dir, 'all_predictions.csv')
            
            # Find maximum length for padding
            max_len = max(len(data.flatten()) if data.ndim > 1 else len(data) for data in all_data.values())
            
            # Create structured data
            csv_data = []
            headers = []
            for method, data in all_data.items():
                flat_data = data.flatten() if data.ndim > 1 else data
                # Pad with NaN if needed
                padded_data = np.pad(flat_data, (0, max_len - len(flat_data)), constant_values=np.nan)
                csv_data.append(padded_data)
                headers.append(method)
            
            # Save as CSV
            csv_array = np.column_stack(csv_data)
            np.savetxt(csv_file, csv_array, delimiter=',', header=','.join(headers), comments='')
            print(f"Saved all predictions to {csv_file}")
    
    def save_models(self, training_results: Dict, case_name: str = ""):
        """Save trained models."""
        models_dir = os.path.join(self.results_dir, 'models')
        if case_name:
            models_dir = os.path.join(models_dir, case_name)
        os.makedirs(models_dir, exist_ok=True)
        
        # Save advection model
        if 'advection' in training_results and 'model' in training_results['advection']:
            advection_path = os.path.join(models_dir, 'advection_model.pth')
            torch.save(training_results['advection']['model'].state_dict(), advection_path)
            print(f"Saved advection model to {advection_path}")
        
        # Save diffusion model
        if 'diffusion' in training_results and 'model' in training_results['diffusion']:
            diffusion_path = os.path.join(models_dir, 'diffusion_model.pth')
            torch.save(training_results['diffusion']['model'].state_dict(), diffusion_path)
            print(f"Saved diffusion model to {diffusion_path}")
        
        # Save training configuration and results
        config_path = os.path.join(models_dir, 'training_config.json')
        with open(config_path, 'w') as f:
            # Extract serializable data
            config_data = {
                'advection': {
                    'history': training_results['advection']['history'],
                    'config': training_results['advection'].get('config', {})
                } if 'advection' in training_results else {},
                'diffusion': {
                    'history': training_results['diffusion']['history'],
                    'config': training_results['diffusion'].get('config', {})
                } if 'diffusion' in training_results else {}
            }
            json.dump(config_data, f, indent=2)
        print(f"Saved training configuration to {config_path}")
        
    def run_parameter_grid_test(self, base_train_config: Optional[Dict] = None) -> Dict:
        """
        Run comprehensive parameter grid evaluation across all combinations.
        
        Args:
            base_train_config: Base training configuration to modify for each parameter set
            
        Returns:
            Complete parameter grid test results
        """
        print("="*60)
        print("NEURAL OPERATOR SPLITTING - PARAMETER GRID EVALUATION")
        print(f"Testing {len(PARAMETER_GRID['advection_speeds'])} × {len(PARAMETER_GRID['viscosities'])} × {len(PARAMETER_GRID['num_steps'])} = {len(PARAMETER_GRID['advection_speeds']) * len(PARAMETER_GRID['viscosities']) * len(PARAMETER_GRID['num_steps'])} combinations")
        print("="*60)
        
        # Default base training configuration
        if base_train_config is None:
            test_dt=0.02
            test_nt=50
            base_train_config = {
                'nx': self.nx,
                'L': self.L,
                'hidden_dim': 128,
                'n_layers': 3,
                'num_epochs': 100,
                'batch_size': 64,
                'training_size': 512,
                'learning_rate': 1e-3,
                'method': 'rk4',
                'verbose': True,
                "dt": test_dt,
                "T": test_dt*test_nt,
            }
        test_dt = base_train_config['dt']
        test_nt = int(base_train_config['T']/test_dt)
        
        all_results = []
        total_combinations = len(PARAMETER_GRID['advection_speeds']) * len(PARAMETER_GRID['viscosities']) * len(PARAMETER_GRID['num_steps'])
        current_combination = 0
        
        # Create comprehensive results directory structure
        grid_results_dir = os.path.join(self.results_dir, 'parameter_grid')
        os.makedirs(grid_results_dir, exist_ok=True)
        
        for beta in PARAMETER_GRID['advection_speeds']:
            for nu in PARAMETER_GRID['viscosities']:
                for num_steps in PARAMETER_GRID['num_steps']:
                    current_combination += 1
                    combination_name = f"adv_{beta}_nu_{nu}_steps_{num_steps}"
                    
                    print(f"\n[{current_combination}/{total_combinations}] Testing combination: {combination_name}")
                    print("-" * 50)
                    
                    # Configure training for this parameter combination
                    train_config = base_train_config.copy()
                    train_config.update({
                        'beta': beta,
                        'nu': nu,
                        'num_steps': num_steps,
                        'save_dir': os.path.join(grid_results_dir, 'models', combination_name)
                    })
                    
                    # Train neural operators for this combination
                    print(f"Training neural operators for beta={beta}, nu={nu}, num_steps={num_steps}")
                    start_time = time.time()
                    training_results = train_neural_operators(**train_config)
                    training_time = time.time() - start_time
                    
                    # Save models for this combination
                    self.save_models(training_results, combination_name)
                    
                    # Create neural splitting framework
                    neural_splitting = NeuralOperatorSplitting(
                        training_results['advection']['model'],
                        training_results['diffusion']['model']
                    )
                    
                    # Create comparison framework with current parameters
                    comparison_framework = ComparisonFramework(
                        neural_splitting, self.nx, self.L, beta, nu
                    )
                    
                    # Test on multiple initial conditions
                    test_cases = self._create_test_cases()
                    combination_results = {
                        'parameters': {'beta': beta, 'nu': nu, 'num_steps': num_steps},
                        'combination_name': combination_name,
                        'training_time': training_time,
                        'training_losses': {
                            'advection': training_results['advection']['history']['best_val_loss'],
                            'diffusion': training_results['diffusion']['history']['best_val_loss']
                        },
                        'test_cases': {}
                    }
                    
                    # Test on validation set using batch processing
                    print(f"  Testing on validation set (batch processing)")
                    
                    # Get validation data from training results
                    val_data = training_results['advection']['val_dataset']
                    val_data = np.array(val_data)
                    u0_batch = val_data[:, 0]  # All initial conditions [batch_size, nx]
                    print(f"    Validation batch shape: {u0_batch.shape}, val_data shape: {val_data.shape}")
                    
                    # Compare methods with adjusted time stepping for num_steps
                    #dt_effective = test_dt / num_steps
                    #nt_effective = test_nt * num_steps
                    
                    # Batch process all validation samples
                    #batch_results = self._compare_methods_batch(
                    #    comparison_framework, u0_batch, val_data, dt_effective, nt_effective,
                    #    methods=['neural_lie', 'neural_strang', 'classical_lie', 'classical_strang', 'ground_truth']
                    #)

                    batch_results = comparison_framework.compare_methods(u0_batch, dt=test_dt, nt=test_nt, methods=['neural_lie', 'classical_lie', 'ground_truth'])                   
                    print("batch_results.keys():", batch_results.keys())
                              
                    # Compute average errors across validation set
                    case_metrics = {}
                    for method in ['neural_lie', 'classical_lie']:
                        if method in batch_results and batch_results['errors']:
                            errors = batch_results['errors'][method]
                            case_metrics[method] = errors  #{
                                #'l2_error': #np.mean([e['l2_error'] for e in errors]),
                                #'linf_error': #np.mean([e['linf_error'] for e in errors]),
                                #'std_l2_error': #np.std([e['l2_error'] for e in errors]),
                                #'std_linf_error': np.std([e['linf_error'] for e in errors]),
                                #'num_samples': len(errors)
                            #}
                        else:
                            case_metrics[method] = None
                    
                    # Save predictions for the first validation sample as example
                    example_results = batch_results 
                    
                    case_dir = f"{combination_name}_validation"
                    #self.save_predictions_csv(example_results, case_dir, test_dt, num_steps, idx=0)
                    self.save_predictions_png(example_results, case_dir, test_dt, num_steps, idx=0)
                    
                    combination_results['test_cases']['validation'] = {
                        'metrics': case_metrics,
                        'success': True,
                        'validation_set_size': len(val_data)
                    }
                    
                    print(f"    ✓ Success - Batch tested on {len(val_data)} validation samples")
                    for method, metrics in case_metrics.items():
                        if metrics:
                            print(f"      {method}: avg L2 = {metrics['l2_error']:.2e}")
                    
                    combination_results['success'] = True
                    print(f"✓ Combination completed in {training_time:.2f}s")
                    
                    all_results.append(combination_results)
                    
                    # Save intermediate results after each combination
                    self._save_parameter_grid_results(all_results, grid_results_dir)
        
        # Create comprehensive DataFrame
        print(f"\nCreating comprehensive results DataFrame...")
        results_df = self._create_comprehensive_dataframe(all_results)
        
        # Save final results
        final_results = {
            'parameter_grid': PARAMETER_GRID,
            'total_combinations': total_combinations,
            'results': all_results,
            'results_dataframe': results_df
        }
        
        self._save_final_parameter_grid_results(final_results, grid_results_dir)
        
        print(f"\nParameter grid evaluation completed!")
        print(f"Results saved to: {grid_results_dir}")
        
        return final_results
    
    def _save_parameter_grid_results(self, results: List[Dict], results_dir: str):
        """Save intermediate parameter grid results."""
        intermediate_file = os.path.join(results_dir, 'intermediate_results.json')
        
        # Create serializable version
        serializable_results = []
        for result in results:
            serializable_result = result.copy()
            # Remove any non-serializable data if present
            serializable_results.append(serializable_result)
        
        with open(intermediate_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    
    def _create_comprehensive_dataframe(self, results: List[Dict]) -> pd.DataFrame:
        """Create comprehensive DataFrame from parameter grid results."""
        df_rows = []
        
        for result in results:
            if not result.get('success', False):
                continue
                
            params = result['parameters']
            base_row = {
                'beta': params['beta'],
                'nu': params['nu'], 
                'num_steps': params['num_steps'],
                'training_time': result.get('training_time', None),
                'advection_loss': result.get('training_losses', {}).get('advection', None),
                'diffusion_loss': result.get('training_losses', {}).get('diffusion', None)
            }
            
            # Add metrics for each test case and method
            for case_name, case_data in result.get('test_cases', {}).items():
                if not case_data.get('success', False):
                    continue
                    
                for method, metrics in case_data.get('metrics', {}).items():
                    if metrics is not None:
                        row = base_row.copy()
                        row.update({
                            'test_case': case_name,
                            'method': method,
                            'l2_error': metrics['l2_error'],
                            'linf_error': metrics['linf_error']
                        })
                        df_rows.append(row)
        
        return pd.DataFrame(df_rows)
    
    def _save_final_parameter_grid_results(self, results: Dict, results_dir: str):
        """Save final comprehensive parameter grid results."""
        # Save DataFrame
        if 'results_dataframe' in results and not results['results_dataframe'].empty:
            df_file = os.path.join(results_dir, 'parameter_grid_results.csv')
            results['results_dataframe'].to_csv(df_file, index=False)
            print(f"Results DataFrame saved to: {df_file}")
        
        # Save summary JSON (without DataFrame)
        summary_results = results.copy()
        summary_results.pop('results_dataframe', None)  # Remove DataFrame for JSON serialization
        
        summary_file = os.path.join(results_dir, 'parameter_grid_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary_results, f, indent=2)
        print(f"Summary results saved to: {summary_file}")
        
        # Create performance comparison summary
        self._create_performance_summary(results, results_dir)
    
    def _create_performance_summary(self, results: Dict, results_dir: str):
        """Create performance comparison summary and visualizations."""
        df = results.get('results_dataframe')
        if df is None or df.empty:
            return
            
        summary_file = os.path.join(results_dir, 'performance_summary.txt')
        with open(summary_file, 'w') as f:
            f.write("PARAMETER GRID EVALUATION - PERFORMANCE SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            # Overall statistics
            f.write("OVERALL STATISTICS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total parameter combinations tested: {len(results.get('results', []))}\n")
            f.write(f"Successful combinations: {len([r for r in results.get('results', []) if r.get('success', False)])}\n")
            f.write(f"Total test cases per combination: {len(df['test_case'].unique()) if 'test_case' in df.columns else 0}\n")
            f.write(f"Methods compared: {', '.join(df['method'].unique()) if 'method' in df.columns else 'None'}\n\n")
            
            # Best performing configurations
            if 'l2_error' in df.columns and not df['l2_error'].isna().all():
                f.write("BEST PERFORMING CONFIGURATIONS (by L2 error):\n")
                f.write("-" * 45 + "\n")
                
                for method in df['method'].unique():
                    method_df = df[df['method'] == method]
                    if not method_df.empty:
                        best_config = method_df.loc[method_df['l2_error'].idxmin()]
                        f.write(f"{method}:\n")
                        f.write(f"  Beta: {best_config['beta']}, Nu: {best_config['nu']}, Steps: {best_config['num_steps']}\n")
                        f.write(f"  L2 Error: {best_config['l2_error']:.2e}\n")
                        f.write(f"  Test Case: {best_config['test_case']}\n\n")
        
        print(f"Performance summary saved to: {summary_file}")
    
    def _create_test_cases(self) -> Dict[str, np.ndarray]:
        """Create different test cases with various initial conditions."""
        x = np.linspace(0, self.L, self.nx, endpoint=False)
        
        test_cases = {
            'gaussian_narrow': np.exp(-(x - self.L/4)**2 / (2 * 0.05**2)),
            'gaussian_wide': np.exp(-(x - self.L/2)**2 / (2 * 0.2**2)),
            'sine_wave': np.sin(2 * np.pi * x / self.L),
            'double_sine': np.sin(2 * np.pi * x / self.L) + 0.5 * np.sin(4 * np.pi * x / self.L),
            'step_function': np.where((x > self.L/3) & (x < 2*self.L/3), 1.0, 0.0),
            'high_frequency': np.sin(8 * np.pi * x / self.L) * np.exp(-(x - self.L/2)**2 / (2 * 0.1**2))
        }
        
        return test_cases
    # 
    def quick_test(self, 
                   train_epochs: int = 20,
                   test_dt: float = 0.02,
                   test_nt: int = 25) -> Dict:
        """
        Run a quick test with minimal training for debugging.
        
        Args:
            train_epochs: Number of training epochs
            test_dt: Test time step
            test_nt: Number of test time steps
            
        Returns:
            Quick test results
        """
        print("Running quick test...")
        
        # Quick training config
        quick_config = {
            'nx': 128,  # Smaller grid
            'L': self.L,
            'beta': self.beta,
            'nu': self.nu,
            'hidden_dim': 64,  # Smaller network
            'n_layers': 3,
            'num_epochs': train_epochs,
            'batch_size': 64,
            'training_size': 1024,
            'learning_rate': 1e-3,
            'method': "rk4", #'euler', #rk4 #euler,
            'num_steps': 1,
            'save_dir': os.path.join(self.results_dir, 'quick_models'),
            'verbose': True,
            'sequence_length': 10,
            "dt": test_dt,
            "T": test_dt*test_nt,
        }
        
        # Train models
        training_results = train_neural_operators(**quick_config)
        
        # Save models
        self.save_models(training_results, "quick_test")
        
        # Quick test
        neural_splitting = NeuralOperatorSplitting(
            training_results['advection']['model'],
            training_results['diffusion']['model']
        )
        
        # Test on simple Gaussian
        #x = np.linspace(0, self.L, quick_config['nx'], endpoint=False)
        #u0 = np.exp(-(x - self.L/4)**2 / (2 * 0.1**2))
        advection_dataset = training_results['advection']['dataset']['trajectories']
        advection_dataset = np.array(advection_dataset)
        print('advection dataset', advection_dataset.shape)
        u0 = advection_dataset[0, 0] # first element of the batch and first timestamp
        
        comparison = ComparisonFramework(neural_splitting, quick_config['nx'], self.L, self.beta, self.nu)
        
        results = comparison.compare_methods(
            u0, dt=test_dt, nt=test_nt,
            methods=['neural_lie', 'classical_lie', 'ground_truth']
        )
        
        # Save predictions from quick test
        self.save_predictions_png(results, "solution", test_dt, test_nt)
        
        # Demonstrate new neural ODE default T=1 functionality
        print("\nTesting neural ODE with T=1 default and intermediate steps:")
        neural_predictions = neural_splitting.predict_with_default_T(
            u0, operator='both', T=1.0, n_intermediate=10, method=quick_config['method']
        )
        
        # Save these predictions as well
        self.save_predictions_png(neural_predictions, "neural_T1_predictions", 0.1, 11)
        print("Neural ODE T=1 predictions saved successfully!")
        
        # Demonstrate new neural splitting with prediction control
        print("\nTesting neural splitting with prediction frequency control:")
        splitting_predictions = neural_splitting.lie_splitting_with_predictions(
            u0, T_total=1.0, num_predictions=10, method=quick_config['method']
        )
        
        # Save these predictions
        solutions_for_save = {'lie_splitting_T1': splitting_predictions['solutions']}
        dt_for_save = splitting_predictions['dt']
        nt_for_save = splitting_predictions['num_predictions']
        
        self.save_predictions_png(solutions_for_save, "neural_splitting_T1_controlled", dt_for_save, nt_for_save)
        print(f"Neural splitting T=1 controlled predictions saved! Shape: {splitting_predictions['solutions'].shape}")
        
        # Compare neural Lie splitting with varying number of steps
        print("\nTesting neural Lie splitting with varying number of steps:")

        # Get ground truth solution for comparison
        ground_truth = comparison.classical_solver.fft_ground_truth(u0, test_dt, test_nt)
        
        step_comparison_results = {}
        for num_steps in [1, 5, 10, 20]:
            print(f"  Testing with {num_steps} steps...")
            output = comparison.neural_splitting.lie_splitting(
                u0, test_dt/num_steps, test_nt*num_steps, quick_config['method'],
                save_intermediate=True, num_save_steps=test_nt
            )
            output = np.array(output)
            #print('step', num_steps, output.shape)
            
            # Calculate errors compared to ground truth
            if output is not None and len(output) > 0:
                final_solution = output[-1]  # Get final time step
                final_ground_truth = ground_truth[-1]
                
                # Compute L2 and L∞ errors
                l2_error = np.linalg.norm(final_solution - final_ground_truth) / np.linalg.norm(final_ground_truth)
                linf_error = np.max(np.abs(final_solution - final_ground_truth)) / np.max(np.abs(final_ground_truth))
                
                step_comparison_results[f"steps_{num_steps}"] = {
                    'solution': output,
                    'num_steps': num_steps,
                    'final_state': final_solution,
                    'l2_error': l2_error,
                    'linf_error': linf_error,
                    'success': True
                }
                print(f"    ✓ Success - L2 error: {l2_error:.2e}, L∞ error: {linf_error:.2e}")
            else:
                step_comparison_results[f"steps_{num_steps}"] = {
                    'num_steps': num_steps,
                    'error': 'No solution obtained',
                    'success': False
                }
                print(f"    ✗ Failed - No solution obtained")

        # Print results comparison
        print("\nStep Comparison Results:")
        print("=" * 40)
        for step_key, step_result in step_comparison_results.items():
            if step_result['success']:
                num_steps = step_result['num_steps']
                l2_error = step_result['l2_error']
                linf_error = step_result['linf_error']
                final_norm = np.linalg.norm(step_result['final_state'])
                print(f"Steps {num_steps:2d}: L2 error = {l2_error:.2e}, L∞ error = {linf_error:.2e}, Final norm = {final_norm:.4f}")
            else:
                print(f"Steps {step_result['num_steps']:2d}: ERROR - {step_result['error']}")
        
        # Save step comparison visualizations
        step_solutions = {'ground_truth': ground_truth}  # Include ground truth
        for step_key, step_result in step_comparison_results.items():
            if step_result['success'] and 'solution' in step_result:
                step_solutions[f"neural_lie_{step_key}"] = step_result['solution']
        
        if step_solutions:
            self.save_predictions_png(step_solutions, "neural_lie_step_comparison", test_dt, test_nt)
            print("Step comparison visualizations saved!")
        
        print("Quick test completed!")
        print({
            'testing': results.get("errors", {}),
            'neural_T1_shapes': {k: v.shape for k, v in neural_predictions.items()},
            'splitting_predictions_shape': splitting_predictions.get('solutions', np.array([])).shape,
            'step_comparison_keys': list(step_comparison_results.keys())
        })
        return {
            'training': training_results,
            'testing': results,
            'neural_T1_predictions': neural_predictions,
            'splitting_predictions': splitting_predictions,
            'step_comparison': step_comparison_results
        }


def main():
    """Main function to run comprehensive neural operator splitting tests."""
    parser = argparse.ArgumentParser(description='Neural Operator Splitting Test Suite')
    parser.add_argument('--test-type', choices=['full', 'quick', 'debug'], default='full',
                        help='Type of test to run (default: full)')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='Advection speed coefficient (default: 1.0)')
    parser.add_argument('--nu', type=float, default=0.1,
                        help='Diffusion viscosity coefficient (default: 0.1)')
    args = parser.parse_args()
    
    print("Neural Operator Splitting - Comprehensive Test Suite")
    print("=" * 60)
    print(f"Running {args.test_type} test...")
    
    # Create test suite with parameterized values
    test_suite = NeuralSplittingTestSuite(
        nx=128, 
        L=16,
        beta=args.beta,
        nu=args.nu,
        results_dir='/mnt/home/lserrano/disco-ball/neural-operator-splitting/test_results_dopri5'
    )
    
    # Choose test type based on command line argument
    test_type = args.test_type
    
    if test_type == 'quick':
        print("\\nRunning quick test...")
        results = test_suite.quick_test(train_epochs=10, test_dt=0.05, test_nt=20)
        
    elif test_type == 'debug':
        print("\\nRunning minimal debug test...")
        results = test_suite.quick_test(train_epochs=50, test_dt=0.02, test_nt=50) #10 steps 
        
    else:  # full test - parameter grid evaluation
        print("\\nRunning parameter grid evaluation...")
        
        # Base training configuration for parameter grid
        test_dt=0.1 #0.02
        test_nt=100
        base_train_config = {
            'nx': 128,
            'L': 16, #2*np.pi,
            'hidden_dim': 128,
            'n_layers': 3,
            'num_epochs': 500,
            'batch_size': 64,
            'training_size': 1024,
            'learning_rate': 1e-3,
            'verbose': True,
            'dt': test_dt,
            'T': test_dt*test_nt,
            'method': 'dopri5', #euler
            'use_adjoint': False
        }
        
        results = test_suite.run_parameter_grid_test(
            base_train_config=base_train_config
        )
    
    print("\\nTest suite completed successfully!")
    print(f"Results saved to: {test_suite.results_dir}")


if __name__ == "__main__":
    main()

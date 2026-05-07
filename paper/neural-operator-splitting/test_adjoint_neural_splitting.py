"""
Adjoint Solver Testing for Neural Operator Splitting

This module extends the comprehensive test suite to include adjoint solver testing,
demonstrating the difference between "discretize-then-optimize" vs "optimize-then-discretize"
approaches for neural ODEs.
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
from test_neural_splitting import NeuralSplittingTestSuite

# Import from operator splitting
try:
    from operator_splitting.operator_splitting_1d import AdvectionDiffusion1DSolver
except ImportError:
    # Fallback: import directly from the actual directory
    operator_splitting_path = os.path.join(parent_dir, 'operator-splitting')
    sys.path.insert(0, operator_splitting_path)
    from operator_splitting_1d import AdvectionDiffusion1DSolver


class AdjointNeuralSplittingTestSuite(NeuralSplittingTestSuite):
    """Extended test suite for adjoint solver testing in neural operator splitting."""
    
    def __init__(self, *args, **kwargs):
        """Initialize the adjoint test suite."""
        super().__init__(*args, **kwargs)
        
    def test_adjoint_solver(self, 
                           train_epochs: int = 50,
                           test_dt: float = 0.02,
                           test_nt: int = 25) -> Dict:
        """
        Test the adjoint solver implementation for neural ODEs.
        Compares standard vs adjoint method for both accuracy and memory usage.
        
        Args:
            train_epochs: Number of training epochs
            test_dt: Test time step
            test_nt: Number of test time steps
            
        Returns:
            Adjoint solver test results
        """
        print("="*50)
        print("ADJOINT SOLVER TEST")
        print("="*50)
        print("Testing adjoint method vs standard method for neural ODEs")
        print("Adjoint method uses 'optimize-then-discretize' approach")
        print("Standard method uses 'discretize-then-optimize' approach")
        print("="*50)
        
        # Training config for adjoint test
        adjoint_config = {
            'nx': 128,  # Smaller for faster testing
            'L': self.L,
            'beta': self.beta,
            'nu': self.nu,
            'hidden_dim': 64,  
            'n_layers': 2,
            'num_epochs': train_epochs,
            'batch_size': 32,
            'training_size': 512,
            'learning_rate': 1e-3,
            'method': 'dopri5',
            'num_steps': 1,
            'save_dir': os.path.join(self.results_dir, 'adjoint_models'),
            'verbose': True,
            'sequence_length': 10,
            "dt": test_dt,
            "T": test_dt*test_nt,
            "use_adjoint": True
        }
        
        # Train models
        print("Training neural operators for adjoint test...")
        training_results = train_neural_operators(**adjoint_config)
        
        # Save models
        self.save_models(training_results, "adjoint_test")
        
        # Create splitting framework
        neural_splitting = NeuralOperatorSplitting(
            training_results['advection']['model'],
            training_results['diffusion']['model']
        )
        
        # Test initial condition
        advection_dataset = training_results['advection']['dataset']['trajectories']
        advection_dataset = np.array(advection_dataset)
        u0 = advection_dataset[:, 0]  # first element of batch and first timestamp
        
        print(f"\nTesting adjoint vs standard solver on initial condition of shape: {u0.shape}")
        
        # Test 1: Compare solutions with standard vs adjoint method
        print("\n1. SOLUTION ACCURACY COMPARISON")
        print("-" * 30)
        
        methods_to_test = ['standard', 'adjoint']
        solutions = {}
        
        for solver_method in methods_to_test:
            use_adjoint = (solver_method == 'adjoint')
            print(f"Testing {solver_method} solver...")
            
            # Test Lie splitting
            lie_solution = neural_splitting.lie_splitting(
                u0, test_dt, test_nt, method='dopri5', use_adjoint=use_adjoint
            )
            
            # Test Strang splitting  
            strang_solution = neural_splitting.strang_splitting(
                u0, test_dt, test_nt, method='dopri5', use_adjoint=use_adjoint
            )
            
            solutions[solver_method] = {
                'lie': np.array(lie_solution),
                'strang': np.array(strang_solution)
            }
            
            print(f"  Lie solution shape: {solutions[solver_method]['lie'].shape}")
            print(f"  Strang solution shape: {solutions[solver_method]['strang'].shape}")
        
        # Compare solutions between standard and adjoint
        print("\n2. SOLUTION DIFFERENCES")
        print("-" * 25)
        
        comparison_results = {}
        for splitting_type in ['lie', 'strang']:
            standard_sol = solutions['standard'][splitting_type]
            adjoint_sol = solutions['adjoint'][splitting_type]
            
            # Compute differences
            final_diff = np.abs(standard_sol[-1] - adjoint_sol[-1])
            max_diff = np.max(final_diff)
            mean_diff = np.mean(final_diff)
            relative_diff = max_diff / (np.max(np.abs(standard_sol[-1])) + 1e-12)
            
            comparison_results[splitting_type] = {
                'max_absolute_diff': max_diff,
                'mean_absolute_diff': mean_diff,
                'relative_diff': relative_diff,
                'standard_final_norm': np.linalg.norm(standard_sol[-1]),
                'adjoint_final_norm': np.linalg.norm(adjoint_sol[-1])
            }
            
            print(f"{splitting_type.upper()} splitting:")
            print(f"  Max absolute difference: {max_diff:.2e}")
            print(f"  Mean absolute difference: {mean_diff:.2e}")
            print(f"  Relative difference: {relative_diff:.2e}")
            print(f"  Standard final norm: {comparison_results[splitting_type]['standard_final_norm']:.4f}")
            print(f"  Adjoint final norm: {comparison_results[splitting_type]['adjoint_final_norm']:.4f}")
        
        # Add ground truth to solutions for visualization
        neural_solutions['ground_truth'] = ground_truth
        
        # Test 4: Save visualization of differences
        print("\n4. SAVING VISUALIZATIONS")
        print("-" * 26)
        
        # Save comparison visualizations
        viz_results = {
            'standard_lie': solutions['standard']['lie'],
            'adjoint_lie': solutions['adjoint']['lie'],
            'standard_strang': solutions['standard']['strang'],
            'adjoint_strang': solutions['adjoint']['strang']
        }
        
        self.save_predictions_png(viz_results, "adjoint_comparison", test_dt, test_nt)
        print("Adjoint comparison visualizations saved!")
        
        # Save difference plots
        difference_results = {}
        for splitting_type in ['lie', 'strang']:
            standard_sol = solutions['standard'][splitting_type]
            adjoint_sol = solutions['adjoint'][splitting_type]
            difference_results[f'{splitting_type}_difference'] = np.abs(standard_sol - adjoint_sol)
        
        self.save_predictions_png(difference_results, "adjoint_differences", test_dt, test_nt)
        print("Difference visualizations saved!")
        
        # Test 5: Performance characteristics
        print("\n5. PERFORMANCE SUMMARY")
        print("-" * 22)
        
        for splitting_type, metrics in comparison_results.items():
            print(f"{splitting_type.upper()} splitting adjoint test:")
            if metrics['relative_diff'] < 1e-6:
                print(f"  ✓ PASS - Solutions match within tolerance ({metrics['relative_diff']:.2e})")
            elif metrics['relative_diff'] < 1e-3:
                print(f"  ⚠ WARN - Small differences detected ({metrics['relative_diff']:.2e})")
            else:
                print(f"  ✗ FAIL - Large differences detected ({metrics['relative_diff']:.2e})")
        
        print("\nAdjoint solver benefits:")
        print("  • Memory efficient for long sequences")
        print("  • Mathematically consistent gradients")
        print("  • Better for training deep ODEs")
        print("  • Implements 'optimize-then-discretize'")
        
        print("\nStandard solver benefits:")
        print("  • Faster for short sequences")
        print("  • Simpler implementation")
        print("  • Direct gradient computation")
        
        final_results = {
            'solutions': solutions,
            'comparison_metrics': comparison_results,
            'training_results': training_results,
            'test_config': {
                'test_dt': test_dt,
                'test_nt': test_nt,
                'u0_shape': u0.shape
            }
        }
        
        print("\nAdjoint solver test completed!")
        return final_results

    def test_gradient_accuracy(self, 
                              train_epochs: int = 30,
                              test_dt: float = 0.05,
                              test_nt: int = 10) -> Dict:
        """
        Test gradient accuracy between standard and adjoint methods.
        This test demonstrates the 'optimize-then-discretize' vs 'discretize-then-optimize' difference.
        
        Args:
            train_epochs: Number of training epochs
            test_dt: Test time step
            test_nt: Number of test time steps
            
        Returns:
            Gradient accuracy test results
        """
        print("\n" + "="*50)
        print("GRADIENT ACCURACY TEST")
        print("="*50)
        print("Comparing gradients from standard vs adjoint methods")
        print("This demonstrates discretize-then-optimize vs optimize-then-discretize")
        
        # Training config
        gradient_config = {
            'nx': 32,  # Small for faster gradient computation
            'L': self.L,
            'beta': self.beta,
            'nu': self.nu,
            'hidden_dim': 32,  
            'n_layers': 2,
            'num_epochs': train_epochs,
            'batch_size': 16,
            'training_size': 256,
            'learning_rate': 1e-3,
            'method': 'dopri5',
            'num_steps': 1,
            'save_dir': os.path.join(self.results_dir, 'gradient_models'),
            'verbose': False,
            'sequence_length': 5,
            "dt": test_dt,
            "T": test_dt*test_nt,
        }
        
        # Train models
        print("Training models for gradient test...")
        training_results = train_neural_operators(**gradient_config)
        
        # Create models for testing
        advection_model = training_results['advection']['model']
        diffusion_model = training_results['diffusion']['model']
        
        # Test data
        advection_dataset = training_results['advection']['dataset']['trajectories']
        u0 = torch.from_numpy(np.array(advection_dataset)[0, 0]).float().unsqueeze(0)
        
        print(f"Testing gradient computation on tensor of shape: {u0.shape}")
        
        # Enable gradients
        u0.requires_grad_(True)
        
        # Test both models with both methods
        gradient_results = {}
        
        for model_name, model in [('advection', advection_model), ('diffusion', diffusion_model)]:
            print(f"\nTesting {model_name} model gradients...")
            
            gradient_results[model_name] = {}
            
            for method_name, use_adjoint in [('standard', False), ('adjoint', True)]:
                print(f"  Computing {method_name} gradients...")
                
                # Forward pass
                u0_copy = u0.clone().detach().requires_grad_(True)
                
                _, solution = model(u0_copy, T=test_dt*test_nt, num_steps=test_nt, 
                                  method='dopri5', use_adjoint=use_adjoint)
                
                # Compute loss (simple L2 norm of final state)
                loss = torch.sum(solution[-1]**2)
                
                # Backward pass
                loss.backward()
                
                # Store gradients
                gradient_results[model_name][method_name] = {
                    'loss': loss.item(),
                    'gradient': u0_copy.grad.clone().detach().numpy() if u0_copy.grad is not None else None,
                    'final_state': solution[-1].detach().numpy()
                }
                
                print(f"    Loss: {loss.item():.6f}")
                if u0_copy.grad is not None:
                    grad_norm = torch.norm(u0_copy.grad).item()
                    print(f"    Gradient norm: {grad_norm:.6f}")
                else:
                    print("    No gradients computed")
        
        # Compare gradients
        print("\nGRADIENT COMPARISON:")
        print("-" * 20)
        
        comparison_metrics = {}
        for model_name in gradient_results:
            print(f"\n{model_name.upper()} model:")
            
            standard_grad = gradient_results[model_name]['standard']['gradient']
            adjoint_grad = gradient_results[model_name]['adjoint']['gradient']
            
            if standard_grad is not None and adjoint_grad is not None:
                # Compute differences
                grad_diff = np.abs(standard_grad - adjoint_grad)
                max_grad_diff = np.max(grad_diff)
                mean_grad_diff = np.mean(grad_diff)
                relative_grad_diff = max_grad_diff / (np.max(np.abs(standard_grad)) + 1e-12)
                
                comparison_metrics[model_name] = {
                    'max_gradient_diff': max_grad_diff,
                    'mean_gradient_diff': mean_grad_diff,
                    'relative_gradient_diff': relative_grad_diff,
                    'standard_grad_norm': np.linalg.norm(standard_grad),
                    'adjoint_grad_norm': np.linalg.norm(adjoint_grad)
                }
                
                print(f"  Max gradient difference: {max_grad_diff:.2e}")
                print(f"  Mean gradient difference: {mean_grad_diff:.2e}")
                print(f"  Relative gradient difference: {relative_grad_diff:.2e}")
                print(f"  Standard gradient norm: {comparison_metrics[model_name]['standard_grad_norm']:.6f}")
                print(f"  Adjoint gradient norm: {comparison_metrics[model_name]['adjoint_grad_norm']:.6f}")
                
                # Check if gradients match
                if relative_grad_diff < 1e-6:
                    print(f"  ✓ PASS - Gradients match within tolerance")
                elif relative_grad_diff < 1e-3:
                    print(f"  ⚠ WARN - Small gradient differences detected")
                else:
                    print(f"  ✗ FAIL - Large gradient differences detected")
            else:
                print(f"  ✗ ERROR - Could not compute gradients")
                comparison_metrics[model_name] = None
        
        print("\nGRADIENT ACCURACY SUMMARY:")
        print("-" * 30)
        print("The adjoint method should produce identical gradients to the standard method")
        print("for the same computation. Small differences may arise from numerical precision.")
        print("Large differences indicate potential implementation issues.")
        
        return {
            'gradient_results': gradient_results,
            'comparison_metrics': comparison_metrics,
            'test_config': gradient_config
        }


def main():
    """Main function to run adjoint neural operator splitting tests."""
    parser = argparse.ArgumentParser(description='Adjoint Neural Operator Splitting Test Suite')
    parser.add_argument('--test-type', choices=['adjoint', 'gradient', 'both'], default='adjoint',
                        help='Type of adjoint test to run (default: adjoint)')
    parser.add_argument('--beta', type=float, default=5.0,
                        help='Advection speed coefficient (default: 1.0)')
    parser.add_argument('--nu', type=float, default=0.1,
                        help='Diffusion viscosity coefficient (default: 0.1)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    args = parser.parse_args()
    
    print("Neural Operator Splitting - Adjoint Solver Test Suite")
    print("=" * 60)
    print(f"Running {args.test_type} test...")
    print("This demonstrates discretize-then-optimize vs optimize-then-discretize")
    
    # Create test suite
    test_suite = AdjointNeuralSplittingTestSuite(
        nx=128, 
        L=16,
        beta=args.beta,
        nu=args.nu,
        results_dir='./paper/neural-operator-splitting/test_results_adjoint'
    )
    
    results = {}
    
    if args.test_type in ['adjoint', 'both']:
        print("\nRunning adjoint solver test...")
        results['adjoint'] = test_suite.test_adjoint_solver(
            train_epochs=args.epochs, 
            test_dt=0.1, 
            test_nt=100
        )
    
    if args.test_type in ['gradient', 'both']:
        print("\nRunning gradient accuracy test...")
        results['gradient'] = test_suite.test_gradient_accuracy(
            train_epochs=max(30, args.epochs//2),
            test_dt=0.1,
            test_nt=100
        )
    
    print("\nAdjoint test suite completed successfully!")
    print(f"Results saved to: {test_suite.results_dir}")
    
    return results


if __name__ == "__main__":
    main()
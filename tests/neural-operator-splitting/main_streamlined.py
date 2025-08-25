"""
Streamlined main script for neural operator splitting experiments.
Uses modular components for better maintainability.
"""

import argparse
import os
import time
import json
import random
import numpy as np
import torch
from typing import Dict, List

# Import the modular components (absolute imports to work when running directly)
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from experiment_core import NeuralOperatorSplittingExperiment  
from visualization import create_plots, create_coefficient_estimation_plots
from results_management import organize_results_by_parameters, save_results, convert_for_json


def organize_aggregate_directory(first_result: Dict, num_integration_steps: int, n_test_trajectories: int = None, n_trajectories_per_operator: int = None) -> str:
    """Create organized directory for aggregate results."""
    test_ranges = first_result.get('test_parameter_ranges', {})
    v_range = test_ranges.get('v_range', (0, 1))
    D_range = test_ranges.get('D_range', (0, 1))
    v_min, v_max = v_range
    D_min, D_max = D_range
    
    # Get trajectory parameters from results if not provided
    if n_test_trajectories is None:
        n_test_trajectories = first_result.get('n_test_trajectories', 1)
    if n_trajectories_per_operator is None:
        n_trajectories_per_operator = first_result.get('n_trajectories_per_operator', 1)
    
    param_dir = f"v_{v_min:.1f}_{v_max:.1f}_D_{D_min:.3f}_{D_max:.3f}"
    operator_dir = f"operators_{first_result.get('num_operators', 0)}"
    test_traj_dir = f"test_traj_{n_test_trajectories}"
    traj_per_op_dir = f"traj_per_op_{n_trajectories_per_operator}"
    integration_dir = f"integration_steps_{num_integration_steps}" if num_integration_steps else "integration_steps_default"
    
    aggregate_parent = os.path.join("./results/neural_operator_splitting", param_dir, operator_dir, test_traj_dir, traj_per_op_dir, integration_dir)
    os.makedirs(aggregate_parent, exist_ok=True)
    
    return aggregate_parent


def aggregate_statistics(all_results: List[Dict], base_output_dir: str, args) -> None:
    """Compute and save aggregate statistics across multiple runs."""
    print(f"\n" + "="*60)
    print("AGGREGATING STATISTICS")
    print("="*60)
    
    # Basic performance metrics
    validation_errors = [min(r['selection_history']['errors']) for r in all_results]
    final_errors = [r['evaluation_after_finetuning']['error'] for r in all_results]
    before_finetuning_errors = [r['evaluation_before_finetuning']['error'] for r in all_results]
    composition_lengths = [r['composition_length'] for r in all_results]
    improvement_ratios = [r['finetuning_effectiveness']['improvement_ratio'] for r in all_results]
    
    # Coefficient estimation metrics
    coefficient_metrics = {
        'v_estimates': [],
        'D_estimates': [],
        'v_ground_truth': [],
        'D_ground_truth': [],
        'v_errors': [],
        'D_errors': [],
        'v_squared_errors': [],
        'D_squared_errors': []
    }
    
    # Collect coefficient estimation data from all runs
    for r in all_results:
        if 'coefficient_estimates' in r and 'ground_truth_params' in r and r['ground_truth_params']:
            coeff_est = r['coefficient_estimates']
            gt_params = r['ground_truth_params'][0]
            
            est_v = coeff_est.get('estimated_v', 0.0)
            est_D = coeff_est.get('estimated_D', 0.0)
            gt_v, gt_D = gt_params
            
            coefficient_metrics['v_estimates'].append(est_v)
            coefficient_metrics['D_estimates'].append(est_D)
            coefficient_metrics['v_ground_truth'].append(gt_v)
            coefficient_metrics['D_ground_truth'].append(gt_D)
            coefficient_metrics['v_errors'].append(est_v - gt_v)
            coefficient_metrics['D_errors'].append(est_D - gt_D)
            coefficient_metrics['v_squared_errors'].append((est_v - gt_v) ** 2)
            coefficient_metrics['D_squared_errors'].append((est_D - gt_D) ** 2)
    
    # Compute coefficient statistics
    coeff_stats = None
    if coefficient_metrics['v_estimates']:
        coeff_stats = {
            'v': {
                'mse': np.mean(coefficient_metrics['v_squared_errors']),
                'mae': np.mean(np.abs(coefficient_metrics['v_errors'])),
                'rmse': np.sqrt(np.mean(coefficient_metrics['v_squared_errors'])),
                'bias': np.mean(coefficient_metrics['v_errors']),
                'relative_mae': np.mean(np.abs(coefficient_metrics['v_errors'])) / max(np.mean(coefficient_metrics['v_ground_truth']), 1e-8) * 100,
                'mean_estimate': np.mean(coefficient_metrics['v_estimates']),
                'std_estimate': np.std(coefficient_metrics['v_estimates']),
                'mean_ground_truth': np.mean(coefficient_metrics['v_ground_truth'])
            },
            'D': {
                'mse': np.mean(coefficient_metrics['D_squared_errors']),
                'mae': np.mean(np.abs(coefficient_metrics['D_errors'])),
                'rmse': np.sqrt(np.mean(coefficient_metrics['D_squared_errors'])),
                'bias': np.mean(coefficient_metrics['D_errors']),
                'relative_mae': np.mean(np.abs(coefficient_metrics['D_errors'])) / max(np.mean(coefficient_metrics['D_ground_truth']), 1e-8) * 100,
                'mean_estimate': np.mean(coefficient_metrics['D_estimates']),
                'std_estimate': np.std(coefficient_metrics['D_estimates']),
                'mean_ground_truth': np.mean(coefficient_metrics['D_ground_truth'])
            },
            'combined': {
                'mse': np.mean(coefficient_metrics['v_squared_errors']) + np.mean(coefficient_metrics['D_squared_errors']),
                'mae': np.mean(np.abs(coefficient_metrics['v_errors'])) + np.mean(np.abs(coefficient_metrics['D_errors']))
            }
        }
    
    # Aggregate statistics
    stats = {
        'num_runs': len(all_results),
        'validation_error': {
            'mean': np.mean(validation_errors),
            'std': np.std(validation_errors),
            'min': np.min(validation_errors),
            'max': np.max(validation_errors)
        },
        'before_finetuning_error': {
            'mean': np.mean(before_finetuning_errors),
            'std': np.std(before_finetuning_errors),
            'min': np.min(before_finetuning_errors),
            'max': np.max(before_finetuning_errors)
        },
        'after_finetuning_error': {
            'mean': np.mean(final_errors),
            'std': np.std(final_errors),
            'min': np.min(final_errors),
            'max': np.max(final_errors)
        },
        'finetuning_improvement': {
            'mean': np.mean(improvement_ratios),
            'std': np.std(improvement_ratios),
            'min': np.min(improvement_ratios),
            'max': np.max(improvement_ratios)
        },
        'composition_length': {
            'mean': np.mean(composition_lengths),
            'std': np.std(composition_lengths),
            'min': np.min(composition_lengths),
            'max': np.max(composition_lengths)
        },
        'generalization_ratio': {
            'mean': np.mean([f/v for f, v in zip(final_errors, validation_errors)]),
            'std': np.std([f/v for f, v in zip(final_errors, validation_errors)]),
            'min': np.min([f/v for f, v in zip(final_errors, validation_errors)]),
            'max': np.max([f/v for f, v in zip(final_errors, validation_errors)])
        },
        'coefficient_estimation': coeff_stats
    }
    
    # Save aggregate statistics in organized structure  
    aggregate_parent = organize_aggregate_directory(all_results[0], args.num_integration_steps, args.n_test_trajectories, args.n_trajectories_per_operator)
    stats_file = os.path.join(aggregate_parent, 'aggregate_statistics.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2, default=lambda x: x.item() if hasattr(x, 'item') else x)
    
    # Also save copy in timestamped directory for backward compatibility
    stats_file_old = os.path.join(base_output_dir, 'aggregate_statistics.json')  
    with open(stats_file_old, 'w') as f:
        json.dump(stats, f, indent=2, default=lambda x: x.item() if hasattr(x, 'item') else x)
    
    # Save detailed coefficient estimation data
    if coeff_stats:
        coeff_aggregate_file = os.path.join(aggregate_parent, 'coefficient_estimation_aggregate.json')
        detailed_coeff_data = {
            'statistics': coeff_stats,
            'raw_data': {
                'v_estimates': coefficient_metrics['v_estimates'],
                'D_estimates': coefficient_metrics['D_estimates'],
                'v_ground_truth': coefficient_metrics['v_ground_truth'],
                'D_ground_truth': coefficient_metrics['D_ground_truth'],
                'v_errors': coefficient_metrics['v_errors'],
                'D_errors': coefficient_metrics['D_errors']
            },
            'run_details': []
        }
        
        # Add per-run details
        for i, r in enumerate(all_results):
            if 'coefficient_estimates' in r and 'ground_truth_params' in r:
                run_detail = {
                    'run_index': i,
                    'coefficient_estimates': r['coefficient_estimates'],
                    'ground_truth_params': r['ground_truth_params'],
                    'best_composition': r.get('best_composition', []),
                    'validation_error': min(r['selection_history']['errors']),
                    'before_finetuning_error': r['evaluation_before_finetuning']['error'],
                    'after_finetuning_error': r['evaluation_after_finetuning']['error'],
                    'improvement_ratio': r['finetuning_effectiveness']['improvement_ratio'],
                    'test_parameter_ranges': r.get('test_parameter_ranges', {})
                }
                detailed_coeff_data['run_details'].append(run_detail)
        
        with open(coeff_aggregate_file, 'w') as f:
            json.dump(convert_for_json(detailed_coeff_data), f, indent=2)
        
        # Also save copy in timestamped directory for backward compatibility
        coeff_aggregate_file_old = os.path.join(base_output_dir, 'coefficient_estimation_aggregate.json')
        with open(coeff_aggregate_file_old, 'w') as f:
            json.dump(convert_for_json(detailed_coeff_data), f, indent=2)
    
    # Print statistics
    print(f"Aggregate Statistics ({args.num_runs} runs):")
    print(f"  Validation Error:         {stats['validation_error']['mean']:.6f} ± {stats['validation_error']['std']:.6f}")
    print(f"  Before Finetuning Error:  {stats['before_finetuning_error']['mean']:.6f} ± {stats['before_finetuning_error']['std']:.6f}")
    print(f"  After Finetuning Error:   {stats['after_finetuning_error']['mean']:.6f} ± {stats['after_finetuning_error']['std']:.6f}")
    print(f"  Finetuning Improvement:   {stats['finetuning_improvement']['mean']:.2f}x ± {stats['finetuning_improvement']['std']:.2f}")
    print(f"  Composition Length:       {stats['composition_length']['mean']:.1f} ± {stats['composition_length']['std']:.1f}")
    print(f"  Generalization Ratio:     {stats['generalization_ratio']['mean']:.2f} ± {stats['generalization_ratio']['std']:.2f}")
    print(f"  Best After-Finetuning Error: {stats['after_finetuning_error']['min']:.6f}")
    print(f"  Best Improvement Ratio:   {stats['finetuning_improvement']['max']:.2f}x")
    print(f"  Length Range: {int(stats['composition_length']['min'])} - {int(stats['composition_length']['max'])}")

    if coeff_stats:
        print(f"\n  Coefficient Estimation Metrics ({len(coefficient_metrics['v_estimates'])} runs with data):")
        print(f"  Advection Coefficient (v):")
        print(f"    MSE:  {coeff_stats['v']['mse']:.6f}")
        print(f"    MAE:  {coeff_stats['v']['mae']:.6f}")
        print(f"    RMSE: {coeff_stats['v']['rmse']:.6f}")
        print(f"    Bias: {coeff_stats['v']['bias']:.6f}")
        print(f"    Relative MAE: {coeff_stats['v']['relative_mae']:.1f}%")
        print(f"    Mean Estimate: {coeff_stats['v']['mean_estimate']:.3f} ± {coeff_stats['v']['std_estimate']:.3f}")
        print(f"    Mean Ground Truth: {coeff_stats['v']['mean_ground_truth']:.3f}")
        print(f"  Diffusion Coefficient (D):")
        print(f"    MSE:  {coeff_stats['D']['mse']:.6f}")
        print(f"    MAE:  {coeff_stats['D']['mae']:.6f}")
        print(f"    RMSE: {coeff_stats['D']['rmse']:.6f}")
        print(f"    Bias: {coeff_stats['D']['bias']:.6f}")
        print(f"    Relative MAE: {coeff_stats['D']['relative_mae']:.1f}%")
        print(f"    Mean Estimate: {coeff_stats['D']['mean_estimate']:.3f} ± {coeff_stats['D']['std_estimate']:.3f}")
        print(f"    Mean Ground Truth: {coeff_stats['D']['mean_ground_truth']:.3f}")
        print(f"  Overall:")
        print(f"    Combined MSE: {coeff_stats['combined']['mse']:.6f}")
        print(f"    Combined MAE: {coeff_stats['combined']['mae']:.6f}")

    print(f"\nExperiment completed successfully!")
    print(f"Results saved to: {base_output_dir}")
    print(f"Organized results in: {aggregate_parent}")


def main():
    """Main function for neural operator splitting experiments."""
    parser = argparse.ArgumentParser(description='Neural Operator Splitting Experiment')
    
    # Model and data parameters
    parser.add_argument('--checkpoint-path', required=True, help='Path to DISCO checkpoint')
    parser.add_argument('--num-operators', type=int, default=10, help='Number of operators to encode')
    parser.add_argument('--n-trajectories-per-operator', type=int, default=1, help='Number of trajectories per operator')
    
    # Optimization parameters
    parser.add_argument('--rel-loss-coeff', type=float, default=1, help='Relative L2 loss coefficient')
    parser.add_argument('--sparsity-coeff', type=float, default=0.01, help='Sparsity coefficient')
    parser.add_argument('--max-operators', type=int, default=5, help='Maximum operators in composition')
    parser.add_argument('--num-integration-steps', type=int, default=1, help='Number of integration steps')
    
    # Finetuning parameters
    parser.add_argument('--enable-finetuning', action='store_true', default=True, help='Enable finetuning')
    parser.add_argument('--optimize-latent', action='store_true', help='Optimize in latent space')
    parser.add_argument('--finetune-epochs', type=int, default=5, help='Number of finetuning epochs')
    parser.add_argument('--preservation-coeff', type=float, default=0.1, help='Preservation loss coefficient')
    parser.add_argument('--noise-level', type=float, default=0.001, help='Noise level for regularization')
    
    # Test parameters
    parser.add_argument('--n-test-trajectories', type=int, default=1, help='Number of test trajectories')
    parser.add_argument('--test-v-min', type=float, default=0.1, help='Test advection velocity min')
    parser.add_argument('--test-v-max', type=float, default=1.0, help='Test advection velocity max')
    parser.add_argument('--test-D-min', type=float, default=0.1, help='Test diffusion coefficient min')
    parser.add_argument('--test-D-max', type=float, default=1.0, help='Test diffusion coefficient max')
    
    # Experiment parameters
    parser.add_argument('--output-dir', default='./results/neural_operator_splitting', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num-runs', type=int, default=1, help='Number of experimental runs')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Create configuration
    config = {
        'checkpoint_path': args.checkpoint_path,
        'num_operators': args.num_operators,
        'rel_loss_2_coeff': args.rel_loss_coeff,
        'sparsity_coeff': args.sparsity_coeff,
        'max_operators': args.max_operators,
        'num_integration_steps': args.num_integration_steps,
        'enable_finetuning': args.enable_finetuning,
        'optimize_latent': args.optimize_latent,
        'finetune_epochs': args.finetune_epochs,
        'preservation_coeff': args.preservation_coeff,
        'noise_level': args.noise_level,
        'seed': args.seed,
        'batch_size': 4,
        'n_input_frames': 16,
        'n_output_frames': 34,
        'n_trajectories_per_operator': args.n_trajectories_per_operator,
        'operator_v_range': (0.01, 1.0),
        'operator_D_range': (0.001, 1.0),
        'n_test_trajectories': args.n_test_trajectories,
        'test_v_range': (args.test_v_min, args.test_v_max),
        'test_D_range': (args.test_D_min, args.test_D_max),
    }
    
    # Create base timestamped results directory
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    base_output_dir = os.path.join(args.output_dir, f'run_{timestamp}')
    config['output_dir'] = base_output_dir
    
    print("Neural Operator Splitting Experiment")
    print("=" * 60)
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 60)
    
    if args.num_runs == 1:
        # Single run
        experiment = NeuralOperatorSplittingExperiment(config)
        results = experiment.run_experiment()
        
        # Organize results with improved structure
        organized_output_dir = organize_results_by_parameters(results, base_output_dir, timestamp, args.num_integration_steps, args.n_test_trajectories, args.n_trajectories_per_operator)
        
        # Save results and create plots in organized directory
        save_results(results, organized_output_dir)
        create_plots(results, organized_output_dir)
        
        # Also save a copy in the timestamped directory for easy access
        save_results(results, base_output_dir)
        
    else:
        # Multiple runs for statistics
        all_results = []
        
        for run_idx in range(args.num_runs):
            print(f"\n" + "="*60)
            print(f"RUN {run_idx + 1}/{args.num_runs}")
            print("="*60)
            
            # Update config for this run
            run_config = config.copy()
            run_config['seed'] = args.seed + run_idx
            
            # Set seed for this run
            torch.manual_seed(run_config['seed'])
            np.random.seed(run_config['seed'])
            random.seed(run_config['seed'])
            
            # Run experiment
            experiment = NeuralOperatorSplittingExperiment(run_config)
            results = experiment.run_experiment()
            
            # Organize results with improved structure
            organized_output_dir = organize_results_by_parameters(results, base_output_dir, f"{timestamp}_run_{run_idx:03d}", args.num_integration_steps, args.n_test_trajectories, args.n_trajectories_per_operator)
            
            # Save individual run results in organized structure
            save_results(results, organized_output_dir)
            create_plots(results, organized_output_dir)
            
            all_results.append(results)
        
        # Create aggregate plots for coefficient estimation metrics in organized structure
        if len(all_results) > 1:
            aggregate_parent = organize_aggregate_directory(all_results[0], args.num_integration_steps, args.n_test_trajectories, args.n_trajectories_per_operator)
            create_coefficient_estimation_plots(all_results, aggregate_parent)
        
        # Compute and save aggregate statistics
        aggregate_statistics(all_results, base_output_dir, args)


if __name__ == "__main__":
    main()
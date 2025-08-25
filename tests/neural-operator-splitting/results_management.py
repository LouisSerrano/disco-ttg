"""
Results Management Module

Contains functions for organizing and saving experimental results from neural operator splitting experiments.
Extracted from tests/neural-operator-splitting/random_approach_iterative_finetune_operators.py
"""

import os
import json
import numpy as np
import torch
from typing import Dict


def organize_results_by_parameters(results: Dict, base_output_dir: str, run_timestamp: str, num_integration_steps: int = None, n_test_trajectories: int = None, n_trajectories_per_operator: int = None) -> str:
    """
    Organize results by parameter ranges, operator count, trajectory parameters, integration steps, and run timestamp.
    
    Structure: v_min_max_D_min_max/operators_N/test_traj_N/traj_per_op_N/integration_steps_N/run_TIMESTAMP/
    
    Args:
        results: experiment results dictionary
        base_output_dir: base directory for results
        run_timestamp: timestamp for this experimental run
        num_integration_steps: number of integration steps used
        n_test_trajectories: number of test trajectories
        n_trajectories_per_operator: number of trajectories per operator
        
    Returns:
        organized output directory path
    """
    # Extract parameter information
    test_ranges = results.get('test_parameter_ranges', {})
    v_range = test_ranges.get('v_range', (0, 1))
    D_range = test_ranges.get('D_range', (0, 1))
    num_operators = results.get('num_operators', 0)
    
    # Get trajectory parameters from results if not provided
    if n_test_trajectories is None:
        n_test_trajectories = results.get('n_test_trajectories', 1)
    if n_trajectories_per_operator is None:
        n_trajectories_per_operator = results.get('n_trajectories_per_operator', 1)
    
    # Create organized directory structure: v_range/D_range -> operators -> test_traj -> traj_per_op -> integration_steps -> run_timestamp
    v_min, v_max = v_range
    D_min, D_max = D_range
    
    # Format parameter ranges for directory names
    param_dir = f"v_{v_min:.1f}_{v_max:.1f}_D_{D_min:.3f}_{D_max:.3f}"
    operator_dir = f"operators_{num_operators}"
    test_traj_dir = f"test_traj_{n_test_trajectories}"
    traj_per_op_dir = f"traj_per_op_{n_trajectories_per_operator}"
    integration_dir = f"integration_steps_{num_integration_steps}" if num_integration_steps else "integration_steps_default"
    run_dir = f"run_{run_timestamp}"
    
    # Create full organized path: param_dir/operator_dir/test_traj_dir/traj_per_op_dir/integration_dir/run_dir
    organized_dir = os.path.join(base_output_dir, param_dir, operator_dir, test_traj_dir, traj_per_op_dir, integration_dir, run_dir)
    os.makedirs(organized_dir, exist_ok=True)
    
    return organized_dir


def convert_for_json(obj):
    """Convert numpy arrays and tensors to JSON-serializable format."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_for_json(item) for item in obj]
    else:
        return obj


def save_results(results: Dict, output_dir: str):
    """Save experiment results with organized structure."""
    
    # Save detailed results
    results_file = os.path.join(output_dir, 'results.json')
    with open(results_file, 'w') as f:
        json.dump(convert_for_json(results), f, indent=2)
    
    # Enhanced summary with before/after finetuning comparison
    before_error = results.get('evaluation_before_finetuning', {}).get('error', None)
    after_error = results.get('evaluation_after_finetuning', {}).get('error', None)
    finetuning_effectiveness = results.get('finetuning_effectiveness', {})
    
    summary = {
        'num_operators': results.get('num_operators', None),
        'composition_length': results.get('composition_length', None),
        'best_composition': results.get('best_composition', []),
        'test_parameter_ranges': results.get('test_parameter_ranges', {}),
        'coefficient_estimates': results.get('coefficient_estimates', {}),
        'ground_truth_params': results.get('ground_truth_params', []),
        'before_finetuning_error': before_error,
        'after_finetuning_error': after_error,
        'finetuning_improvement_ratio': finetuning_effectiveness.get('improvement_ratio', None),
        'finetuning_relative_improvement_pct': finetuning_effectiveness.get('relative_improvement', None),
        'error_reduction': finetuning_effectiveness.get('error_reduction', None)
    }
    
    # Add coefficient estimation accuracy if ground truth available
    if results.get('ground_truth_params') and results.get('coefficient_estimates'):
        gt_params = results['ground_truth_params'][0] if results['ground_truth_params'] else (0, 0)
        estimates = results['coefficient_estimates']
        
        gt_v, gt_D = gt_params
        est_v, est_D = estimates.get('estimated_v', 0), estimates.get('estimated_D', 0)
        
        v_error = abs(gt_v - est_v) / max(gt_v, 1e-8) if gt_v != 0 else abs(est_v)
        D_error = abs(gt_D - est_D) / max(gt_D, 1e-8) if gt_D != 0 else abs(est_D)
        
        # Calculate detailed metrics for single run
        v_absolute_error = abs(est_v - gt_v)
        D_absolute_error = abs(est_D - gt_D)
        v_squared_error = (est_v - gt_v) ** 2
        D_squared_error = (est_D - gt_D) ** 2
        
        summary['coefficient_estimation_accuracy'] = {
            'ground_truth_v': gt_v,
            'ground_truth_D': gt_D,
            'estimated_v': est_v,
            'estimated_D': est_D,
            'v_absolute_error': v_absolute_error,
            'D_absolute_error': D_absolute_error,
            'v_squared_error': v_squared_error,
            'D_squared_error': D_squared_error,
            'v_relative_error': v_error,
            'D_relative_error': D_error,
            'v_bias': est_v - gt_v,
            'D_bias': est_D - gt_D,
            'combined_mae': v_absolute_error + D_absolute_error,
            'combined_mse': v_squared_error + D_squared_error,
            'combined_rmse': np.sqrt(v_squared_error + D_squared_error)
        }
    
    summary_file = os.path.join(output_dir, 'summary.json')
    with open(summary_file, 'w') as f:
        json.dump(convert_for_json(summary), f, indent=2)
    
    # Save detailed coefficient estimation results if available
    if results.get('coefficient_estimates'):
        coeff_file = os.path.join(output_dir, 'coefficient_estimation.json')
        coeff_data = {
            'coefficient_estimates': results['coefficient_estimates'],
            'ground_truth_params': results.get('ground_truth_params', []),
            'estimation_accuracy': summary.get('coefficient_estimation_accuracy', {}),
            'operator_contributions': results['coefficient_estimates'].get('operator_contributions', []),
            'best_composition': results.get('best_composition', []),
            'operator_params': results.get('operator_params', [])
        }
        with open(coeff_file, 'w') as f:
            json.dump(convert_for_json(coeff_data), f, indent=2)
    
    print(f"Results saved to {output_dir}")
    
    # Enhanced summary printout
    before_str = f"{before_error:.6f}" if before_error is not None else "N/A"
    after_str = f"{after_error:.6f}" if after_error is not None else "N/A"
    improvement_str = f"{finetuning_effectiveness.get('improvement_ratio', 1.0):.2f}x" if finetuning_effectiveness.get('improvement_ratio') else "N/A"
    
    print(f"Summary:")
    print(f"  Operators: {results.get('num_operators', 'N/A')}, Composition length: {results.get('composition_length', 'N/A')}")
    print(f"  Before finetuning error: {before_str}")
    print(f"  After finetuning error:  {after_str}")
    print(f"  Finetuning improvement: {improvement_str}")
    
    if 'coefficient_estimation_accuracy' in summary:
        acc = summary['coefficient_estimation_accuracy']
        print(f"  Coefficient estimation - v: {acc['estimated_v']:.3f} (GT: {acc['ground_truth_v']:.3f}, err: {acc['v_relative_error']*100:.1f}%)")
        print(f"                          - D: {acc['estimated_D']:.3f} (GT: {acc['ground_truth_D']:.3f}, err: {acc['D_relative_error']*100:.1f}%)")
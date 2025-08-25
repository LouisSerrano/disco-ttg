"""
Visualization utilities for neural operator splitting experiments.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, List


def create_trajectory_snapshots(predictions, ground_truth, output_dir: str, n_snapshots: int = 6):
    """
    Create trajectory snapshots showing prediction and ground truth in separate plots at different timestamps.
    
    Args:
        predictions: [time, batch, channel, spatial] prediction tensor
        ground_truth: [time, batch, channel, spatial] ground truth tensor
        output_dir: Directory to save plots
        n_snapshots: Number of time snapshots to show
    """
    sample_idx = 0  # Use first sample
    
    # Select time indices to plot
    time_indices = np.linspace(0, predictions.shape[0] - 1, n_snapshots, dtype=int)
    colors = plt.cm.viridis(np.linspace(0, 1, n_snapshots))
    
    # Find global min/max for consistent color scaling
    vmin = min(np.min(predictions[:, sample_idx, 0, :]), np.min(ground_truth[:, sample_idx, 0, :]))
    vmax = max(np.max(predictions[:, sample_idx, 0, :]), np.max(ground_truth[:, sample_idx, 0, :]))
    
    # Create figure with 2 rows (prediction and ground truth) and 2 columns
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Get spatial coordinates (assuming uniform grid)
    x = np.arange(predictions.shape[-1])
    
    # Row 1: Neural Prediction
    # Left: Heatmap of trajectory evolution
    pred_heatmap = predictions[:, sample_idx, 0, :].T  # Transpose for imshow
    im1 = axes[0, 0].imshow(pred_heatmap, aspect='auto', cmap='RdBu_r', 
                           extent=[0, predictions.shape[0]-1, x[-1], x[0]], 
                           vmin=vmin, vmax=vmax)
    axes[0, 0].set_title('Neural Prediction: Trajectory Evolution')
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Spatial coordinate (x)')
    
    # Add vertical lines at snapshot times
    for t_idx in time_indices:
        axes[0, 0].axvline(x=t_idx, color='white', linestyle='--', alpha=0.8, linewidth=1)
    
    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=axes[0, 0])
    cbar1.set_label('Solution u(x,t)')
    
    # Right: Trajectory snapshots at different times
    for i, (t_idx, color) in enumerate(zip(time_indices, colors)):
        axes[0, 1].plot(x, predictions[t_idx, sample_idx, 0, :], 
                       color=color, linewidth=2, alpha=0.8, 
                       label=f't = {t_idx}')
    
    axes[0, 1].set_title('Neural Prediction: Trajectory Snapshots')
    axes[0, 1].set_xlabel('Spatial coordinate (x)')
    axes[0, 1].set_ylabel('u(x,t)')
    axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Row 2: Ground Truth
    # Left: Heatmap of trajectory evolution
    truth_heatmap = ground_truth[:, sample_idx, 0, :].T  # Transpose for imshow
    im2 = axes[1, 0].imshow(truth_heatmap, aspect='auto', cmap='RdBu_r', 
                           extent=[0, ground_truth.shape[0]-1, x[-1], x[0]], 
                           vmin=vmin, vmax=vmax)
    axes[1, 0].set_title('Ground Truth: Trajectory Evolution')
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Spatial coordinate (x)')
    
    # Add vertical lines at snapshot times
    for t_idx in time_indices:
        axes[1, 0].axvline(x=t_idx, color='white', linestyle='--', alpha=0.8, linewidth=1)
    
    # Add colorbar
    cbar2 = plt.colorbar(im2, ax=axes[1, 0])
    cbar2.set_label('Solution u(x,t)')
    
    # Right: Trajectory snapshots at different times
    for i, (t_idx, color) in enumerate(zip(time_indices, colors)):
        axes[1, 1].plot(x, ground_truth[t_idx, sample_idx, 0, :], 
                       color=color, linewidth=2, alpha=0.8, 
                       label=f't = {t_idx}')
    
    axes[1, 1].set_title('Ground Truth: Trajectory Snapshots')
    axes[1, 1].set_xlabel('Spatial coordinate (x)')
    axes[1, 1].set_ylabel('u(x,t)')
    axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'trajectory_snapshots.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_coefficient_estimation_plots(all_results: List[Dict], output_dir: str):
    """Create plots for coefficient estimation metrics across multiple runs."""
    print("Creating coefficient estimation plots...")
    
    plots_dir = os.path.join(output_dir, 'coefficient_estimation_plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Collect data from all runs
    v_estimates, D_estimates = [], []
    v_ground_truth, D_ground_truth = [], []
    v_errors, D_errors = [], []
    run_indices = []
    
    for i, r in enumerate(all_results):
        if 'coefficient_estimates' in r and 'ground_truth_params' in r and r['ground_truth_params']:
            coeff_est = r['coefficient_estimates']
            gt_params = r['ground_truth_params'][0]
            
            est_v = coeff_est.get('estimated_v', 0.0)
            est_D = coeff_est.get('estimated_D', 0.0)
            gt_v, gt_D = gt_params
            
            v_estimates.append(est_v)
            D_estimates.append(est_D)
            v_ground_truth.append(gt_v)
            D_ground_truth.append(gt_D)
            v_errors.append(est_v - gt_v)
            D_errors.append(est_D - gt_D)
            run_indices.append(i)
    
    if not v_estimates:  # No data to plot
        return
    
    # Create comprehensive figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Estimates vs Ground Truth for Advection
    axes[0, 0].scatter(v_ground_truth, v_estimates, alpha=0.7, color='red')
    min_v, max_v = min(min(v_ground_truth), min(v_estimates)), max(max(v_ground_truth), max(v_estimates))
    axes[0, 0].plot([min_v, max_v], [min_v, max_v], 'k--', alpha=0.7, label='Perfect estimation')
    axes[0, 0].set_xlabel('Ground Truth Advection (v)')
    axes[0, 0].set_ylabel('Estimated Advection (v)')
    axes[0, 0].set_title('Advection Coefficient: Estimates vs Ground Truth')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Estimates vs Ground Truth for Diffusion
    axes[0, 1].scatter(D_ground_truth, D_estimates, alpha=0.7, color='blue')
    min_D, max_D = min(min(D_ground_truth), min(D_estimates)), max(max(D_ground_truth), max(D_estimates))
    axes[0, 1].plot([min_D, max_D], [min_D, max_D], 'k--', alpha=0.7, label='Perfect estimation')
    axes[0, 1].set_xlabel('Ground Truth Diffusion (D)')
    axes[0, 1].set_ylabel('Estimated Diffusion (D)')
    axes[0, 1].set_title('Diffusion Coefficient: Estimates vs Ground Truth')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Error Distribution for Advection
    axes[0, 2].hist(v_errors, bins=min(10, len(v_errors)), alpha=0.7, color='red', edgecolor='black')
    axes[0, 2].axvline(0, color='black', linestyle='--', alpha=0.7, label='Perfect estimation')
    axes[0, 2].set_xlabel('Estimation Error (v)')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Advection Coefficient Error Distribution')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Error Distribution for Diffusion
    axes[1, 0].hist(D_errors, bins=min(10, len(D_errors)), alpha=0.7, color='blue', edgecolor='black')
    axes[1, 0].axvline(0, color='black', linestyle='--', alpha=0.7, label='Perfect estimation')
    axes[1, 0].set_xlabel('Estimation Error (D)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Diffusion Coefficient Error Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Errors over Runs
    axes[1, 1].plot(run_indices, v_errors, 'ro-', alpha=0.7, label='Advection Error', markersize=4)
    axes[1, 1].plot(run_indices, D_errors, 'bo-', alpha=0.7, label='Diffusion Error', markersize=4)
    axes[1, 1].axhline(0, color='black', linestyle='--', alpha=0.7)
    axes[1, 1].set_xlabel('Run Index')
    axes[1, 1].set_ylabel('Estimation Error')
    axes[1, 1].set_title('Estimation Errors Across Runs')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Combined Error Metrics
    v_mae = np.mean(np.abs(v_errors))
    D_mae = np.mean(np.abs(D_errors))
    v_mse = np.mean(np.array(v_errors) ** 2)
    D_mse = np.mean(np.array(D_errors) ** 2)
    v_rmse = np.sqrt(v_mse)
    D_rmse = np.sqrt(D_mse)
    
    metrics = ['MAE', 'MSE', 'RMSE']
    v_metrics = [v_mae, v_mse, v_rmse]
    D_metrics = [D_mae, D_mse, D_rmse]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    axes[1, 2].bar(x - width/2, v_metrics, width, label='Advection (v)', alpha=0.7, color='red')
    axes[1, 2].bar(x + width/2, D_metrics, width, label='Diffusion (D)', alpha=0.7, color='blue')
    axes[1, 2].set_ylabel('Error Value')
    axes[1, 2].set_title('Error Metrics Summary')
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(metrics)
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_yscale('log')
    
    plt.suptitle(f'Coefficient Estimation Analysis ({len(v_estimates)} runs)', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'coefficient_estimation_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Coefficient estimation plots saved to {plots_dir}")


def create_plots(results: Dict, output_dir: str):
    """Create comprehensive plots of the experiment results."""
    print("Creating plots...")
    
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot 0: NEW - Before/After Finetuning Comparison
    if 'evaluation_before_finetuning' in results and 'evaluation_after_finetuning' in results:
        fig = plt.figure(figsize=(16, 5))
        plt.clf()  # Clear any previous figure content
        
        before_error = results['evaluation_before_finetuning']['error']
        after_error = results['evaluation_after_finetuning']['error']
        improvement_ratio = results.get('finetuning_effectiveness', {}).get('improvement_ratio', 1.0)
        
        # Plot 1: Error comparison
        plt.subplot(1, 3, 1)
        errors = [before_error, after_error]
        labels = ['Before Finetuning', 'After Finetuning']
        colors = ['red', 'green']
        bars = plt.bar(labels, errors, color=colors, alpha=0.7)
        plt.ylabel('Relative L2 Error')
        plt.title('Finetuning Effectiveness')
        plt.yscale('log')
        
        # Add value labels on bars (positioned better)
        for bar, error in zip(bars, errors):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{error:.4f}', ha='center', va='bottom', fontsize=9)
        
        # Add improvement text (positioned in upper area)
        plt.text(0.5, 0.8, f'Improvement: {improvement_ratio:.2f}x', 
                ha='center', fontsize=11, transform=plt.gca().transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Coefficient Estimation Accuracy
        plt.subplot(1, 3, 2)
        if 'coefficient_estimates' in results and 'ground_truth_params' in results and results['ground_truth_params']:
            coeff_est = results['coefficient_estimates']
            gt_params = results['ground_truth_params'][0]
            
            gt_v, gt_D = gt_params
            est_v, est_D = coeff_est.get('estimated_v', 0), coeff_est.get('estimated_D', 0)
            
            categories = ['Advection (v)', 'Diffusion (D)']
            ground_truth = [gt_v, gt_D]
            estimated = [est_v, est_D]
            
            x = np.arange(len(categories))
            width = 0.35
            
            plt.bar(x - width/2, ground_truth, width, label='Ground Truth', alpha=0.7, color='blue')
            plt.bar(x + width/2, estimated, width, label='Estimated', alpha=0.7, color='orange')
            
            plt.ylabel('Coefficient Value')
            plt.title('Coefficient Estimation Accuracy')
            plt.xticks(x, categories)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Add error text
            v_error = abs(gt_v - est_v) / max(gt_v, 1e-8) if gt_v != 0 else abs(est_v)
            D_error = abs(gt_D - est_D) / max(gt_D, 1e-8) if gt_D != 0 else abs(est_D)
            plt.text(0, max(max(ground_truth), max(estimated)) * 0.8, 
                    f'v error: {v_error*100:.1f}%\nD error: {D_error*100:.1f}%', 
                    ha='center', fontsize=10, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        else:
            plt.text(0.5, 0.5, 'No coefficient estimation data', ha='center', va='center', 
                    transform=plt.gca().transAxes, fontsize=12)
            plt.title('Coefficient Estimation Accuracy')
        
        # Plot 3: Operator Contributions
        plt.subplot(1, 3, 3)
        if 'coefficient_estimates' in results and 'operator_contributions' in results['coefficient_estimates']:
            contributions = results['coefficient_estimates']['operator_contributions']
            
            if contributions:
                op_indices = [c['operator_index'] for c in contributions]
                v_contributions = [c['v_contribution'] for c in contributions]
                D_contributions = [c['D_contribution'] for c in contributions]
                colors = ['red' if c['operator_type'] == 'advection' else 'blue' if c['operator_type'] == 'diffusion' else 'green' 
                         for c in contributions]
                
                # Stack v and D contributions
                x_pos = range(len(op_indices))
                plt.bar(x_pos, v_contributions, color='red', alpha=0.7, label='Advection (v)')
                plt.bar(x_pos, D_contributions, bottom=v_contributions, color='blue', alpha=0.7, label='Diffusion (D)')
                
                plt.xlabel('Position in Composition')
                plt.ylabel('Parameter Value')
                plt.title('Operator Parameter Contributions')
                plt.xticks(x_pos, [f'Op {idx}' for idx in op_indices], rotation=45)
                
                # Add legend
                plt.legend(loc='upper right')
                plt.grid(True, alpha=0.3)
            else:
                plt.text(0.5, 0.5, 'No operator contribution data', ha='center', va='center', 
                        transform=plt.gca().transAxes, fontsize=12)
        else:
            plt.text(0.5, 0.5, 'No operator contribution data', ha='center', va='center', 
                    transform=plt.gca().transAxes, fontsize=12)
            plt.title('Individual Operator Contributions')
        
        plt.tight_layout()
        try:
            plt.savefig(os.path.join(plots_dir, 'finetuning_and_coefficients.png'), dpi=300, bbox_inches='tight')
        except ValueError as e:
            print(f"Warning: Could not save finetuning plot due to size constraints: {e}")
            # Save at lower DPI to avoid memory issues
            plt.savefig(os.path.join(plots_dir, 'finetuning_and_coefficients.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    # Plot 1: Greedy selection results
    if 'selection_history' in results:
        fig = plt.figure(figsize=(15, 10))
        plt.clf()
        
        selection_history = results['selection_history']
        
        # Plot composition errors over time
        plt.subplot(2, 3, 1)
        plt.plot(selection_history['errors'])
        plt.xlabel('Composition Evaluated')
        plt.ylabel('Validation Error')
        plt.title('Greedy Selection: Error per Composition')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        # Plot improvement per step
        plt.subplot(2, 3, 2)
        improvements = selection_history['improvement_per_step']
        composition_lengths = list(range(1, len(improvements) + 1))
        plt.bar(composition_lengths, improvements)
        plt.xlabel('Composition Length')
        plt.ylabel('Improvement (%)')
        plt.title('Improvement per Composition Length')
        plt.grid(True, alpha=0.3)
        
        # Plot best error per length
        plt.subplot(2, 3, 3)
        best_per_length = selection_history['best_composition_per_length']
        lengths = list(best_per_length.keys())
        errors = [best_per_length[l]['error'] for l in lengths]
        plt.plot(lengths, errors, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Composition Length')
        plt.ylabel('Best Error')
        plt.title('Best Error vs Composition Length')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        # Plot composition frequencies
        plt.subplot(2, 3, 4)
        compositions = selection_history['compositions']
        comp_lengths = [len(comp) for comp in compositions]
        length_counts = {}
        for length in comp_lengths:
            length_counts[length] = length_counts.get(length, 0) + 1
        
        lengths = sorted(length_counts.keys())
        counts = [length_counts[l] for l in lengths]
        plt.bar(lengths, counts)
        plt.xlabel('Composition Length')
        plt.ylabel('Number of Compositions Tested')
        plt.title('Compositions Tested by Length')
        plt.grid(True, alpha=0.3)
        
        # Plot candidates tested per length
        plt.subplot(2, 3, 5)
        lengths = list(best_per_length.keys())
        candidates = [best_per_length[l]['candidates_tested'] for l in lengths]
        plt.bar(lengths, candidates, alpha=0.7)
        plt.xlabel('Composition Length')
        plt.ylabel('Candidates Tested')
        plt.title('Search Effort per Length')
        plt.grid(True, alpha=0.3)
        
        # Show final best composition
        plt.subplot(2, 3, 6)
        best_composition = results.get('best_composition', [])
        if best_composition:
            plt.bar(range(len(best_composition)), best_composition, alpha=0.7)
            plt.xlabel('Position in Composition')
            plt.ylabel('Operator Index')
            plt.title(f'Best Composition: {best_composition}')
            plt.xticks(range(len(best_composition)))
            for i, op_idx in enumerate(best_composition):
                plt.text(i, op_idx + 0.1, str(op_idx), ha='center', va='bottom')
        
        plt.tight_layout()
        try:
            plt.savefig(os.path.join(plots_dir, 'greedy_selection.png'), dpi=300, bbox_inches='tight')
        except ValueError as e:
            print(f"Warning: Could not save greedy selection plot: {e}")
            plt.savefig(os.path.join(plots_dir, 'greedy_selection.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    # Plot 2: Best composition information
    if 'best_composition' in results and 'operator_params' in results:
        fig = plt.figure(figsize=(15, 4))
        plt.clf()
        
        best_composition = results['best_composition']
        operator_params = results['operator_params']
        
        # Plot operator parameters with best composition highlighted
        v_params = [param[0] for param in operator_params]
        d_params = [param[1] for param in operator_params]
        
        plt.subplot(1, 3, 1)
        plt.scatter(v_params, d_params, alpha=0.6, s=50, label='All operators')
        
        # Highlight operators in best composition
        colors = plt.cm.Set1(np.linspace(0, 1, len(best_composition)))
        for i, op_idx in enumerate(best_composition):
            plt.scatter([operator_params[op_idx][0]], [operator_params[op_idx][1]], 
                       color=colors[i], s=150, 
                       label=f'Pos {i+1}: Op {op_idx}', 
                       edgecolor='black', linewidth=2)
        
        plt.xlabel('Advection Parameter (v)')
        plt.ylabel('Diffusion Parameter (D)')
        plt.title('Operator Parameter Space')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 2)
        # Plot operator types in composition
        op_types = []
        for v, D in operator_params:
            if D == 0:
                op_types.append('Advection')
            elif v == 0:
                op_types.append('Diffusion')
            else:
                op_types.append('Mixed')
        
        comp_types = [op_types[idx] for idx in best_composition]
        type_counts = {'Advection': comp_types.count('Advection'), 
                      'Diffusion': comp_types.count('Diffusion'),
                      'Mixed': comp_types.count('Mixed')}
        
        colors = ['blue', 'green', 'purple']
        bars = plt.bar(type_counts.keys(), type_counts.values(), color=colors, alpha=0.6)
        
        plt.xlabel('Operator Type')
        plt.ylabel('Count in Best Composition')
        plt.title('Operator Types in Best Composition')
        plt.grid(True, alpha=0.3)
        
        # Plot composition sequence
        plt.subplot(1, 3, 3)
        position_labels = [f"Pos {i+1}" for i in range(len(best_composition))]
        colors = ['red' if op_types[idx] == 'Advection' else 'blue' if op_types[idx] == 'Diffusion' else 'green' 
                 for idx in best_composition]
        
        bars = plt.bar(position_labels, best_composition, color=colors, alpha=0.7)
        plt.xlabel('Position in Composition')
        plt.ylabel('Operator Index')
        plt.title('Composition Sequence')
        
        # Add value labels on bars
        for bar, op_idx in zip(bars, best_composition):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    str(op_idx), ha='center', va='bottom')
        
        plt.tight_layout()
        try:
            plt.savefig(os.path.join(plots_dir, 'best_composition.png'), dpi=300, bbox_inches='tight')
        except ValueError as e:
            print(f"Warning: Could not save best composition plot: {e}")
            plt.savefig(os.path.join(plots_dir, 'best_composition.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    # Plot 3: Trajectory snapshots for predictions and ground truth
    eval_key = 'evaluation_after_finetuning' if 'evaluation_after_finetuning' in results else 'evaluation'
    if eval_key in results:
        eval_data = results[eval_key]
        predictions = eval_data['predictions']
        ground_truth = eval_data['ground_truth']
        
        # Create trajectory snapshots using the new function
        create_trajectory_snapshots(predictions, ground_truth, plots_dir)
        
        # Additional plot: Error evolution over time
        fig = plt.figure(figsize=(10, 6))
        plt.clf()
        errors_over_time = []
        sample_idx = 0
        for t in range(predictions.shape[0]):
            pred_t = predictions[t, sample_idx, 0, :]
            true_t = ground_truth[t, sample_idx, 0, :]
            error_t = np.linalg.norm(pred_t - true_t) / np.linalg.norm(true_t)
            errors_over_time.append(error_t)
        
        plt.plot(errors_over_time, 'g-', linewidth=2)
        plt.xlabel('Time Step')
        plt.ylabel('Relative L2 Error')
        plt.title('Error Evolution Over Time')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        try:
            plt.savefig(os.path.join(plots_dir, 'error_evolution.png'), dpi=300, bbox_inches='tight')
        except ValueError as e:
            print(f"Warning: Could not save error evolution plot: {e}")
            plt.savefig(os.path.join(plots_dir, 'error_evolution.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    print(f"Plots saved to {plots_dir}")
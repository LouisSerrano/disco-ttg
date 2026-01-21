#!/usr/bin/env python3
"""
Simple script to compare predictions from different methods on a single plot.
Takes a list of result directories (one per method) and method names.

Usage:
  python plot_method_comparison.py \
    --dirs ./results_strang ./results_lie ./results_model1_only \
    --names "Strang" "Lie" "Model1 Only" \
    --sample_index 1
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import argparse


def load_method_data(result_dir, sample_index):
    """Load prediction data for a specific sample from a method's result directory."""
    # Look for the sample subdirectory
    sample_dir = os.path.join(result_dir, f'sample_{sample_index:03d}')
    
    if not os.path.exists(sample_dir):
        print(f"Sample directory not found: {sample_dir}")
        return None
    
    # Find any .npz file in the sample directory (should be only one per method)
    import glob
    npz_files = glob.glob(os.path.join(sample_dir, "*.npz"))
    
    if not npz_files:
        print(f"No .npz files found in: {sample_dir}")
        return None
    
    # Load the first (and hopefully only) .npz file
    data = np.load(npz_files[0])
    
    return {
        'prediction': data['prediction'],
        'ground_truth': data['ground_truth'],
        'time_points': data['time_points'],
        'rollout_error': data['rollout_error'].item(),
        'next_step_error': data['next_step_error'].item() if 'next_step_error' in data else None
    }


def plot_method_comparison(result_dirs, method_names, sample_index, save_path=None, num_snapshots=6):
    """Plot comparison of different methods for a single sample."""
    
    # Load data for all methods
    methods_data = {}
    ground_truth = None
    time_points = None
    
    for result_dir, method_name in zip(result_dirs, method_names):
        data = load_method_data(result_dir, sample_index)
        if data is not None:
            methods_data[method_name] = data
            if ground_truth is None:
                ground_truth = data['ground_truth']
                time_points = data['time_points']
    
    if not methods_data:
        print(f"No data found for sample {sample_index}")
        return
    
    print(f"Found methods: {list(methods_data.keys())}")
    for method_name, data in methods_data.items():
        print(f"  {method_name}: rollout_error={data['rollout_error']:.6f}")
    
    # Determine time indices for snapshots
    max_time = min(len(ground_truth), min(len(data['prediction']) for data in methods_data.values()))
    time_indices = np.linspace(0, max_time-1, num_snapshots, dtype=int)
    
    num_methods = len(methods_data)
    
    # Create figure: GT + methods + error for best method
    fig, axes = plt.subplots(2 + num_methods, len(time_indices), 
                            figsize=(3*len(time_indices), 3*(2 + num_methods)))
    
    if len(time_indices) == 1:
        axes = axes.reshape(-1, 1)
    
    # Calculate global min/max for consistent colorbars
    all_data = [ground_truth[time_indices]]
    for method_data in methods_data.values():
        all_data.append(method_data['prediction'][time_indices])
    
    data_min = min(data.min() for data in all_data)
    data_max = max(data.max() for data in all_data)
    
    # Row 0: Ground Truth
    for j, t_idx in enumerate(time_indices):
        ax = axes[0, j] if len(time_indices) > 1 else axes[0]
        im = ax.imshow(ground_truth[t_idx], cmap='viridis', origin='lower', 
                      vmin=data_min, vmax=data_max)
        ax.set_title(f"Ground Truth\nt={t_idx}", fontweight='bold', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        if j == 0:
            ax.set_ylabel("Ground Truth", fontweight='bold')
    
    # Rows 1 to num_methods: Method predictions
    for i, (method_name, method_data) in enumerate(methods_data.items()):
        prediction = method_data['prediction']
        rollout_error = method_data['rollout_error']
        
        for j, t_idx in enumerate(time_indices):
            ax = axes[1 + i, j] if len(time_indices) > 1 else axes[1 + i]
            im = ax.imshow(prediction[t_idx], cmap='viridis', origin='lower', 
                          vmin=data_min, vmax=data_max)
            ax.set_title(f"{method_name}\nt={t_idx}", fontweight='bold', fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(f"{method_name}\nError: {rollout_error:.4f}", fontweight='bold')
    
    # Last row: Error map for best method
    best_method = min(methods_data.keys(), key=lambda k: methods_data[k]['rollout_error'])
    best_prediction = methods_data[best_method]['prediction']
    
    error_maps = []
    for t_idx in time_indices:
        error_map = np.abs(ground_truth[t_idx] - best_prediction[t_idx])
        error_maps.append(error_map)
    
    error_min, error_max = 0, max(em.max() for em in error_maps)
    
    for j, t_idx in enumerate(time_indices):
        ax = axes[1 + num_methods, j] if len(time_indices) > 1 else axes[1 + num_methods]
        error_map = np.abs(ground_truth[t_idx] - best_prediction[t_idx])
        im_error = ax.imshow(error_map, cmap='hot', origin='lower', vmin=error_min, vmax=error_max)
        ax.set_title(f"Error ({best_method})\nt={t_idx}", fontweight='bold', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        if j == 0:
            ax.set_ylabel(f"Abs Error\n({best_method})", fontweight='bold')
    
    # Add colorbars
    # Data colorbar
    cbar_ax1 = fig.add_axes([0.92, 0.4, 0.02, 0.5])
    cbar1 = fig.colorbar(im, cax=cbar_ax1)
    cbar1.set_label('Field Value', fontweight='bold')
    
    # Error colorbar  
    cbar_ax2 = fig.add_axes([0.92, 0.1, 0.02, 0.25])
    cbar2 = fig.colorbar(im_error, cax=cbar_ax2)
    cbar2.set_label('Absolute Error', fontweight='bold')
    
    # Main title with error summary
    error_summary = " | ".join([f"{name}: {data['rollout_error']:.4f}" 
                               for name, data in sorted(methods_data.items(), 
                                                      key=lambda x: x[1]['rollout_error'])])
    plt.suptitle(f"Method Comparison - Sample {sample_index}\n{error_summary}", 
                fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.9, top=0.92)
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
        plt.close()
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Compare predictions from different methods')
    
    parser.add_argument('--dirs', nargs='+', required=True,
                        help='List of result directories, one per method')
    parser.add_argument('--names', nargs='+', required=True,
                        help='List of method names corresponding to directories')
    parser.add_argument('--sample_index', type=int, default=1,
                        help='Index of sample to plot (1-based)')
    parser.add_argument('--num_snapshots', type=int, default=6,
                        help='Number of time snapshots to show')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Path to save plot (if not specified, plot is shown)')
    
    args = parser.parse_args()
    
    # Check arguments
    if len(args.dirs) != len(args.names):
        print("Error: Number of directories must match number of names")
        return
    
    for result_dir in args.dirs:
        if not os.path.exists(result_dir):
            print(f"Directory not found: {result_dir}")
            return
    
    # Plot comparison
    plot_method_comparison(args.dirs, args.names, args.sample_index, args.save_path, args.num_snapshots)


if __name__ == "__main__":
    main()
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import argparse
from matplotlib.animation import FuncAnimation, PillowWriter


def load_prediction_data(sample_dir):
    """Load prediction data from a sample directory"""
    input_data = np.load(os.path.join(sample_dir, "input.npy"))
    ground_truth = np.load(os.path.join(sample_dir, "ground_truth.npy"))
    prediction = np.load(os.path.join(sample_dir, "prediction.npy"))
    
    # Load metadata
    metadata_path = os.path.join(sample_dir, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}
    
    return input_data, ground_truth, prediction, metadata


def plot_timesteps(input_data, ground_truth, prediction, metadata, output_path, timesteps=None, both_channels=False):
    """Plot specific timesteps of the data including input with improved styling"""
    
    # Set style for better aesthetics
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'sans-serif',
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'figure.titlesize': 14
    })
    
    if both_channels:
        # Improved layout with better proportions
        fig = plt.figure(figsize=(13, 10))
        
        # Create gridspec with better spacing
        import matplotlib.gridspec as gridspec
        gs = gridspec.GridSpec(4, 5, figure=fig,
                              height_ratios=[1, 1, 1, 1],  # 4 rows: A GT, A Pred, B GT, B Pred
                              width_ratios=[1, 1, 1, 1, 1],
                              hspace=0.2, wspace=0,
                              left=0.05, right=0.95, top=0.92, bottom=0.12)
        
        # Better color schemes
        cmap_ch0 = 'viridis'
        cmap_ch1 = 'plasma'
        
        # Find global min/max for consistent colorscale per channel
        vmin_ch0 = min(input_data[:, 0].min(), ground_truth[:, 0].min(), prediction[:, 0].min())
        vmax_ch0 = max(input_data[:, 0].max(), ground_truth[:, 0].max(), prediction[:, 0].max())
        vmin_ch1 = min(input_data[:, 1].min(), ground_truth[:, 1].min(), prediction[:, 1].min())
        vmax_ch1 = max(input_data[:, 1].max(), ground_truth[:, 1].max(), prediction[:, 1].max())
        
        # Select timesteps for output visualization
        output_timesteps = [0, ground_truth.shape[0]//2, ground_truth.shape[0]-1]
        
        # Channel 0 Section
        # Add section label
        fig.text(0.02, 0.75, 'Channel 0', rotation=90, fontsize=14, fontweight='bold', 
                ha='center', va='center', color='#2E8B57')
        
        # Input snapshots for Channel 0
        ax00 = fig.add_subplot(gs[0, 0])
        im0_ch0 = ax00.imshow(input_data[0, 0], cmap=cmap_ch0, vmin=vmin_ch0, vmax=vmax_ch0, aspect='equal')
        ax00.set_title('Input\n(t=0)', fontsize=10, pad=8, fontweight='bold')
        ax00.set_xticks([])
        ax00.set_yticks([])
        # Add subtle border
        for spine in ax00.spines.values():
            spine.set_edgecolor('gray')
            spine.set_linewidth(0.5)
        
        ax01 = fig.add_subplot(gs[0, 1])
        im1_ch0 = ax01.imshow(input_data[7, 0], cmap=cmap_ch0, vmin=vmin_ch0, vmax=vmax_ch0, aspect='equal')
        ax01.set_title('Input\n(t=8)', fontsize=10, pad=8, fontweight='bold')
        ax01.set_xticks([])
        ax01.set_yticks([])
        for spine in ax01.spines.values():
            spine.set_edgecolor('gray')
            spine.set_linewidth(0.5)
        
        # Ground truth and predictions for Channel 0
        for idx, t in enumerate(output_timesteps):
            col_idx = idx + 2
            
            # Ground truth
            ax_gt_ch0 = fig.add_subplot(gs[0, col_idx])
            im_gt_ch0 = ax_gt_ch0.imshow(ground_truth[t, 0], cmap=cmap_ch0, vmin=vmin_ch0, vmax=vmax_ch0, aspect='equal')
            ax_gt_ch0.set_title(f'Ground Truth\n(t={16+t})', fontsize=10, pad=8, fontweight='bold')
            ax_gt_ch0.set_xticks([])
            ax_gt_ch0.set_yticks([])
            for spine in ax_gt_ch0.spines.values():
                spine.set_edgecolor('gray')
                spine.set_linewidth(0.5)
            
            # Prediction
            ax_pred_ch0 = fig.add_subplot(gs[1, col_idx])
            im_pred_ch0 = ax_pred_ch0.imshow(prediction[t, 0], cmap=cmap_ch0, vmin=vmin_ch0, vmax=vmax_ch0, aspect='equal')
            ax_pred_ch0.set_title(f'Prediction\n(t={16+t})', fontsize=10, pad=8, fontweight='bold')
            ax_pred_ch0.set_xticks([])
            ax_pred_ch0.set_yticks([])
            for spine in ax_pred_ch0.spines.values():
                spine.set_edgecolor('gray')
                spine.set_linewidth(0.5)
        
        # Channel 1 Section
        # Add section label
        fig.text(0.02, 0.35, 'Channel 1', rotation=90, fontsize=14, fontweight='bold',
                ha='center', va='center', color='#8B008B')
        
        # Input snapshots for Channel 1
        ax20 = fig.add_subplot(gs[2, 0])
        im0_ch1 = ax20.imshow(input_data[0, 1], cmap=cmap_ch1, vmin=vmin_ch1, vmax=vmax_ch1, aspect='equal')
        ax20.set_title('Input\n(t=0)', fontsize=10, pad=8, fontweight='bold')
        ax20.set_xticks([])
        ax20.set_yticks([])
        for spine in ax20.spines.values():
            spine.set_edgecolor('gray')
            spine.set_linewidth(0.5)
        
        ax21 = fig.add_subplot(gs[2, 1])
        im1_ch1 = ax21.imshow(input_data[7, 1], cmap=cmap_ch1, vmin=vmin_ch1, vmax=vmax_ch1, aspect='equal')
        ax21.set_title('Input\n(t=8)', fontsize=10, pad=8, fontweight='bold')
        ax21.set_xticks([])
        ax21.set_yticks([])
        for spine in ax21.spines.values():
            spine.set_edgecolor('gray')
            spine.set_linewidth(0.5)
        
        # Ground truth and predictions for Channel 1
        for idx, t in enumerate(output_timesteps):
            col_idx = idx + 2
            
            # Ground truth
            ax_gt_ch1 = fig.add_subplot(gs[2, col_idx])
            im_gt_ch1 = ax_gt_ch1.imshow(ground_truth[t, 1], cmap=cmap_ch1, vmin=vmin_ch1, vmax=vmax_ch1, aspect='equal')
            ax_gt_ch1.set_title(f'Ground Truth\n(t={16+t})', fontsize=10, pad=8, fontweight='bold')
            ax_gt_ch1.set_xticks([])
            ax_gt_ch1.set_yticks([])
            for spine in ax_gt_ch1.spines.values():
                spine.set_edgecolor('gray')
                spine.set_linewidth(0.5)
            
            # Prediction
            ax_pred_ch1 = fig.add_subplot(gs[3, col_idx])
            im_pred_ch1 = ax_pred_ch1.imshow(prediction[t, 1], cmap=cmap_ch1, vmin=vmin_ch1, vmax=vmax_ch1, aspect='equal')
            ax_pred_ch1.set_title(f'Prediction\n(t={16+t})', fontsize=10, pad=8, fontweight='bold')
            ax_pred_ch1.set_xticks([])
            ax_pred_ch1.set_yticks([])
            for spine in ax_pred_ch1.spines.values():
                spine.set_edgecolor('gray')
                spine.set_linewidth(0.5)
        
        # Improved colorbars with better positioning
        # Channel 0 colorbar
        cbar_ax_ch0 = fig.add_axes([0.15, 0.06, 0.3, 0.02])
        cb_ch0 = plt.colorbar(im0_ch0, cax=cbar_ax_ch0, orientation='horizontal')
        cb_ch0.set_label('Channel 0 Values', fontsize=11, fontweight='bold')
        cb_ch0.ax.tick_params(labelsize=9)
        
        # Channel 1 colorbar
        cbar_ax_ch1 = fig.add_axes([0.55, 0.06, 0.3, 0.02])
        cb_ch1 = plt.colorbar(im0_ch1, cax=cbar_ax_ch1, orientation='horizontal')
        cb_ch1.set_label('Channel 1 Values', fontsize=11, fontweight='bold')
        cb_ch1.ax.tick_params(labelsize=9)
        
    else:
        # Improved single channel layout
        fig = plt.figure(figsize=(18, 8))
        
        # Create gridspec with better spacing
        import matplotlib.gridspec as gridspec
        gs = gridspec.GridSpec(2, 5, figure=fig, 
                              height_ratios=[1, 1], 
                              width_ratios=[1, 1, 1, 1, 1], 
                              hspace=0.25, wspace=0,
                              left=0.05, right=0.95, top=0.88, bottom=0.15)
        
        # Find global min/max for consistent colorscale
        vmin = min(input_data.min(), ground_truth.min(), prediction.min())
        vmax = max(input_data.max(), ground_truth.max(), prediction.max())
        
        # Select timesteps for output visualization
        output_timesteps = [0, ground_truth.shape[0]//2, ground_truth.shape[0]-1]
        
        # Input columns with improved styling
        ax00 = fig.add_subplot(gs[0, 0])
        im0 = ax00.imshow(input_data[0, 0], cmap='viridis', vmin=vmin, vmax=vmax, aspect='equal')
        ax00.set_title('Input\n(t=0)', fontsize=12, pad=10, fontweight='bold')
        ax00.set_xticks([])
        ax00.set_yticks([])
        for spine in ax00.spines.values():
            spine.set_edgecolor('gray')
            spine.set_linewidth(0.5)
        
        ax01 = fig.add_subplot(gs[0, 1])
        im1 = ax01.imshow(input_data[7, 0], cmap='viridis', vmin=vmin, vmax=vmax, aspect='equal')
        ax01.set_title('Input\n(t=8)', fontsize=12, pad=10, fontweight='bold')
        ax01.set_xticks([])
        ax01.set_yticks([])
        for spine in ax01.spines.values():
            spine.set_edgecolor('gray')
            spine.set_linewidth(0.5)
        
        # Empty cells for bottom row of input columns
        ax10 = fig.add_subplot(gs[1, 0])
        ax10.axis('off')
        ax11 = fig.add_subplot(gs[1, 1])
        ax11.axis('off')
        
        # Output columns with improved styling
        for idx, t in enumerate(output_timesteps):
            col_idx = idx + 2
            
            # Ground truth
            ax_gt = fig.add_subplot(gs[0, col_idx])
            im_gt = ax_gt.imshow(ground_truth[t, 0], cmap='viridis', vmin=vmin, vmax=vmax, aspect='equal')
            ax_gt.set_title(f'Ground Truth\n(t={16+t})', fontsize=12, pad=10, fontweight='bold')
            ax_gt.set_xticks([])
            ax_gt.set_yticks([])
            for spine in ax_gt.spines.values():
                spine.set_edgecolor('gray')
                spine.set_linewidth(0.5)
            
            # Prediction
            ax_pred = fig.add_subplot(gs[1, col_idx])
            im_pred = ax_pred.imshow(prediction[t, 0], cmap='viridis', vmin=vmin, vmax=vmax, aspect='equal')
            ax_pred.set_title(f'Prediction\n(t={16+t})', fontsize=12, pad=10, fontweight='bold')
            ax_pred.set_xticks([])
            ax_pred.set_yticks([])
            for spine in ax_pred.spines.values():
                spine.set_edgecolor('gray')
                spine.set_linewidth(0.5)
        
        # Improved colorbar
        cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.03])
        cb = plt.colorbar(im0, cax=cbar_ax, orientation='horizontal')
        cb.set_label('Values', fontsize=12, fontweight='bold')
        cb.ax.tick_params(labelsize=10)
    
    # Improved title with better formatting
    title_parts = []
    if 'sample_idx' in metadata:
        title_parts.append(f"Sample {metadata['sample_idx']}")
    if 'composition' in metadata:
        title_parts.append(f"Composition: {metadata['composition']}")
    if 'error' in metadata:
        title_parts.append(f"Error: {metadata['error']:.4f}")
    if both_channels:
        title_parts.append("Dual Channel Analysis")
    
    title = " | ".join(title_parts)
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.96)
    
    # Save with higher quality
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"Saved improved plot to {output_path}")

def plot_timesteps_simple(input_data, ground_truth, prediction, metadata, output_path, timesteps=None, both_channels=False):
    """Plot specific timesteps using simple plt.subplots approach - much more compact!"""
    
    if both_channels:
        # Simple approach: 4 rows (A GT, A Pred, B GT, B Pred), 5 columns
        fig, axs = plt.subplots(
            nrows=4, 
            ncols=5, 
            figsize=(10, 8)  # Much more reasonable size
        )
        
        # Find global min/max for consistent colorscale per channel
        vmin_ch0 = min(input_data[:, 0].min(), ground_truth[:, 0].min(), prediction[:, 0].min())
        vmax_ch0 = max(input_data[:, 0].max(), ground_truth[:, 0].max(), prediction[:, 0].max())
        vmin_ch1 = min(input_data[:, 1].min(), ground_truth[:, 1].min(), prediction[:, 1].min())
        vmax_ch1 = max(input_data[:, 1].max(), ground_truth[:, 1].max(), prediction[:, 1].max())
        
        # Select timesteps for output visualization
        output_timesteps = [0, ground_truth.shape[0]//2, ground_truth.shape[0]-1]
        
        # CHANNEL 0 - Row 0 (GT) and Row 1 (Pred)
        # Input snapshots
        axs[0, 0].imshow(input_data[0, 0], cmap='viridis', vmin=vmin_ch0, vmax=vmax_ch0)
        axs[0, 0].set_title('Input A (t=0)', fontsize=9)
        axs[0, 0].axis('off')
        
        axs[0, 1].imshow(input_data[7, 0], cmap='viridis', vmin=vmin_ch0, vmax=vmax_ch0)
        axs[0, 1].set_title('Input A (t=8)', fontsize=9)
        axs[0, 1].axis('off')
        
        # Ground truth and predictions for timesteps
        for idx, t in enumerate(output_timesteps):
            col_idx = idx + 2
            
            # Ground truth A
            im_ch0 = axs[0, col_idx].imshow(ground_truth[t, 0], cmap='viridis', vmin=vmin_ch0, vmax=vmax_ch0)
            axs[0, col_idx].set_title(f'GT A (t={16+t})', fontsize=9)
            axs[0, col_idx].axis('off')
            
            # Prediction A
            axs[1, col_idx].imshow(prediction[t, 0], cmap='viridis', vmin=vmin_ch0, vmax=vmax_ch0)
            axs[1, col_idx].set_title(f'Pred A (t={16+t})', fontsize=9)
            axs[1, col_idx].axis('off')
        
        # CHANNEL 1 - Row 2 (GT) and Row 3 (Pred)
        # Input snapshots
        axs[2, 0].imshow(input_data[0, 1], cmap='plasma', vmin=vmin_ch1, vmax=vmax_ch1)
        axs[2, 0].set_title('Input B (t=0)', fontsize=9)
        axs[2, 0].axis('off')
        
        axs[2, 1].imshow(input_data[7, 1], cmap='plasma', vmin=vmin_ch1, vmax=vmax_ch1)
        axs[2, 1].set_title('Input B (t=8)', fontsize=9)
        axs[2, 1].axis('off')
        
        # Ground truth and predictions for timesteps
        for idx, t in enumerate(output_timesteps):
            col_idx = idx + 2
            
            # Ground truth B
            im_ch1 = axs[2, col_idx].imshow(ground_truth[t, 1], cmap='plasma', vmin=vmin_ch1, vmax=vmax_ch1)
            axs[2, col_idx].set_title(f'GT B (t={16+t})', fontsize=9)
            axs[2, col_idx].axis('off')
            
            # Prediction B
            axs[3, col_idx].imshow(prediction[t, 1], cmap='plasma', vmin=vmin_ch1, vmax=vmax_ch1)
            axs[3, col_idx].set_title(f'Pred B (t={16+t})', fontsize=9)
            axs[3, col_idx].axis('off')
        
        # Hide unused subplots in input columns
        axs[1, 0].axis('off')  # Empty cell below input A t=0
        axs[1, 1].axis('off')  # Empty cell below input A t=8
        axs[3, 0].axis('off')  # Empty cell below input B t=0
        axs[3, 1].axis('off')  # Empty cell below input B t=8
        
        # Add vertical colorbars completely to the right side
        # First, adjust subplot layout to make room for colorbars
        plt.subplots_adjust(right=0.85)
        
        # Channel 0 colorbar (positioned to the right of all plots)
        cbar_ax_ch0 = fig.add_axes([1.0, 0.5, 0.02, 0.35])  # [left, bottom, width, height]
        cbar_ch0 = plt.colorbar(im_ch0, cax=cbar_ax_ch0)
        cbar_ch0.set_label('Channel 0', fontsize=10, rotation=270, labelpad=15)
        
        # Channel 1 colorbar (positioned to the right of all plots)
        cbar_ax_ch1 = fig.add_axes([1.0, 0.0, 0.02, 0.35])  # [left, bottom, width, height]
        cbar_ch1 = plt.colorbar(im_ch1, cax=cbar_ax_ch1)
        cbar_ch1.set_label('Channel 1', fontsize=10, rotation=270, labelpad=15)
        
    else:
        # Single channel version - also simplified
        fig, axs = plt.subplots(
            nrows=2, 
            ncols=5, 
            figsize=(10, 4)
        )
        
        # Find global min/max for consistent colorscale
        vmin = min(input_data.min(), ground_truth.min(), prediction.min())
        vmax = max(input_data.max(), ground_truth.max(), prediction.max())
        
        output_timesteps = [0, ground_truth.shape[0]//2, ground_truth.shape[0]-1]
        
        # Input snapshots
        axs[0, 0].imshow(input_data[0, 0], cmap='viridis', vmin=vmin, vmax=vmax)
        axs[0, 0].set_title('Input (t=0)', fontsize=10)
        axs[0, 0].axis('off')
        
        axs[0, 1].imshow(input_data[7, 0], cmap='viridis', vmin=vmin, vmax=vmax)
        axs[0, 1].set_title('Input (t=8)', fontsize=10)
        axs[0, 1].axis('off')
        
        # Hide unused input cells
        axs[1, 0].axis('off')
        axs[1, 1].axis('off')
        
        # Ground truth and predictions
        for idx, t in enumerate(output_timesteps):
            col_idx = idx + 2
            
            # Ground truth
            im = axs[0, col_idx].imshow(ground_truth[t, 0], cmap='viridis', vmin=vmin, vmax=vmax)
            axs[0, col_idx].set_title(f'Ground Truth (t={16+t})', fontsize=10)
            axs[0, col_idx].axis('off')
            
            # Prediction
            axs[1, col_idx].imshow(prediction[t, 0], cmap='viridis', vmin=vmin, vmax=vmax)
            axs[1, col_idx].set_title(f'Prediction (t={16+t})', fontsize=10)
            axs[1, col_idx].axis('off')
        
        # Add single vertical colorbar completely to the right
        plt.subplots_adjust(right=0.85)
        
        cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        cbar = plt.colorbar(im, cax=cbar_ax)
        cbar.set_label('Values', fontsize=10, rotation=270, labelpad=15)
    
    # Create title
    title = f"Sample {metadata.get('sample_idx', '?')}"
    if 'composition' in metadata:
        title += f" | Composition: {metadata['composition']}"
    if 'error' in metadata:
        title += f" | Error: {metadata['error']:.4f}"
    if both_channels:
        title += " | Both Channels"
    
    fig.suptitle(title, fontsize=12, y=0.98)
    
    # THE KEY: Use tight_layout like in your simple example
    plt.tight_layout()
    
    # Save with high quality
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved compact plot to {output_path}")




def save_individual_plots(input_data, ground_truth, prediction, metadata, sample_dir, both_channels=False):
    """Save individual plots over time in the sample folder"""
    
    if both_channels:
        # Create subdirectories for predictions and inputs with channel-specific folders
        pred_dir_ch0 = os.path.join(sample_dir, "predictions_ch0")
        pred_dir_ch1 = os.path.join(sample_dir, "predictions_ch1")
        input_dir_ch0 = os.path.join(sample_dir, "inputs_ch0")
        input_dir_ch1 = os.path.join(sample_dir, "inputs_ch1")
        os.makedirs(pred_dir_ch0, exist_ok=True)
        os.makedirs(pred_dir_ch1, exist_ok=True)
        os.makedirs(input_dir_ch0, exist_ok=True)
        os.makedirs(input_dir_ch1, exist_ok=True)
        
        # Find global min/max for consistent colorscale per channel
        vmin_ch0 = min(input_data[:, 0].min(), ground_truth[:, 0].min(), prediction[:, 0].min())
        vmax_ch0 = max(input_data[:, 0].max(), ground_truth[:, 0].max(), prediction[:, 0].max())
        vmin_ch1 = min(input_data[:, 1].min(), ground_truth[:, 1].min(), prediction[:, 1].min())
        vmax_ch1 = max(input_data[:, 1].max(), ground_truth[:, 1].max(), prediction[:, 1].max())
        
        # Save prediction images for each timestep and channel
        n_timesteps = ground_truth.shape[0]
        for t in range(n_timesteps):
            # Channel 0 prediction
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            im = ax.imshow(prediction[t, 0], cmap='viridis', vmin=vmin_ch0, vmax=vmax_ch0)
            ax.axis('off')
            pred_path_ch0 = os.path.join(pred_dir_ch0, f"prediction_ch0_t{t:04d}.png")
            plt.savefig(pred_path_ch0, dpi=150, bbox_inches='tight', pad_inches=0)
            plt.close()
            
            # Channel 1 prediction
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            im = ax.imshow(prediction[t, 1], cmap='plasma', vmin=vmin_ch1, vmax=vmax_ch1)
            ax.axis('off')
            pred_path_ch1 = os.path.join(pred_dir_ch1, f"prediction_ch1_t{t:04d}.png")
            plt.savefig(pred_path_ch1, dpi=150, bbox_inches='tight', pad_inches=0)
            plt.close()
        
        # Save input images for each timestep and channel
        n_input_timesteps = input_data.shape[0]
        for t in range(n_input_timesteps):
            # Channel 0 input
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            im = ax.imshow(input_data[t, 0], cmap='viridis', vmin=vmin_ch0, vmax=vmax_ch0)
            ax.axis('off')
            input_path_ch0 = os.path.join(input_dir_ch0, f"input_ch0_t{t:04d}.png")
            plt.savefig(input_path_ch0, dpi=150, bbox_inches='tight', pad_inches=0)
            plt.close()
            
            # Channel 1 input
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            im = ax.imshow(input_data[t, 1], cmap='plasma', vmin=vmin_ch1, vmax=vmax_ch1)
            ax.axis('off')
            input_path_ch1 = os.path.join(input_dir_ch1, f"input_ch1_t{t:04d}.png")
            plt.savefig(input_path_ch1, dpi=150, bbox_inches='tight', pad_inches=0)
            plt.close()
        
        print(f"Saved {n_timesteps} prediction images for A to {pred_dir_ch0}")
        print(f"Saved {n_timesteps} prediction images for B to {pred_dir_ch1}")
        print(f"Saved {n_input_timesteps} input images for A to {input_dir_ch0}")
        print(f"Saved {n_input_timesteps} input images for B to {input_dir_ch1}")
        
    else:
        # Original single channel behavior
        # Create subdirectories for predictions and inputs
        pred_dir = os.path.join(sample_dir, "predictions")
        input_dir = os.path.join(sample_dir, "inputs")
        os.makedirs(pred_dir, exist_ok=True)
        os.makedirs(input_dir, exist_ok=True)
        
        # Find global min/max for consistent colorscale
        vmin = min(input_data.min(), ground_truth.min(), prediction.min())
        vmax = max(input_data.max(), ground_truth.max(), prediction.max())
        
        # Save prediction-only images for each timestep
        n_timesteps = ground_truth.shape[0]
        for t in range(n_timesteps):
            # Save prediction only - minimalist layout
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            im = ax.imshow(prediction[t, 0], cmap='viridis', vmin=vmin, vmax=vmax)
            ax.axis('off')
            
            pred_path = os.path.join(pred_dir, f"prediction_t{t:04d}.png")
            plt.savefig(pred_path, dpi=150, bbox_inches='tight', pad_inches=0)
            plt.close()
        
        # Save input-only images for each timestep in input data
        n_input_timesteps = input_data.shape[0]
        for t in range(n_input_timesteps):
            # Save input only - minimalist layout
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            im = ax.imshow(input_data[t, 0], cmap='viridis', vmin=vmin, vmax=vmax)
            ax.axis('off')
            
            input_path = os.path.join(input_dir, f"input_t{t:04d}.png")
            plt.savefig(input_path, dpi=150, bbox_inches='tight', pad_inches=0)
            plt.close()
        
        print(f"Saved {n_timesteps} prediction images to {pred_dir}")
        print(f"Saved {n_input_timesteps} input images to {input_dir}")


def main():
    parser = argparse.ArgumentParser(description='Visualize beam search predictions')
    parser.add_argument('--prediction_dir', type=str, required=True, 
                        help='Directory containing beam search predictions')
    parser.add_argument('--sample_idx', type=int, default=None,
                        help='Specific sample index to visualize (default: all)')
    parser.add_argument('--create_animation', action='store_true',
                        help='Create animation GIFs')
    parser.add_argument('--fps', type=int, default=10,
                        help='Frames per second for animations (default: 10)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Custom output directory for plots (default: timestamped folder in prediction_dir)')
    parser.add_argument('--both_channels', action='store_true',
                        help='Show both channels in visualizations (default: channel 0 only)')
    args = parser.parse_args()
    
    # Create output directory for plots with timestamp
    from datetime import datetime
    if args.output_dir:
        plot_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_dir = os.path.join(args.prediction_dir, f'plots_{timestamp}')
    os.makedirs(plot_dir, exist_ok=True)
    
    # Get list of sample directories
    sample_dirs = []
    if args.sample_idx is not None:
        # Visualize specific sample
        sample_dir = os.path.join(args.prediction_dir, f"sample_{args.sample_idx:04d}")
        if os.path.exists(sample_dir):
            sample_dirs.append(sample_dir)
        else:
            print(f"Sample directory not found: {sample_dir}")
            return
    else:
        # Visualize all samples
        for item in sorted(os.listdir(args.prediction_dir)):
            if item.startswith('sample_') and os.path.isdir(os.path.join(args.prediction_dir, item)):
                sample_dirs.append(os.path.join(args.prediction_dir, item))
    
    print(f"Found {len(sample_dirs)} samples to visualize")
    
    # Process each sample
    for sample_dir in sample_dirs:
        sample_name = os.path.basename(sample_dir)
        print(f"\nProcessing {sample_name}...")
        
        try:
            # Load data
            input_data, ground_truth, prediction, metadata = load_prediction_data(sample_dir)
            
            # Create static plot
            plot_path = os.path.join(plot_dir, f"{sample_name}_comparison.png")
            plot_timesteps_simple(input_data, ground_truth, prediction, metadata, plot_path, both_channels=args.both_channels)
            
            # Save individual plots over time in the sample folder
            save_individual_plots(input_data, ground_truth, prediction, metadata, sample_dir, both_channels=args.both_channels)
            
            # Create animation if requested
            if args.create_animation:
                anim_path = os.path.join(plot_dir, f"{sample_name}_animation.gif")
                create_animation(input_data, ground_truth, prediction, metadata, 
                               anim_path, fps=args.fps, both_channels=args.both_channels)
                
        except Exception as e:
            print(f"Error processing {sample_name}: {e}")
            continue
    
    print(f"\nVisualization complete. Plots saved to: {plot_dir}")


if __name__ == "__main__":
    main()
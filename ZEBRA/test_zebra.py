import os
import argparse
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
import wandb
import matplotlib.pyplot as plt
from transformers import PreTrainedTokenizer

from zebra.models.llama.model import Zebra

from zebra.training.tokenizer_trainer import TokenizerTrainer
from zebra.training.llama_trainer import LLaMATrainer
from zebra.utils.data import get_data, load_wave2d, load_vort, TemporalDataset, TemporalDatasetWithContext, tokenize_dataset

# Import custom data utilities
from ZEBRA.data_utils import get_hdf5_files, get_hdf5_files_gs, TemporalBatchDatasetFly, HDF5TemporalDataset, GrayScottDatasetWrapper
from ZEBRA.data_utils_llama import RawDatasetWithContextFly, RawDatasetWithContext
from ZEBRA.test_utils import get_test_metrics_1d, get_test_metrics_2d


def parse_args():
    parser = argparse.ArgumentParser(description="Test ZEBRA model")
    
    # Essential arguments
    parser.add_argument("--dataset_name", type=str, required=True, 
                        choices=["advection-diffusion", "combined-equation", "gray-scott"],
                        help="Name of the dataset to test")
    parser.add_argument("--output_dir", type=str, default="./ZEBRA/results",
                        help="Output directory for results")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    # Model and experiment arguments
    parser.add_argument("--model_path", type=str, 
                        help="Path to the model checkpoint")
    parser.add_argument("--experiment", type=str,
                        help="Experiment configuration (required for advection-diffusion and combined-equation)")
    
    # Batch processing arguments
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for testing")
    parser.add_argument("--num_samples", type=int, default=512,
                        help="Number of samples to test")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of dataloader workers")
    
    # Additional parameters that might be needed
    
    args = parser.parse_args()
    
    # Validate experiment argument
    if args.dataset_name in ["advection-diffusion", "combined-equation"] and not args.experiment:
        parser.error(f"--experiment is required when dataset_name is '{args.dataset_name}'")
    
    return args

def plot_predictions_over_time(pred, gt, output_dir, dataset_name, n_input_frames, num_samples=5):
    """Plot predictions vs ground truth over time for selected samples and spatial indices."""
    
    if dataset_name in ['advection-diffusion', 'combined-equation']:
        # 1D case: pred shape is (batch, height, channels, time)
        plot_1d_predictions(pred, gt, output_dir, n_input_frames, num_samples)
    elif dataset_name == 'gray-scott':
        # 2D case: pred shape is (batch, channels, height, width, time)  
        plot_2d_predictions(pred, gt, output_dir, n_input_frames, num_samples)

def plot_1d_predictions(pred, gt, output_dir, n_input_frames, num_samples=3):
    """Plot 1D spatial predictions with different time steps as different lines."""
    print(f"Pred shape: {pred.shape}")
    print(f"GT shape: {gt.shape}")
    
    # Looking at the test_utils.py output:
    # upred = rearrange(upred, '(b t) c h -> b c h t', t=rollout_len) - line 133
    # So pred shape should be (batch, channels, height, time)
    # And gt shape from images should be (batch, channels, height, time) 
    
    batch_size = pred.shape[0]
    
    # Select random samples to plot
    sample_indices = np.random.choice(batch_size, min(num_samples, batch_size), replace=False)
    
    for sample_idx in sample_indices:
        # Create subplots: GT vs Predictions, and an overlay plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        try:
            # Extract data for this sample
            # Try the expected layout first: (batch, channels, height, time)
            if pred.shape[1] == 1:  # Single channel
                pred_data = pred[sample_idx, 0, :, :]  # Shape: (height, time)
            else:
                pred_data = pred[sample_idx, :, :, :]  # Shape: (channels, height, time) - take first channel
                pred_data = pred_data[0, :, :]  # (height, time)
                
            print(f"DEBUG: pred_data shape: {pred_data.shape}")
                
            # Extract GT data - need to handle the portion that corresponds to predictions
            if len(gt.shape) == 4:  # (batch, channels, height, time)
                if gt.shape[1] == 1:
                    # Single channel case
                    gt_full = gt[sample_idx, 0, :, :]  # (height, time)
                else:
                    gt_full = gt[sample_idx, 0, :, :]  # Take first channel (height, time)
            elif len(gt.shape) == 5:  # (batch, channels, height, width, time) - take middle slice
                gt_full = gt[sample_idx, 0, :, gt.shape[3]//2, :]  # (height, time)
            else:
                print(f"Unsupported GT shape: {gt.shape}")
                continue
                
            # Extract the prediction portion from GT
            pred_start_time = n_input_frames
            pred_end_time = n_input_frames + pred_data.shape[1]
            gt_pred_portion = gt_full[:, pred_start_time:pred_end_time]
            
            print(f"DEBUG: gt_full shape: {gt_full.shape}")
            print(f"DEBUG: gt_pred_portion shape: {gt_pred_portion.shape}")
            print(f"DEBUG: pred_start_time: {pred_start_time}, pred_end_time: {pred_end_time}")
            
            height = pred_data.shape[0]
            spatial_positions = np.arange(height)
            
            # Debug: Check if spatial shift exists by looking at peak locations
            final_gt = gt_pred_portion[:, -1]
            final_pred = pred_data[:, -1]
            gt_peak_idx = np.argmax(np.abs(final_gt))
            pred_peak_idx = np.argmax(np.abs(final_pred))
            print(f"DEBUG: GT peak at spatial index {gt_peak_idx}, Pred peak at spatial index {pred_peak_idx}")
            print(f"DEBUG: Spatial shift = {pred_peak_idx - gt_peak_idx} grid points")
            
            # Just record the shift for debugging - don't correct it
            spatial_shift = pred_peak_idx - gt_peak_idx
            
            # Select time steps to plot (evenly spaced)
            n_time_steps = min(8, pred_data.shape[1])  # Show up to 8 time steps
            time_indices = np.linspace(0, pred_data.shape[1]-1, n_time_steps, dtype=int)
            
            # Plot 1: Ground Truth spatial profiles at different times
            for i, t_idx in enumerate(time_indices):
                actual_time = pred_start_time + t_idx
                color = plt.cm.viridis(i / (n_time_steps - 1))
                axes[0].plot(spatial_positions, gt_pred_portion[:, t_idx], 
                           color=color, linewidth=2, label=f'Time {actual_time}')
            
            axes[0].set_title('Ground Truth - Spatial Profiles')
            axes[0].set_xlabel('Spatial Position')
            axes[0].set_ylabel('Field Value')
            axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[0].grid(True, alpha=0.3)
            
            # Plot 2: Prediction spatial profiles at different times  
            for i, t_idx in enumerate(time_indices):
                actual_time = pred_start_time + t_idx
                color = plt.cm.viridis(i / (n_time_steps - 1))
                axes[1].plot(spatial_positions, pred_data[:, t_idx], 
                           color=color, linewidth=2, label=f'Time {actual_time}')
            
            axes[1].set_title('Predictions - Spatial Profiles')
            axes[1].set_xlabel('Spatial Position')
            axes[1].set_ylabel('Field Value')
            axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[1].grid(True, alpha=0.3)
            
            # Plot 3: Overlay comparison at final time
            final_t_idx = time_indices[-1]
            final_time = pred_start_time + final_t_idx
            axes[2].plot(spatial_positions, gt_pred_portion[:, final_t_idx], 
                        'b-', linewidth=3, label=f'GT at time {final_time}')
            axes[2].plot(spatial_positions, pred_data[:, final_t_idx], 
                        'r--', linewidth=3, label=f'Pred at time {final_time}')
            
            axes[2].set_title(f'Final Time Comparison (t={final_time}, shift={spatial_shift})')
            axes[2].set_xlabel('Spatial Position')  
            axes[2].set_ylabel('Field Value')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            
        except Exception as e:
            print(f"Error creating plot for sample {sample_idx}: {e}")
            continue
            
        plt.suptitle(f'Sample {sample_idx}: 1D Spatial Profiles Over Time')
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, f'sample_{sample_idx}_spatial_profiles.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    print(f"1D spatial profile plots saved for {len(sample_indices)} samples")

def plot_2d_predictions(pred, gt, output_dir, n_input_frames, num_samples=3):
    """Plot 2D predictions over time (heatmaps for selected time steps)."""
    batch_size, channels, height, width, pred_time = pred.shape
    gt_time = gt.shape[-1]
    
    # Select random samples to plot
    sample_indices = np.random.choice(batch_size, min(num_samples, batch_size), replace=False)
    
    # Select time steps to show
    time_steps = [0, pred_time//4, pred_time//2, 3*pred_time//4, pred_time-1]
    
    for sample_idx in sample_indices:
        fig, axes = plt.subplots(2, len(time_steps), figsize=(4*len(time_steps), 8))
        
        for i, t_step in enumerate(time_steps):
            actual_time = n_input_frames + t_step
            
            # Ground truth at this time
            gt_frame = gt[sample_idx, 0, :, :, actual_time]  # Channel 0
            pred_frame = pred[sample_idx, 0, :, :, t_step]
            
            # Plot ground truth
            im1 = axes[0, i].imshow(gt_frame, cmap='viridis', aspect='auto')
            axes[0, i].set_title(f'GT - Time {actual_time}')
            axes[0, i].set_xlabel('Width')
            axes[0, i].set_ylabel('Height')
            plt.colorbar(im1, ax=axes[0, i])
            
            # Plot prediction
            im2 = axes[1, i].imshow(pred_frame, cmap='viridis', aspect='auto')
            axes[1, i].set_title(f'Pred - Time {actual_time}')
            axes[1, i].set_xlabel('Width')
            axes[1, i].set_ylabel('Height')
            plt.colorbar(im2, ax=axes[1, i])
            
        plt.suptitle(f'Sample {sample_idx}: 2D Predictions vs Ground Truth')
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, f'sample_{sample_idx}_2d_predictions.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    print(f"2D prediction plots saved for {len(sample_indices)} samples")

def test(args):
    # Set up logging
    L.seed_everything(args.seed)
    
    # Get dataset name first
    dataset_name = args.dataset_name
    
    # Create logger with descriptive run name
    run = wandb.init(project="disco-baselines")
    run.tags = (
            ("zebra",)
            + ("test",)
            + (dataset_name,)
        )
    run_name = wandb.run.name
    output_ckpt_dir = f"{args.output_dir}/{dataset_name}/{run_name}/"
    os.makedirs(output_ckpt_dir, exist_ok=True)
    wandb_logger = WandbLogger(
        project="disco-baselines",
        config=vars(args),
        name=run_name
    )


    # load data
    if dataset_name=="advection-diffusion":
        # Advection-diffusion: always tokenize on the fly
        n_batches = int(args.num_samples//args.batch_size)  # or set as needed for your epoch size
        
        EXPERIMENT_CONFIGS = {
        'E_AD_v_easy': {
            'v_range': (0.01, 1.0),  # Grid of advection and diffusion in [0,1]
            'D_range': (0.0, 0.0),
            'description': 'Just for testing zebra diffusion in [0,1] range'
        },
        'E_AD_ALL': {
            'v_range': (0.01, 1.0),  # Grid of advection and diffusion in [0,1]
            'D_range': (0.01, 1.0),
            'description': 'Both advection and diffusion in [0,1] range'
        },
        'E_AD_v': {
            'v_range': (1.0, 3.0),   # Advection speed in [1,3]
            'D_range': (0.0, 0.0),   # No diffusion (pure advection)
            'description': 'High advection speed [1,3], no diffusion'
        },
        'E_AD_D': {
            'v_range': (0.0, 0.0),   # No advection (pure diffusion)
            'D_range': (1.0, 3.0),   # Diffusion in [1,3]
            'description': 'High diffusion [1,3], no advection'
            }
        }

        N_INPUT_FRAMES = 16
        N_OUTPUT_FRAMES = 34
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)

        # Model loading is handled later

        # Load datasets based on experiment type
        experiment_config = EXPERIMENT_CONFIGS[args.experiment]
        print(f"\nLoading datasets for experiment: {args.experiment}")
        print(f"Description: {experiment_config['description']}")
        print(f"v_range: {experiment_config['v_range']}, D_range: {experiment_config['D_range']}")

        # Create test dataset (using experiment-specific parameter ranges)
        # Calculate batches needed for desired number of samples
        test_n_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
        
        test_dataset = TemporalBatchDatasetFly(
            n_batches=test_n_batches,
            batch_size=args.batch_size,
            sub_x=1,
            sub_t=1,
            split='test',
            input_frames=N_INPUT_FRAMES,
            output_frames=N_OUTPUT_FRAMES,
            L=16.0,
            nx=256,
            nt=100,
            T=10.0,
            v_range=experiment_config['v_range'],
            D_range=experiment_config['D_range'],
            fractal_degree=256,
            fractal_power_range=3,
            seed=124
        )
        
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=None, num_workers=args.num_workers, prefetch_factor=4, pin_memory=False)
        
    elif dataset_name=="combined-equation":

        EXPERIMENT_FILES = {

        'E_DEBUG': {
            'train': '/mnt/home/lserrano/disco-ball/datasets/combined_equation/E_HEAT_valid.h5',
            #'test': '/mnt/home/lserrano/disco-ball/datasets/combined_equation/E_BG_test.h5'
        },    
        'E_BG': {
            'train': '/mnt/home/lserrano/ceph/E_BG_train_gridparam512.h5',
            #'test': '/mnt/home/lserrano/disco-ball/datasets/combined_equation/E_BG_test.h5'
        },
        'E_ED': {
            'train': '/mnt/home/lserrano/ceph/E_ED_train_gridparam512.h5',
            #'test': '/mnt/home/lserrano/disco-ball/datasets/combined_equation/ood/E_ED_test.h5'
        },
        'E_HE': {
            'train': '/mnt/home/lserrano/ceph/E_HE_train_gridparam512.h5',
            #'test': '/mnt/home/lserrano/disco-ball/datasets/combined_equation/ood/E_HE_test.h5'
        },
        'E_ALL': {
            'train': '/mnt/home/lserrano/ceph/E_ALL_train_gridparam512.h5',
            #'test': '/mnt/home/lserrano/disco-ball/datasets/combined_equation/test.h5'
        },
        'E_EULER_OOD': {
            'train': '/mnt/home/lserrano/ceph/E_EULER_OOD_train_envsize16.h5',
        },
        'E_DISP_OOD': {
            'train': '/mnt/home/lserrano/ceph/E_DISP_OOD_train_envsize16.h5',
            }
        }

        N_INPUT_FRAMES = 16
        N_OUTPUT_FRAMES = 50
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)

        # Model loading is handled later

        # Load datasets based on experiment type
        print(f"\nLoading datasets for experiment: {args.experiment}...")
        #train_file = EXPERIMENT_DATASETS[args.experiment]['train']
        test_file = EXPERIMENT_FILES[args.experiment]['train']

        if not os.path.exists(test_file):
            print(f"Test file not found: {test_file}")
            return

        # For testing, we don't need train dataset
        # train_dataset = HDF5TemporalDataset(
        #     hdf5_files=TRAINING_FILES,
        #     input_frames=N_INPUT_FRAMES,
        #     output_frames=N_OUTPUT_FRAMES,
        #     sub_x=1,
        #     sub_t=1,
        #     split='train'
        # )

        test_dataset = HDF5TemporalDataset(
            hdf5_files=[test_file],
            input_frames=N_INPUT_FRAMES,
            output_frames=N_OUTPUT_FRAMES,
            sub_x=1,
            sub_t=1,
            split='train',
            mode="test"
        )

        # For testing, we don't need train loader

        test_loader = DataLoader(
            test_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=4,
            prefetch_factor=2,
            pin_memory=True
        )

        # print(f"Train dataset: {len(train_dataset)} samples")  # Commented out since train_dataset not defined
        print(f"Test dataset: {len(test_dataset)} samples")
        
    elif dataset_name=="gray-scott":
        # Gray-Scott dataset
        TEST_FILES = ["/mnt/home/lserrano/gray-scott-python/data/gray_scott_10x10_params_16traj_each.hdf5"]
        N_INPUT_FRAMES = 16
        N_OUTPUT_FRAMES = 32
        
        test_ds = GrayScottDatasetWrapper(
        hdf5_files=TEST_FILES,
        split='test',
        input_frames=N_INPUT_FRAMES,
        output_frames=N_OUTPUT_FRAMES,
        sub_x=1,
        sub_t=1,
        trajectories_per_environment=16,
        mode="test")
        
        test_loader = DataLoader(
            test_ds, 
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            prefetch_factor=2,
            pin_memory=True,
            
        )

    checkpoint_dict = {
        "advection-diffusion": f"/mnt/home/lserrano/ceph/zebra/llama/{dataset_name}/radiant-planet-99/last.ckpt", # best.ckpt before
        "combined-equation":f"/mnt/home/lserrano/ceph/zebra/llama/{dataset_name}/skilled-plasma-98/best.ckpt",
        "gray-scott": f"/mnt/home/lserrano/ceph/zebra/llama/{dataset_name}/dashing-violet-94/best.ckpt"
    }
    checkpoint_path = checkpoint_dict[dataset_name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lit_model = LLaMATrainer.load_from_checkpoint(checkpoint_path, map_location=device)
    vocab_size = lit_model.model.model.vocab_size - 8 # vocab_size here is the non-syntaxic tokens

    if dataset_name in ['advection-diffusion', "combined-equation"]:
        avg_loss, pred, gt = get_test_metrics_1d(test_loader, lit_model, input_size=N_INPUT_FRAMES, output_size=N_OUTPUT_FRAMES, num_examples=0, vocab_size=vocab_size)

    elif dataset_name in ['gray-scott']:
        avg_loss, pred, gt = get_test_metrics_2d(test_loader, lit_model, input_size=N_INPUT_FRAMES, output_size=N_OUTPUT_FRAMES, max_len_size=31, num_examples=0, vocab_size=vocab_size)

    print(f"Dataset: {dataset_name}, Average Loss: {avg_loss}")
    
    # Save results
    results = {
        'dataset_name': dataset_name,
        'average_loss': float(avg_loss),
        'num_samples': args.num_samples,
        'batch_size': args.batch_size,
        'seed': args.seed,
        'checkpoint_path': checkpoint_path,
    }
    
    if dataset_name in ['advection-diffusion', 'combined-equation']:
        results['experiment'] = args.experiment
    
    # Save results to JSON
    results_path = os.path.join(output_ckpt_dir, 'test_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")
    
    # Save predictions and ground truth
    predictions_path = os.path.join(output_ckpt_dir, 'predictions.npz')
    np.savez(predictions_path, predictions=pred, ground_truth=gt)
    print(f"Predictions saved to: {predictions_path}")
    
    # Create and save plots
    plot_predictions_over_time(pred, gt, output_ckpt_dir, dataset_name, N_INPUT_FRAMES)
    
    # Log to wandb
    wandb.log(results)
    wandb.finish()
    
if __name__ == "__main__":
    args = parse_args()
    test(args) 

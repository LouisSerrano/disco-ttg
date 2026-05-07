import torch
import argparse
import os
import time
from datetime import datetime
from torch.utils.data import DataLoader
import sys
import numpy as np
from einops import rearrange
import copy
import csv

sys.path.append("/mnt/home/lserrano/disco-ttg/test_time_compute")

from ttc_utils import (
    save_results,
    CombinedHDF5TemporalDataset,
    DEVICE
)

from train.train import TemporalBatchDatasetFly
from train_advection_diffusion import GEPSLightning as GEPSAdvectionDiffusionLightning
from train_combined_equation import GEPSLightning as GEPSCombinedLightning
from train_gray_scott import GEPSLightning as GEPSGrayScottLightning, GrayScottDatasetWrapper
from src.utils.database import RelativeL2
import random

from fvcore.nn import FlopCountAnalysis

def count_flops_inference(model, model_input, time_grid, env_codes):
    """Count FLOPs for a single model inference call"""
    
    # Create a wrapper module that calls forward_with_codes in its forward method
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model, time_grid, env_codes):
            super().__init__()
            self.model = model
            self.time_grid = time_grid
            self.env_codes = env_codes
        
        def forward(self, x):
            return self.model.forward_with_codes(x, self.time_grid, self.env_codes)
    
    # Wrap the model
    wrapped_model = ModelWrapper(model, time_grid, env_codes)
    
    # Count FLOPs using the wrapped model
    flops = FlopCountAnalysis(wrapped_model, model_input)
    total_flops = flops.total()
    
    return total_flops


def inference_with_flops(model, model_input, time_grid, env_codes, cumulative_flops=0, cumulative_time=0):
    """Perform inference while tracking FLOPs and time"""
    start_time = time.time()
    
    # Count FLOPs for this inference
    current_flops = count_flops_inference(model, model_input, time_grid, env_codes.detach())
    
    # Perform actual inference
    pred = model.forward_with_codes(model_input, time_grid, env_codes)
    
    inference_time = time.time() - start_time
    
    # Update cumulative metrics
    total_flops = cumulative_flops + current_flops
    total_time = cumulative_time + inference_time
    
    return pred, total_flops, total_time

def load_geps_model_from_checkpoint(checkpoint_path, equation_type):
    """Load GEPS model from Lightning checkpoint"""
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return None, None

    try:
        # Select the appropriate Lightning class based on equation type
        if equation_type == 'advection_diffusion':
            lit_model = GEPSAdvectionDiffusionLightning.load_from_checkpoint(checkpoint_path, map_location=DEVICE)
        elif equation_type == 'combined_equation':
            lit_model = GEPSCombinedLightning.load_from_checkpoint(checkpoint_path, map_location=DEVICE)
        elif equation_type == 'gray_scott':
            lit_model = GEPSGrayScottLightning.load_from_checkpoint(checkpoint_path, map_location=DEVICE)
        else:
            raise ValueError(f"Unknown equation type: {equation_type}")
            
        lit_model.eval()
        model = lit_model.model.to(DEVICE)
        model.eval()

        print(f"GEPS model loaded successfully from {checkpoint_path}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        return model, lit_model

    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None


def optimize_codes_for_trajectory(model, input_sequence, test_target, cfg, n_output_frames, n_optimization_steps=50, is_1d=True, lr=0.001, weight_decay=0.0, n_pred=1):
      """
      Create and optimize fresh environment codes for a specific trajectory.
      
      Args:
          model: GEPS forecaster model
          input_sequence: Input trajectory sequence [T, C, H] or [T, C, H, W]
          test_target: Target sequence for testing [T, C, H] or [T, C, H, W]
          cfg: Model configuration
          n_output_frames: Number of output frames to predict during testing
          n_optimization_steps: Number of optimization steps
          is_1d: Whether data is 1D (advection_diffusion, combined_equation) or 2D (gray_scott)
          lr: Learning rate for code optimization
          weight_decay: Weight decay for code optimization
          n_pred: Number of prediction steps to use during optimization (default: 1)
          
      Returns:
          optimized_codes: The optimized environment codes tensor
          test_error: Final test error
          total_optimization_flops: Total FLOPs used during optimization
          total_optimization_time: Total wall-clock time for entire optimization process
          total_forward_time: Time spent only in forward passes
      """

      if input_sequence.shape[0] < 2:
          print("Warning: Not enough frames for code optimization")
          # Return random codes if not enough frames
          code_dim = cfg.model.code_c
          empty_scaling_metrics = {
              'step_flops_history': [0],
              'step_cumulative_flops_history': [0],
              'step_forward_times': [0.0],
              'step_total_times': [0.0],
              'step_cumulative_forward_times': [0.0],
              'step_cumulative_total_times': [0.0],
              'step_losses': [0.0],
              'n_steps': 0
          }
          return torch.randn(1, code_dim, device=DEVICE, requires_grad=False), 0.0, 0, 0.0, 0.0, empty_scaling_metrics

      # Create fresh environment codes for this trajectory
      n_samples = input_sequence.shape[0]  # Fixed: was input_sequences

      code_dim = cfg.model.code_c
      env_codes = (torch.randn(n_samples, code_dim, device=DEVICE)*0.01).requires_grad_(True)
      
      # Initialize tracking variables
      total_optimization_flops = 0
      total_forward_time = 0.0  # Time spent in forward passes only
      total_optimization_time = 0.0  # Total wall-clock time for entire optimization
      optimization_start_time = time.time()  # Start timing the entire optimization process
      
      # Per-step tracking for scaling analysis
      step_flops_history = []  # FLOPs at each step
      step_cumulative_flops_history = []  # Cumulative FLOPs up to each step
      step_forward_times = []  # Forward time for each step
      step_total_times = []  # Total time for each step
      step_cumulative_forward_times = []  # Cumulative forward time up to each step
      step_cumulative_total_times = []  # Cumulative total time up to each step
      step_losses = []  # Loss at each step
      
      # Handle no-optimization baseline case
      if n_optimization_steps == 0:
          print("No optimization (baseline case)")
          # Test with random codes
          test_input = input_sequence[:, -1:].clone()
          
          if is_1d:
              model_input = rearrange(test_input, "b t c h -> b c h t")
          else:
              model_input = rearrange(test_input, "b t c h w -> b c h w t")
          
          dt = cfg.model.default_integration_time
          time_grid = torch.tensor([i * dt for i in range(n_output_frames + 1)], device=DEVICE)
          
          # Use inference_with_flops for tracking even in no-optimization case
          pred, flops, inference_time = inference_with_flops(model, model_input, time_grid, env_codes.detach())
          pred = pred[..., 1:]
          total_optimization_flops += flops
          total_forward_time += inference_time
          total_optimization_time = time.time() - optimization_start_time
          
          if is_1d:
              test_predictions = rearrange(pred, "b c h t -> b t c h")
          else:
              test_predictions = rearrange(pred, "b c h w t -> b t c h w")
          
          rel_loss = RelativeL2()
          test_error = rel_loss(test_predictions, test_target)
          print(f"No optimization test error: {test_error.item():.6f}")
      
          # Create empty scaling metrics for no-optimization case
          scaling_metrics = {
              'step_flops_history': [total_optimization_flops],
              'step_cumulative_flops_history': [total_optimization_flops],
              'step_forward_times': [total_forward_time],
              'step_total_times': [total_optimization_time],
              'step_cumulative_forward_times': [total_forward_time],
              'step_cumulative_total_times': [total_optimization_time],
              'step_losses': [0.0],
              'n_steps': 1
          }
          return env_codes.detach(), test_error.item(), total_optimization_flops, total_optimization_time, total_forward_time, scaling_metrics

      # Set up optimizer for the codes
      optimizer = torch.optim.Adam([env_codes], lr=lr, weight_decay=weight_decay)
      scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_optimization_steps)

      rel_loss = RelativeL2()

      input_length = input_sequence.shape[1]

      # Optimize codes on input sequence transitions
      print(f"Optimizing codes for trajectory with {input_sequence.shape[0]} frames...")

      for step in range(n_optimization_steps):
          step_start_time = time.time()  # Time for this entire optimization step
          
          optimizer.zero_grad()

          # Sample starting position ensuring we have enough frames for n_pred predictions
          max_start = input_length - n_pred - 1
          if max_start < 0:
              print(f"Warning: Not enough frames for n_pred={n_pred} predictions. Using available frames.")
              n_pred_actual = min(n_pred, input_length - 1)
              i = 0
          else:
              n_pred_actual = n_pred
              i = random.randint(0, max_start)
          
          current_input = input_sequence[:,i:i+1]
          target = input_sequence[:,i+1:i+1+n_pred_actual]  # Get n_pred_actual target frames

          # Adjust dimensions based on data type (following exact training format)
          if is_1d:  # 1D case: advection_diffusion, combined_equation
              current_input = rearrange(current_input, "b t c h -> b c h t")
              target = rearrange(target, "b t c h -> b c h t")
          else:  # 2D case: gray_scott
              current_input = rearrange(current_input, "b t c h w -> b c h w t")
              target = rearrange(target, "b t c h w -> b c h w t")

          # Create time tensor for n_pred_actual step predictions
          dt = cfg.model.default_integration_time
          time_grid = torch.tensor([j * dt for j in range(n_pred_actual + 1)], device=DEVICE)
          
          # Forward pass with current codes - track FLOPs during optimization
          pred, step_flops, step_forward_time = inference_with_flops(model, current_input, time_grid, env_codes, 
                                                                    total_optimization_flops, total_forward_time)
          
          # Calculate step-specific metrics
          step_only_flops = step_flops - total_optimization_flops  # FLOPs for just this step
          step_only_forward_time = step_forward_time - total_forward_time  # Forward time for just this step
          
          # Update cumulative totals
          total_optimization_flops = step_flops
          total_forward_time = step_forward_time
          pred = pred[..., 1:]  # Remove initial timepoint

          # Calculate loss
          loss = rel_loss(pred, target)

          # Backward pass
          loss.backward()  # Fixed: was total_loss.backward()
          optimizer.step()
          scheduler.step()
          
          # Track total time for this step (includes forward, backward, optimizer step)
          step_total_time = time.time() - step_start_time
          cumulative_total_time = time.time() - optimization_start_time
          
          # Record step metrics for scaling analysis
          step_flops_history.append(step_only_flops)
          step_cumulative_flops_history.append(total_optimization_flops)
          step_forward_times.append(step_only_forward_time)
          step_total_times.append(step_total_time)
          step_cumulative_forward_times.append(total_forward_time)
          step_cumulative_total_times.append(cumulative_total_time)
          step_losses.append(loss.item())

          if step % 100 == 0 or step == n_optimization_steps - 1:
              print(f"Step {step}: Loss = {loss.item():.6f}")
              
              # Test current codes on prediction task
              # Use last frame of input sequence as starting point
              test_input = input_sequence[:, -1:].clone()  # [n_samples, 1, C, H] or [n_samples, 1, C, H, W]
              
              # Adjust dimensions for model input
              if is_1d:
                  model_input = rearrange(test_input, "b t c h -> b c h t")
              else:
                  model_input = rearrange(test_input, "b t c h w -> b c h w t")
              
              # Create temporal grid as tensor
              dt = cfg.model.default_integration_time
              time_grid = torch.tensor([i * dt for i in range(n_output_frames + 1)], device=DEVICE)  # [0, dt, 2*dt, 3*dt, ...]
              
              # Single forward pass with all timestamps - track FLOPs for test inference
              pred, step_flops, step_forward_time = inference_with_flops(model, model_input, time_grid, env_codes.detach(),
                                                                        total_optimization_flops, total_forward_time)
              total_optimization_flops = step_flops
              total_forward_time = step_forward_time
              pred = pred[..., 1:]  # Remove initial timepoint, get [n_samples, C, H, n_output_frames] or [n_samples, C, H, W, n_output_frames]
              
              # Reshape predictions to match test_target format
              if is_1d:
                  test_predictions = rearrange(pred, "b c h t -> b t c h")
              else:
                  test_predictions = rearrange(pred, "b c h w t -> b t c h w")
              
              test_error = rel_loss(test_predictions, test_target)
              error_through_time = [rel_loss(test_predictions[:, t], test_target[:, t]).item() for t in range(n_output_frames)]
              
              # Calculate elapsed wall-clock time
              elapsed_optimization_time = time.time() - optimization_start_time
              
              print(f"  Test error: {test_error.item():.6f}")
              print(f"  Test error through time: {error_through_time}")
              print(f"  Cumulative FLOPs: {total_optimization_flops/1e9:.3f}G")
              print(f"  Forward time: {total_forward_time:.2f}s, Total time: {elapsed_optimization_time:.2f}s")
              print(f"  This step - FLOPs: {step_only_flops/1e9:.3f}G, Forward: {step_only_forward_time:.4f}s, Total: {step_total_time:.4f}s")


      # Calculate final total optimization time
      total_optimization_time = time.time() - optimization_start_time
      
      print(f"Code optimization completed. Final training loss: {loss.item():.6f}, Final test error: {test_error.item():.6f}")
      print(f"Total optimization FLOPs: {total_optimization_flops/1e9:.3f}G")
      print(f"Forward time: {total_forward_time:.2f}s, Total optimization time: {total_optimization_time:.2f}s")
      
      # Package step-by-step metrics for scaling analysis
      scaling_metrics = {
          'step_flops_history': step_flops_history,
          'step_cumulative_flops_history': step_cumulative_flops_history,
          'step_forward_times': step_forward_times,
          'step_total_times': step_total_times,
          'step_cumulative_forward_times': step_cumulative_forward_times,
          'step_cumulative_total_times': step_cumulative_total_times,
          'step_losses': step_losses,
          'n_steps': len(step_losses)
      }
      
      return env_codes.detach(), test_error.item(), total_optimization_flops, total_optimization_time, total_forward_time, scaling_metrics


def test_geps_inference(model, test_loader, equation_type, cfg, n_output_frames, n_optimization_steps=50, lr=0.001, weight_decay=0.0, n_pred=1):
    """Test GEPS inference with code optimization and FLOP counting"""
    model.eval()
    rel_loss = RelativeL2()
    
    all_errors = []
    all_optimization_times = []
    all_forward_times = []
    all_inference_flops = []
    all_optimization_flops = []
    all_scaling_metrics = []  # Store scaling metrics from each batch
    total_error = 0
    total_samples = 0
    
    for batch_idx, batch in enumerate(test_loader):
        start_time = time.time()
        
        # Extract batch data based on equation type with proper key names
        if equation_type == 'advection_diffusion':
            input_seq = batch['input'].to(DEVICE)  # [B, T, C, H]
            target_seq = batch['target'].to(DEVICE)  # [B, T, C, H]
            is_1d = True
            
        elif equation_type == 'combined_equation':
            input_seq = batch['input'].to(DEVICE)  # [B, T, C, H]
            target_seq = batch['output'].to(DEVICE)  # [B, T, C, H]
            is_1d = True
            
        else:  # gray_scott
            input_seq = batch['input'].to(DEVICE)  # [B, T, C, H, W]
            target_seq = batch['output'].to(DEVICE)  # [B, T, C, H, W]
            is_1d = False
        
        batch_predictions = []
        batch_errors = []
        
        # Process each sample in the batch
        
        # Step 1: Create and optimize fresh codes for this trajectory
        optimized_codes, test_error, optimization_flops, optimization_time, forward_time, scaling_metrics = optimize_codes_for_trajectory(
            model, input_seq, target_seq, cfg, n_output_frames, n_optimization_steps, is_1d, lr, weight_decay, n_pred
        )
        
        # Step 2: Final inference FLOPs are already included in optimization_flops
        # No need for separate final inference counting since it's tracked during optimization
        
        # Calculate error for this sample
        batch_errors.append(test_error)
        n_samples = input_seq.shape[0]

        # Normalize scaling traces to per-sample values before aggregation
        if scaling_metrics.get('n_steps', 0) > 0:
            scaling_metrics['step_cumulative_flops_history'] = [
                flops / n_samples for flops in scaling_metrics['step_cumulative_flops_history']
            ]
            scaling_metrics['step_flops_history'] = [
                flops / n_samples for flops in scaling_metrics['step_flops_history']
            ]

        all_inference_flops.append(optimization_flops / n_samples)  # Divide by batch size for per-sample FLOPs
        all_optimization_flops.append(optimization_flops / n_samples)  # Divide by batch size for per-sample FLOPs
        all_optimization_times.append(optimization_time / n_samples)  # Also divide timing for consistency
        all_forward_times.append(forward_time / n_samples)  # Also divide timing for consistency
        all_scaling_metrics.append(scaling_metrics)
        total_samples += n_samples
        total_error += test_error*n_samples
        
        print(f"Batch {batch_idx}: Error={test_error:.6f}, Total FLOPs={optimization_flops/1e9:.3f}G")
        print(f"  Forward time: {forward_time:.2f}s, Total time: {optimization_time:.2f}s")
        print(f"  Optimization steps: {scaling_metrics['n_steps']}")
    
    avg_error = total_error / total_samples
    avg_optimization_time = np.mean(all_optimization_times) if all_optimization_times else 0
    avg_forward_time = np.mean(all_forward_times) if all_forward_times else 0
    avg_inference_flops = np.mean(all_inference_flops) if all_inference_flops else 0
    avg_optimization_flops = np.mean(all_optimization_flops) if all_optimization_flops else 0
    
    return avg_error, avg_optimization_time, avg_forward_time, avg_inference_flops, avg_optimization_flops, all_scaling_metrics


def main():
    parser = argparse.ArgumentParser(description='Test GEPS inference with code optimization')
    parser.add_argument('--model_path', type=str, required=True, help='Path to GEPS model checkpoint')
    parser.add_argument('--equation_type', type=str, required=True, 
                        choices=['advection_diffusion', 'combined_equation', 'gray_scott'],
                        help='Type of equation/dataset')
    parser.add_argument('--output_dir', type=str, default='./GEPS/results', help='Output directory')
    parser.add_argument('--num_samples', type=int, default=64, help='Number of test samples') #512 before
    parser.add_argument('--experiment', type=str, help='Experiment identifier')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--n_optimization_steps', type=int, default=500, help='Steps for code optimization') # 2000 for adv-diff, 500 as default, 100 if it diverges
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for code optimization')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay for code optimization')
    parser.add_argument('--n_pred', type=int, default=1, help='Number of predictions to use during optimization')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print(f"Loading GEPS {args.equation_type} model...")
    model, lit_model = load_geps_model_from_checkpoint(args.model_path, args.equation_type)
    if model is None:
        print("Failed to load model")
        return

    cfg = lit_model.cfg
    
    # Define experiment configurations for each equation type
    if args.equation_type == 'advection_diffusion':
        EXPERIMENT_CONFIGS = {
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
        
    elif args.equation_type == 'combined_equation':
        EXPERIMENT_FILES = {
            'E_BG': {
                'train': '/mnt/home/lserrano/ceph/E_BG_train_gridparam512.h5',
            },
            'E_ED': {
                'train': '/mnt/home/lserrano/ceph/E_ED_train_gridparam512.h5',
            },
            'E_HE': {
                'train': '/mnt/home/lserrano/ceph/E_HE_train_gridparam512.h5',
            },
            'E_ALL': {
                'train': '/mnt/home/lserrano/ceph/E_ALL_train_gridparam512.h5',
            },
            'E_HEAT':{
                'train': '/mnt/home/lserrano/ceph/E_HEAT_train_gridparam8192.h5',
            },
            'E_EULER':
            {
                'train': '/mnt/home/lserrano/disco-ttg/datasets/combined_equation/E_EULER_valid.h5',
            },
             'E_EULER_OOD': {
                'train': '/mnt/home/lserrano/ceph/E_EULER_OOD_train_envsize16.h5',},
            'E_DISP_OOD': {
            'train': '/mnt/home/lserrano/ceph/E_DISP_OOD_train_envsize16.h5',
            }
        }
        N_INPUT_FRAMES = 16
        N_OUTPUT_FRAMES = 50
        
    else:  # gray_scott
        TEST_FILES = ["/mnt/home/lserrano/gray-scott-python/data/gray_scott_10x10_params_16traj_each.hdf5"]
        N_INPUT_FRAMES = 16
        N_OUTPUT_FRAMES = 32

    # Create test dataset based on equation type
    print(f"Creating test dataset for {args.equation_type}...")
    
    if args.equation_type == 'advection_diffusion':
        if not args.experiment or args.experiment not in EXPERIMENT_CONFIGS:
            raise ValueError(f"--experiment required for advection_diffusion. Options: {list(EXPERIMENT_CONFIGS.keys())}")
            
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
        test_loader = test_dataset
        
    elif args.equation_type == 'combined_equation':
        if not args.experiment or args.experiment not in EXPERIMENT_FILES:
            raise ValueError(f"--experiment required for combined_equation. Options: {list(EXPERIMENT_FILES.keys())}")
            
        print(f"\nLoading datasets for experiment: {args.experiment}...")
        test_file = EXPERIMENT_FILES[args.experiment]['train']

        if not os.path.exists(test_file):
            print(f"Test file not found: {test_file}")
            return

        test_dataset = CombinedHDF5TemporalDataset(
            hdf5_files=[test_file],
            input_frames=N_INPUT_FRAMES,
            output_frames=N_OUTPUT_FRAMES,
            sub_x=1,
            sub_t=1,
            split='train'  # Using train split for evaluation as specified
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            prefetch_factor=2,
            pin_memory=True
        )
        
    else:  # gray_scott
        print("\nLoading test dataset...")
        # Gray-Scott uses a single dataset configuration
        if args.experiment:
            print(f"Note: Gray-Scott uses single dataset configuration. Ignoring experiment: {args.experiment}")
            
        test_dataset = GrayScottDatasetWrapper(
            hdf5_files=TEST_FILES,
            split='test',
            input_frames=N_INPUT_FRAMES,
            output_frames=N_OUTPUT_FRAMES,
            sub_x=1,
            sub_t=1,
            trajectories_per_environment=16
        )

        test_loader = DataLoader(
            test_dataset, 
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            prefetch_factor=2,
            pin_memory=True
        )

    print(f"Test dataset created with {len(test_dataset) if hasattr(test_dataset, '__len__') else 'streaming'} samples")

    # Run GEPS inference
    print(f"\nTesting GEPS inference for {args.equation_type}...")
    start_time = time.time()
    
    geps_error, geps_optimization_time, geps_forward_time, geps_flops, geps_optimization_flops, all_scaling_metrics = test_geps_inference(
        model, test_loader, args.equation_type, cfg, 
        N_OUTPUT_FRAMES, args.n_optimization_steps, args.lr, args.weight_decay, args.n_pred
    )
    
    total_time = time.time() - start_time
    
    # Analyze scaling across all batches
    if all_scaling_metrics and len(all_scaling_metrics) > 0:
        # Average across all batches to get typical scaling behavior
        max_steps = max(metrics['n_steps'] for metrics in all_scaling_metrics if metrics['n_steps'] > 0)
        if max_steps > 0:
            # Create step-wise averages
            step_avg_cumulative_flops = []
            step_avg_cumulative_times = []
            step_avg_losses = []
            
            for step in range(max_steps):
                step_flops = []
                step_times = []
                step_losses = []
                
                for metrics in all_scaling_metrics:
                    if step < len(metrics['step_cumulative_flops_history']):
                        step_flops.append(metrics['step_cumulative_flops_history'][step])
                        step_times.append(metrics['step_cumulative_total_times'][step])
                        step_losses.append(metrics['step_losses'][step])
                
                if step_flops:  # Only if we have data for this step
                    step_avg_cumulative_flops.append(np.mean(step_flops))
                    step_avg_cumulative_times.append(np.mean(step_times))
                    step_avg_losses.append(np.mean(step_losses))
            
            print(f"\n{'='*50}")
            print(f"SCALING ANALYSIS (averaged across {len(all_scaling_metrics)} batches):")
            print(f"{'='*50}")
            
            # Show scaling at key points
            scaling_points = [0, max_steps//4, max_steps//2, 3*max_steps//4, max_steps-1] if max_steps > 4 else list(range(max_steps))
            scaling_points = [p for p in scaling_points if p < len(step_avg_cumulative_flops)]
            
            for step in scaling_points:
                if step < len(step_avg_cumulative_flops):
                    print(f"Step {step:4d}: FLOPs={step_avg_cumulative_flops[step]/1e9:8.3f}G, Time={step_avg_cumulative_times[step]:8.2f}s, Loss={step_avg_losses[step]:8.6f}")
            
            # Calculate FLOP and time efficiency
            if len(step_avg_cumulative_flops) >= 2:
                flops_per_step = (step_avg_cumulative_flops[-1] - step_avg_cumulative_flops[0]) / max(1, max_steps-1)
                time_per_step = (step_avg_cumulative_times[-1] - step_avg_cumulative_times[0]) / max(1, max_steps-1)
                print(f"\nAverage per step: {flops_per_step/1e9:.3f}G FLOPs, {time_per_step:.4f}s")
                print(f"Linear scaling: {max_steps * flops_per_step/1e9:.3f}G FLOPs expected vs {step_avg_cumulative_flops[-1]/1e9:.3f}G actual")
        else:
            step_avg_cumulative_flops = []
            step_avg_cumulative_times = []
            step_avg_losses = []
    else:
        step_avg_cumulative_flops = []
        step_avg_cumulative_times = []
        step_avg_losses = []
    
    results = {
        'equation_type': args.equation_type,
        'model_type': 'GEPS',
        'experiment': args.experiment or 'default',
        'timestamp': datetime.now().isoformat(),
        'num_samples': args.num_samples,
        'n_optimization_steps': args.n_optimization_steps,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'n_pred': args.n_pred,
        'geps_inference': {
            'error': geps_error,
            'avg_optimization_time_per_batch': geps_optimization_time,
            'avg_forward_time_per_batch': geps_forward_time,
            'total_time': total_time,
            'avg_inference_flops': geps_flops,
            'avg_optimization_flops': geps_optimization_flops,
            'gflops': geps_flops / 1e9,
            'optimization_gflops': geps_optimization_flops / 1e9,
            'flops_per_error': geps_flops / geps_error if geps_error > 0 else float('inf'),
            'optimization_flops_per_error': geps_optimization_flops / geps_error if geps_error > 0 else float('inf'),
            'forward_time_fraction': geps_forward_time / geps_optimization_time if geps_optimization_time > 0 else 0
        },
        'scaling_analysis': {
            'step_avg_cumulative_flops': step_avg_cumulative_flops,
            'step_avg_cumulative_times': step_avg_cumulative_times,
            'step_avg_losses': step_avg_losses,
            'max_steps': max_steps if 'max_steps' in locals() else 0,
            'flops_per_step': (step_avg_cumulative_flops[-1] - step_avg_cumulative_flops[0]) / max(1, max_steps-1) if len(step_avg_cumulative_flops) >= 2 and 'max_steps' in locals() else 0,
            'time_per_step': (step_avg_cumulative_times[-1] - step_avg_cumulative_times[0]) / max(1, max_steps-1) if len(step_avg_cumulative_times) >= 2 and 'max_steps' in locals() else 0
        }
    }
    
    # Summary
    print("\n" + "="*50)
    print(f"GEPS INFERENCE RESULTS ({args.equation_type}):")
    print(f"Optimization Steps: {args.n_optimization_steps}")
    print(f"Average Error: {geps_error:.6f}")
    print(f"Average Total FLOPs (including optimization): {geps_optimization_flops/1e9:.3f} G")
    print(f"Total FLOPs per Error: {geps_optimization_flops/geps_error/1e6:.2f} M" if geps_error > 0 else "Total FLOPs per Error: inf")
    print(f"Average Optimization Time per Batch: {geps_optimization_time:.2f}s")
    print(f"Average Forward Time per Batch: {geps_forward_time:.2f}s")
    print(f"Forward Time Fraction: {geps_forward_time/geps_optimization_time*100:.1f}%" if geps_optimization_time > 0 else "Forward Time Fraction: N/A")
    print(f"Total Time: {total_time:.2f}s")
    print("="*50)
    
    # Save results
    exp_name = args.experiment or 'default'
    output_file = os.path.join(
        args.output_dir, 
        f"geps_{args.equation_type}_{exp_name}_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    save_results(results, output_file)
    
    # Save CSV for table generation
    csv_file = os.path.join(
        args.output_dir, 
        f"geps_{args.equation_type}_{exp_name}_table_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Optimization_Steps', 'Total_FLOPs_G', 'Optimization_Time_ms', 'Forward_Time_ms', 'Forward_Fraction_%', 'Error', 'Total_FLOPs_per_Error_M'])
        writer.writerow([
            args.n_optimization_steps,
            f"{geps_optimization_flops/1e9:.3f}",
            f"{geps_optimization_time*1000:.1f}",  # Convert to ms
            f"{geps_forward_time*1000:.1f}",  # Convert to ms
            f"{geps_forward_time/geps_optimization_time*100:.1f}" if geps_optimization_time > 0 else "0",
            f"{geps_error:.6f}",
            f"{geps_optimization_flops/geps_error/1e6:.2f}" if geps_error > 0 else "inf"
        ])
    
    # Save scaling analysis CSV
    scaling_csv_file = os.path.join(
        args.output_dir, 
        f"geps_{args.equation_type}_{exp_name}_scaling_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    
    if step_avg_cumulative_flops and len(step_avg_cumulative_flops) > 0:
        with open(scaling_csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Step', 'Cumulative_FLOPs_G', 'Cumulative_Time_s', 'Loss', 'FLOPs_per_Step_G', 'Time_per_Step_s'])
            
            for step in range(len(step_avg_cumulative_flops)):
                flops_per_step_val = step_avg_cumulative_flops[step] / max(1, step + 1) if step >= 0 else 0
                time_per_step_val = step_avg_cumulative_times[step] / max(1, step + 1) if step >= 0 else 0
                
                writer.writerow([
                    step,
                    f"{step_avg_cumulative_flops[step]/1e9:.3f}",
                    f"{step_avg_cumulative_times[step]:.3f}",
                    f"{step_avg_losses[step]:.6f}",
                    f"{flops_per_step_val/1e9:.3f}",
                    f"{time_per_step_val:.3f}"
                ])
    
    print(f"\nResults saved to:")
    print(f"JSON: {output_file}")
    print(f"CSV: {csv_file}")
    print(f"Scaling CSV: {scaling_csv_file}")


if __name__ == "__main__":
    main()

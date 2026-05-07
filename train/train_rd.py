"""Train DISCO on reaction-diffusion (Gray-Scott). Config: configs/config_rd.yaml."""
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import _LRScheduler
import wandb
from src.operators.disco import DISCOHouse
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from src.utils.database import RelativeL2
import h5py
import time
import random
import math
from einops import rearrange
from datetime import datetime
import logging
from typing import Dict, List, Union

def add_weight_decay(params, weight_decay=1e-5, skip_list=()):
    """ From Ross Wightman at:
    https://discuss.pytorch.org/t/weight-decay-in-the-optimizers-is-a-bad-idea-especially-with-batchnorm/16994/3 
    
    Goes through the parameter list and if the squeeze dim is 1 or 0 (usually means bias or scale) 
    then don't apply weight decay. 
    """
    decay = []
    no_decay = []
    for name, param in params:
        if not param.requires_grad:
            continue
        if (len(param.squeeze().shape) <= 1 or name in skip_list):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.,},
        {'params': decay, 'weight_decay': weight_decay}
    ]


class CosineWithWarmupScheduler(_LRScheduler):
    """Cosine annealing with linear warmup scheduler"""
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Calculate learning rate for current epoch"""
        if self.last_epoch <= self.warmup_steps:
            # Linear warmup
            lr_scale = self.last_epoch / self.warmup_steps if self.warmup_steps > 0 else 1.0
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)  # Clamp to [0, 1]
            lr_scale = self.min_lr_ratio + 0.5 * (1 - self.min_lr_ratio) * (1 + math.cos(math.pi * progress))
        
        return [base_lr * lr_scale for base_lr in self.base_lrs]


def compute_gradient_stats(model):
    """Compute detailed gradient statistics per layer"""
    grad_stats = {}
    total_norm = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad.detach()
            param_norm = grad.norm().item()
            total_norm += param_norm ** 2
            
            grad_stats[f'grad_norm/{name}'] = param_norm
            grad_stats[f'grad_mean/{name}'] = grad.mean().item()
            grad_stats[f'grad_std/{name}'] = grad.std().item()
    
    grad_stats['grad_norm/total'] = total_norm ** 0.5
    return grad_stats


def get_dataset_info(filename: str, group_name: str) -> Dict:
    """Get dataset information from HDF5 file."""
    with h5py.File(filename, 'r') as f:
        if group_name not in f:
            raise ValueError(f"Group '{group_name}' not found in {filename}")
        
        group = f[group_name]
        info = {
            'n_trajectories': group.attrs.get('n_trajectories', len(group['trajectory_a'])),
            'n_spatial_x': group.attrs.get('n_spatial_x', 128),
            'n_spatial_y': group.attrs.get('n_spatial_y', 128),
            'n_timesteps': group.attrs.get('n_timesteps', group['trajectory_a'].shape[1])
        }
    return info


class GrayScottHDF5Dataset:
    """
    Efficient dataset class for loading Gray-Scott trajectories from HDF5 files.
    Supports multiple files and keeps file handles open for fast access.
    """
    
    def __init__(self, filenames: Union[str, List[str]], group_name: str = 'train', 
                 reshape_to_spatial: bool = True, keep_file_open: bool = True):
        """
        Initialize the dataset.
        
        Args:
            filenames: Path(s) to HDF5 file(s)
            group_name: HDF5 group name
            reshape_to_spatial: Whether to reshape flattened data to spatial dimensions
            keep_file_open: Whether to keep file handle open for faster access
        """
        print(f"DEBUG: GrayScottHDF5Dataset received filenames: {filenames} (type: {type(filenames)})")
        self.filenames = filenames if isinstance(filenames, list) else [filenames]
        print(f"DEBUG: self.filenames after processing: {self.filenames}")
        self.group_name = group_name
        self.reshape_to_spatial = reshape_to_spatial
        self.keep_file_open = keep_file_open
        
        # Build file index for efficient access
        print("Building file index for Gray-Scott dataset...")
        start_time = time.time()
        self.total_trajectories = self._build_file_index()
        index_time = time.time() - start_time
        print(f"Dataset length calculation took {index_time:.2f}s for {self.total_trajectories} trajectories")
        
        # Open file handles if requested
        self.file_handles = {}
        if keep_file_open:
            for file_path in self.filenames:
                if os.path.exists(file_path):
                    self.file_handles[file_path] = h5py.File(file_path, 'r')
    
    def _build_file_index(self):
        """Pre-compute file offsets for efficient __len__ and __getitem__"""
        self.file_offsets = []
        total_trajectories = 0
        self.n_x = None
        self.n_y = None
        self.n_t = None
        
        for file_path in self.filenames:
            if not os.path.exists(file_path):
                print(f"Warning: HDF5 file not found: {file_path}")
                continue
                
            try:
                with h5py.File(file_path, 'r') as f:
                    # Try to find the appropriate group
                    group_to_use = self.group_name
                    if self.group_name not in f:
                        # If requested group doesn't exist, try 'train' as fallback
                        if 'train' in f:
                            group_to_use = 'train'
                            print(f"Warning: Group '{self.group_name}' not found in {file_path}, using 'train' instead")
                        else:
                            print(f"Warning: Neither '{self.group_name}' nor 'train' group found in {file_path}")
                            print(f"Available groups: {list(f.keys())}")
                            continue
                    
                    # Get dataset info for this file
                    info = get_dataset_info(file_path, group_to_use)
                    n_trajectories = info['n_trajectories']
                    
                    # Store spatial dimensions (should be same for all files)
                    if self.n_x is None:
                        self.n_x = info['n_spatial_x']
                        self.n_y = info['n_spatial_y']
                        self.n_t = info['n_timesteps']
                    else:
                        # Verify dimensions match across files
                        if (self.n_x != info['n_spatial_x'] or 
                            self.n_y != info['n_spatial_y'] or 
                            self.n_t != info['n_timesteps']):
                            print(f"Warning: Dimensions mismatch in {file_path}. Skipping.")
                            continue
                    
                    self.file_offsets.append((file_path, total_trajectories, n_trajectories, group_to_use))
                    total_trajectories += n_trajectories
                    print(f"Added {n_trajectories} trajectories from {file_path} (group: {group_to_use})")
                    
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue
                
        if total_trajectories == 0:
            raise ValueError("No valid trajectories found in any HDF5 files!")
            
        return total_trajectories
    
    def _get_file_and_local_idx(self, idx):
        """Convert global index to file path, local index, and group name"""
        for entry in self.file_offsets:
            # Handle both old format (3-tuple) and new format (4-tuple)
            if len(entry) == 3:
                file_path, offset, n_trajectories = entry
                group_name = self.group_name  # Use default group name
            else:
                file_path, offset, n_trajectories, group_name = entry
            
            if idx < offset + n_trajectories:
                local_idx = idx - offset
                return file_path, local_idx, group_name
        raise IndexError(f"Index {idx} out of range for dataset size {self.total_trajectories}")
    
    def __len__(self):
        return self.total_trajectories
    
    def __getitem__(self, idx: Union[int, List[int]]) -> Dict:
        """
        Get trajectory(ies) by index.
        
        Args:
            idx: Single index or list of indices
            
        Returns:
            Dictionary with trajectory data
        """
        single_item = isinstance(idx, int)
        
        if single_item:
            # Single index case
            file_path, local_idx, group_name = self._get_file_and_local_idx(idx)
            
            # Get file handle or open temporarily
            if self.keep_file_open and file_path in self.file_handles:
                f = self.file_handles[file_path]
                close_file = False
            else:
                f = h5py.File(file_path, 'r')
                close_file = True
            
            try:
                group = f[group_name]
                a_data = group['trajectory_a'][local_idx]
                b_data = group['trajectory_b'][local_idx]
                f_values = group['f'][local_idx] if 'f' in group else None
                k_values = group['k'][local_idx] if 'k' in group else None
                init_cond = group['initial_conditions'][local_idx] if 'initial_conditions' in group else None
            finally:
                if close_file:
                    f.close()
        else:
            # Multiple indices case - need to handle indices from different files
            all_a_data = []
            all_b_data = []
            all_f_values = []
            all_k_values = []
            all_init_cond = []
            
            for i in idx:
                file_path, local_idx, group_name = self._get_file_and_local_idx(i)
                
                if self.keep_file_open and file_path in self.file_handles:
                    f = self.file_handles[file_path]
                    close_file = False
                else:
                    f = h5py.File(file_path, 'r')
                    close_file = True
                
                try:
                    group = f[group_name]
                    all_a_data.append(group['trajectory_a'][local_idx])
                    all_b_data.append(group['trajectory_b'][local_idx])
                    all_f_values.append(group['f'][local_idx] if 'f' in group else None)
                    all_k_values.append(group['k'][local_idx] if 'k' in group else None)
                    all_init_cond.append(group['initial_conditions'][local_idx] if 'initial_conditions' in group else None)
                finally:
                    if close_file:
                        f.close()
            
            # Stack the data
            a_data = np.stack(all_a_data)
            b_data = np.stack(all_b_data)
            f_values = np.stack(all_f_values) if all_f_values[0] is not None else None
            k_values = np.stack(all_k_values) if all_k_values[0] is not None else None
            init_cond = np.stack(all_init_cond) if all_init_cond[0] is not None else None
        
        # Reshape if requested
        if self.reshape_to_spatial:
            # Handle different data shapes
            if len(a_data.shape) == 2:
                # Single trajectory: (n_timesteps, n_spatial) -> (n_timesteps, n_x, n_y)
                n_t, n_spatial = a_data.shape
                a_data = a_data.reshape(n_t, self.n_x, self.n_y)
                b_data = b_data.reshape(n_t, self.n_x, self.n_y)
                
                if init_cond is not None:
                    if len(init_cond.shape) == 2:
                        n_spatial, n_channels = init_cond.shape
                        init_cond = init_cond.reshape(self.n_x, self.n_y, n_channels)
            elif len(a_data.shape) == 3:
                if single_item:
                    # Single item with batch dimension: (1, n_timesteps, n_spatial) -> squeeze and reshape
                    a_data = a_data.squeeze(0)
                    b_data = b_data.squeeze(0)
                    if init_cond is not None:
                        init_cond = init_cond.squeeze(0)
                    
                    n_t, n_spatial = a_data.shape
                    a_data = a_data.reshape(n_t, self.n_x, self.n_y)
                    b_data = b_data.reshape(n_t, self.n_x, self.n_y)
                    
                    if init_cond is not None:
                        n_spatial, n_channels = init_cond.shape
                        init_cond = init_cond.reshape(self.n_x, self.n_y, n_channels)
                else:
                    # Multiple trajectories: (n_batch, n_timesteps, n_spatial) -> (n_batch, n_timesteps, n_x, n_y)
                    n_batch, n_t, n_spatial = a_data.shape
                    a_data = a_data.reshape(n_batch, n_t, self.n_x, self.n_y)
                    b_data = b_data.reshape(n_batch, n_t, self.n_x, self.n_y)
                    
                    if init_cond is not None:
                        n_batch, n_spatial, n_channels = init_cond.shape
                        init_cond = init_cond.reshape(n_batch, self.n_x, self.n_y, n_channels)
        
        # Remove batch dimension for single items (parameters only)
        if single_item:
            if f_values is not None:
                f_values = f_values[0] if hasattr(f_values, '__len__') else f_values
                k_values = k_values[0] if hasattr(k_values, '__len__') else k_values
        
        return {
            'a': a_data,
            'b': b_data,
            'f': f_values,
            'k': k_values,
            'initial_conditions': init_cond
        }
    
    def get_batch(self, indices: List[int]) -> Dict:
        """Get multiple trajectories efficiently."""
        return self.__getitem__(indices)
    
    def close(self):
        """Close all file handles if open."""
        for file_path, file_handle in self.file_handles.items():
            if file_handle is not None:
                file_handle.close()
        self.file_handles = {}
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.close()


class GrayScottDatasetWrapper(Dataset):
    """Efficient wrapper using GrayScottHDF5Dataset with optimized file access."""
    
    def __init__(self, hdf5_files, split='train', input_frames=16, output_frames=2, 
                 sub_x=1, sub_t=1):
        logging.info(f"Initializing efficient dataset from {hdf5_files}")
        self.hdf5_files = hdf5_files if isinstance(hdf5_files, list) else [hdf5_files]
        self.split = split
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.sub_x = sub_x
        self.sub_t = sub_t
        
        # Use the efficient GrayScottHDF5Dataset
        self.dataset = GrayScottHDF5Dataset(
            self.hdf5_files, 
            group_name=split,
            reshape_to_spatial=True,
            keep_file_open=True
        )
        
        self.n_samples = len(self.dataset)
        
        # Get time information from first file metadata
        first_file = self.hdf5_files[0] if self.hdf5_files else None
        if first_file and os.path.exists(first_file):
            with h5py.File(first_file, 'r') as f:
                if split in f:
                    self.n_timesteps = f[split].attrs.get('n_timesteps', 50)
                    logging.info(f"Found {self.n_samples} trajectories with {self.n_timesteps} timesteps each")
                else:
                    self.n_timesteps = 50
                    logging.info("Single trajectory file detected")
        else:
            self.n_timesteps = 50
            logging.info("Using default timestep count")
        
        # Generate time points based on the number of time steps
        self.time_points = torch.tensor(np.linspace(0, 50.0, self.n_timesteps), dtype=torch.float32)
        
        logging.info(f"Efficient dataset initialized with {self.n_samples} trajectories from {len(self.hdf5_files)} file(s)")
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        # Load using the efficient dataset
        data = self.dataset[idx]
        
        # Extract a and b channels and convert to tensors
        # Skip first timestep: data shape is (n_timesteps, n_x, n_y)
        a_full = data['a'][1:]  # Skip first timestep
        b_full = data['b'][1:]  # Skip first timestep
        
        # Apply temporal subsampling if specified
        if self.sub_t > 1:
            a_full = a_full[::self.sub_t]
            b_full = b_full[::self.sub_t]
        
        # Apply spatial subsampling if specified
        if self.sub_x > 1:
            a_full = a_full[:, ::self.sub_x, ::self.sub_x]
            b_full = b_full[:, ::self.sub_x, ::self.sub_x]
        
        # Convert to tensors
        a_tensor = torch.from_numpy(a_full).float()
        b_tensor = torch.from_numpy(b_full).float()
        
        # Sample temporal window randomly
        total_frames_needed = self.input_frames + self.output_frames
        max_start = a_tensor.shape[0] - total_frames_needed
        
        #print(f"DEBUG: idx={idx}, a_tensor.shape={a_tensor.shape}, total_frames_needed={total_frames_needed}, max_start={max_start}")
        
        if max_start <= 0:
            # If not enough frames, use what we have
            start_idx = 0
            available_frames = a_tensor.shape[0]
            actual_input_frames = min(self.input_frames, available_frames // 2)
            actual_output_frames = available_frames - actual_input_frames
            #print(f"DEBUG: Not enough frames - available={available_frames}, input={actual_input_frames}, output={actual_output_frames}")
        else:
            # Random temporal sampling
            start_idx = np.random.randint(0, max_start + 1)
            actual_input_frames = self.input_frames
            actual_output_frames = self.output_frames
            #print(f"DEBUG: Enough frames - start_idx={start_idx}, input={actual_input_frames}, output={actual_output_frames}")
        
        # Extract input and output sequences
        input_end = start_idx + actual_input_frames
        #output_end = input_end + actual_output_frames
        
        a_input = a_tensor[start_idx:input_end]
        b_input = b_tensor[start_idx:input_end]

        # WARNING: this version does not assumes input and then output (output is after start_idx)
        max_start = a_tensor.shape[0] - actual_output_frames
        output_start = np.random.randint(start_idx, max_start + 1)
        output_end = output_start + actual_output_frames

        a_output = a_tensor[output_start:output_end]
        b_output = b_tensor[output_start:output_end]
        
        # Stack channels: (n_timesteps, n_channels, n_x, n_y)
        input_trajectory = torch.stack([a_input, b_input], dim=1)
        output_trajectory = torch.stack([a_output, b_output], dim=1)
        
        #print(f"DEBUG: Final shapes - input: {input_trajectory.shape}, output: {output_trajectory.shape}")
        
        # For compatibility with existing training code that expects 'trajectory'
        # we'll concatenate input and output
        #full_trajectory = torch.cat([input_trajectory, output_trajectory], dim=0)
        
        # Convert parameters to tensors
        f_val = torch.tensor(data['f'], dtype=torch.float32) if data['f'] is not None else torch.tensor(0.029, dtype=torch.float32)
        k_val = torch.tensor(data['k'], dtype=torch.float32) if data['k'] is not None else torch.tensor(0.057, dtype=torch.float32)
        
        return {
            #'trajectory': full_trajectory,
            'input': input_trajectory,
            'output': output_trajectory,
            'f': f_val,
            'k': k_val,
            #'time_points': self.time_points[start_idx:output_end]
        }
    
    def close(self):
        """Close the underlying dataset file handle."""
        if hasattr(self.dataset, 'close'):
            self.dataset.close()



class DISCOLitModule(L.LightningModule):
    def __init__(self, model_cfg, training_cfg):
        super().__init__()
        self.save_hyperparameters()
        self.loss_fn = RelativeL2()
        self.model = DISCOHouse(**model_cfg)
        for k, v in training_cfg.items():
            setattr(self, k, v)
        self.automatic_optimization = False  # Enable manual optimization
        
        # In-context learning parameter (default to False)
        self.in_context = getattr(self, 'in_context', False)
        self.in_context_progressive = getattr(self, 'in_context_progressive', False)
        
        if self.in_context_progressive:
            print("Progressive in-context learning enabled: probability will increase from ~0 to ~1 during training")
        else:
            print(f"In-context learning: {'enabled' if self.in_context else 'disabled'}")
        
        # Progressive step refinement parameters
        self.progressive_steps = getattr(self, 'progressive_steps', False)
        if self.progressive_steps:
            # Default schedule: 80% at 1 step, then 5% each at 2,4,8,16 steps
            self.step_schedule = getattr(self, 'step_schedule', [1, 2, 4, 8, 16])
            self.step_percentages = getattr(self, 'step_percentages', [0.8, 0.05, 0.05, 0.05, 0.05])
            
            # Calculate step boundaries
            self.step_boundaries = []
            cumulative = 0
            for pct in self.step_percentages:
                cumulative += pct
                self.step_boundaries.append(int(cumulative * self.max_steps))
            
            print(f"Progressive steps enabled:")
            for i, (steps, boundary) in enumerate(zip(self.step_schedule, self.step_boundaries)):
                start = self.step_boundaries[i-1] if i > 0 else 0
                print(f"  Steps {start}-{boundary}: {steps} integration steps")
        
        # Track current number of integration steps
        self.current_integration_steps = 1

    def forward(self, x, y):
        y_pred, _ = self.model(x, y)
        return y_pred
    
    def _update_integration_steps(self, global_step):
        """Update the current number of integration steps based on training progress"""
        if not self.progressive_steps:
            return
        
        # Find which stage we're in
        for i, boundary in enumerate(self.step_boundaries):
            if global_step < boundary:
                new_steps = self.step_schedule[i]
                if new_steps != self.current_integration_steps:
                    self.current_integration_steps = new_steps
                    # Update the model's max_steps parameter
                    self.model.max_steps = new_steps
                    print(f"Step {global_step}: Updated integration steps to {new_steps}")
                break

    def training_step(self, batch, batch_idx):
        # Update integration steps based on training progress
        self._update_integration_steps(self.global_step)
        
        # Use the separate input and output from the dataset
        input = batch['input']  # Shape: (batch, input_frames, channels, height, width)
        output = batch['output']  # Shape: (batch, output_frames, channels, height, width)
        
        # For teacher forcing, we need the output sequence
        target = output
        
        # Determine whether to use in-context learning for this step
        use_in_context = self.in_context
        if self.in_context_progressive:
            # Linear schedule from ~0 to ~1 over the course of training
            # Start with 0.01 probability and reach 0.99 at the end
            progress = self.global_step / self.max_steps
            in_context_prob = 0.01 + 0.98 * progress
            use_in_context = torch.rand(1).item() < in_context_prob
            
            # Log the probability every 100 steps
            if batch_idx % 100 == 0:
                self.log('in_context_prob', in_context_prob, on_step=True, prog_bar=True)

        # Add Gaussian noise to first timestamp during training if noise_level is set
        #if hasattr(self, 'noise_level') and self.noise_level is not None:
        #    target[:, 0, ...] += torch.randn_like(target[:, 0, ...]) * self.noise_level

        # Reshape for model: need (batch_time, 1, channels, spatial)
        # For teacher forcing, we need target[:-1] as input and target[1:] as output
        target_inp = rearrange(target[:, :-1], 'b t c h w -> (b t) 1 c h w')
        target_out = rearrange(target[:, 1:], 'b t c h w -> (b t) 1 c h w')

        #input, target = batch
        state_labels = torch.tensor([0, 1], device=input.device)
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()
        optimizer.zero_grad()
        y_pred, metadata = self.model(input, state_labels, y=target_inp, n_future_steps=1)
        loss = self.loss_fn(y_pred, target_out)
        self.manual_backward(loss)
        
        # Safe gradient monitoring and clipping
        if batch_idx % 100 == 0:  # Log every 100 steps
            try:
                grad_stats = compute_gradient_stats(self.model)
                self.log_dict(grad_stats, on_step=True, logger=True)
            except Exception:
                # Fail silently to not break training
                pass
        
        # Add gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('integration_steps', self.current_integration_steps, on_step=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input = batch['input']
        target = batch['output']
        state_labels = torch.tensor([0, 1], device=input.device)
        target_inp = rearrange(target[:, :-1], 'b t c h w -> (b t) 1 c h w')
        target_out = rearrange(target[:, 1:], 'b t c h w-> (b t) 1 c h w')

        y_pred, metadata = self.model(input, state_labels, y=target_inp, n_future_steps=1)

        loss = self.loss_fn(y_pred, target_out)

        loss = self.loss_fn(y_pred, target_out)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        # Use warmup steps if specified, otherwise default to 5% of max_steps
        warmup_steps = getattr(self, 'warmup_steps', int(0.05 * self.max_steps))
        warmup_steps = int(0.05 * self.max_steps) if warmup_steps is None else warmup_steps
        min_lr_ratio = getattr(self, 'min_lr_ratio', 0.0)
        
        scheduler = CosineWithWarmupScheduler(
            optimizer, 
            warmup_steps=warmup_steps, 
            total_steps=self.max_steps, 
            min_lr_ratio=min_lr_ratio
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def on_train_epoch_start(self):
        pass  # No longer needed, handled by callback


def get_run_name(cfg: DictConfig) -> str:
    """Create a descriptive run name for DISCO training with timestamp for uniqueness."""
    # Get dataset name
    dataset_name = cfg.data.dataset_name
    
    # Get model parameters
    model_params = [
        f"solver{cfg.model.solver}",
        f"adj{cfg.model.use_adjoint}",
        f"h{cfg.model.hidden_dim}",
        f"t{cfg.model.theta_dim}",
        f"steps{cfg.model.max_steps}",
        f"init{cfg.model.principled_initialization}",
    ]
    
    # Add training parameters
    train_params = [
        f"bs{cfg.training.batch_size}",
        f"lr{cfg.training.lr}",
        f"hdf5",  # Indicate HDF5 data loading
    ]
    
    # Add noise level if specified
    if hasattr(cfg.training, 'noise_level') and cfg.training.noise_level is not None:
        train_params.append(f"noise{cfg.training.noise_level}")
    
    # Add data parameters
    data_params = [
        f"inframes{cfg.data.n_input_frames}",
        f"outframes{cfg.data.n_output_frames}",
        f"subx{cfg.data.sub_x}",
        f"subt{cfg.data.sub_t}",
    ]
    
    # Add checkpoint suffix if resuming from checkpoint
    suffix = ""
    if hasattr(cfg.training, 'checkpoint_path') and cfg.training.checkpoint_path:
        suffix = "_resumed"
    
    # Add timestamp at the end for uniqueness
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Combine all parts with timestamp at the end
    return f"DISCO_{dataset_name}_{'_'.join(model_params)}_{'_'.join(train_params)}_{'_'.join(data_params)}{suffix}_{timestamp}"

@hydra.main(config_path="../configs", config_name="config_rd")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    run_name = get_run_name(cfg)
    
    # Print the unique run name for clarity
    print(f"\n{'='*60}")
    print(f"Run name (unique): {run_name}")
    print(f"{'='*60}\n")
    
    # Create output directory
    output_dir = cfg.data.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Get HDF5 files for different splits
    def get_hdf5_files(cfg, split_name):
        """Get HDF5 files for a specific split (train/val/test)"""
        print(f"DEBUG: Getting HDF5 files for split: {split_name}")
        
        # Try to get split-specific files first
        split_key = f'{split_name}_hdf5_files'
        print(f"DEBUG: Looking for config key: {split_key}")
        
        if hasattr(cfg.data, split_key):
            files = getattr(cfg.data, split_key)
            print(f"DEBUG: Found {split_key} = {files} (type: {type(files)})")
            if files:
                result = list(files) if not isinstance(files, str) else [files]
                print(f"DEBUG: Returning split-specific files: {result}")
                return result
        
        # Fall back to general hdf5_files for backward compatibility
        if hasattr(cfg.data, 'hdf5_files'):
            files = cfg.data.hdf5_files
            print(f"DEBUG: Found general hdf5_files = {files} (type: {type(files)})")
            result = list(files) if not isinstance(files, str) else [files]
            print(f"DEBUG: Returning general files: {result}")
            return result
        
        # Default fallback
        default_files = ["./datasets/mp-neural/E_EULER_train_1024.h5"]
        print(f"Warning: No HDF5 files specified for {split_name}, using default: {default_files}")
        return default_files

    train_hdf5_files = get_hdf5_files(cfg, 'train')
    val_hdf5_files = get_hdf5_files(cfg, 'val')
    test_hdf5_files = get_hdf5_files(cfg, 'test')
    
    print(f"Train HDF5 files: {train_hdf5_files}")
    print(f"Val HDF5 files: {val_hdf5_files}")
    print(f"Test HDF5 files: {test_hdf5_files}")
    
    wandb_logger = WandbLogger(
        project=cfg.training.project,
        config=OmegaConf.to_container(cfg, resolve=True),
        name=run_name
    )
    
    # Configure enhanced wandb logging for gradients
    wandb_logger.experiment.define_metric("grad_norm/*", step_metric="trainer/global_step")
    wandb_logger.experiment.define_metric("grad_mean/*", step_metric="trainer/global_step")
    wandb_logger.experiment.define_metric("grad_std/*", step_metric="trainer/global_step")

    # Create separate HDF5 datasets for train/val/test using GrayScottDatasetWrapper
    # GrayScottDatasetWrapper now supports multiple files
    train_ds = GrayScottDatasetWrapper(
        hdf5_files=train_hdf5_files,
        split='train',
        input_frames=getattr(cfg.data, 'n_input_frames', 16),
        output_frames=getattr(cfg.data, 'n_output_frames', 16),
        sub_x=getattr(cfg.data, 'sub_x', 1),
        sub_t=getattr(cfg.data, 'sub_t', 1)
    )
    
    val_ds = GrayScottDatasetWrapper(
        hdf5_files=val_hdf5_files,
        split='val',
        input_frames=getattr(cfg.data, 'n_input_frames', 16),
        output_frames=getattr(cfg.data, 'n_output_frames', 16),
        sub_x=getattr(cfg.data, 'sub_x', 1),
        sub_t=getattr(cfg.data, 'sub_t', 1)
    )

    train_loader = DataLoader(
        train_ds, 
        batch_size=cfg.training.batch_size, 
        shuffle=True,
        num_workers=getattr(cfg.training, 'num_workers', 4), 
        prefetch_factor=getattr(cfg.training, 'prefetch_factor', 2), 
        persistent_workers=True, 
        drop_last=True,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=cfg.training.batch_size, 
        shuffle=False,
        num_workers=getattr(cfg.training, 'num_workers', 4), 
        prefetch_factor=getattr(cfg.training, 'prefetch_factor', 2), 
        pin_memory=True
    )

    model = DISCOLitModule(cfg.model, cfg.training)

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(output_dir, run_name),
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        filename="best-checkpoint",
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = L.Trainer(
        max_steps=cfg.training.max_steps,
        logger=wandb_logger,
        accelerator='gpu',
        devices=1 if torch.cuda.is_available() else None,
        log_every_n_steps=100,
        check_val_every_n_epoch=5,
        callbacks=[checkpoint_callback, lr_monitor],
    )
    
    # Print dataset loading statistics
    print(f"\nDataset Statistics:")
    print(f"Training samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")
    
    # Check if we should resume from a checkpoint
    checkpoint_path = None
    if hasattr(cfg.training, 'checkpoint_path') and cfg.training.checkpoint_path:
        checkpoint_path = cfg.training.checkpoint_path
        if os.path.exists(checkpoint_path):
            print(f"Resuming training from checkpoint: {checkpoint_path}")
        else:
            print(f"Warning: Checkpoint path does not exist: {checkpoint_path}")
            checkpoint_path = None
    
    # Start training
    trainer.fit(model, train_loader, val_loader, ckpt_path=checkpoint_path)
    
    # Dataset loading is complete
    
    trainer.save_checkpoint(os.path.join(output_dir, run_name, "final.ckpt"))
    wandb.finish()

if __name__ == "__main__":
    main()
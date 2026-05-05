"""Train DISCO on aggregated combined-equation data. Config: configs/config_hdf5.yaml."""
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
import torch.nn.functional as F

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


class HDF5TemporalDataset(Dataset):
    """Dataset for loading pre-computed trajectory data from HDF5 files"""
    
    def __init__(self, hdf5_files, input_frames=16, output_frames=16, 
                 sub_x=1, sub_t=1, split='train', trajectories_per_environment=16):
        """
        Args:
            hdf5_files: List of HDF5 file paths to load data from
            input_frames: Number of input time frames
            output_frames: Number of output time frames  
            sub_x: Spatial subsampling factor
            sub_t: Temporal subsampling factor
            split: Dataset split ('train', 'val', 'test')
            trajectories_per_environment: Number of trajectories per environment (default: 16)
        """
        self.hdf5_files = hdf5_files if isinstance(hdf5_files, list) else [hdf5_files]
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.sub_x = sub_x
        self.sub_t = sub_t
        self.split = split
        self.trajectories_per_environment = trajectories_per_environment
        
        # Build file index for efficient access
        print("Building file index...")
        start_time = time.time()
        self.total_samples = self._build_file_index()
        index_time = time.time() - start_time
        print(f"Dataset length calculation took {index_time:.2f}s for {self.total_samples} samples")
        
        # Track loading times for performance assessment
        self.loading_times = []
        
    def _build_file_index(self):
        """Pre-compute file offsets for efficient __len__ and __getitem__"""
        self.file_offsets = []
        total_samples = 0
        
        for file_path in self.hdf5_files:
            if not os.path.exists(file_path):
                print(f"Warning: HDF5 file not found: {file_path}")
                continue
                
            try:
                with h5py.File(file_path, 'r') as f:
                    # Try different group names based on split
                    data_group = None
                    dataset_path = None
                    
                    # Check for split-specific groups first, then fall back to 'train'
                    possible_groups = [self.split, 'train', 'valid', 'test']
                    for group_name in possible_groups:
                        if group_name in f and 'pde_250-256' in f[group_name]:
                            data_group = group_name
                            dataset_path = f'{group_name}/pde_250-256'
                            break
                    
                    if data_group is None:
                        print(f"Warning: No valid dataset structure found in {file_path}. Checked groups: {possible_groups}")
                        continue
                        
                    n_samples = f[dataset_path].shape[0]
                    n_timesteps = f[dataset_path].shape[1]
                    
                    # Verify we have enough timesteps for input + output frames
                    min_timesteps_needed = (self.input_frames + self.output_frames) * self.sub_t
                    if n_timesteps < min_timesteps_needed:
                        print(f"Warning: Not enough timesteps in {file_path}. "
                              f"Need {min_timesteps_needed}, got {n_timesteps}")
                        continue
                    
                    self.file_offsets.append((file_path, total_samples, n_samples, dataset_path))
                    total_samples += n_samples
                    print(f"Added {n_samples} samples from {file_path} (using {dataset_path})")
                    
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue
                
        if total_samples == 0:
            raise ValueError("No valid samples found in any HDF5 files!")
            
        return total_samples
    
    def _get_file_and_local_idx(self, idx):
        """Convert global index to file path and local index"""
        for file_path, offset, n_samples, dataset_path in self.file_offsets:
            if idx < offset + n_samples:
                local_idx = idx - offset
                return file_path, local_idx, dataset_path
        raise IndexError(f"Index {idx} out of range for dataset size {self.total_samples}")
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        start_time = time.time()
        # Find which file and local index
        file_path, local_idx, dataset_path = self._get_file_and_local_idx(idx)
        
        # Calculate environment index (each environment has trajectories_per_environment trajectories)
        environment_idx = idx // self.trajectories_per_environment
        
        min_target_index = (local_idx//self.trajectories_per_environment)*self.trajectories_per_environment
        max_target_index = min_target_index + self.trajectories_per_environment - 1
        target_local_idx = random.randint(min_target_index, max_target_index)
        
        try:
            with h5py.File(file_path, 'r') as f:
                # Get the group containing the data
                group_name = dataset_path.split('/')[0]
                
                # Load trajectory data - shape: (n_timesteps, n_spatial)
                trajectory = f[dataset_path][local_idx].copy()[::self.sub_t]
                target_trajectory = f[dataset_path][target_local_idx].copy()[::self.sub_t]
                
                # Load PDE parameters (alpha, beta, gamma) for this sample
                alpha = f[group_name]['alpha'][local_idx]
                beta = f[group_name]['beta'][local_idx]
                gamma = f[group_name]['gamma'][local_idx]
                
                # Sample temporal window randomly
                total_frames_needed = self.input_frames + self.output_frames
                max_start = trajectory.shape[0] - total_frames_needed
                if max_start <= 0:
                    # If not enough frames, use what we have
                    start_idx = 0
                    available_frames = trajectory.shape[0] 
                    actual_input_frames = min(self.input_frames, available_frames // 2)
                    actual_output_frames = available_frames - actual_input_frames
                else:
                    start_idx = np.random.randint(0, max_start + 1)
                    actual_input_frames = self.input_frames
                    actual_output_frames = self.output_frames
                
                # Apply temporal subsampling and extract sequences
                start_t = start_idx 
                input_end_t = start_t + actual_input_frames 
                #output_end_t = input_end_t + actual_output_frames
                
                input_seq = trajectory[start_t:input_end_t, ::self.sub_x]

                # WARNING big change
                output_start_t = np.random.randint(start_idx, trajectory.shape[0] - actual_output_frames + 1)
                output_seq = trajectory[output_start_t:output_start_t+actual_output_frames, ::self.sub_x]

                # for target, the start is after the frames from the context 
                start_t = np.random.randint(start_idx, max_start + 1)
                input_end_t = start_t + actual_input_frames 
                output_end_t = input_end_t + actual_output_frames
                
                target_input_seq = target_trajectory[start_t:input_end_t, ::self.sub_x]
                target_output_seq = target_trajectory[input_end_t:output_end_t, ::self.sub_x]

                target_input_tensor = torch.from_numpy(target_input_seq).unsqueeze(-2).float()
                target_output_tensor = torch.from_numpy(target_output_seq).unsqueeze(-2).float()
                
                # Add channel dimension and convert to torch tensors
                # Expected format: (time, channels, spatial)
                input_tensor = torch.from_numpy(input_seq).unsqueeze(-2).float()
                output_tensor = torch.from_numpy(output_seq).unsqueeze(-2).float()
                
                # Track loading time
                loading_time = time.time() - start_time
                if len(self.loading_times) < 1000:  # Collect first 1000 samples
                    self.loading_times.append(loading_time)
                
                return {
                    'input': input_tensor, 
                    'output': output_tensor,
                    'target_input': target_input_tensor,
                    'target_output': target_output_tensor,
                    'alpha': float(alpha),
                    'beta': float(beta),
                    'gamma': float(gamma),
                    'environment_idx': environment_idx
                }
                
        except Exception as e:
            print(f"Error loading sample {idx} from {file_path}: {e}")
            # Return dummy data to avoid training crash
            dummy_input = torch.zeros(self.input_frames, 1, 256 // self.sub_x)
            dummy_output = torch.zeros(self.output_frames, 1, 256 // self.sub_x)
            return {
                'input': dummy_input, 
                'output': dummy_output,
                'target_input': dummy_input,
                'target_output': dummy_output,
                'alpha': 0.0,
                'beta': 0.0,
                'gamma': 0.0,
                'environment_idx': environment_idx
            }
    
    def get_loading_stats(self):
        """Return loading performance statistics"""
        if not self.loading_times:
            return {}
        
        return {
            'avg_loading_time': np.mean(self.loading_times),
            'min_loading_time': np.min(self.loading_times),
            'max_loading_time': np.max(self.loading_times),
            'samples_per_second': 1.0 / np.mean(self.loading_times),
            'total_samples_timed': len(self.loading_times)
        }


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
        
        # Initialize codebook for variance reduction
        #self.num_environments = 32 * 3#512 * 3  # 512*3 training environments
        # for testing
        self.num_environments = 128 * 3#512 * 3  # 512*3 training environments
        self.codebook_dim = getattr(model_cfg, 'theta_dim', 256)  # Use theta_dim from model config
        self.codebook_momentum = getattr(training_cfg, 'codebook_momentum', 0.99)  # EMA momentum
        
        # Initialize codebook with normal distribution
        self.register_buffer('codebook', torch.randn(self.num_environments, self.codebook_dim) * 0.02)
        self.register_buffer('codebook_usage', torch.zeros(self.num_environments))
        
        print(f"Initialized codebook: {self.num_environments} entries of dimension {self.codebook_dim}")
        print(f"Codebook EMA momentum: {self.codebook_momentum}")

    def forward(self, x, y):
        y_pred, _ = self.model(x, y)
        return y_pred
    
    def codebook_lookup_with_straight_through(self, theta_latent, environment_idx):
        """
        Perform index-based codebook lookup with straight-through estimator.
        
        Args:
            theta_latent: Encoder output, shape (B, D)
            environment_idx: Environment indices, shape (B,)
        
        Returns:
            quantized: Quantized embeddings from codebook
            theta_latent: Original embeddings (for straight-through gradient)
        """
        # Get codebook embeddings for the given indices
        quantized = self.codebook[environment_idx]  # (B, D)
        
        # Straight-through estimator: forward pass uses quantized, backward uses original
        quantized = theta_latent + (quantized - theta_latent).detach()
        
        return quantized, theta_latent
    
    def update_codebook_ema(self, theta_latent, environment_idx):
        """
        Update codebook entries using exponential moving average.
        
        Args:
            theta_latent: Encoder output, shape (B, D)
            environment_idx: Environment indices, shape (B,)
        """
        with torch.no_grad():
            # Update usage counts
            self.codebook_usage.index_add_(0, environment_idx, torch.ones_like(environment_idx, dtype=torch.float))
            
            # Handle potential duplicates in the batch
            unique_idx = torch.unique(environment_idx)
            
            if len(unique_idx) == len(environment_idx):
                # No duplicates - fast path
                self.codebook[environment_idx] = self.codebook_momentum * self.codebook[environment_idx] + \
                                                (1 - self.codebook_momentum) * theta_latent
            else:
                # Duplicates present - need to average them
                for idx in unique_idx:
                    mask = environment_idx == idx
                    if mask.sum() == 1:
                        # Single occurrence
                        self.codebook[idx] = self.codebook_momentum * self.codebook[idx] + \
                                           (1 - self.codebook_momentum) * theta_latent[mask].squeeze(0)
                    else:
                        # Multiple occurrences - average them
                        avg_theta = theta_latent[mask].mean(dim=0)
                        self.codebook[idx] = self.codebook_momentum * self.codebook[idx] + \
                                           (1 - self.codebook_momentum) * avg_theta
    
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
        
        input = batch['input']
        environment_idx = batch['environment_idx']
        
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
        
        # Use target_output for in-context learning, output for regular training
        target = batch['target_output'] if use_in_context else batch['output']

        # Add Gaussian noise to first timestamp during training if noise_level is set
        #if hasattr(self, 'noise_level') and self.noise_level is not None:
        #    target[:, 0, ...] += torch.randn_like(target[:, 0, ...]) * self.noise_level

        target_inp = rearrange(target[:, :-1], 'b t c h -> (b t) 1 c h')
        target_out = rearrange(target[:, 1:], 'b t c h -> (b t) 1 c h')

        #input, target = batch
        state_labels = torch.tensor([0], device=input.device)
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()
        optimizer.zero_grad()
        
        # Encode input to get theta_latent
        theta_latent, encode_metadata = self.model.encode_theta_latent(input, state_labels)
        
        # Randomly decide whether to use codebook for each sample (50% probability)
        batch_size = theta_latent.shape[0]
        use_codebook_mask = torch.rand(batch_size, device=theta_latent.device) < 0.5 #0.5
        
        # Initialize theta_to_use with original theta_latent
        theta_to_use = theta_latent.clone()
        
        # Apply codebook lookup only for samples where use_codebook_mask is True
        if use_codebook_mask.any():
            # Get codebook embeddings for samples that should use it
            codebook_indices = environment_idx[use_codebook_mask]
            codebook_embeddings = self.codebook[codebook_indices]
            
            # Apply straight-through estimator for codebook samples
            theta_to_use[use_codebook_mask] = theta_latent[use_codebook_mask] + \
                                               (codebook_embeddings - theta_latent[use_codebook_mask]).detach()
            
            # Update codebook with EMA only for samples that used it
            self.update_codebook_ema(theta_latent[use_codebook_mask].detach(), codebook_indices)
        
        # Track codebook usage rate
        codebook_usage_rate = use_codebook_mask.float().mean().item()
        
        # Use custom forward pass with the chosen theta
        B, T = input.shape[:2]
        spatial = input.shape[3:]
        dim = len(spatial)
        
        # Decode theta from the chosen latent (either codebook or original)
        theta = self.model.decode_theta(theta_to_use, dim)


        print('theta', theta.shape)
        print('target_inp', target_inp[:, 0].shape)
        
        # Run the ODE solver with decoded theta
        y_pred, metadata = self.model.solve_ode(
            target_inp[:, 0],
            theta, 
            state_labels, 
            dim, 
            integration_time=self.model.default_integration_time,
            n_future_steps=1,
            metadata=encode_metadata,
        )
        
        loss = self.loss_fn(y_pred, target_out)
        
        # Compute codebook loss only for samples that used the codebook
        codebook_loss = torch.tensor(0.0, device=loss.device)
        if use_codebook_mask.any():
            # Calculate L2 distance between original and codebook embeddings
            codebook_embeddings = self.codebook[environment_idx[use_codebook_mask]]
            codebook_loss = F.mse_loss(theta_latent[use_codebook_mask], codebook_embeddings.detach())
        
        # Total loss
        #total_loss = loss + 0.25 * codebook_loss  # You can adjust the weight
        total_loss = loss + 0.5 * codebook_loss  # You can adjust the weight
        
        self.manual_backward(total_loss)
        
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
        
        # Log losses and metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('codebook_loss', codebook_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('total_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('integration_steps', self.current_integration_steps, on_step=True, prog_bar=True)
        self.log('codebook_usage_rate', codebook_usage_rate, on_step=True, on_epoch=True, prog_bar=True)
        
        # Log codebook statistics
        if batch_idx % 100 == 0:
            with torch.no_grad():
                # Compute codebook utilization
                used_codes = (self.codebook_usage > 0).sum().item()
                codebook_utilization = used_codes / self.num_environments
                
                # Compute average L2 distance only for samples that used codebook
                avg_distance = 0.0
                if use_codebook_mask.any():
                    codebook_embeddings = self.codebook[environment_idx[use_codebook_mask]]
                    avg_distance = F.mse_loss(theta_latent[use_codebook_mask], codebook_embeddings).item()
                
                self.log('codebook_utilization', codebook_utilization, on_step=True, prog_bar=True)
                self.log('codebook_avg_distance', avg_distance, on_step=True, prog_bar=True)
                self.log('codebook_used_codes', used_codes, on_step=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        input = batch['input']
        target = batch['output']
        state_labels = torch.tensor([0], device=input.device)
        target_inp = rearrange(target[:, :-1], 'b t c h -> (b t) 1 c h')
        target_out = rearrange(target[:, 1:], 'b t c h -> (b t) 1 c h')

        y_pred, metadata = self.model(input, state_labels, y=target_inp, n_future_steps=1)

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

@hydra.main(config_path="../configs", config_name="config_hdf5")
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
        # Try to get split-specific files first
        split_key = f'{split_name}_hdf5_files'
        if hasattr(cfg.data, split_key):
            files = getattr(cfg.data, split_key)
            if files:
                return list(files) if not isinstance(files, str) else [files]
        
        # Fall back to general hdf5_files for backward compatibility
        if hasattr(cfg.data, 'hdf5_files'):
            files = cfg.data.hdf5_files
            return list(files) if not isinstance(files, str) else [files]
        
        # Default fallback
        default_files = ["/mnt/home/lserrano/MP-Neural-PDE-Solvers/data/E_EULER_train_1024.h5"]
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

    # Create separate HDF5 datasets for train/val/test
    train_ds = HDF5TemporalDataset(
        hdf5_files=train_hdf5_files,
        input_frames=cfg.data.n_input_frames,
        output_frames=cfg.data.n_output_frames,
        sub_x=cfg.data.sub_x,
        sub_t=cfg.data.sub_t,
        split='train',
        trajectories_per_environment=getattr(cfg.data, 'trajectories_per_environment', 16)
    )
    
    val_ds = HDF5TemporalDataset(
        hdf5_files=val_hdf5_files,
        input_frames=cfg.data.n_input_frames,
        output_frames=cfg.data.n_output_frames,
        sub_x=cfg.data.sub_x,
        sub_t=cfg.data.sub_t,
        split='val',
        trajectories_per_environment=getattr(cfg.data, 'trajectories_per_environment', 16)
    )

    train_loader = DataLoader(
        train_ds, 
        batch_size=cfg.training.batch_size, 
        shuffle=True,
        num_workers=getattr(cfg.training, 'num_workers', 4), 
        prefetch_factor=getattr(cfg.training, 'prefetch_factor', 2), 
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
    
    # Print loading performance statistics
    train_stats = train_ds.get_loading_stats()
    if train_stats:
        print(f"\nData Loading Performance:")
        print(f"Average loading time: {train_stats['avg_loading_time']:.4f}s per sample")
        print(f"Samples per second: {train_stats['samples_per_second']:.1f}")
        print(f"Min/Max loading time: {train_stats['min_loading_time']:.4f}s / {train_stats['max_loading_time']:.4f}s")
    
    trainer.save_checkpoint(os.path.join(output_dir, run_name, "final.ckpt"))
    wandb.finish()

if __name__ == "__main__":
    main()
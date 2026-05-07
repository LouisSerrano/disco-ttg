"""Train DISCO on Euler/Navier-Stokes + diffusion aggregated data. Config: configs/config_euler.yaml."""
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import _LRScheduler
import wandb
from src.operators.disco import DISCOHouse
from src.utils.euler_ns_dataset import EulerDiffusionDatasetWrapper
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
        # num_environments should be set in training config based on dataset (1 Euler + m viscosities)
        self.num_environments = getattr(self, 'num_environments', 17)
        self.codebook_dim = getattr(model_cfg, 'theta_dim', 256)  # Use theta_dim from model config
        self.codebook_momentum = getattr(training_cfg, 'codebook_momentum', 0.99)  # EMA momentum
        self.codebook_prob = getattr(training_cfg, 'codebook_prob', 0.0)  # Probability of using codebook lookup
        
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
        
        # Use the separate input and output from the dataset
        input = batch['input']  # Shape: (batch, input_frames, channels, height, width)
        output = batch['output']  # Shape: (batch, output_frames, channels, height, width)
        environment_idx = batch['environment_idx']  # Environment indices for codebook lookup
        
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
        # Euler/Diffusion has 1 channel (vorticity), so state_labels = [0]
        state_labels = torch.tensor([0], device=input.device)
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()
        optimizer.zero_grad()
        
        # Encode input to get theta_latent
        theta_latent, encode_metadata = self.model.encode_theta_latent(input, state_labels)
        
        # Randomly decide whether to use codebook for each sample
        batch_size = theta_latent.shape[0]
        use_codebook_mask = torch.rand(batch_size, device=theta_latent.device) < self.codebook_prob
        
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
        #total_loss = loss + 1.0 * codebook_loss  # You can adjust the weight
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
        # Euler/Diffusion has 1 channel (vorticity), so state_labels = [0]
        state_labels = torch.tensor([0], device=input.device)
        target_inp = rearrange(target[:, :-1], 'b t c h w -> (b t) 1 c h w')
        target_out = rearrange(target[:, 1:], 'b t c h w-> (b t) 1 c h w')

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

    # Add T_limit if specified
    if hasattr(cfg.data, 'T_limit') and cfg.data.T_limit is not None:
        data_params.append(f"Tlim{cfg.data.T_limit}")
    
    # Add checkpoint suffix if resuming from checkpoint
    suffix = ""
    if hasattr(cfg.training, 'checkpoint_path') and cfg.training.checkpoint_path:
        suffix = "_resumed"

    # Add overfit test suffix
    if hasattr(cfg.training, 'overfit_test') and cfg.training.overfit_test:
        overfit_ntrain = getattr(cfg.training, 'overfit_ntrain', 1)
        suffix += f"_OVERFIT_n{overfit_ntrain}"

    # Add timestamp at the end for uniqueness
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Combine all parts with timestamp at the end
    return f"DISCO_{dataset_name}_{'_'.join(model_params)}_{'_'.join(train_params)}_{'_'.join(data_params)}{suffix}_{timestamp}"

@hydra.main(config_path="../configs", config_name="config_euler", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Overfit test mode: use only Euler (label 0) with limited trajectories
    overfit_test = getattr(cfg.training, 'overfit_test', False)
    overfit_ntrain = getattr(cfg.training, 'overfit_ntrain', 1)

    if overfit_test:
        print(f"\n{'='*60}")
        print(f"OVERFIT TEST MODE ENABLED")
        print(f"Using only Euler (label 0) with {overfit_ntrain} trajectory(ies)")
        print(f"{'='*60}\n")
        # Force filter_labels to only include Euler (label 0)
        cfg.data.filter_labels = [0]
        # Adjust batch size to not exceed number of samples (otherwise drop_last=True drops everything)
        if cfg.training.batch_size > overfit_ntrain:
            print(f"Adjusting batch_size from {cfg.training.batch_size} to {overfit_ntrain} for overfit test")
            cfg.training.batch_size = overfit_ntrain

    run_name = get_run_name(cfg)
    
    # Print the unique run name for clarity
    print(f"\n{'='*60}")
    print(f"Run name (unique): {run_name}")
    print(f"{'='*60}\n")
    
    # Create output directory
    output_dir = cfg.data.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    wandb_logger = WandbLogger(
        project=cfg.training.project,
        config=OmegaConf.to_container(cfg, resolve=True),
        name=run_name
    )
    
    # Configure enhanced wandb logging for gradients
    wandb_logger.experiment.define_metric("grad_norm/*", step_metric="trainer/global_step")
    wandb_logger.experiment.define_metric("grad_mean/*", step_metric="trainer/global_step")
    wandb_logger.experiment.define_metric("grad_std/*", step_metric="trainer/global_step")

    # Create separate HDF5 datasets for train/val/test using EulerDiffusionDatasetWrapper
    file_dir = getattr(cfg.data, 'file_dir', "./datasets/euler_ns_short/")
    num_gpus = getattr(cfg.data, 'num_gpus', 8)
    val_fraction = 0.1
    seed = 42

    # Set max_samples for overfit test mode
    train_max_samples = overfit_ntrain if overfit_test else None
    val_max_samples = overfit_ntrain if overfit_test else None  # Use same samples for val in overfit mode

    train_ds = EulerDiffusionDatasetWrapper(
        file_dir=file_dir,
        num_gpus=num_gpus,
        split='train',
        input_frames=getattr(cfg.data, 'n_input_frames', 16),
        output_frames=getattr(cfg.data, 'n_output_frames', 2),
        sub_x=getattr(cfg.data, 'sub_x', 1),
        sub_t=getattr(cfg.data, 'sub_t', 1),
        val_fraction=val_fraction if not overfit_test else 0.0,  # No val split in overfit mode
        seed=seed,
        vorticity_scale=getattr(cfg.data, 'vorticity_scale', 20.0),
        filter_labels=getattr(cfg.data, 'filter_labels', None),
        max_samples=train_max_samples,
        T_limit=getattr(cfg.data, 'T_limit', None),
    )

    # In overfit mode, use training data for validation too
    if overfit_test:
        val_ds = train_ds
    else:
        val_ds = EulerDiffusionDatasetWrapper(
            file_dir=file_dir,
            num_gpus=num_gpus,
            split='val',
            input_frames=getattr(cfg.data, 'n_input_frames', 16),
            output_frames=getattr(cfg.data, 'n_output_frames', 2),
            sub_x=getattr(cfg.data, 'sub_x', 1),
            sub_t=getattr(cfg.data, 'sub_t', 1),
            val_fraction=val_fraction,
            seed=seed,
            vorticity_scale=getattr(cfg.data, 'vorticity_scale', 20.0),
            filter_labels=getattr(cfg.data, 'filter_labels', None),
            T_limit=getattr(cfg.data, 'T_limit', None),
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

    # Set num_environments from dataset (1 Euler + m viscosities)
    # In overfit mode, we only have Euler (1 environment)
    if overfit_test:
        cfg.training.num_environments = 1
    else:
        cfg.training.num_environments = train_ds.num_environments

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


import os
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, IterableDataset
from torch.optim.lr_scheduler import _LRScheduler
import wandb
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from einops import rearrange
from src.utils.database import RelativeL2
from src.utils.advection_diffusion import Fractaloid, FractaloidPhase, AdvectionDiffusionExplicit
import random
import math

from geps.utils import fix_seed, count_parameters, init_weights
from geps.model.forecasters import *
from geps.losses import *
import h5py
import time
import logging
from train.train_rd import GrayScottHDF5Dataset


class GrayScottDatasetWrapper(Dataset):
    """Efficient wrapper using GrayScottHDF5Dataset with optimized file access."""
    
    def __init__(self, hdf5_files, split='train', input_frames=16, output_frames=2, 
                 sub_x=1, sub_t=1, trajectories_per_environment=512):
        logging.info(f"Initializing efficient dataset from {hdf5_files}")
        self.hdf5_files = hdf5_files if isinstance(hdf5_files, list) else [hdf5_files]
        self.split = split
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.sub_x = sub_x
        self.sub_t = sub_t
        self.trajectories_per_environment = trajectories_per_environment
        
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

        environment_idx = idx // self.trajectories_per_environment
        
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
        output_end = input_end + actual_output_frames
        
        a_input = a_tensor[start_idx:input_end]
        b_input = b_tensor[start_idx:input_end]

        # WARNING: this version does not assumes input and then output (output is after start_idx)

        a_output = a_tensor[input_end:output_end]
        b_output = b_tensor[input_end:output_end]
        
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
            'environment_idx': environment_idx
            #'time_points': self.time_points[start_idx:output_end]
        }
    
    def close(self):
        """Close the underlying dataset file handle."""
        if hasattr(self.dataset, 'close'):
            self.dataset.close()

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
        print(f"Warning: No HDF5 files specified for {split_name}")



class GEPSLightning(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.model = Forecaster(cfg.model.dataset_name,
                                cfg.model.state_c,
                                cfg.model.hidden_c,
                                cfg.model.code_c,
                                cfg.model.factor,
                                cfg.model.num_env,
                                cfg.model.is_complete,
                                cfg.model.type_augment,
                                cfg.model.method,
                                cfg.model.options)

        init_weights(self.model, init_config=cfg.model.init_type)
        self.myloss = RelativeL2()
        self.automatic_optimization = False

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        self.model.train()
        opt = self.optimizers()
        sch = self.lr_schedulers()

        input = batch['input']
        target = batch['output']
        env = batch['environment_idx']
        t = torch.tensor([0, self.cfg.model.default_integration_time])
        
        input = rearrange(input, "b t c h w ->  b c h w t")
        target = rearrange(target, "b t c h w ->  b c h w t")

        pred = self.model(input, t, env) # pred of shape B, C, H, W index_t+2 is excluded 
        pred = pred[..., 1:]

        #print('pred', pred.shape, target.shape)
        #print('pred', pred.shape)

        rel_loss = self.myloss(pred, target)

        opt.zero_grad()
        self.manual_backward(rel_loss)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
        opt.step()
        sch.step()
        
        self.log('train_l2', rel_loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        
        input = batch['input']
        target = batch['output']
        
        env = batch['environment_idx']
        t = torch.tensor([0, self.cfg.model.default_integration_time])
        #t = [0, self.cfg.model.default_integration_time]
        
        input = rearrange(input, "b t c h w ->  b c h w t")
        target = rearrange(target, "b t c h w ->  b c h w t")
        
        pred = self.model(input, t, env) # pred of shape B, C, H, W index_t+2 is excluded 
        pred = pred[..., 1:]

        #print('pred', pred.shape, target.shape)
        #print('pred', pred.shape)

        rel_loss = self.myloss(pred, target)

        self.log('val_l2', rel_loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        

    def configure_optimizers(self):
        lr = self.cfg.train.lr 
        #param_groups = [
            #{'params': self.model.tokenizer.encoder_layers.parameters(), 'lr': lr},
            #{'params': self.model.tokenizer.decoder_layers.parameters(), 'lr': lr},
            #{'params': self.model.tokenizer.quantizers.parameters(), 'lr': lr},#3e-3
        #]
        opt_gen = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4) # 1e-2 before
        scheduler_ae = torch.optim.lr_scheduler.CosineAnnealingLR(opt_gen, self.cfg.train.max_steps)

        return [opt_gen], [scheduler_ae] #{"optimizer": opt_gen, "lr_scheduler": scheduler_ae}, {"optimizer": None, "lr_scheduler": None}

    def get_last_layer(self):
        return self.model.get_last_layer()


@hydra.main(config_path="config/model", config_name="gray_scott.yaml")
def main(cfg):
    torch.set_default_dtype(torch.float32)
    dataset_name = cfg.data.dataset_name

    train_hdf5_files = get_hdf5_files(cfg, 'train')
    val_hdf5_files = get_hdf5_files(cfg, 'val')
    test_hdf5_files = get_hdf5_files(cfg, 'test')
    
    print(f"Train HDF5 files: {train_hdf5_files}")
    print(f"Val HDF5 files: {val_hdf5_files}")
    print(f"Test HDF5 files: {test_hdf5_files}")
    
    model = GEPSLightning(cfg).cuda()

    train_ds = GrayScottDatasetWrapper(
        hdf5_files=train_hdf5_files,
        split='train',
        input_frames=getattr(cfg.data, 'n_input_frames', 1),
        output_frames=getattr(cfg.data, 'n_output_frames', 1),
        sub_x=getattr(cfg.data, 'sub_x', 1),
        sub_t=getattr(cfg.data, 'sub_t', 1),
        trajectories_per_environment=512
        #trajectories_per_environment=getattr(cfg.data, 'trajectories_per_environment', 512)
    )
    
    val_ds = GrayScottDatasetWrapper(
        hdf5_files=val_hdf5_files,
        split='val',
        input_frames=getattr(cfg.data, 'n_input_frames', 1),
        output_frames=getattr(cfg.data, 'n_output_frames', 1),
        sub_x=getattr(cfg.data, 'sub_x', 1),
        sub_t=getattr(cfg.data, 'sub_t', 1),
        trajectories_per_environment=8
        #trajectories_per_environment=getattr(cfg.data, 'trajectories_per_environment', 512)
    )

    train_loader = DataLoader(
        train_ds, 
        batch_size=cfg.train.batch_size, 
        shuffle=True,
        num_workers=4,
        prefetch_factor=2,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=cfg.train.batch_size, 
        shuffle=False,
        num_workers=4,
        prefetch_factor=2,
        pin_memory=False
    )
    
    run = wandb.init(project="disco-baselines")
    run.tags = (
            ("geps",)
            + (dataset_name,)
        )
    run_name = wandb.run.name
    output_ckpt_dir = f"{cfg.data.output_dir}/{dataset_name}/{run_name}/"
    os.makedirs(output_ckpt_dir, exist_ok=True)
    wandb_logger = WandbLogger(
        project="disco-baselines",
        config=OmegaConf.to_container(cfg, resolve=True),
        name=run_name
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(dirpath=output_ckpt_dir, save_top_k=1, save_last=True, verbose=True, every_n_train_steps=1000, filename='{step}-{train_l2:.4f}', monitor="train_l2")
    trainer = L.Trainer(
        max_steps=cfg.train.max_steps,
        logger=wandb_logger,
        accelerator='gpu',
        devices=1 if torch.cuda.is_available() else None,
        log_every_n_steps=100,
        check_val_every_n_epoch=5,
        callbacks=[checkpoint_callback, lr_monitor],
    )

    model = GEPSLightning(cfg)

    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()

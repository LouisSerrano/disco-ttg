
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
        #default_files = ["/mnt/home/lserrano/MP-Neural-PDE-Solvers/data/E_EULER_train_1024.h5"]
        print(f"Warning: No HDF5 files specified for {split_name}, using default: {default_files}")

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
                #output_start_t = np.random.randint(start_idx, trajectory.shape[0] - actual_output_frames + 1)
                output_seq = trajectory[input_end_t:input_end_t+actual_output_frames, ::self.sub_x]

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
        
        input = rearrange(input, "b t c h ->  b c h t")
        target = rearrange(target, "b t c h ->  b c h t")

        pred = self.model(input, t, env) # pred of shape B, C, H, W index_t+2 is excluded 
        pred = pred[..., 1:]
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
        
        input = rearrange(input, "b t c h ->  b c h t")
        target = rearrange(target, "b t c h ->  b c h t")
        
        pred = self.model(input, t, env) # pred of shape B, C, H, W index_t+2 is excluded 
        pred = pred[..., 1:]
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


@hydra.main(config_path="config/model", config_name="combined_equation.yaml")
def main(cfg):
    torch.set_default_dtype(torch.float32)
    dataset_name = cfg.data.dataset_name

    train_hdf5_files = get_hdf5_files(cfg, 'train')
    val_hdf5_files = get_hdf5_files(cfg, 'val')
    test_hdf5_files = get_hdf5_files(cfg, 'test')
    trajectories_per_environment = 64

    train_ds = HDF5TemporalDataset(
        hdf5_files=train_hdf5_files,
        input_frames=cfg.data.n_input_frames,
        output_frames=cfg.data.n_output_frames,
        sub_x=cfg.data.sub_x,
        sub_t=cfg.data.sub_t,
        split='train',
        trajectories_per_environment=trajectories_per_environment
    )
    
    val_ds = HDF5TemporalDataset(
        hdf5_files=val_hdf5_files,
        input_frames=cfg.data.n_input_frames,
        output_frames=cfg.data.n_output_frames,
        sub_x=cfg.data.sub_x,
        sub_t=cfg.data.sub_t,
        split='val',
        trajectories_per_environment=trajectories_per_environment
    )

    train_loader = DataLoader(
        train_ds, 
        batch_size=cfg.train.batch_size, 
        shuffle=True,
        num_workers=4,
        prefetch_factor=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=cfg.train.batch_size, 
        shuffle=False,
        num_workers=4,
        prefetch_factor=2,
        pin_memory=True
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

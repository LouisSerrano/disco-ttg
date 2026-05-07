
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, IterableDataset
from torch.optim.lr_scheduler import _LRScheduler
import wandb
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from einops import rearrange
from src.utils.database import RelativeL2
import random
import math

from train.train_combined_vqvae import HDF5TemporalDataset
from src.multiple_physics_pretraining.models.avit import AViT1d


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
        #default_files = ["./datasets/mp-neural/E_EULER_train_1024.h5"]
        print(f"Warning: No HDF5 files specified for {split_name}, using default: {default_files}")

class MPPLightning(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.model = AViT1d(embed_dim=512, processor_blocks=6)
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
        
        input = rearrange(input, "b t c h ->  t b c h")
        pred = self.model(input) # pred of shape B, C, H, W index_t+2 is excluded 

        rel_loss = self.myloss(pred, target)

        opt.zero_grad()
        self.manual_backward(rel_loss)
        opt.step()
        sch.step()
        
        self.log('train_l2', rel_loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        
        input = batch['input']
        target = batch['output'] 
        
        input = rearrange(input, "b t c h ->  t b c h")
        pred = self.model(input) # pred of shape B, C, H, W index_t+2 is excluded 

        rel_loss = self.myloss(pred, target)

        self.log('val_l2', rel_loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        

    def configure_optimizers(self):
        lr = self.cfg.train.learning_rate 
        #param_groups = [
            #{'params': self.model.tokenizer.encoder_layers.parameters(), 'lr': lr},
            #{'params': self.model.tokenizer.decoder_layers.parameters(), 'lr': lr},
            #{'params': self.model.tokenizer.quantizers.parameters(), 'lr': lr},#3e-3
        #]
        max_steps = self.cfg.train.max_steps
        warmup_steps = getattr(self, 'warmup_steps', int(0.05 * max_steps))
        warmup_steps = int(0.05 * max_steps) if warmup_steps is None else warmup_steps

        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4) # 1e-2 before
        scheduler = CosineWithWarmupScheduler(
            opt, 
            warmup_steps=warmup_steps, 
            total_steps=self.cfg.train.max_steps, 
            min_lr_ratio=1e-3
        )

        return [opt], [scheduler] #{"optimizer": opt_gen, "lr_scheduler": scheduler_ae}, {"optimizer": None, "lr_scheduler": None}

    def get_last_layer(self):
        return self.model.get_last_layer()


@hydra.main(config_path="configs", config_name="mpp_combined_equation.yaml")
def main(cfg):
    torch.set_default_dtype(torch.float32)
    dataset_name = cfg.data.dataset_name

    train_hdf5_files = get_hdf5_files(cfg, 'train')
    val_hdf5_files = get_hdf5_files(cfg, 'val')
    test_hdf5_files = get_hdf5_files(cfg, 'test')
    


    train_ds = HDF5TemporalDataset(
        hdf5_files=train_hdf5_files,
        input_frames=cfg.data.n_input_frames,
        output_frames=cfg.data.n_output_frames,
        sub_x=cfg.data.sub_x,
        sub_t=cfg.data.sub_t,
        split='train'
    )
    
    val_ds = HDF5TemporalDataset(
        hdf5_files=val_hdf5_files,
        input_frames=cfg.data.n_input_frames,
        output_frames=cfg.data.n_output_frames,
        sub_x=cfg.data.sub_x,
        sub_t=cfg.data.sub_t,
        split='val'
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
            ("mpp",)
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

    model = MPPLightning(cfg)

    wandb_logger = WandbLogger(project="disco-baselines")
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()

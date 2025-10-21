
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
from src.utils.advection_diffusion import Fractaloid, FractaloidPhase, AdvectionDiffusionExplicit
import random
import math

from train.train import TemporalBatchDatasetFly
from src.multiple_physics_pretraining.models.avit import AViT1d

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
        target = batch['target'] 
        
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
        target = batch['target'] 
        
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
        opt_gen = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4) # 1e-2 before
        scheduler_ae = torch.optim.lr_scheduler.CosineAnnealingLR(opt_gen, self.cfg.train.max_steps)

        return [opt_gen], [scheduler_ae] #{"optimizer": opt_gen, "lr_scheduler": scheduler_ae}, {"optimizer": None, "lr_scheduler": None}

    def get_last_layer(self):
        return self.model.get_last_layer()


@hydra.main(config_path="configs", config_name="mpp_advection_diffusion.yaml")
def main(cfg):
    torch.set_default_dtype(torch.float32)
    dataset_name = cfg.data.dataset_name

    n_batches = int(10000//cfg.train.batch_size)  # or set as needed for your epoch size

    train_ds = TemporalBatchDatasetFly(
        n_batches=n_batches,
        batch_size=cfg.train.batch_size,
        sub_x=cfg.data.sub_x,
        sub_t=cfg.data.sub_t,
        input_frames=cfg.data.n_input_frames,
        output_frames=cfg.data.n_output_frames,
        split='train',
        L=cfg.data.L,
        nx=cfg.data.nx,
        nt=cfg.data.nt,
        T=cfg.data.T,
        fractal_power_range=tuple(cfg.data.fractal_power_range),
        fractal_degree=cfg.data.fractal_degree,
        v_range=tuple(cfg.data.v_range),
        D_range=tuple(cfg.data.D_range),
    )
    val_ds = TemporalBatchDatasetFly(
        n_batches=n_batches//10,
        batch_size=cfg.train.batch_size,
        sub_x=cfg.data.sub_x,
        sub_t=cfg.data.sub_t,
        input_frames=cfg.data.n_input_frames,
        output_frames=cfg.data.n_output_frames,
        split='train',
        L=cfg.data.L,
        nx=cfg.data.nx,
        nt=cfg.data.nt,
        T=cfg.data.T,
        fractal_power_range=tuple(cfg.data.fractal_power_range),
        fractal_degree=cfg.data.fractal_degree,
        v_range=tuple(cfg.data.v_range),
        D_range=tuple(cfg.data.D_range),
    ) 

    train_loader = DataLoader(train_ds, batch_size=None, num_workers=4, prefetch_factor=4, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=None, num_workers=4, prefetch_factor=4, pin_memory=False)
    
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

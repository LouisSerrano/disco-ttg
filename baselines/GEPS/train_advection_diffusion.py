
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

from train.train import advection_diffusion_analytical
from geps.utils import fix_seed, count_parameters, init_weights
from geps.model.forecasters import *
from geps.losses import *

class TemporalBatchDatasetFly(IterableDataset):
    def __init__(self, n_batches, batch_size, sub_x, sub_t, split='train', input_frames=16, output_frames=16,
                 L=16.0, nx=256, nt=100, T=10.0,
                 v_range=(0.01, 1.0), D_range=(0.01, 1.0),
                 fractal_degree=8, fractal_power_range=2, seed=None, in_context=True, num_envs=200):
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.sub_x = sub_x
        self.sub_t = sub_t
        self.split = split
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.L = L
        self.nx = nx
        self.nt = nt
        self.T = T
        self.v_range = v_range
        self.D_range = D_range
        self.fractal_degree = fractal_degree
        self.fractal_power_range = fractal_power_range
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.in_context = in_context
        self.num_envs = num_envs

    def __iter__(self):
        for _ in range(self.n_batches):
            #input_frames = self.rng.integers(2, self.input_frames + 1)
            input_frames = self.input_frames
            batch_inputs = []
            batch_targets = []
            batch_context_inputs = []
            batch_context_targets = []
            batch_advection_speed = []
            batch_diffusion = []
            env_indices = []
            for _ in range(self.batch_size):
                # Sample advection speed and viscosity
                if self.split == 'train':
                    if random.random() < 0.5:
                        v = self.rng.uniform(*self.v_range) if isinstance(self.v_range, (tuple, list)) else float(self.v_range)
                        D = 0
                        # Map v to an environment index (0 to num_v_envs-1)
                        v_min, v_max = self.v_range
                        # Normalize v to [0, 1], then scale to number of v environments
                        normalized_v = (v - v_min) / (v_max - v_min)
                        env = int(normalized_v * (self.num_envs // 2))
                        # Ensure env doesn't exceed bounds due to floating point
                        env = min(env, self.num_envs // 2 - 1)
                        env_indices.append(torch.tensor([env]))
                    else:
                        v = 0
                        D = self.rng.uniform(*self.D_range) if isinstance(self.D_range, (tuple, list)) else float(self.D_range)
                        # Map D to an environment index (num_v_envs to total_envs-1)
                        D_min, D_max = self.D_range
                        normalized_D = (D - D_min) / (D_max - D_min)
                        env = int(normalized_D * (self.num_envs // 2))
                        env = min(env, self.num_envs // 2 - 1)
                        env_indices.append(torch.tensor([self.num_envs // 2 + env]))# the envs are split between advection and diffusion
                else:
                    v = self.rng.uniform(*self.v_range) if isinstance(self.v_range, (tuple, list)) else float(self.v_range)
                    D = self.rng.uniform(*self.D_range) if isinstance(self.D_range, (tuple, list)) else float(self.D_range)

                batch_advection_speed.append(v)
                batch_diffusion.append(D)

                # Generate fractaloid initial condition
                fractal_power = self.rng.uniform(*self.fractal_power_range) if isinstance(self.fractal_power_range, (tuple, list)) else float(self.fractal_power_range)
                fractaloid = FractaloidPhase(
                    degree=self.fractal_degree,
                    power=fractal_power,
                    size=self.nx,
                    patch_size=self.nx
                )
                u0 = fractaloid.generate(batch_size=1, seed=None).squeeze(0).numpy()
                u0 = (u0 - u0.mean()) / (u0.std() + 1e-8)
                u_xt, x, t = advection_diffusion_analytical(
                    u0, L=self.L, v=v, D=D, nt=self.nt, T=self.T
                )
                u_xt = u_xt[::self.sub_t, ::self.sub_x]
                max_start_index_input = u_xt.shape[0] - input_frames - self.output_frames
                if max_start_index_input < 0:
                    raise ValueError("Input frames size is larger than the sequence length.")
                start_index_enc = self.rng.integers(0, max_start_index_input + 1)
                input = u_xt[start_index_enc:start_index_enc + input_frames].copy()
                #max_start_index_target = u_xt.shape[0] - self.output_frames
                #if max_start_index_target < 0:
                #    raise ValueError("Output frames size is larger than the sequence length.")
                #start_index_dec = self.rng.integers(0, max_start_index_target + 1)
                #target = u_xt[start_index_dec:start_index_dec + self.output_frames].copy()
                target = u_xt[start_index_enc + input_frames: start_index_enc + input_frames + self.output_frames].copy()
                batch_inputs.append(torch.from_numpy(input).unsqueeze(-2).float())
                batch_targets.append(torch.from_numpy(target).unsqueeze(-2).float())

                # Second trajectory (context)
                fractal_power_ctx = self.rng.uniform(*self.fractal_power_range) if isinstance(self.fractal_power_range, (tuple, list)) else float(self.fractal_power_range)
                fractaloid_ctx = Fractaloid(
                    degree=self.fractal_degree,
                    power=fractal_power_ctx,
                    size=self.nx,
                    patch_size=self.nx
                )
                u0_ctx = fractaloid_ctx.generate(batch_size=1, seed=None).squeeze(0).numpy()
                u0_ctx = (u0_ctx - u0_ctx.mean()) / (u0_ctx.std() + 1e-8)
                u_xt_ctx, x_ctx, t_ctx = advection_diffusion_analytical(
                    u0_ctx, L=self.L, v=v, D=D, nt=self.nt, T=self.T
                )
                u_xt_ctx = u_xt_ctx[::self.sub_t, ::self.sub_x]
                max_start_index_input_ctx = u_xt_ctx.shape[0] - input_frames
                if max_start_index_input_ctx < 0:
                    raise ValueError("Input frames size is larger than the sequence length (context).")
                start_index_enc_ctx = self.rng.integers(0, max_start_index_input_ctx + 1)
                input_ctx = u_xt_ctx[start_index_enc_ctx:start_index_enc_ctx + input_frames].copy()
                #input_ctx = u_xt_ctx[start_index_enc:start_index_enc + input_frames].copy()
                max_start_index_target_ctx = u_xt_ctx.shape[0] - self.output_frames
                if max_start_index_target_ctx < 0:
                    raise ValueError("Output frames size is larger than the sequence length (context).")
                start_index_dec_ctx = self.rng.integers(0, max_start_index_target_ctx + 1)
                target_ctx = u_xt_ctx[start_index_dec_ctx:start_index_dec_ctx + self.output_frames].copy()
                #target_ctx = u_xt_ctx[start_index_enc + input_frames: start_index_enc + input_frames + self.output_frames].copy()
                batch_context_inputs.append(torch.from_numpy(input_ctx).unsqueeze(-2).float())
                batch_context_targets.append(torch.from_numpy(target_ctx).unsqueeze(-2).float())

            batch = {
                'input': torch.stack(batch_inputs),
                'target': torch.stack(batch_targets),
                'context_input': torch.stack(batch_context_inputs),
                'context_target': torch.stack(batch_context_targets),
                'advection_speed': batch_advection_speed,
                'diffusion': batch_diffusion,
                "env": torch.cat(env_indices)
            }
            yield batch



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
        target = batch['target']
        env = batch['env']
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
        target = batch['target']
        
        env = batch['env']
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


@hydra.main(config_path="config/model", config_name="advection_diffusion.yaml")
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
        num_envs=cfg.model.num_env
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
        num_envs=cfg.model.num_env
    ) 

    train_loader = DataLoader(train_ds, batch_size=None, num_workers=4, prefetch_factor=4, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=None, num_workers=4, prefetch_factor=4, pin_memory=False)
    
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

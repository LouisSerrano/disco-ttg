"""Generic DISCO training script — codebook + environments, no in-context trick.

A clean, well-documented entry point for training DISCO on any HDF5 dataset
(local or on the HuggingFace Hub). Modeled on train/train_combined_aggregate.py
but stripped of advection-diffusion-specific in-context learning.

Expected HDF5 format (per file):
    trajectories: (N, T, C, *spatial)   float32
    env_id:       (N,)                  int64

Config: configs/config_generic.yaml.
"""
import math
import os
import random
import time
from datetime import datetime

import h5py
import hydra
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from einops import rearrange
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, Dataset

from src.operators.disco import DISCOHouse
from src.utils.database import RelativeL2


# ---------- Helpers ----------

def add_weight_decay(params, weight_decay=1e-5, skip_list=()):
    """Apply weight decay only to multi-dim params (skip biases / norms)."""
    decay, no_decay = [], []
    for name, param in params:
        if not param.requires_grad:
            continue
        if len(param.squeeze().shape) <= 1 or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]


class CosineWithWarmupScheduler(_LRScheduler):
    """Cosine annealing with linear warmup."""

    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch <= self.warmup_steps:
            scale = self.last_epoch / self.warmup_steps if self.warmup_steps > 0 else 1.0
        else:
            progress = min(
                (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps),
                1.0,
            )
            scale = self.min_lr_ratio + 0.5 * (1 - self.min_lr_ratio) * (1 + math.cos(math.pi * progress))
        return [base * scale for base in self.base_lrs]


# ---------- Dataset ----------

def _resolve_files(cfg_data, split):
    """Return a list of HDF5 file paths for the given split.

    If cfg_data.hf_repo_id is set, files are downloaded from the Hub on first use
    (requires `huggingface_hub`). Otherwise the local paths from cfg are used.
    """
    files = list(getattr(cfg_data, f"{split}_files", []) or [])

    repo_id = getattr(cfg_data, "hf_repo_id", None)
    if repo_id is None:
        return files

    from huggingface_hub import hf_hub_download

    revision = getattr(cfg_data, "hf_revision", None)
    resolved = []
    for f in files:
        # Treat each entry as a relative path inside the HF repo (if not absolute).
        if os.path.isabs(f) and os.path.exists(f):
            resolved.append(f)
        else:
            local = hf_hub_download(
                repo_id=repo_id,
                filename=os.path.basename(f),
                repo_type="dataset",
                revision=revision,
            )
            resolved.append(local)
    return resolved


class GenericHDF5Dataset(Dataset):
    """Loads (trajectories, env_id) from one or more HDF5 files.

    Each __getitem__ returns:
        input:  (n_input_frames,  C, *spatial)
        output: (n_output_frames, C, *spatial)
        environment_idx: int
    """

    def __init__(
        self,
        hdf5_files,
        n_input_frames=16,
        n_output_frames=16,
        sub_x=1,
        sub_t=1,
        trajectories_key="trajectories",
        env_id_key="env_id",
    ):
        self.files = hdf5_files if isinstance(hdf5_files, list) else [hdf5_files]
        self.n_input_frames = n_input_frames
        self.n_output_frames = n_output_frames
        self.sub_x = sub_x
        self.sub_t = sub_t
        self.trajectories_key = trajectories_key
        self.env_id_key = env_id_key

        self.file_offsets = []  # list of (path, global_offset, n_samples)
        total = 0
        for path in self.files:
            if not os.path.exists(path):
                raise FileNotFoundError(f"HDF5 file not found: {path}")
            with h5py.File(path, "r") as f:
                if trajectories_key not in f:
                    raise KeyError(f"{path}: missing dataset '{trajectories_key}'")
                if env_id_key not in f:
                    raise KeyError(f"{path}: missing dataset '{env_id_key}'")
                n = f[trajectories_key].shape[0]
            self.file_offsets.append((path, total, n))
            total += n
        self.total_samples = total

    def __len__(self):
        return self.total_samples

    def _resolve(self, idx):
        for path, offset, n in self.file_offsets:
            if idx < offset + n:
                return path, idx - offset
        raise IndexError(idx)

    def __getitem__(self, idx):
        path, local_idx = self._resolve(idx)
        with h5py.File(path, "r") as f:
            traj = f[self.trajectories_key][local_idx][:: self.sub_t]
            env = int(f[self.env_id_key][local_idx])

        traj = torch.from_numpy(np.ascontiguousarray(traj)).float()
        # Spatial subsampling on the last dim(s)
        if traj.ndim >= 3 and self.sub_x != 1:
            traj = traj[..., :: self.sub_x]

        total_needed = self.n_input_frames + self.n_output_frames
        if traj.shape[0] < total_needed:
            raise ValueError(
                f"Trajectory in {path} has {traj.shape[0]} frames, need {total_needed}."
            )

        max_start = traj.shape[0] - total_needed
        start = random.randint(0, max_start)
        inp = traj[start : start + self.n_input_frames]
        out = traj[start + self.n_input_frames : start + total_needed]

        # 'output' is the canonical key used by train_generic.py;
        # 'target' is an alias so test_time_compute methods (which expect 'target') work as-is.
        return {"input": inp, "output": out, "target": out, "environment_idx": env}


# ---------- Lightning module ----------

class DISCOLitModule(L.LightningModule):
    """DISCO with codebook + environment-indexed variance reduction.

    Each sample carries an environment_idx. With probability `codebook_prob`,
    the encoder's theta_latent is replaced (via a straight-through estimator)
    by the codebook entry for that environment. The codebook is updated with
    EMA on the encoder outputs that used it.

    There is no in-context trick — input and output come from the same trajectory.
    """

    def __init__(self, model_cfg, training_cfg, num_environments=384):
        super().__init__()
        self.save_hyperparameters()
        self.loss_fn = RelativeL2()
        self.model = DISCOHouse(**model_cfg)
        for k, v in training_cfg.items():
            setattr(self, k, v)
        self.automatic_optimization = False

        # Codebook for variance reduction across environments.
        self.num_environments = num_environments
        self.codebook_dim = getattr(model_cfg, "theta_dim", 256)
        self.codebook_prob = getattr(training_cfg, "codebook_prob", 0.5)
        self.codebook_momentum = getattr(training_cfg, "codebook_momentum", 0.99)
        self.codebook_loss_weight = getattr(training_cfg, "codebook_loss_weight", 0.5)
        self.register_buffer(
            "codebook", torch.randn(self.num_environments, self.codebook_dim) * 0.02
        )
        self.register_buffer("codebook_usage", torch.zeros(self.num_environments))

    def forward(self, x, y):
        y_pred, _ = self.model(x, y)
        return y_pred

    def update_codebook_ema(self, theta_latent, env_idx):
        """EMA-update codebook entries for the given env indices."""
        with torch.no_grad():
            self.codebook_usage.index_add_(
                0, env_idx, torch.ones_like(env_idx, dtype=torch.float)
            )
            unique = torch.unique(env_idx)
            for idx in unique:
                mask = env_idx == idx
                avg = theta_latent[mask].mean(dim=0) if mask.sum() > 1 else theta_latent[mask].squeeze(0)
                self.codebook[idx] = (
                    self.codebook_momentum * self.codebook[idx]
                    + (1 - self.codebook_momentum) * avg
                )

    def training_step(self, batch, batch_idx):
        input = batch["input"]
        target = batch["output"]
        env_idx = batch["environment_idx"].long()

        target_inp = rearrange(target[:, :-1], "b t c h -> (b t) 1 c h")
        target_out = rearrange(target[:, 1:], "b t c h -> (b t) 1 c h")
        state_labels = torch.tensor([0], device=input.device)

        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()
        optimizer.zero_grad()

        # Encode → optionally replace with codebook entry (straight-through).
        theta_latent, encode_metadata = self.model.encode_theta_latent(input, state_labels)

        B = theta_latent.shape[0]
        use_codebook_mask = torch.rand(B, device=theta_latent.device) < self.codebook_prob
        theta_to_use = theta_latent.clone()
        if use_codebook_mask.any():
            cb_idx = env_idx[use_codebook_mask]
            cb_emb = self.codebook[cb_idx]
            theta_to_use[use_codebook_mask] = (
                theta_latent[use_codebook_mask]
                + (cb_emb - theta_latent[use_codebook_mask]).detach()
            )
            self.update_codebook_ema(theta_latent[use_codebook_mask].detach(), cb_idx)

        # Decode and roll out one ODE step.
        spatial = input.shape[3:]
        dim = len(spatial)
        theta = self.model.decode_theta(theta_to_use, dim)
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
        codebook_loss = torch.tensor(0.0, device=loss.device)
        if use_codebook_mask.any():
            cb_emb = self.codebook[env_idx[use_codebook_mask]]
            codebook_loss = F.mse_loss(theta_latent[use_codebook_mask], cb_emb.detach())
        total_loss = loss + self.codebook_loss_weight * codebook_loss

        self.manual_backward(total_loss)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("codebook_loss", codebook_loss, on_step=True, on_epoch=True)
        self.log("total_loss", total_loss, on_step=True, on_epoch=True)
        self.log(
            "codebook_usage_rate",
            use_codebook_mask.float().mean(),
            on_step=True,
            on_epoch=True,
        )
        return total_loss

    def validation_step(self, batch, batch_idx):
        input = batch["input"]
        target = batch["output"]
        env_idx = batch["environment_idx"].long()

        target_inp = rearrange(target[:, :-1], "b t c h -> (b t) 1 c h")
        target_out = rearrange(target[:, 1:], "b t c h -> (b t) 1 c h")
        state_labels = torch.tensor([0], device=input.device)

        theta_latent, encode_metadata = self.model.encode_theta_latent(input, state_labels)
        # Validation always uses the codebook (frozen) so behaviour is deterministic.
        theta_to_use = self.codebook[env_idx]

        spatial = input.shape[3:]
        dim = len(spatial)
        theta = self.model.decode_theta(theta_to_use, dim)
        y_pred, _ = self.model.solve_ode(
            target_inp[:, 0],
            theta,
            state_labels,
            dim,
            integration_time=self.model.default_integration_time,
            n_future_steps=1,
            metadata=encode_metadata,
        )
        loss = self.loss_fn(y_pred, target_out)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        params = add_weight_decay(self.named_parameters(), weight_decay=self.weight_decay)
        optimizer = torch.optim.AdamW(params, lr=self.lr)
        scheduler = CosineWithWarmupScheduler(
            optimizer,
            warmup_steps=self.warmup_steps,
            total_steps=self.max_steps,
            min_lr_ratio=getattr(self, "min_lr_ratio", 1e-3),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


# ---------- Entry point ----------

def get_run_name(cfg: DictConfig) -> str:
    if cfg.training.run_name:
        return cfg.training.run_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        f"DISCO_{cfg.data.dataset_name}_h{cfg.model.hidden_dim}"
        f"_t{cfg.model.theta_dim}_steps{cfg.model.max_steps}"
        f"_bs{cfg.training.batch_size}_lr{cfg.training.lr}_{timestamp}"
    )


@hydra.main(config_path="../configs", config_name="config_generic", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    train_files = _resolve_files(cfg.data, "train")
    val_files = _resolve_files(cfg.data, "val")

    train_ds = GenericHDF5Dataset(
        train_files,
        n_input_frames=cfg.data.n_input_frames,
        n_output_frames=cfg.data.n_output_frames,
        sub_x=cfg.data.sub_x,
        sub_t=cfg.data.sub_t,
        trajectories_key=cfg.data.trajectories_key,
        env_id_key=cfg.data.env_id_key,
    )
    val_ds = GenericHDF5Dataset(
        val_files,
        n_input_frames=cfg.data.n_input_frames,
        n_output_frames=cfg.data.n_output_frames,
        sub_x=cfg.data.sub_x,
        sub_t=cfg.data.sub_t,
        trajectories_key=cfg.data.trajectories_key,
        env_id_key=cfg.data.env_id_key,
    )
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        prefetch_factor=cfg.training.prefetch_factor,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
    )

    run_name = get_run_name(cfg)
    output_dir = os.path.join(cfg.data.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)

    wandb_logger = WandbLogger(project=cfg.training.project, name=run_name, save_dir=output_dir)
    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir,
            filename="best-checkpoint",
            monitor="val_loss",
            mode="min",
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    module = DISCOLitModule(
        model_cfg=cfg.model,
        training_cfg=cfg.training,
        num_environments=cfg.data.num_environments,
    )

    trainer = L.Trainer(
        max_steps=cfg.training.max_steps,
        callbacks=callbacks,
        logger=wandb_logger,
        log_every_n_steps=10,
        precision="32",
    )
    trainer.fit(
        module,
        train_loader,
        val_loader,
        ckpt_path=cfg.training.checkpoint_path,
    )


if __name__ == "__main__":
    main()

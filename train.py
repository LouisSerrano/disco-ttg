import os
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import wandb
from models import DISCOHouse
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from utils import RelativeL2

class TemporalDataset(torch.utils.data.Dataset):
    def __init__(self, u, sub_x, sub_t, input_frames=16, output_frames=16):
        self.u = u 
        self.sub_x = sub_x
        self.sub_t = sub_t
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.slice_size = input_frames + output_frames

    def __len__(self):
        return len(self.u)

    def __getitem__(self, idx):
        images = torch.from_numpy(self.u[idx]).unsqueeze(-2) # add channel dimension
        images = images[::self.sub_t, ..., ::self.sub_x]
        max_start_index = images.shape[0] - self.slice_size
        if max_start_index < 0:
            raise ValueError("Slice size is larger than the sequence length.")
        start_index = np.random.randint(0, max_start_index + 1)
        images = images[start_index:start_index + self.slice_size]
        input = images[:self.input_frames]
        target = images[self.input_frames:]

        return input, target


def load_data(data_cfg):
    train = np.load(data_cfg.train_path)
    val = np.load(data_cfg.val_path)
    test = np.load(data_cfg.test_path)
    # Assume arrays are named 'x' and 'y' in the .npz files
    train_ds = TemporalDataset(train['trajectories'], data_cfg.sub_x, data_cfg.sub_t, data_cfg.n_input_frames, data_cfg.n_output_frames)
    val_ds = TemporalDataset(val['trajectories'], data_cfg.sub_x, data_cfg.sub_t, data_cfg.n_input_frames, data_cfg.n_output_frames)
    test_ds = TemporalDataset(test['trajectories'], data_cfg.sub_x, data_cfg.sub_t, data_cfg.n_input_frames, data_cfg.n_output_frames)
    return train_ds, val_ds, test_ds


class DISCOLitModule(L.LightningModule):
    def __init__(self, model_cfg, training_cfg):
        super().__init__()
        self.save_hyperparameters()
        self.loss_fn = RelativeL2()
        self.model = DISCOHouse(**model_cfg)
        for k, v in training_cfg.items():
            setattr(self, k, v)
        self.automatic_optimization = False  # Enable manual optimization

    def forward(self, x, y):
        y_pred, _ = self.model(x, y)
        return y_pred

    def training_step(self, batch, batch_idx):
        input, target = batch
        state_labels = torch.tensor([0], device=input.device)
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()
        optimizer.zero_grad()
        y_pred, metadata = self.model(input, state_labels, n_future_steps=target.shape[1])
        loss = self.loss_fn(y_pred, target)
        self.manual_backward(loss)
        optimizer.step()
        scheduler.step()
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input, target = batch
        state_labels = torch.tensor([0], device=input.device)
        y_pred, metadata = self.model(input, state_labels, n_future_steps=target.shape[1])
        loss = self.loss_fn(y_pred, target)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_steps)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    run_name = f"DISCO_adj{cfg.model.use_adjoint}_h{cfg.model.hidden_dim}_t{cfg.model.theta_dim}_lr{cfg.training.lr}_steps{cfg.model.max_steps}"  # Example name
    wandb_logger = WandbLogger(
        project=cfg.training.project,
        config=OmegaConf.to_container(cfg, resolve=True),
        name=run_name
    )

    train_ds, val_ds, test_ds = load_data(cfg.data)
    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True, num_workers=4, prefetch_factor=4)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size, num_workers=4, prefetch_factor=4)

    model = DISCOLitModule(cfg.model, cfg.training)

    checkpoint_callback = ModelCheckpoint(
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
    trainer.fit(model, train_loader, val_loader)
    trainer.save_checkpoint("model_final.ckpt")
    wandb.finish()

if __name__ == "__main__":
    main() 
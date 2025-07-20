import os
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, IterableDataset
import wandb
from models import DISCOExpert
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from utils import RelativeL2
from advection_diffusion import Fractaloid, AdvectionDiffusionExplicit
import random

def decorrelation_loss_univariate(features_1: torch.Tensor, features_2: torch.Tensor, lambda_val: float = 0.1) -> torch.Tensor:
    """
    Calculates a decorrelation loss term for two univariate feature tensors.

    Args:
        features_1 (torch.Tensor): A tensor of shape (N, 1).
        features_2 (torch.Tensor): A tensor of shape (N, 1).
        lambda_val (float): The weighting hyperparameter for the loss term.

    Returns:
        torch.Tensor: The calculated decorrelation loss.
    """

    features_1 = features_1.squeeze(1)
    features_2 = features_2.squeeze(1)

    # Ensure inputs have the correct shape
    #if features_1.ndim != 2 or features_1.shape[1] != 1 or \
    #   features_2.ndim != 2 or features_2.shape[1] != 1:
    #    raise ValueError("Input tensors must both have shape (N, 1).")

    # Get the number of samples (N)
    N = features_1.size(-1)
    
    # Center the features by subtracting the mean
    f1_centered = features_1 - features_1.mean(-1, keepdim=True)
    f2_centered = features_2 - features_2.mean(-1, keepdim=True)
    
    # Calculate the empirical covariance (dot product)
    # The result is a scalar (1x1 tensor)
    #cov = torch.einsum(f1_centered, f2_centered, 'b n, b n -> b') / N
    cov = (f1_centered * f2_centered).sum(-1) / N
    #cov = torch.dot(f1_centered.squeeze(), f2_centered.squeeze()) / N

    # The loss is the squared covariance
    loss = (cov**2).mean(0)
    
    # Apply the weighting hyperparameter and return the loss
    return loss

def advection_diffusion_analytical(u0, L=16.0, v=0.1, D=0.5, nt=100, T=10.0):
    """
    Compute the analytical solution of the 1D advection-diffusion equation
    with periodic boundary conditions using the Fourier spectral method.

    Parameters:
        u0 (ndarray): Initial condition, array of shape (nx,)
        L (float): Domain length
        v (float): Advection speed
        D (float): Diffusion coefficient
        nt (int): Number of time steps
        T (float): Final time

    Returns:
        u_xt (ndarray): Solution array of shape (nt, nx)
        x (ndarray): Spatial grid of shape (nx,)
        t (ndarray): Time grid of shape (nt,)
    """
    nx = len(u0)  # infer spatial resolution from input
    x = np.linspace(0, L, nx, endpoint=False)
    t = np.linspace(0, T, nt)

    # Fourier wavenumbers
    k = np.fft.fftfreq(nx, d=L / nx) * 2 * np.pi
    k = 1j * k  # complex wavenumber for exponential form

    # FFT of initial condition
    u0_hat = np.fft.fft(u0)

    # Allocate solution array
    u_xt = np.zeros((nt, nx))

    # Time evolution in spectral space
    for i, ti in enumerate(t):
        decay = np.exp(D * (k**2) * ti) * np.exp(-k * v * ti)
        u_hat_t = u0_hat * decay
        u_xt[i] = np.fft.ifft(u_hat_t).real  # keep only real part

    return u_xt, x, t

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
        images = torch.from_numpy(self.u[idx]).unsqueeze(-2).float() # add channel dimension
        images = images[::self.sub_t, ..., ::self.sub_x]

        # we sample the input frames from a uniform distribution between 2 and self.input_frames
        input_frames = np.random.randint(2, self.input_frames)
        # input_frames = self.input_frames
        max_start_index = images.shape[0] - input_frames
        if max_start_index < 0:
            raise ValueError("Input frames size is larger than the sequence length.")

        start_index_enc = np.random.randint(0, max_start_index + 1)
        input = images[start_index_enc:start_index_enc + input_frames].clone()

        max_start_index = images.shape[0] - self.output_frames
        if max_start_index < 0:
            raise ValueError("Output frames size is larger than the sequence length.")

        start_index_dec = np.random.randint(0, max_start_index + 1)
        target = images[start_index_dec:start_index_dec + self.output_frames].clone()

        return input, target


class TemporalBatchDatasetFly(IterableDataset):
    def __init__(self, n_batches, batch_size, sub_x, sub_t, split='train', input_frames=16, output_frames=16,
                 L=16.0, nx=256, nt=100, T=10.0,
                 v_range=(0.01, 1.0), D_range=(0.01, 1.0),
                 fractal_degree=8, fractal_power_range=2, seed=None):
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

    def __iter__(self):
        for _ in range(self.n_batches):
            #input_frames = self.rng.integers(2, self.input_frames + 1)
            input_frames = self.input_frames


            batch_inputs = []
            batch_targets = []
            for _ in range(self.batch_size):
                # Sample advection speed and viscosity
                if self.split == 'train':
                    if random.random() < 0.5:
                        v = self.rng.uniform(*self.v_range) if isinstance(self.v_range, (tuple, list)) else float(self.v_range)
                        D = 0
                    else:
                        v = 0
                        D = self.rng.uniform(*self.D_range) if isinstance(self.D_range, (tuple, list)) else float(self.D_range)
                else:
                    v = self.rng.uniform(*self.v_range) if isinstance(self.v_range, (tuple, list)) else float(self.v_range)
                    D = self.rng.uniform(*self.D_range) if isinstance(self.D_range, (tuple, list)) else float(self.D_range)
                # Generate fractaloid initial condition
                fractal_power = self.rng.uniform(*self.fractal_power_range) if isinstance(self.fractal_power_range, (tuple, list)) else float(self.fractal_power_range)
                fractaloid = Fractaloid(
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
                max_start_index_input = u_xt.shape[0] - input_frames
                if max_start_index_input < 0:
                    raise ValueError("Input frames size is larger than the sequence length.")
                start_index_enc = self.rng.integers(0, max_start_index_input + 1)
                input = u_xt[start_index_enc:start_index_enc + input_frames].copy()
                max_start_index_target = u_xt.shape[0] - self.output_frames
                if max_start_index_target < 0:
                    raise ValueError("Output frames size is larger than the sequence length.")
                start_index_dec = self.rng.integers(0, max_start_index_target + 1)
                target = u_xt[start_index_dec:start_index_dec + self.output_frames].copy()
                batch_inputs.append(torch.from_numpy(input).unsqueeze(-2).float())
                batch_targets.append(torch.from_numpy(target).unsqueeze(-2).float())
            batch = {
                'input': torch.stack(batch_inputs),
                'target': torch.stack(batch_targets),
            }
            yield batch


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
        self.model = DISCOExpert(**model_cfg)
        for k, v in training_cfg.items():
            setattr(self, k, v)
        self.automatic_optimization = False  # Enable manual optimization

    def forward(self, x, y):
        y_pred, _ = self.model(x, y)
        return y_pred

    def training_step(self, batch, batch_idx):
        input = batch['input']
        target = batch['target']

        #input, target = batch
        state_labels = torch.tensor([0], device=input.device)
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()
        optimizer.zero_grad()
        y_pred, metadata = self.model(input, state_labels, y=target, n_future_steps=target.shape[1]-1)
        #gating_weights = metadata['gating_weights']
        expert_outputs = metadata['expert_outputs']
        decorrelation_loss = decorrelation_loss_univariate(expert_outputs[..., 0], expert_outputs[..., 1])
        #L_sparsity = torch.mean(torch.abs(gating_weights[..., 0])) + torch.mean(gating_weights[..., 1])
        loss = self.loss_fn(y_pred, target[:,1:]) + self.sparsity_alpha*decorrelation_loss #+ self.sparsity_alpha*L_sparsity
        self.manual_backward(loss)
        optimizer.step()
        scheduler.step()
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_decorrelation', decorrelation_loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input = batch['input']
        target = batch['target']
        state_labels = torch.tensor([0], device=input.device)
        y_pred, metadata = self.model(input, state_labels, y=target, n_future_steps=target.shape[1]-1)

        loss = self.loss_fn(y_pred, target[:,1:])
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_steps)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def on_train_epoch_start(self):
        pass  # No longer needed, handled by callback


@hydra.main(config_path="configs", config_name="expert")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    run_name = f"DISCO_adj{cfg.model.use_adjoint}_h{cfg.model.hidden_dim}_nexperts{cfg.model.n_experts}_lr{cfg.training.lr}_steps{cfg.model.max_steps}"  # Example name
    wandb_logger = WandbLogger(
        project=cfg.training.project,
        config=OmegaConf.to_container(cfg, resolve=True),
        name=run_name
    )

    n_batches = int(10000//cfg.training.batch_size)  # or set as needed for your epoch size
    train_ds = TemporalBatchDatasetFly(
        n_batches=n_batches,
        batch_size=cfg.training.batch_size,
        sub_x=cfg.data.sub_x,
        sub_t=cfg.data.sub_t,
        input_frames=cfg.data.n_input_frames,
        output_frames=cfg.data.n_output_frames,
        split='train',
        L=16.0,
        nx=256,
        nt=100,
        T=10.0,
        fractal_power_range=(1.0, 8.0),
        fractal_degree=256, # nx
        v_range=(0.01, 1.0),
        D_range=(0.001, 1.0),
    )
    val_ds = train_ds  # You may want a separate validation dataset

    train_loader = DataLoader(train_ds, batch_size=None, num_workers=4, prefetch_factor=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=None, num_workers=4, prefetch_factor=4, pin_memory=True)

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
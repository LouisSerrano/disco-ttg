import os
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, IterableDataset
from torch.optim.lr_scheduler import _LRScheduler
import wandb
from src.disco_ablations import DiscoAblationsUNet
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from src.utils import RelativeL2
from src.advection_diffusion import Fractaloid, AdvectionDiffusionExplicit
import random
import math

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
                 fractal_degree=8, fractal_power_range=2, seed=None, in_context=True):
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

    def __iter__(self):
        for _ in range(self.n_batches):
            #input_frames = self.rng.integers(2, self.input_frames + 1)
            input_frames = self.input_frames


            batch_inputs = []
            batch_targets = []
            batch_context_inputs = []
            batch_context_targets = []
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
                #max_start_index_input_ctx = u_xt_ctx.shape[0] - input_frames
                #if max_start_index_input_ctx < 0:
                #    raise ValueError("Input frames size is larger than the sequence length (context).")
                #start_index_enc_ctx = self.rng.integers(0, max_start_index_input_ctx + 1)
                #input_ctx = u_xt_ctx[start_index_enc_ctx:start_index_enc_ctx + input_frames].copy()
                input_ctx = u_xt_ctx[start_index_enc:start_index_enc + input_frames].copy()
                #max_start_index_target_ctx = u_xt_ctx.shape[0] - self.output_frames
                #if max_start_index_target_ctx < 0:
                #    raise ValueError("Output frames size is larger than the sequence length (context).")
                #start_index_dec_ctx = self.rng.integers(0, max_start_index_target_ctx + 1)
                #target_ctx = u_xt_ctx[start_index_dec_ctx:start_index_dec_ctx + self.output_frames].copy()
                target_ctx = u_xt_ctx[start_index_enc + input_frames: start_index_enc + input_frames + self.output_frames].copy()
                batch_context_inputs.append(torch.from_numpy(input_ctx).unsqueeze(-2).float())
                batch_context_targets.append(torch.from_numpy(target_ctx).unsqueeze(-2).float())

            batch = {
                'input': torch.stack(batch_inputs),
                'target': torch.stack(batch_targets),
                'context_input': torch.stack(batch_context_inputs),
                'context_target': torch.stack(batch_context_targets),
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


class DISCOUNetLitModule(L.LightningModule):
    def __init__(self, model_cfg, training_cfg):
        super().__init__()
        self.save_hyperparameters()
        self.loss_fn = RelativeL2()
        self.model = DiscoAblationsUNet(**model_cfg)
        for k, v in training_cfg.items():
            setattr(self, k, v)
        self.automatic_optimization = False  # Enable manual optimization

    def forward(self, x, y):
        y_pred, _ = self.model(x, y)
        return y_pred

    def training_step(self, batch, batch_idx):
        input = batch['context_input'] if self.in_context else batch['input']
        target = batch['target'] 

        #input, target = batch
        state_labels = torch.tensor([0], device=input.device)
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()
        optimizer.zero_grad()
        y_pred, metadata = self.model(input, state_labels, y=target, n_future_steps=target.shape[1]-1)
        loss = self.loss_fn(y_pred, target[:,1:])
        self.manual_backward(loss)
        
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
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
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
        #parameters_standard = self.named_parameters()
        #parameters = add_weight_decay(parameters_standard, self.weight_decay) 
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        # Use warmup steps if specified, otherwise default to 10% of max_steps
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
    """Create a descriptive run name for DISCO UNet training."""
    # Get dataset name
    dataset_name = cfg.data.dataset_name
    
    # Get model parameters
    model_params = [
        f"solver{cfg.model.solver}",
        f"adj{cfg.model.use_adjoint}",
        f"h{cfg.model.hidden_dim}",
        f"codedim{cfg.model.code_dim}",
        f"steps{cfg.model.max_steps}",
        f"uchans{cfg.model.hidden_channels}",
        f"init{cfg.model.principled_initialization}",
    ]
    
    # Add training parameters
    train_params = [
        f"bs{cfg.training.batch_size}",
        f"lr{cfg.training.lr}",
        f"ctx{cfg.training.in_context}",
    ]
    
    # Add data parameters
    data_params = [
        f"inframes{cfg.data.n_input_frames}",
        f"outframes{cfg.data.n_output_frames}",
        f"T{cfg.data.T}",
    ]
    
    # Combine all parts
    return f"AblationsUNet_DISCO_{dataset_name}_{'_'.join(model_params)}_{'_'.join(train_params)}_{'_'.join(data_params)}"

@hydra.main(config_path="../configs", config_name="ablations_unet")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    run_name = get_run_name(cfg)
    
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

    n_batches = int(10000//cfg.training.batch_size)  # or set as needed for your epoch size
    train_ds = TemporalBatchDatasetFly(
        n_batches=n_batches,
        batch_size=cfg.training.batch_size,
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
    val_ds = train_ds  # You may want a separate validation dataset

    train_loader = DataLoader(train_ds, batch_size=None, num_workers=4, prefetch_factor=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=None, num_workers=4, prefetch_factor=4, pin_memory=True)

    model = DISCOUNetLitModule(cfg.model, cfg.training)

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
    trainer.fit(model, train_loader, val_loader)
    trainer.save_checkpoint(os.path.join(output_dir, run_name, "final.ckpt"))
    wandb.finish()

if __name__ == "__main__":
    main() 
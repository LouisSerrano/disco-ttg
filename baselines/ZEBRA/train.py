import os
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

from zebra.models.tokenizer.vqvae1d import VQVAE1D
from zebra.models.tokenizer.vqvae2d import VQVAE2D
from zebra.training.tokenizer_trainer import TokenizerTrainer
#from zebra.utils.data import get_data, load_wave2d, load_vort, TemporalDataset
from baselines.ZEBRA.data_utils import get_hdf5_files, TemporalBatchDatasetFly, HDF5TemporalDataset, GrayScottDatasetWrapper
from torch.utils.data import DataLoader


def get_wandb_run_name(cfg: DictConfig) -> str:
    """Create a descriptive wandb run name for tokenizer training."""
    # Get dataset name from path
    dataset_name = cfg.data.dataset_name
    
    # Get model parameters
    model_params = [
        f"emb{cfg.model.codebook_size}",
        f"dim{cfg.model.code_dim}"
    ]
    
    # Combine all parts
    return f"tokenizer_{dataset_name}_{'_'.join(model_params)}"

@hydra.main(version_base=None, config_path="configs/tokenizer", config_name="vqvae1d.yaml")
def train(cfg: DictConfig):
    # Set up logging
    print(cfg.training)
    #L.seed_everything(cfg.training.seed)
    
    # Create logger with descriptive run name
    run_name=get_wandb_run_name(cfg)
    logger = WandbLogger(
        project=cfg.logging.project,
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    dataset_name = cfg.data.dataset_name

    # model class
    if dataset_name in ['gray-scott', 'euler-ns']:
       model_class = VQVAE2D
    else:
        model_class = VQVAE1D
    # load data
    if dataset_name=="advection-diffusion":
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
        val_ds = TemporalBatchDatasetFly(
            n_batches=n_batches//10,
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

        train_loader = DataLoader(train_ds, batch_size=None, num_workers=4, prefetch_factor=4, pin_memory=False)
        val_loader = DataLoader(val_ds, batch_size=None, num_workers=4, prefetch_factor=4, pin_memory=False)
        
    elif dataset_name=="combined-equation":
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
            batch_size=cfg.training.batch_size, 
            shuffle=True,
            num_workers=4,
            prefetch_factor=2,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=4,
            prefetch_factor=2,
            pin_memory=True
        )
    elif dataset_name == "euler-ns":
        # Euler/Navier-Stokes dataset
        from src.utils.euler_ns_dataset import EulerNSDatasetWrapperZEBRA

        train_ds = EulerNSDatasetWrapperZEBRA(
            file_dir=cfg.data.file_dir,
            num_gpus=cfg.data.num_gpus,
            split='train',
            input_frames=getattr(cfg.data, 'n_input_frames', 16),
            output_frames=getattr(cfg.data, 'n_output_frames', 1),
            sub_x=getattr(cfg.data, 'sub_x', 1),
            sub_t=getattr(cfg.data, 'sub_t', 1),
            vorticity_scale=getattr(cfg.data, 'vorticity_scale', 20.0),
        )

        val_ds = EulerNSDatasetWrapperZEBRA(
            file_dir=cfg.data.file_dir,
            num_gpus=cfg.data.num_gpus,
            split='val',
            input_frames=getattr(cfg.data, 'n_input_frames', 16),
            output_frames=getattr(cfg.data, 'n_output_frames', 1),
            sub_x=getattr(cfg.data, 'sub_x', 1),
            sub_t=getattr(cfg.data, 'sub_t', 1),
            vorticity_scale=getattr(cfg.data, 'vorticity_scale', 20.0),
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=4,
            prefetch_factor=2,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=4,
            prefetch_factor=2,
            pin_memory=True
        )
    else:
        train_ds = GrayScottDatasetWrapper(
        hdf5_files=train_hdf5_files,
        split='train',
        input_frames=getattr(cfg.data, 'n_input_frames', 16),
        output_frames=getattr(cfg.data, 'n_output_frames', 1),
        sub_x=getattr(cfg.data, 'sub_x', 1),
        sub_t=getattr(cfg.data, 'sub_t', 1),
        #trajectories_per_environment=getattr(cfg.data, 'trajectories_per_environment', 512)
    )
        val_ds = GrayScottDatasetWrapper(
            hdf5_files=val_hdf5_files,
            split='val',
            input_frames=getattr(cfg.data, 'n_input_frames', 16),
            output_frames=getattr(cfg.data, 'n_output_frames', 1),
            sub_x=getattr(cfg.data, 'sub_x', 1),
            sub_t=getattr(cfg.data, 'sub_t', 1),
            #trajectories_per_environment=getattr(cfg.data, 'trajectories_per_environment', 512)
        )

        train_loader = DataLoader(
            train_ds, 
            batch_size=cfg.training.batch_size, 
            shuffle=True,
            num_workers=4,
            prefetch_factor=2,
            pin_memory=False
        )
        
        val_loader = DataLoader(
            val_ds, 
            batch_size=cfg.training.batch_size, 
            shuffle=False,
            num_workers=4,
            prefetch_factor=2,
            pin_memory=False
        )

    # Create model
    model = model_class(**cfg.model)

    # Create trainer
    trainer = TokenizerTrainer(
        model=model,
        training_config=cfg.training
    )

    # Create callbacks
    print(cfg.logging.output_dir)
    print(run_name)
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(cfg.logging.output_dir, run_name),
            filename="{step}-{val_rel_loss:.2f}",
            monitor="val_rel_loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # Create Lightning trainer
    pl_trainer = L.Trainer(
        max_steps=cfg.training.max_steps,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        strategy=cfg.training.strategy,
        precision=cfg.training.precision,
        #gradient_clip_val=cfg.training.gradient_clip_val,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        check_val_every_n_epoch=10,
        callbacks=callbacks,
        logger=logger,
    )

    # Train
    pl_trainer.fit(
        trainer,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

if __name__ == "__main__":
    train() 

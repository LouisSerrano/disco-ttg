import os
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
import wandb
from transformers import PreTrainedTokenizer

from zebra.models.llama.model import Zebra

from zebra.training.tokenizer_trainer import TokenizerTrainer
from zebra.training.llama_trainer import LLaMATrainer
from zebra.utils.data import get_data, load_wave2d, load_vort, TemporalDataset, TemporalDatasetWithContext, tokenize_dataset

# Import custom data utilities
from ZEBRA.data_utils import get_hdf5_files, get_hdf5_files_gs, TemporalBatchDatasetFly, HDF5TemporalDataset, GrayScottDatasetWrapper
from ZEBRA.data_utils_llama import RawDatasetWithContextFly, RawDatasetWithContext

def get_wandb_run_name(cfg: DictConfig) -> str:
    """Create a descriptive wandb run name for LLaMA pretraining."""
    # Get dataset name from path
    dataset_name = cfg.data.dataset_name
    
    # Get model parameters
    model_params = [
        f"dim{cfg.model.hidden_size}",
        f"layers{cfg.model.num_hidden_layers}",
        f"heads{cfg.model.num_attention_heads}",
        f"ctx{cfg.data.num_context_trajectories}",
    ]
    # Add training parameters
    train_params = [
        f"bs{cfg.training.batch_size}",
        f"lr{cfg.training.learning_rate}",
    ]
    # Combine all parts
    return f"llama_{dataset_name}_{'_'.join(model_params)}_{'_'.join(train_params)}"

@hydra.main(config_path="./configs/llama", config_name="base.yaml")
def train(cfg: DictConfig):
    # Set up logging
    L.seed_everything(cfg.training.seed)
    
    # Get dataset name first
    dataset_name = cfg.data.dataset_name
    
    # Create logger with descriptive run name
    run = wandb.init(project="disco-baselines")
    run.tags = (
            ("zebra",)
            + ("llama",)
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

    tokenizer = TokenizerTrainer.load_from_checkpoint(cfg.data.tokenizer_path)
    tokenizer = tokenizer.model.eval()
    tokenizer.to("cuda") if cfg.training.devices>0 else tokenizer.to("cpu")

    # load data
    if dataset_name=="advection-diffusion":
        # Advection-diffusion: always tokenize on the fly
        n_batches = int(10000//cfg.training.batch_size)  # or set as needed for your epoch size
        
        train_ds = TemporalBatchDatasetFly(
            n_batches=n_batches,
            batch_size=cfg.training.batch_size,  # We'll handle batching in the wrapper
            sub_x=cfg.data.sub_x,
            sub_t=cfg.data.sub_t,  # Subsampling handled in wrapper
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
            split='val',
            L=cfg.data.L,
            nx=cfg.data.nx,
            nt=cfg.data.nt,
            T=cfg.data.T,
            fractal_power_range=tuple(cfg.data.fractal_power_range),
            fractal_degree=cfg.data.fractal_degree,
            v_range=tuple(cfg.data.v_range),
            D_range=tuple(cfg.data.D_range),
        )
        
        # Wrap with context - returns raw trajectories, tokenization happens in trainer
        train_dataset = RawDatasetWithContextFly(
            base_dataset=train_ds,
            sub_t=1,
            slice_size=cfg.data.slice_size,
            num_context_trajectories=cfg.data.num_context_trajectories
        )
        
        val_dataset = RawDatasetWithContextFly(
            base_dataset=val_ds,
            sub_t=1,
            slice_size=cfg.data.slice_size,
            num_context_trajectories=cfg.data.num_context_trajectories
        )
        
        # Create data loaders
        #train_loader = torch.utils.data.DataLoader(
        #    train_dataset,
        #    batch_size=cfg.training.batch_size,
        #    num_workers=0,  # IterableDataset
        #    pin_memory=True,
        #)
        #val_loader = torch.utils.data.DataLoader(
        #    val_dataset,
        #    batch_size=cfg.training.batch_size,
        #    num_workers=0,
        #    pin_memory=True,
        #)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=None, num_workers=4, prefetch_factor=4, pin_memory=False)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=None, num_workers=4, prefetch_factor=4, pin_memory=False)
        
    elif dataset_name=="combined-equation":
        # Combined equation dataset
        train_hdf5_files = get_hdf5_files(cfg, 'train')
        val_hdf5_files = get_hdf5_files(cfg, 'val')
        
        train_base_ds = HDF5TemporalDataset(
            hdf5_files=train_hdf5_files,
            input_frames=cfg.data.n_input_frames,
            output_frames=cfg.data.n_output_frames,
            sub_x=cfg.data.sub_x,
            sub_t=1,  # Temporal subsampling in wrapper
            split='train'
        )
        
        val_base_ds = HDF5TemporalDataset(
            hdf5_files=val_hdf5_files,
            input_frames=cfg.data.n_input_frames,
            output_frames=cfg.data.n_output_frames,
            sub_x=cfg.data.sub_x,
            sub_t=1,
            split='val'
        )
        
        if cfg.training.tokenize_on_the_fly:
            # Use wrapper that returns raw data, tokenization happens in trainer
            train_dataset = RawDatasetWithContext(
                base_dataset=train_base_ds,
                sub_t=cfg.data.sub_t,
                slice_size=cfg.data.slice_size,
                num_context_trajectories=cfg.data.num_context_trajectories,
                trajectories_per_environment=getattr(cfg.data, 'trajectories_per_environment', 16)
            )
            
            val_dataset = RawDatasetWithContext(
                base_dataset=val_base_ds,
                sub_t=cfg.data.sub_t,
                slice_size=cfg.data.slice_size,
                num_context_trajectories=cfg.data.num_context_trajectories,
                trajectories_per_environment=getattr(cfg.data, 'trajectories_per_environment', 16)
            )
        else:
            # Pre-tokenize the dataset
            temp_train_loader = torch.utils.data.DataLoader(
                train_base_ds,
                batch_size=cfg.training.batch_size,
                shuffle=False,
                num_workers=cfg.training.num_workers,
                pin_memory=True,
            )
            temp_val_loader = torch.utils.data.DataLoader(
                val_base_ds,
                batch_size=cfg.training.batch_size,
                shuffle=False,
                num_workers=cfg.training.num_workers,
                pin_memory=True,
            )
            
            # Dummy test loader for tokenize_dataset function
            temp_test_loader = temp_val_loader
            
            token_train, token_val, token_test = tokenize_dataset(
                cfg.data.token_dataset_path, 
                run_name, 
                temp_train_loader, 
                temp_val_loader, 
                temp_test_loader, 
                tokenizer, 
                device=torch.device("cuda") if cfg.training.devices>0 else torch.device("cpu")
            )
            
            train_dataset = TemporalDatasetWithContext(
                token_train, 
                sub_t=cfg.data.sub_t, 
                slice_size=cfg.data.slice_size, 
                num_context_trajectories=cfg.data.num_context_trajectories
            )
            val_dataset = TemporalDatasetWithContext(
                token_val, 
                sub_t=cfg.data.sub_t, 
                slice_size=cfg.data.slice_size, 
                num_context_trajectories=cfg.data.num_context_trajectories
            )
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=cfg.training.num_workers,
            pin_memory=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=cfg.training.num_workers,
            pin_memory=True,
        )
        
    elif dataset_name=="gray-scott":
        # Gray-Scott dataset
        train_hdf5_files = get_hdf5_files_gs(cfg, 'train')
        val_hdf5_files = get_hdf5_files_gs(cfg, 'val')
        
        train_base_ds = GrayScottDatasetWrapper(
            hdf5_files=train_hdf5_files,
            split='train',
            input_frames=getattr(cfg.data, 'n_input_frames', 16),
            output_frames=getattr(cfg.data, 'n_output_frames', 1),
            sub_x=getattr(cfg.data, 'sub_x', 1),
            sub_t=1,  # Temporal subsampling in wrapper
        )
        
        val_base_ds = GrayScottDatasetWrapper(
            hdf5_files=val_hdf5_files,
            split='val',
            input_frames=getattr(cfg.data, 'n_input_frames', 16),
            output_frames=getattr(cfg.data, 'n_output_frames', 1),
            sub_x=getattr(cfg.data, 'sub_x', 1),
            sub_t=1,
        )
        
        if cfg.training.tokenize_on_the_fly:
            # Use wrapper that returns raw data, tokenization happens in trainer
            train_dataset = RawDatasetWithContext(
                base_dataset=train_base_ds,
                sub_t=cfg.data.sub_t,
                slice_size=cfg.data.slice_size,
                num_context_trajectories=cfg.data.num_context_trajectories,
                trajectories_per_environment=getattr(cfg.data, 'trajectories_per_environment', 512)
            )
            
            val_dataset = RawDatasetWithContext(
                base_dataset=val_base_ds,
                sub_t=cfg.data.sub_t,
                slice_size=cfg.data.slice_size,
                num_context_trajectories=cfg.data.num_context_trajectories,
                trajectories_per_environment=getattr(cfg.data, 'trajectories_per_environment', 512)
            )
        else:
            # Pre-tokenize the dataset
            temp_train_loader = torch.utils.data.DataLoader(
                train_base_ds,
                batch_size=cfg.training.batch_size,
                shuffle=False,
                num_workers=cfg.training.num_workers,
                pin_memory=True,
            )
            temp_val_loader = torch.utils.data.DataLoader(
                val_base_ds,
                batch_size=cfg.training.batch_size,
                shuffle=False,
                num_workers=cfg.training.num_workers,
                pin_memory=True,
            )
            
            # Dummy test loader for tokenize_dataset function
            temp_test_loader = temp_val_loader
            
            token_train, token_val, token_test = tokenize_dataset(
                cfg.data.token_dataset_path, 
                run_name, 
                temp_train_loader, 
                temp_val_loader, 
                temp_test_loader, 
                tokenizer, 
                device=torch.device("cuda") if cfg.training.devices>0 else torch.device("cpu")
            )
            
            train_dataset = TemporalDatasetWithContext(
                token_train, 
                sub_t=cfg.data.sub_t, 
                slice_size=cfg.data.slice_size, 
                num_context_trajectories=cfg.data.num_context_trajectories
            )
            val_dataset = TemporalDatasetWithContext(
                token_val, 
                sub_t=cfg.data.sub_t, 
                slice_size=cfg.data.slice_size, 
                num_context_trajectories=cfg.data.num_context_trajectories
            )
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=cfg.training.num_workers,
            pin_memory=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=cfg.training.num_workers,
            pin_memory=True,
        )
    
    elif dataset_name=="wave2d":
        if not cfg.training.tokenize_on_the_fly:
            train_loader, val_loader, test_loader = load_wave2d(cfg.data.data_dir, cfg.training.batch_size, cfg.training.batch_size, sub_t=1, slice_size=30)
            token_train, token_val, token_test = tokenize_dataset(cfg.data.token_dataset_path, run_name, train_loader, val_loader, test_loader, tokenizer, device=torch.device("cuda") if cfg.training.devices>0 else torch.device("cpu"))
            train_dataset = TemporalDatasetWithContext(token_train, sub_t=cfg.data.sub_t, slice_size=cfg.data.slice_size, num_context_trajectories=cfg.data.num_context_trajectories)
            val_dataset = TemporalDatasetWithContext(token_val, sub_t=cfg.data.sub_t, slice_size=cfg.data.slice_size, num_context_trajectories=cfg.data.num_context_trajectories)
            test_dataset = TemporalDatasetWithContext(token_test, sub_t=cfg.data.sub_t, slice_size=cfg.data.slice_size, num_context_trajectories=cfg.data.num_context_trajectories)

            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=cfg.training.batch_size,
                shuffle=True,
                num_workers=cfg.training.num_workers,
                pin_memory=True,
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=cfg.training.batch_size,
                shuffle=False,
                num_workers=cfg.training.num_workers,
                pin_memory=True,
            )

    elif dataset_name=="vorticity":
        if not cfg.training.tokenize_on_the_fly:
            train_loader, val_loader, test_loader = load_vort(cfg.data.data_dir, cfg.training.batch_size, cfg.training.batch_size, sub_t=1, slice_size=30)
            token_train, token_val, token_test = tokenize_dataset(cfg.data.token_dataset_path, run_name, train_loader, val_loader, test_loader, tokenizer, device=torch.device("cuda") if cfg.training.devices>0 else torch.device("cpu"))
            train_dataset = TemporalDatasetWithContext(token_train, sub_t=cfg.data.sub_t, slice_size=cfg.data.slice_size, num_context_trajectories=cfg.data.num_context_trajectories)
            val_dataset = TemporalDatasetWithContext(token_val, sub_t=cfg.data.sub_t, slice_size=cfg.data.slice_size, num_context_trajectories=cfg.data.num_context_trajectories)
            test_dataset = TemporalDatasetWithContext(token_test, sub_t=cfg.data.sub_t, slice_size=cfg.data.slice_size, num_context_trajectories=cfg.data.num_context_trajectories)

            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=cfg.training.batch_size,
                shuffle=True,
                num_workers=cfg.training.num_workers,
                pin_memory=True,
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=cfg.training.batch_size,
                shuffle=False,
                num_workers=cfg.training.num_workers,
                pin_memory=True,
            )
    else:
        u_train, u_val, u_test = get_data(cfg.data.data_dir, dataset_name, return_params=False)

        if not cfg.training.tokenize_on_the_fly:
            train_loader = torch.utils.data.DataLoader(
                u_train,
                batch_size=cfg.training.batch_size,
                shuffle=False,
                num_workers=cfg.training.num_workers,
                pin_memory=True,
            )
            val_loader = torch.utils.data.DataLoader(
                u_val,
                batch_size=cfg.training.batch_size,
                shuffle=False,
                num_workers=cfg.training.num_workers,
                pin_memory=True,
            )
            test_loader = torch.utils.data.DataLoader(
                u_test,
                batch_size=cfg.training.batch_size,
                shuffle=False,
                num_workers=cfg.training.num_workers,
                pin_memory=True,
            )
            token_train, token_val, token_test = tokenize_dataset(cfg.data.token_dataset_path, run_name, train_loader, val_loader, test_loader, tokenizer, device=torch.device("cuda") if cfg.training.devices>0 else torch.device("cpu"))
            print("token_train.shape", token_train.shape)
            train_dataset = TemporalDatasetWithContext(token_train, sub_t=cfg.data.sub_t, slice_size=cfg.data.slice_size, num_context_trajectories=cfg.data.num_context_trajectories)
            val_dataset = TemporalDatasetWithContext(token_val, sub_t=cfg.data.sub_t, slice_size=cfg.data.slice_size, num_context_trajectories=cfg.data.num_context_trajectories)
            test_dataset = TemporalDatasetWithContext(token_test, sub_t=cfg.data.sub_t, slice_size=cfg.data.slice_size, num_context_trajectories=cfg.data.num_context_trajectories)

            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=cfg.training.batch_size,
                shuffle=True,
                num_workers=cfg.training.num_workers,
                pin_memory=True,
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=cfg.training.batch_size,
                shuffle=False,
                num_workers=cfg.training.num_workers,
                pin_memory=True,
            )

        else:
            train_dataset = TemporalDatasetWithContext(u_train, sub_t=cfg.data.sub_t, slice_size=cfg.data.slice_size, num_context_trajectories=cfg.data.num_context_trajectories)
            val_dataset = TemporalDatasetWithContext(u_val, sub_t=cfg.data.sub_t, slice_size=cfg.data.slice_size, num_context_trajectories=cfg.data.num_context_trajectories)
            test_dataset = TemporalDatasetWithContext(u_test, sub_t=cfg.data.sub_t, slice_size=cfg.data.slice_size, num_context_trajectories=cfg.data.num_context_trajectories)

            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=cfg.training.batch_size,
                shuffle=True,
                num_workers=cfg.training.num_workers,
                pin_memory=True,
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=cfg.training.batch_size,
                shuffle=False,
                num_workers=cfg.training.num_workers,
                pin_memory=True,
            )

    try:
        codebook_size = tokenizer.codebook_size
    except:
        codebook_size = tokenizer.quantizers.codebook_size
        
    cfg.model.vocab_size = codebook_size+ 8
    cfg.model.bos_token_id = codebook_size 
    cfg.model.eos_token_id = codebook_size + 1
    cfg.model.context_token_id = codebook_size + 2
    cfg.model.input_token_id = codebook_size + 3 
    cfg.model.target_token_id = codebook_size + 4
    cfg.model.bot_token_id = codebook_size + 5 
    cfg.model.eot_token_id = codebook_size + 6 
    cfg.model.pad_token_id = codebook_size + 7 

    model = Zebra(cfg.model)

    # Create trainer
    trainer = LLaMATrainer(
        model=model,
        tokenizer=tokenizer,
        training_config=cfg.training
    )

    # Create callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=output_ckpt_dir,
            filename="{step}-{val_loss:.2f}",
            monitor="val_loss",
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
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        callbacks=callbacks,
        logger=wandb_logger,
    )

    # Train
    pl_trainer.fit(
        trainer,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    # Save final model
    if cfg.huggingface.push_to_hub:
        trainer.model.push_to_hub(
            repo_name=cfg.huggingface.repo_name,
            private=cfg.huggingface.private,
            commit_message=cfg.huggingface.commit_message,
            model_card=cfg.huggingface.model_card,
            model_card_template=cfg.huggingface.model_card_template,
        )

if __name__ == "__main__":
    train() 

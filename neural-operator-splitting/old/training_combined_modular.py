"""
Modular Training Module for Neural ODE Operators

This module handles the training of neural ODEs to approximate individual
operators using the generated trajectory data. It supports different scenarios
like burgers+heat, dispersion+heat, etc.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import time
from tqdm import tqdm
import os
import argparse
import re
from einops import rearrange

from neural_ode_operators import create_neural_operators
from data_generation import generate_training_data
import h5py


def parse_parameters_from_filename(filename):
    """
    Parse alpha, beta, gamma values from filename.
    Expected format: OP_BURGERS_train_alpha3.0_beta0.0_gamma0.0_1024.h5
    
    Returns:
        dict: Dictionary with 'alpha', 'beta', 'gamma' values
    """
    basename = os.path.basename(filename)
    
    # Pattern to match alpha, beta, gamma values
    pattern = r'alpha(\d+\.?\d*)_beta(\d+\.?\d*)_gamma(\d+\.?\d*)'
    match = re.search(pattern, basename)
    
    if match:
        alpha = float(match.group(1))
        beta = float(match.group(2))
        gamma = float(match.group(3))
        return {'alpha': alpha, 'beta': beta, 'gamma': gamma}
    else:
        # Default values if parsing fails
        return {'alpha': 1.0, 'beta': 0.0, 'gamma': 0.0}


class TrajectoryDataset(Dataset):
    """Dataset for neural ODE training from HDF5 files."""
    
    def __init__(self, 
                 hdf5_file: str,
                 operator_type: str,
                 split: str = 'train'):
        """
        Initialize trajectory dataset from HDF5 file.
        
        Args:
            hdf5_file: Path to HDF5 file
            operator_type: 'heat' or 'dispersion'
            split: 'train', 'valid', or 'test'
        """
        self.hdf5_file = hdf5_file
        self.operator_type = operator_type
        self.split = split
        
        # Load data from HDF5 file
        with h5py.File(hdf5_file, 'r') as f:
            print(f"Loading {operator_type} data from {hdf5_file}")
            print(f"Available groups: {list(f.keys())}")
            
            # Load trajectories - assuming structure similar to train_combined.py
            if split in f:
                group = f[split]
                if 'pde_250-256' in group:
                    self.trajectories = group['pde_250-256'][:]  # (n_samples, n_timesteps, n_spatial)
                    self.alpha = group['alpha'][:]
                    self.beta = group['beta'][:]
                    self.gamma = group['gamma'][:]
                else:
                    raise ValueError(f"Expected 'pde_250-256' not found in group {split}")
            else:
                raise ValueError(f"Split '{split}' not found in file {hdf5_file}")
        
        self.n_samples, self.n_timesteps, self.n_spatial = self.trajectories.shape
        print(f"Loaded {self.n_samples} trajectories with shape ({self.n_timesteps}, {self.n_spatial})")
        
        # Create time points (assuming uniform spacing)
        self.time_points = np.linspace(0, 4.0, self.n_timesteps)
        self.dt = self.time_points[1] - self.time_points[0]
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        # Get appropriate parameter based on operator type
        if self.operator_type == 'heat':
            param_value = self.gamma[idx]  # gamma is typically the diffusion coefficient
        elif self.operator_type == 'dispersion':
            param_value = self.beta[idx]   # beta is typically the advection coefficient
        else:
            param_value = self.alpha[idx]  # fallback
            
        return {
            'u_sequence': torch.from_numpy(self.trajectories[idx]).float(),
            't_sequence': torch.from_numpy(self.time_points).float(),
            'parameter': torch.tensor(param_value).float(),
            'traj_idx': idx
        }


class NeuralODETrainer:
    """Trainer for neural ODE operators."""
    
    def __init__(self, 
                 model: nn.Module,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-5,
                 num_epochs: int = 100,
                 curriculum_start_frames: int = 1,
                 curriculum_end_frames: int = 5,
                 curriculum_warmup_epochs: int = 50):
        """
        Initialize trainer.
        
        Args:
            model: Neural ODE model to train
            device: Device to use for training
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            num_epochs: Number of epochs for cosine annealing scheduler
            curriculum_start_frames: Initial number of target frames to predict
            curriculum_end_frames: Final number of target frames to predict
            curriculum_warmup_epochs: Number of epochs to reach the final target frames
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_epochs, eta_min=0
        )
        self.criterion = self._relative_l2_loss
        
        # Curriculum learning parameters
        self.curriculum_start_frames = curriculum_start_frames
        self.curriculum_end_frames = curriculum_end_frames
        self.curriculum_warmup_epochs = curriculum_warmup_epochs
        self.current_epoch = 0
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_losses_per_step = {}
        self.curriculum_history = []
    
    def _get_current_target_frames(self):
        """Get the current number of target frames based on curriculum schedule."""
        if self.current_epoch >= self.curriculum_warmup_epochs:
            return self.curriculum_end_frames
        
        # Linear interpolation
        progress = self.current_epoch / self.curriculum_warmup_epochs
        current_frames = self.curriculum_start_frames + progress * (self.curriculum_end_frames - self.curriculum_start_frames)
        
        # Return as integer, but use probabilistic rounding to maintain smooth progression
        floor_frames = int(np.floor(current_frames))
        prob_ceil = current_frames - floor_frames
        
        if np.random.rand() < prob_ceil:
            return int(np.ceil(current_frames))
        else:
            return floor_frames
    
    def _relative_l2_loss(self, pred, target):
        """Compute relative L2 loss: ||pred - target||_2 / ||target||_2"""
        diff = pred - target
        l2_diff = torch.norm(diff.flatten(start_dim=1), p=2, dim=1)
        l2_target = torch.norm(target.flatten(start_dim=1), p=2, dim=1)
        # Add small epsilon to avoid division by zero
        relative_error = l2_diff / (l2_target + 1e-8)
        return torch.mean(relative_error)
        
    def train_step(self, batch: Dict, method: str = 'rk4', use_adjoint: bool = False) -> float:
        """Single training step with curriculum learning."""
        self.model.train()
        self.optimizer.zero_grad()
        
        u_sequence = batch['u_sequence'].to(self.device)  # (batch_size, seq_len, nx)
        t_sequence = batch['t_sequence'].to(self.device)  # (batch_size, seq_len)
        
        #print(f"DEBUG: u_sequence shape: {u_sequence.shape}")
        batch_size, seq_len = u_sequence.shape[:2]
        
        # Get current number of target frames from curriculum
        target_frames = self._get_current_target_frames()
        
        # Randomly sample time index for the batch, ensuring we have enough frames
        max_start_idx = seq_len - target_frames - 1  # Ensure we have enough target steps
        if max_start_idx < 0:
            max_start_idx = 0
            target_frames = seq_len - 1  # Adjust if sequence is too short
        
        start_idx = torch.randint(0, max_start_idx + 1, (1,)).item()
        
        # Extract initial condition
        u0 = u_sequence[:, start_idx][:, None, :]  # (batch_size, nx)

        #print(f"DEBUG: u0 shape: {u0.shape}")
        
        # Create time span for autoregressive prediction
        t_start = t_sequence[0, start_idx]  # Use first sample's time (scalar)
        t_span = t_sequence[0, start_idx:start_idx+1+target_frames] - t_start

        #print(f"DEBUG: t_span shape: {t_span.shape}", t_span)
        
        # Predict autoregressive trajectory
        n_steps, pred_trajectory = self.model(u0, t_span=t_span, method=method, use_adjoint=use_adjoint)
        #print(f"DEBUG: pred_trajectory shape: {pred_trajectory.shape}")
        pred_trajectory = rearrange(pred_trajectory[1:], 't b c h -> b t c h').squeeze(-2)
        # pred_trajectory shape: (num_times, batch_size, nx)
        
        # Remove the initial condition (first time step)
        #pred_trajectory = pred_trajectory[1:]  # (num_targets, batch_size, nx)
        
        loss = self.criterion(pred_trajectory, u_sequence[:, start_idx + 1:start_idx + 1 + target_frames])
        #print('loss', loss, pred_trajectory.shape, u_sequence[:, start_idx + 1:start_idx + 1 + target_frames].shape)

        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return loss.item()
    
    def validate(self, val_loader: DataLoader, method: str = 'rk4', use_adjoint: bool = False, validation_steps: List[int] = None) -> Dict[str, float]:
        """Validation step with multiple time steps."""
        if validation_steps is None:
            validation_steps = [1]
        
        self.model.eval()
        losses_per_step = {step: {'total_loss': 0.0, 'num_samples': 0} for step in validation_steps}
        
        with torch.no_grad():
            for batch in val_loader:
                u_sequence = batch['u_sequence'].to(self.device)
                t_sequence = batch['t_sequence'].to(self.device)
        
                batch_size, seq_len = u_sequence.shape[:2]
                
                # Use fixed starting point (beginning of sequence) for consistency
                start_idx = 0
                u0 = u_sequence[:, start_idx:start_idx+1]  # (batch_size, 1, nx)
                
                # Test on different number of steps
                for num_steps_ahead in validation_steps:
                    if start_idx + num_steps_ahead >= seq_len:
                        continue  # Skip if not enough steps in sequence
                    
                    # Get time for integration
                    t_target = t_sequence[:, start_idx + num_steps_ahead] - t_sequence[:, start_idx]
                    target = u_sequence[:, start_idx + num_steps_ahead:start_idx + num_steps_ahead + 1]
                    
                    # Predict with neural ODE
                    n_steps, pred_trajectory = self.model(u0, T=t_target[0], method=method, use_adjoint=use_adjoint)
                    pred_trajectory = pred_trajectory[-1:]  # Take the last time-stamp
                    
                    if pred_trajectory.dim() > 3:
                        pred_trajectory = pred_trajectory.squeeze(2)
                    
                    # Transpose target to match pred_trajectory shape
                    target_transposed = target.transpose(0, 1)
                    
                    # Compute loss
                    loss = self.criterion(pred_trajectory, target_transposed)
                    
                    losses_per_step[num_steps_ahead]['total_loss'] += loss.item() * batch_size
                    losses_per_step[num_steps_ahead]['num_samples'] += batch_size
        
        # Calculate average losses
        avg_losses = {}
        for step in validation_steps:
            if losses_per_step[step]['num_samples'] > 0:
                avg_losses[f'val_loss_step_{step}'] = (
                    losses_per_step[step]['total_loss'] / losses_per_step[step]['num_samples']
                )
            else:
                avg_losses[f'val_loss_step_{step}'] = float('inf')
        
        # Also return the average across all steps for backward compatibility
        avg_losses['val_loss'] = sum(avg_losses.values()) / len(avg_losses)
        
        return avg_losses
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: DataLoader,
              num_epochs: int = 100,
              method: str = 'rk4',
              save_path: Optional[str] = None,
              verbose: bool = True,
              use_adjoint: bool = False,
              validation_steps: List[int] = None,
              validation_frequency: int = 100) -> Dict:
        """
        Train the neural ODE model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader  
            num_epochs: Number of training epochs
            method: ODE solver method
            save_path: Path to save best model
            verbose: Whether to print progress
            use_adjoint: Whether to use adjoint method during training
            validation_steps: List of steps to validate on (e.g., [1, 10, 50, 100])
            validation_frequency: How often to run validation (default: every 100 epochs)
            
        Returns:
            Training history dictionary
        """
        if validation_steps is None:
            validation_steps = [1, 10, 50, 100]
        best_val_loss = float('inf')
        
        if verbose:
            print(f"Training neural ODE with {method} solver{'(adjoint)' if use_adjoint else ''}...")
            print(f"Device: {self.device}")
            print(f"Number of parameters: {sum(p.numel() for p in self.model.parameters())}")
        
        for epoch in range(num_epochs):
            # Update current epoch for curriculum learning
            self.current_epoch = epoch
            current_target_frames = self._get_current_target_frames()
            self.curriculum_history.append(current_target_frames)
            
            # Training phase
            self.model.train()
            train_loss = 0.0
            num_batches = 0
            
            if verbose:
                pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} (frames: {current_target_frames})')
            else:
                pbar = train_loader
            
            for batch in pbar:
                loss = self.train_step(batch, method, use_adjoint)
                train_loss += loss
                num_batches += 1
                
                if verbose and isinstance(pbar, tqdm):
                    pbar.set_postfix({'loss': f'{loss:.6f}'})
            
            avg_train_loss = train_loss / max(num_batches, 1)
            self.train_losses.append(avg_train_loss)
            
            # Validation phase - only run every validation_frequency epochs or on last epoch
            if (epoch + 1) % validation_frequency == 0 or epoch == num_epochs - 1:
                val_losses = self.validate(val_loader, method, use_adjoint, validation_steps)
                
                # Store validation losses for each step
                for key, value in val_losses.items():
                    if key != 'val_loss':  # Skip the average
                        if key not in self.val_losses_per_step:
                            self.val_losses_per_step[key] = []
                        self.val_losses_per_step[key].append(value)
                
                # Use average loss for best model tracking
                val_loss = val_losses['val_loss']
                self.val_losses.append(val_loss)
                
                # Save best model based on average validation loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if save_path:
                        torch.save(self.model.state_dict(), save_path)
                
                if verbose:
                    print(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}')
                    for step in validation_steps:
                        step_loss = val_losses.get(f'val_loss_step_{step}', float('inf'))
                        print(f'  Val Loss (step {step}): {step_loss:.6f}')
                    print(f'  Average Val Loss: {val_loss:.6f}, Best Val = {best_val_loss:.6f}')
            else:
                if verbose and (epoch + 1) % 10 == 0:  # Print train loss every 10 epochs
                    print(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}')
            
            # Learning rate scheduling
            self.scheduler.step()
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_losses_per_step': self.val_losses_per_step,
            'best_val_loss': best_val_loss
        }
    
    def plot_training_history(self, save_path: Optional[str] = None, validation_frequency: int = 100):
        """Plot training history."""
        # Create figure with subplots
        n_val_steps = len(self.val_losses_per_step)
        if n_val_steps > 0:
            fig = plt.figure(figsize=(15, 10))
            gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1])
        else:
            fig = plt.figure(figsize=(10, 6))
            gs = fig.add_gridspec(1, 2)
        
        # Training loss plot
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(self.train_losses, label='Train', alpha=0.7)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Validation loss (average) plot
        ax2 = fig.add_subplot(gs[0, 1])
        if self.val_losses:
            # Create x-axis values for validation points
            val_epochs = [i * validation_frequency - 1 for i in range(1, len(self.val_losses) + 1)]
            if len(self.train_losses) - 1 not in val_epochs:  # Add last epoch if needed
                val_epochs[-1] = len(self.train_losses) - 1
            ax2.plot(val_epochs, self.val_losses, 'o-', label='Avg Validation', markersize=6)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.set_title('Average Validation Loss')
            ax2.set_yscale('log')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        
        # Per-step validation losses
        if n_val_steps > 0:
            ax3 = fig.add_subplot(gs[1, :])
            val_epochs = [i * validation_frequency - 1 for i in range(1, len(list(self.val_losses_per_step.values())[0]) + 1)]
            if len(self.train_losses) - 1 not in val_epochs:  # Add last epoch if needed
                val_epochs[-1] = len(self.train_losses) - 1
            
            for step_key, losses in sorted(self.val_losses_per_step.items()):
                step_num = int(step_key.split('_')[-1])
                ax3.plot(val_epochs[:len(losses)], losses, 'o-', label=f'{step_num} steps', markersize=6)
            
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Loss')
            ax3.set_title('Validation Loss for Different Step Counts')
            ax3.set_yscale('log')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
            
            # Zoomed view of last validations
            ax4 = fig.add_subplot(gs[2, :])
            last_n = min(5, len(val_epochs))  # Last 5 validations
            start_idx = -last_n
            
            for step_key, losses in sorted(self.val_losses_per_step.items()):
                step_num = int(step_key.split('_')[-1])
                if len(losses) >= last_n:
                    ax4.plot(val_epochs[start_idx:len(losses)], losses[start_idx:], 'o-', 
                            label=f'{step_num} steps', markersize=8)
            
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Loss')
            ax4.set_title(f'Validation Loss (Last {last_n} Validations)')
            ax4.set_yscale('log')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return plt.gcf()


def train_neural_operators(nx: int = 256,
                          L: float = 16.0,
                          hidden_dim: int = 64,
                          n_layers: int = 3,
                          num_epochs: int = 100,
                          batch_size: int = 8,
                          learning_rate: float = 1e-3,
                          method: str = 'rk4',
                          num_steps: int = 1,
                          device: str = 'auto',
                          save_dir: str = './models',
                          verbose: bool = True,
                          use_adjoint: bool = False,
                          scenario: str = 'burgers_heat',
                          operator1_files: Dict = None,
                          operator2_files: Dict = None,
                          curriculum_start_frames: int = 1,
                          curriculum_end_frames: int = 10,
                          curriculum_warmup_epochs: int = 50) -> Dict:
    """
    Train neural operators using LPSDA HDF5 datasets.
    
    Args:
        nx: Number of spatial grid points
        L: Domain length
        hidden_dim: Hidden layer dimension
        n_layers: Number of hidden layers
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        method: ODE solver method
        device: Device to use ('auto', 'cpu', 'cuda')
        save_dir: Directory to save models
        verbose: Whether to print progress
        use_adjoint: Whether to use adjoint method during training
        scenario: Training scenario ('burgers_heat', 'dispersion_heat', etc.)
        operator1_files: Dict with 'train' and 'valid' paths for first operator data
        operator2_files: Dict with 'train' and 'valid' paths for second operator data
        
    Returns:
        Dictionary with trained models and histories
    """
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Define scenario configurations
    scenario_configs = {
        'burgers_heat': {
            'op1_name': 'burgers',
            'op2_name': 'heat',
            'op1_type': 'dispersion',
            'op2_type': 'heat',
            'op1_model_key': 'advection',
            'op2_model_key': 'diffusion',
            'default_op1_files': {
                'train': '/mnt/home/lserrano/LPSDA/data/OP_BURGERS_train_1024.h5',
                'valid': '/mnt/home/lserrano/LPSDA/data/OP_BURGERS_valid.h5'
            },
            'default_op2_files': {
                'train': '/mnt/home/lserrano/LPSDA/data/OP_HEAT_train_1024.h5',
                'valid': '/mnt/home/lserrano/LPSDA/data/OP_HEAT_valid.h5'
            }
        },
        'dispersion_heat': {
            'op1_name': 'dispersion',
            'op2_name': 'heat',
            'op1_type': 'dispersion',
            'op2_type': 'heat',
            'op1_model_key': 'advection',
            'op2_model_key': 'diffusion',
            'default_op1_files': {
                'train': '/mnt/home/lserrano/LPSDA/data/OP_DISP_train_1024.h5',
                'valid': '/mnt/home/lserrano/LPSDA/data/OP_DISP_valid.h5'
            },
            'default_op2_files': {
                'train': '/mnt/home/lserrano/LPSDA/data/OP_HEAT_train_1024.h5',
                'valid': '/mnt/home/lserrano/LPSDA/data/OP_HEAT_valid.h5'
            }
        },
        'burgers_disp': {
            'op1_name': 'burgers',
            'op2_name': 'dispersion',
            'op1_type': 'burgers',
            'op2_type': 'dispersion',
            'op1_model_key': 'advection',
            'op2_model_key': 'diffusion',
            'default_op1_files': {
                'train': '/mnt/home/lserrano/LPSDA/data/OP_BURGERS_train_1024.h5',
                'valid': '/mnt/home/lserrano/LPSDA/data/OP_BURGERS_valid.h5'
            },
            'default_op2_files': {
                'train': '/mnt/home/lserrano/LPSDA/data/OP_DISP_train_1024.h5',
                'valid': '/mnt/home/lserrano/LPSDA/data/OP_DISP_valid.h5'
            }
        }
    }
    
    if scenario not in scenario_configs:
        raise ValueError(f"Unknown scenario '{scenario}'. Available: {list(scenario_configs.keys())}")
    
    config = scenario_configs[scenario]
    
    if verbose:
        print(f"Training neural operators on {device}")
        print(f"Grid size: {nx}, Domain length: {L}")
        print(f"Training scenario: {scenario} ({config['op1_name']} + {config['op2_name']}) operators from LPSDA datasets")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Set default file paths if not provided
    if operator1_files is None:
        operator1_files = config['default_op1_files']
    
    if operator2_files is None:
        operator2_files = config['default_op2_files']
    
    if verbose:
        print("Loading training data from HDF5 files...")
        print(f"{config['op1_name'].capitalize()} files: {operator1_files}")
        print(f"{config['op2_name'].capitalize()} files: {operator2_files}")
    
    # Create models
    operators = create_neural_operators(nx, L, hidden_dim, n_layers, padding_mode="circular", num_steps=num_steps)
    
    results = {}
    
    # Train first operator
    if verbose:
        print("\n" + "="*50)
        print(f"TRAINING {config['op1_name'].upper()} OPERATOR")
        print("="*50)
    
    # Create datasets for first operator
    train_dataset = TrajectoryDataset(
        operator1_files['train'],
        config['op1_type'],
        'train'
    )
    val_dataset = TrajectoryDataset(
        operator1_files['valid'],
        config['op1_type'],
        'valid'
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Train first operator
    op1_trainer = NeuralODETrainer(
        operators[config['op1_model_key']], device, learning_rate, 
        num_epochs=num_epochs,
        curriculum_start_frames=curriculum_start_frames,
        curriculum_end_frames=curriculum_end_frames,
        curriculum_warmup_epochs=curriculum_warmup_epochs
    )
    
    op1_history = op1_trainer.train(
        train_loader, val_loader, num_epochs, method,
        save_path=os.path.join(save_dir, f"{config['op1_name']}_model.pth"),
        verbose=verbose,
        use_adjoint=use_adjoint
    )
    
    results[config['op1_name']] = {
        'model': operators[config['op1_model_key']],
        'trainer': op1_trainer,
        'history': op1_history
    }
    
    # Train second operator - COMMENTED OUT FOR SINGLE OPERATOR TRAINING
    # if verbose:
    #     print("\n" + "="*50)
    #     print(f"TRAINING {config['op2_name'].upper()} OPERATOR")
    #     print("="*50)
    # 
    # # Create datasets for second operator
    # train_dataset = TrajectoryDataset(
    #     operator2_files['train'],
    #     config['op2_type'],
    #     'train'
    # )
    # val_dataset = TrajectoryDataset(
    #     operator2_files['valid'],
    #     config['op2_type'],
    #     'valid'
    # )
    # 
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    # 
    # # Train second operator
    # op2_trainer = NeuralODETrainer(
    #     operators[config['op2_model_key']], device, learning_rate, num_epochs=num_epochs
    # )
    # 
    # op2_history = op2_trainer.train(
    #     train_loader, val_loader, num_epochs, method,
    #     save_path=os.path.join(save_dir, f"{config['op2_name']}_model.pth"),
    #     verbose=verbose,
    #     use_adjoint=use_adjoint
    # )
    # 
    # results[config['op2_name']] = {
    #     'model': operators[config['op2_model_key']],
    #     'trainer': op2_trainer, 
    #     'history': op2_history
    # }
    
    # Plot training histories
    if verbose:
        print("\nPlotting training histories...")
    
    fig_op1 = results[config['op1_name']]['trainer'].plot_training_history(
        os.path.join(save_dir, f"{config['op1_name']}_training.png")
    )
    
    # fig_op2 = results[config['op2_name']]['trainer'].plot_training_history(
    #     os.path.join(save_dir, f"{config['op2_name']}_training.png")
    # )
    
    results['figures'] = {
        f"{config['op1_name']}_training": fig_op1,
        # f"{config['op2_name']}_training": fig_op2
    }
    
    if verbose:
        print(f"\nTraining complete! Models saved to {save_dir}")
        print(f"{config['op1_name'].capitalize()} best val loss: {op1_history['best_val_loss']:.6f}")
        # print(f"{config['op2_name'].capitalize()} best val loss: {op2_history['best_val_loss']:.6f}")
    
    return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Neural ODE Operators')
    
    parser.add_argument('--scenario', type=str, default='burgers_heat',
                        choices=['burgers_heat', 'dispersion_heat', 'burgers_disp'],
                        help='Training scenario (default: burgers_heat)')
    
    parser.add_argument('--nx', type=int, default=256,
                        help='Number of spatial grid points (default: 256)')
    
    parser.add_argument('--L', type=float, default=16.0,
                        help='Domain length (default: 16.0)')
    
    parser.add_argument('--hidden_dim', type=int, default=32,
                        help='Hidden layer dimension (default: 32)')
    
    parser.add_argument('--n_layers', type=int, default=2,
                        help='Number of hidden layers (default: 2)')
    
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs (default: 50)')
    
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size (default: 64)')
    
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    
    parser.add_argument('--method', type=str, default='rk4',
                        choices=['rk4', 'dopri5', 'euler'],
                        help='ODE solver method (default: rk4)')
    
    parser.add_argument('--save_dir', type=str, 
                        default='/mnt/home/lserrano/disco-ball/neural-operator-splitting/models',
                        help='Directory to save models')
    
    parser.add_argument('--use_adjoint', action='store_true',
                        help='Use adjoint method during training')
    
    parser.add_argument('--op1_train', type=str, default=None,
                        help='Custom training file for first operator')
    
    parser.add_argument('--op1_valid', type=str, default=None,
                        help='Custom validation file for first operator')
    
    parser.add_argument('--op2_train', type=str, default=None,
                        help='Custom training file for second operator')
    
    parser.add_argument('--op2_valid', type=str, default=None,
                        help='Custom validation file for second operator')
    
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Print progress (default: True)')
    
    parser.add_argument('--dataset', type=str, default=None,
                        help='Custom dataset path (e.g., OP_BURGERS_train_alpha3.0_beta0.0_gamma0.0_1024.h5)')
    
    parser.add_argument('--valid_dataset', type=str, default=None,
                        help='Custom validation dataset path (e.g., OP_BURGERS_valid_alpha3.0_beta0.0_gamma0.0_1024.h5)')
    
    parser.add_argument('--curriculum_start_frames', type=int, default=1,
                        help='Initial number of target frames in curriculum (default: 1)')
    
    parser.add_argument('--curriculum_end_frames', type=int, default=10,
                        help='Final number of target frames in curriculum (default: 10)')
    
    parser.add_argument('--curriculum_warmup_epochs', type=int, default=50,
                        help='Number of epochs to reach final target frames (default: 50)')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    print("Training Neural ODE Operators...")
    print(f"Configuration: {vars(args)}")
    
    # Prepare custom file paths if provided
    operator1_files = None
    operator2_files = None
    
    # If a custom dataset is provided, use it and parse parameters
    if args.dataset:
        params = parse_parameters_from_filename(args.dataset)
        print(f"Parsed parameters from filename: alpha={params['alpha']}, beta={params['beta']}, gamma={params['gamma']}")
        
        # Determine the validation file path
        if args.valid_dataset:
            # Use explicitly provided validation dataset
            valid_path = args.valid_dataset
            print(f"Using custom validation dataset: {valid_path}")
        else:
            # Try to construct validation filename (assuming it follows similar naming convention)
            dataset_dir = os.path.dirname(args.dataset)
            dataset_basename = os.path.basename(args.dataset)
            
            if 'train' in dataset_basename:
                valid_basename = dataset_basename.replace('train', 'valid')
                valid_path = os.path.join(dataset_dir, valid_basename)
                if not os.path.exists(valid_path):
                    # Fallback to a default validation set
                    valid_path = '/mnt/home/lserrano/LPSDA/data/OP_BURGERS_valid.h5'
                    print(f"Validation file {valid_path} not found, using default: {valid_path}")
            else:
                valid_path = '/mnt/home/lserrano/LPSDA/data/OP_BURGERS_valid.h5'
                print(f"Using default validation dataset: {valid_path}")
        
        # Use custom dataset for first operator (assuming burgers scenario)
        operator1_files = {
            'train': args.dataset,
            'valid': valid_path
        }
        
        # Update save directory to include parameters
        args.save_dir = os.path.join(
            args.save_dir, 
            f"alpha{params['alpha']}_beta{params['beta']}_gamma{params['gamma']}"
        )
    
    elif args.op1_train or args.op1_valid:
        operator1_files = {}
        if args.op1_train:
            operator1_files['train'] = args.op1_train
        if args.op1_valid:
            operator1_files['valid'] = args.op1_valid
    
    if args.op2_train or args.op2_valid:
        operator2_files = {}
        if args.op2_train:
            operator2_files['train'] = args.op2_train
        if args.op2_valid:
            operator2_files['valid'] = args.op2_valid
    
    # Train neural operators
    results = train_neural_operators(
        scenario=args.scenario,
        nx=args.nx,
        L=args.L,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        method=args.method,
        save_dir=args.save_dir,
        use_adjoint=args.use_adjoint,
        operator1_files=operator1_files,
        operator2_files=operator2_files,
        verbose=args.verbose,
        curriculum_start_frames=args.curriculum_start_frames,
        curriculum_end_frames=args.curriculum_end_frames,
        curriculum_warmup_epochs=args.curriculum_warmup_epochs
    )
    
    print("\nTraining completed!")
    
    # Get scenario info for dynamic coefficient printing
    scenario_config = {
        'burgers_heat': ('burgers', 'heat'),
        'dispersion_heat': ('dispersion', 'heat'),
        'burgers_disp': ('burgers', 'dispersion')
    }[args.scenario]
    
    op1_name, op2_name = scenario_config
    
    if op1_name in ['burgers', 'dispersion']:
        print(f"{op1_name.capitalize()} coefficient learned: "
              f"{results[op1_name]['model'].get_advection_coefficient():.4f}")
    
    # if op2_name == 'heat':
    #     print(f"Heat coefficient learned: "
    #           f"{results[op2_name]['model'].get_diffusion_coefficient():.4f}")
    # elif op2_name == 'dispersion':
    #     print(f"Dispersion coefficient learned: "
    #           f"{results[op2_name]['model'].get_advection_coefficient():.4f}")
"""
Training Module for Neural ODE Operators

This module handles the training of neural ODEs to approximate individual
advection and diffusion operators using the generated trajectory data.
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

from neural_ode_operators import create_neural_operators
from data_generation import generate_training_data
import h5py


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
    
    def _create_samples(self):
        """Create training samples from trajectories."""
        for traj_idx, trajectory in enumerate(self.trajectories):
            nt, nx = trajectory.shape
            
            # Create sequences of length sequence_length
            for start_idx in range(0, nt - self.sequence_length, self.sequence_length // 2):
                end_idx = start_idx + self.sequence_length
                if end_idx >= nt:
                    continue
                
                u_sequence = trajectory[start_idx:end_idx]  # (seq_len, nx)
                t_sequence = self.time_points[start_idx:end_idx]
                
                # Get corresponding parameter value
                if self.operator_type == 'advection':
                    param_value = self.parameters[traj_idx]['beta']
                else:  # diffusion
                    param_value = self.parameters[traj_idx]['D']
                
                self.samples.append({
                    'u_sequence': torch.from_numpy(u_sequence).float(),
                    't_sequence': torch.from_numpy(t_sequence).float(),
                    'parameter': torch.tensor(param_value).float(),
                    'traj_idx': traj_idx
                })
    
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
                 num_epochs: int = 100):
        """
        Initialize trainer.
        
        Args:
            model: Neural ODE model to train
            device: Device to use for training
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            num_epochs: Number of epochs for cosine annealing scheduler
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_epochs, eta_min=0
        )
        self.criterion = self._relative_l2_loss
        
        # Training history
        self.train_losses = []
        self.val_losses = []
    
    def _relative_l2_loss(self, pred, target):
        """Compute relative L2 loss: ||pred - target||_2 / ||target||_2"""
        diff = pred - target
        l2_diff = torch.norm(diff.flatten(start_dim=1), p=2, dim=1)
        l2_target = torch.norm(target.flatten(start_dim=1), p=2, dim=1)
        # Add small epsilon to avoid division by zero
        relative_error = l2_diff / (l2_target + 1e-8)
        return torch.mean(relative_error)
        
    def train_step(self, batch: Dict, method: str = 'rk4', use_adjoint: bool = False) -> float:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        u_sequence = batch['u_sequence'].to(self.device)  # (batch_size, seq_len, nx)
        t_sequence = batch['t_sequence'].to(self.device)  # (batch_size, seq_len)
        
        batch_size, seq_len = u_sequence.shape[:2]
        
        # Randomly sample time index for the batch
        max_start_idx = seq_len - 2  # Ensure we have at least one target step
        start_idx = torch.randint(0, max_start_idx + 1, (1,)).item()
        
        # Extract initial condition and target from sampled time index
        u0 = u_sequence[:, start_idx:start_idx+1]  # (batch_size, 1, nx)
        t_seq = t_sequence[:, 1]  # (batch_size, remaining_steps) # we take the first time

        target = u_sequence[:, start_idx+1:start_idx+2]  # (batch_size, remaining_steps, nx)
        
        n_steps, pred_trajectory = self.model(u0, T=t_seq[0], method=method, use_adjoint=use_adjoint)  # (remaining_steps, batch_size, nx)
        
        pred_trajectory = pred_trajectory.squeeze(2)  # Remove extra dimension if present
        pred_trajectory = pred_trajectory[-1:] # we take the last time-stamp
        
        # Transpose target to match pred_trajectory shape: (batch_size, remaining_steps, nx) -> (remaining_steps, batch_size, nx)
        target_transposed = target.transpose(0, 1)  # (remaining_steps, batch_size, nx)
        
        # Compute loss against ground truth
        loss = self.criterion(pred_trajectory, target_transposed)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return loss.item()
    
    def validate(self, val_loader: DataLoader, method: str = 'rk4', use_adjoint: bool = False) -> float:
        """Validation step."""
        self.model.eval()
        total_loss = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                u_sequence = batch['u_sequence'].to(self.device)
                t_sequence = batch['t_sequence'].to(self.device)
        
                batch_size, seq_len = u_sequence.shape[:2]
                
                # Randomly sample time index for the batch
                max_start_idx = seq_len - 2  # Ensure we have at least one target step
                start_idx = torch.randint(0, max_start_idx + 1, (1,)).item()
                
                # Extract initial condition and target from sampled time index
                u0 = u_sequence[:, start_idx:start_idx+1]  # (batch_size, 1, nx)
                t_seq = t_sequence[:, 1]  # (batch_size, remaining_steps)
                target = u_sequence[:, start_idx+1:start_idx+2]  # (batch_size, remaining_steps, nx)
                
                n_steps, pred_trajectory = self.model(u0, T=t_seq[0], method=method, use_adjoint=use_adjoint)  # (remaining_steps, batch_size, nx)
                pred_trajectory = pred_trajectory[-1:] # we take the last time-stamp
                
                pred_trajectory = pred_trajectory.squeeze(2)  # Remove extra dimension if present
                
                # Transpose target to match pred_trajectory shape: (batch_size, remaining_steps, nx) -> (remaining_steps, batch_size, nx)
                target_transposed = target.transpose(0, 1)  # (remaining_steps, batch_size, nx)
                
                # Compute loss against ground truth
                loss = self.criterion(pred_trajectory, target_transposed)

                total_loss += loss.item() * batch_size
                num_samples += batch_size
        
        return total_loss / max(num_samples, 1)
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: DataLoader,
              num_epochs: int = 100,
              method: str = 'rk4',
              save_path: Optional[str] = None,
              verbose: bool = True,
              use_adjoint: bool = False) -> Dict:
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
            
        Returns:
            Training history dictionary
        """
        best_val_loss = float('inf')
        
        if verbose:
            print(f"Training neural ODE with {method} solver{'(adjoint)' if use_adjoint else ''}...")
            print(f"Device: {self.device}")
            print(f"Number of parameters: {sum(p.numel() for p in self.model.parameters())}")
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            num_batches = 0
            
            if verbose:
                pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
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
            
            # Validation phase
            val_loss = self.validate(val_loader, method, use_adjoint)
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if save_path:
                    torch.save(self.model.state_dict(), save_path)
            
            if verbose:
                print(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, '
                      f'Val Loss = {val_loss:.6f}, Best Val = {best_val_loss:.6f}')
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': best_val_loss
        }
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history."""
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train')
        plt.plot(self.val_losses, label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.yscale('log')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        # Plot last 50 epochs for better detail
        start_idx = max(0, len(self.train_losses) - 50)
        plt.plot(range(start_idx, len(self.train_losses)), 
                 self.train_losses[start_idx:], label='Train')
        plt.plot(range(start_idx, len(self.val_losses)), 
                 self.val_losses[start_idx:], label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History (Last 50 Epochs)')
        plt.legend()
        plt.yscale('log')
        plt.grid(True)
        
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
                          heat_files: Dict = None,
                          dispersion_files: Dict = None) -> Dict:
    """
    Train neural operators for heat and dispersion using LPSDA HDF5 datasets.
    
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
        heat_files: Dict with 'train' and 'valid' paths for heat data
        dispersion_files: Dict with 'train' and 'valid' paths for dispersion data
        
    Returns:
        Dictionary with trained models and histories
    """
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if verbose:
        print(f"Training neural operators on {device}")
        print(f"Grid size: {nx}, Domain length: {L}")
        print(f"Training heat and burgers operators from LPSDA datasets")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Set default file paths if not provided
    if heat_files is None:
        heat_files = {
            'train': '/mnt/home/lserrano/LPSDA/data/OP_HEAT_train_1024.h5',
            'valid': '/mnt/home/lserrano/LPSDA/data/OP_HEAT_valid.h5'
        }
    
    if dispersion_files is None:
        dispersion_files = {
            'train': '/mnt/home/lserrano/LPSDA/data/OP_DISP_train_1024.h5',
            'valid': '/mnt/home/lserrano/LPSDA/data/OP_DISP_valid.h5'
        }
    
    if verbose:
        print("Loading training data from HDF5 files...")
        print(f"Heat files: {heat_files}")
        print(f"Burgers files: {dispersion_files}")
    
    # Create models
    operators = create_neural_operators(nx, L, hidden_dim, n_layers, padding_mode="circular", num_steps=num_steps)
    
    results = {}
    
    # Train burgers operator
    if verbose:
        print("\n" + "="*50)
        print("TRAINING DISPERSION OPERATOR")
        print("="*50)
    
    # Create datasets for burgers (advection) operator
    train_dataset = TrajectoryDataset(
        dispersion_files['train'],
        'dispersion',
        'train'
    )
    val_dataset = TrajectoryDataset(
        dispersion_files['valid'],
        'dispersion',
        'valid'
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Train burgers operator
    dispersion_trainer = NeuralODETrainer(
        operators['advection'], device, learning_rate, num_epochs=num_epochs
    )
    
    dispersion_history = dispersion_trainer.train(
        train_loader, val_loader, num_epochs, method,
        save_path=os.path.join(save_dir, 'dispersion_model.pth'),
        verbose=verbose,
        use_adjoint=use_adjoint
    )
    
    results['dispersion'] = {
        'model': operators['advection'],
        'trainer': dispersion_trainer,
        'history': dispersion_history
    }
    
    # Train heat operator
    if verbose:
        print("\n" + "="*50)
        print("TRAINING HEAT OPERATOR")
        print("="*50)
    
    # Create datasets for heat (diffusion) operator
    train_dataset = TrajectoryDataset(
        heat_files['train'],
        'heat',
        'train'
    )
    val_dataset = TrajectoryDataset(
        heat_files['valid'],
        'heat',
        'valid'
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Train heat operator
    heat_trainer = NeuralODETrainer(
        operators['diffusion'], device, learning_rate, num_epochs=num_epochs
    )
    
    heat_history = heat_trainer.train(
        train_loader, val_loader, num_epochs, method,
        save_path=os.path.join(save_dir, 'heat_model.pth'),
        verbose=verbose,
        use_adjoint=use_adjoint
    )
    
    results['heat'] = {
        'model': operators['diffusion'],
        'trainer': heat_trainer, 
        'history': heat_history
    }
    
    # Plot training histories
    if verbose:
        print("\nPlotting training histories...")
    
    fig_burgers = results['dispersion']['trainer'].plot_training_history(
        os.path.join(save_dir, 'dispersion_training.png')
    )
    
    fig_heat = results['heat']['trainer'].plot_training_history(
        os.path.join(save_dir, 'heat_training.png')
    )
    
    results['figures'] = {
        'burgers_training': fig_burgers,
        'heat_training': fig_heat
    }
    
    if verbose:
        print(f"\nTraining complete! Models saved to {save_dir}")
        print(f"Burgers best val loss: {dispersion_history['best_val_loss']:.6f}")
        print(f"Heat best val loss: {heat_history['best_val_loss']:.6f}")
    
    return results


if __name__ == "__main__":
    # Train neural operators
    print("Training Neural ODE Operators...")
    
    # Configuration - aligned with train/train.py
    config = {
        'nx': 256,
        'L': 16.0,
        'hidden_dim': 32,  # Smaller for faster training
        'n_layers': 2,
        'num_epochs': 100,
        'batch_size': 64,
        'learning_rate': 1e-3,
        'method': 'rk4',  # Start with simpler method
        'save_dir': '/mnt/home/lserrano/disco-ball/neural-operator-splitting/models'
    }
    
    results = train_neural_operators(**config)
    
    print("\nTraining completed!")
    print(f"Burgers coefficient learned: "
          f"{results['burgers']['model'].get_advection_coefficient():.4f}")
    print(f"Heat coefficient learned: "
          f"{results['heat']['model'].get_diffusion_coefficient():.4f}")
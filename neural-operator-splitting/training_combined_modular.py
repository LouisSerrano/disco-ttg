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
                          scenario: str = 'burgers_heat',
                          operator1_files: Dict = None,
                          operator2_files: Dict = None) -> Dict:
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
        operators[config['op1_model_key']], device, learning_rate, num_epochs=num_epochs
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
    
    # Train second operator
    if verbose:
        print("\n" + "="*50)
        print(f"TRAINING {config['op2_name'].upper()} OPERATOR")
        print("="*50)
    
    # Create datasets for second operator
    train_dataset = TrajectoryDataset(
        operator2_files['train'],
        config['op2_type'],
        'train'
    )
    val_dataset = TrajectoryDataset(
        operator2_files['valid'],
        config['op2_type'],
        'valid'
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Train second operator
    op2_trainer = NeuralODETrainer(
        operators[config['op2_model_key']], device, learning_rate, num_epochs=num_epochs
    )
    
    op2_history = op2_trainer.train(
        train_loader, val_loader, num_epochs, method,
        save_path=os.path.join(save_dir, f"{config['op2_name']}_model.pth"),
        verbose=verbose,
        use_adjoint=use_adjoint
    )
    
    results[config['op2_name']] = {
        'model': operators[config['op2_model_key']],
        'trainer': op2_trainer, 
        'history': op2_history
    }
    
    # Plot training histories
    if verbose:
        print("\nPlotting training histories...")
    
    fig_op1 = results[config['op1_name']]['trainer'].plot_training_history(
        os.path.join(save_dir, f"{config['op1_name']}_training.png")
    )
    
    fig_op2 = results[config['op2_name']]['trainer'].plot_training_history(
        os.path.join(save_dir, f"{config['op2_name']}_training.png")
    )
    
    results['figures'] = {
        f"{config['op1_name']}_training": fig_op1,
        f"{config['op2_name']}_training": fig_op2
    }
    
    if verbose:
        print(f"\nTraining complete! Models saved to {save_dir}")
        print(f"{config['op1_name'].capitalize()} best val loss: {op1_history['best_val_loss']:.6f}")
        print(f"{config['op2_name'].capitalize()} best val loss: {op2_history['best_val_loss']:.6f}")
    
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
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    print("Training Neural ODE Operators...")
    print(f"Configuration: {vars(args)}")
    
    # Prepare custom file paths if provided
    operator1_files = None
    operator2_files = None
    
    if args.op1_train or args.op1_valid:
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
        verbose=args.verbose
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
    
    if op2_name == 'heat':
        print(f"Heat coefficient learned: "
              f"{results[op2_name]['model'].get_diffusion_coefficient():.4f}")
    elif op2_name == 'dispersion':
        print(f"Dispersion coefficient learned: "
              f"{results[op2_name]['model'].get_advection_coefficient():.4f}")
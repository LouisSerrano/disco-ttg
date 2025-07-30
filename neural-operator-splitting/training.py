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


class TrajectoryDataset(Dataset):
    """Dataset for neural ODE training from trajectory data."""
    
    def __init__(self, 
                 trajectories: List[np.ndarray],
                 parameters: List[Dict],
                 time_points: np.ndarray,
                 operator_type: str,
                 sequence_length: int = 100):
        """
        Initialize trajectory dataset.
        
        Args:
            trajectories: List of trajectory arrays (nt+1, nx)
            parameters: List of parameter dictionaries
            time_points: Time points array
            operator_type: 'advection' or 'diffusion'
            sequence_length: Length of time sequences for training
        """
        self.trajectories = trajectories
        print('trajectories', len(trajectories))
        self.parameters = parameters
        self.time_points = time_points
        self.operator_type = operator_type
        self.sequence_length = sequence_length
        self.dt = time_points[1] - time_points[0]
        
        # Create training samples
        self.samples = []
        #self._create_samples()
    
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
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        key = "beta" if self.operator_type=="advection" else "D"
        return {
                    'u_sequence': torch.from_numpy(self.trajectories[idx]).float(),
                    't_sequence': torch.from_numpy(self.time_points).float(),
                    'parameter': torch.tensor(self.parameters[idx][key]).float(),
                    'traj_idx': idx
                }#self.samples[idx]


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
        
    def train_step(self, batch: Dict, method: str = 'rk4') -> float:
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
        t_seq = t_sequence[:, 1]  # (batch_size, remaining_steps) # we take the first timestamp

        target = u_sequence[:, start_idx+1:start_idx+2]  # (batch_size, remaining_steps, nx)
        
        n_steps, pred_trajectory = self.model(u0, T=t_seq[0], method=method)  # (remaining_steps, batch_size, nx)
        
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
    
    def validate(self, val_loader: DataLoader, method: str = 'rk4') -> float:
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
                
                n_steps, pred_trajectory = self.model(u0, T=t_seq[0], method=method)  # (remaining_steps, batch_size, nx)
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
              verbose: bool = True) -> Dict:
        """
        Train the neural ODE model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader  
            num_epochs: Number of training epochs
            method: ODE solver method
            save_path: Path to save best model
            verbose: Whether to print progress
            
        Returns:
            Training history dictionary
        """
        best_val_loss = float('inf')
        
        if verbose:
            print(f"Training neural ODE with {method} solver...")
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
                loss = self.train_step(batch, method)
                train_loss += loss
                num_batches += 1
                
                if verbose and isinstance(pbar, tqdm):
                    pbar.set_postfix({'loss': f'{loss:.6f}'})
            
            avg_train_loss = train_loss / max(num_batches, 1)
            self.train_losses.append(avg_train_loss)
            
            # Validation phase
            val_loss = self.validate(val_loader, method)
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
                          L: float = 2*np.pi,
                          beta: float = 1.0,
                          nu: float = 0.1,
                          hidden_dim: int = 64,
                          n_layers: int = 3,
                          num_epochs: int = 100,
                          batch_size: int = 8,
                          training_size: int = 8,
                          learning_rate: float = 1e-3,
                          method: str = 'rk4',
                          num_steps: int = 1,
                          device: str = 'auto',
                          save_dir: str = './models',
                          verbose: bool = True,
                          sequence_length=20,
                          dt=0.01,
                          T=1) -> Dict:
    """
    Train neural operators for advection and diffusion.
    
    Args:
        nx: Number of spatial grid points
        L: Domain length
        beta: Advection speed coefficient
        nu: Diffusion viscosity coefficient
        hidden_dim: Hidden layer dimension
        n_layers: Number of hidden layers
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        method: ODE solver method
        device: Device to use ('auto', 'cpu', 'cuda')
        save_dir: Directory to save models
        verbose: Whether to print progress
        
    Returns:
        Dictionary with trained models and histories
    """
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if verbose:
        print(f"Training neural operators on {device}")
        print(f"Grid size: {nx}, Domain length: {L}")
        print(f"Parameters: beta={beta}, nu={nu}")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate training data
    if verbose:
        print("Generating training data...")
    
    advection_data, diffusion_data, _ = generate_training_data(
        nx=nx, L=L, beta_values=[beta], nu_values=[nu], dt=dt, T=T, n_initial_conditions=training_size
    )
    #print('advection_data', advection_data['trajectories'])
    
    # Create models
    operators = create_neural_operators(nx, L, hidden_dim, n_layers, padding_mode="circular", num_steps=num_steps)
    
    results = {}
    
    # Train advection operator
    if verbose:
        print("\n" + "="*50)
        print("TRAINING ADVECTION OPERATOR")
        print("="*50)
    
    # Create datasets
    train_dataset = TrajectoryDataset(
        advection_data['trajectories'],
        advection_data['parameters'],
        advection_data['metadata']['time'],
        'advection',
        sequence_length=sequence_length
    )
    
    # Split into train/val
    print(f"Dataset size before split: {len(train_dataset)}")
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    print(f"Train size: {train_size}, Val size: {val_size}")
    original_dataset = train_dataset
    train_dataset = torch.utils.data.Subset(original_dataset, range(train_size))
    val_dataset = torch.utils.data.Subset(original_dataset, range(train_size, train_size + val_size))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Train advection operator
    advection_trainer = NeuralODETrainer(
        operators['advection'], device, learning_rate, num_epochs=num_epochs
    )
    
    advection_history = advection_trainer.train(
        train_loader, val_loader, num_epochs, method,
        save_path=os.path.join(save_dir, 'advection_model.pth'),
        verbose=verbose
    )
    
    results['advection'] = {
        'model': operators['advection'],
        'trainer': advection_trainer,
        'history': advection_history,
        "dataset": advection_data,
        "val_dataset": advection_data["trajectories"][train_size:train_size+val_size]
    }
    
    # Train diffusion operator
    if verbose:
        print("\n" + "="*50)
        print("TRAINING DIFFUSION OPERATOR")
        print("="*50)
    
    # Create datasets
    train_dataset = TrajectoryDataset(
        diffusion_data['trajectories'],
        diffusion_data['parameters'],
        diffusion_data['metadata']['time'],
        'diffusion',
        sequence_length=sequence_length
    )
    
    # Split into train/val  
    print(f"Dataset size before split: {len(train_dataset)}")
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    print(f"Train size: {train_size}, Val size: {val_size}")
    
    original_dataset = train_dataset
    train_dataset = torch.utils.data.Subset(original_dataset, range(train_size))
    val_dataset = torch.utils.data.Subset(original_dataset, range(train_size, train_size + val_size))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Train diffusion operator
    diffusion_trainer = NeuralODETrainer(
        operators['diffusion'], device, learning_rate, num_epochs=num_epochs
    )
    
    diffusion_history = diffusion_trainer.train(
        train_loader, val_loader, num_epochs, method,
        save_path=os.path.join(save_dir, 'diffusion_model.pth'),
        verbose=verbose
    )
    
    results['diffusion'] = {
        'model': operators['diffusion'],
        'trainer': diffusion_trainer, 
        'history': diffusion_history,
        "dataset": diffusion_data
    }
    
    # Plot training histories
    if verbose:
        print("\nPlotting training histories...")
    
    fig_adv = results['advection']['trainer'].plot_training_history(
        os.path.join(save_dir, 'advection_training.png')
    )
    
    fig_diff = results['diffusion']['trainer'].plot_training_history(
        os.path.join(save_dir, 'diffusion_training.png')
    )
    
    results['figures'] = {
        'advection_training': fig_adv,
        'diffusion_training': fig_diff
    }
    
    if verbose:
        print(f"\nTraining complete! Models saved to {save_dir}")
        print(f"Advection best val loss: {advection_history['best_val_loss']:.6f}")
        print(f"Diffusion best val loss: {diffusion_history['best_val_loss']:.6f}")
    
    return results


if __name__ == "__main__":
    # Train neural operators
    print("Training Neural ODE Operators...")
    
    # Configuration
    config = {
        'nx': 64,
        'L': 2*np.pi,
        'hidden_dim': 32,  # Smaller for faster training
        'n_layers': 2,
        'num_epochs': 50,
        'batch_size': 4,
        'learning_rate': 1e-3,
        'method': 'euler',  # Start with simpler method
        'save_dir': '/mnt/home/lserrano/disco-ball/neural-operator-splitting/models'
    }
    
    results = train_neural_operators(**config)
    
    print("\nTraining completed!")
    print(f"Advection coefficient learned: "
          f"{results['advection']['model'].get_advection_coefficient():.4f}")
    print(f"Diffusion coefficient learned: "
          f"{results['diffusion']['model'].get_diffusion_coefficient():.4f}")
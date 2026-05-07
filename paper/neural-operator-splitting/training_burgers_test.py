"""
Overfitting Test Module for Neural ODE Operators

This module tests the overfitting quality of neural operators on small
contiguous blocks of trajectories from the combined equation dataset.
It helps identify potential data quality issues.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional
import time
import os
import argparse
import pandas as pd

from neural_ode_operators import create_neural_operators
import h5py


class BlockTrajectoryDataset(Dataset):
    """Dataset for overfitting test on contiguous trajectory blocks."""
    
    def __init__(self, 
                 hdf5_file: str,
                 block_idx: int,
                 block_size: int = 16):
        """
        Initialize dataset for a specific block of trajectories.
        
        Args:
            hdf5_file: Path to HDF5 file
            block_idx: Block index (0 to num_blocks-1)
            block_size: Number of trajectories per block (default: 16)
        """
        self.hdf5_file = hdf5_file
        self.block_idx = block_idx
        self.block_size = block_size
        
        # Calculate trajectory indices for this block
        self.start_idx = block_idx * block_size
        self.end_idx = self.start_idx + block_size
        
        # Load data from HDF5 file
        with h5py.File(hdf5_file, 'r') as f:
            # The dataset has a 'train' key at the top level
            print('f', f)
            print('f.keys()', f.keys())
            train_data = f['train']
            self.trajectories = train_data['pde_250-256'][self.start_idx:self.end_idx]  # (block_size, n_timesteps, n_spatial)
            self.alpha = train_data['alpha'][self.start_idx:self.end_idx]
            self.beta = train_data['beta'][self.start_idx:self.end_idx]
            self.gamma = train_data['gamma'][self.start_idx:self.end_idx]
            
            # Store dataset info
            total_samples = train_data['pde_250-256'].shape[0]
            print(f"Block {block_idx}: trajectories {self.start_idx}-{self.end_idx-1} of {total_samples}")
            print(f"Alpha values in block: {self.alpha[0]:.4f} (should be constant)")
            
        self.n_samples, self.n_timesteps, self.n_spatial = self.trajectories.shape
        
        # Create time points (assuming uniform spacing)
        self.time_points = np.linspace(0, 4.0, self.n_timesteps)
        self.dt = self.time_points[1] - self.time_points[0]
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return {
            'u_sequence': torch.from_numpy(self.trajectories[idx]).float(),
            't_sequence': torch.from_numpy(self.time_points).float(),
            'alpha': torch.tensor(self.alpha[idx]).float(),
            'beta': torch.tensor(self.beta[idx]).float(),
            'gamma': torch.tensor(self.gamma[idx]).float(),
            'traj_idx': self.start_idx + idx
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
        
        _, seq_len = u_sequence.shape[:2]
        
        # Randomly sample time index for the batch
        max_start_idx = seq_len - 2  # Ensure we have at least one target step
        start_idx = torch.randint(0, max_start_idx + 1, (1,)).item()
        
        # Extract initial condition and target from sampled time index
        u0 = u_sequence[:, start_idx:start_idx+1]  # (batch_size, 1, nx)
        t_seq = t_sequence[:, 1]  # (batch_size, remaining_steps) # we take the first time

        target = u_sequence[:, start_idx+1:start_idx+2]  # (batch_size, remaining_steps, nx)
        
        _, pred_trajectory = self.model(u0, T=t_seq[0], method=method, use_adjoint=use_adjoint)  # (remaining_steps, batch_size, nx)
        
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
    
    def evaluate_overfitting(self, data_loader: DataLoader, method: str = 'rk4') -> Dict[str, float]:
        """Evaluate overfitting metrics on the training data."""
        self.model.eval()
        total_loss = 0.0
        num_samples = 0
        
        all_alpha_values = []
        
        with torch.no_grad():
            for batch in data_loader:
                u_sequence = batch['u_sequence'].to(self.device)
                t_sequence = batch['t_sequence'].to(self.device)
                alpha_values = batch['alpha'].cpu().numpy()
                all_alpha_values.extend(alpha_values)
        
                batch_size = u_sequence.shape[0]
                
                # Use first timestep as initial condition, predict next step
                u0 = u_sequence[:, 0:1]  # (batch_size, 1, nx)
                target = u_sequence[:, 1:2]  # (batch_size, 1, nx)
                
                # Time step for prediction
                dt = t_sequence[0, 1] - t_sequence[0, 0]
                
                _, pred_trajectory = self.model(u0, T=dt, method=method, use_adjoint=False)
                pred_trajectory = pred_trajectory[-1:]  # Take last timestep
                
                if pred_trajectory.dim() > 3:
                    pred_trajectory = pred_trajectory.squeeze(2)
                
                # Transpose target to match pred_trajectory shape
                target_transposed = target.transpose(0, 1)
                
                # Compute loss
                loss = self.criterion(pred_trajectory, target_transposed)
                
                total_loss += loss.item() * batch_size
                num_samples += batch_size
        
        avg_loss = total_loss / max(num_samples, 1)
        
        # Get unique alpha value (should be constant in block)
        unique_alphas = np.unique(all_alpha_values)
        alpha_value = unique_alphas[0] if len(unique_alphas) == 1 else np.mean(unique_alphas)
        
        return {
            'train_loss': avg_loss,
            'alpha_value': alpha_value,
            'num_samples': num_samples
        }
    
    def train_overfitting(self, 
                         train_loader: DataLoader,
                         num_epochs: int = 500,
                         method: str = 'rk4',
                         verbose: bool = True) -> Dict:
        """
        Train the neural ODE model for overfitting test.
        
        Args:
            train_loader: Training data loader
            num_epochs: Number of training epochs
            method: ODE solver method
            verbose: Whether to print progress
            
        Returns:
            Training history and metrics
        """
        if verbose:
            print(f"Overfitting test with {method} solver...")
            print(f"Device: {self.device}")
            print(f"Number of parameters: {sum(p.numel() for p in self.model.parameters())}")
        
        min_train_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            num_batches = 0
            
            for batch in train_loader:
                loss = self.train_step(batch, method, use_adjoint=False)
                train_loss += loss
                num_batches += 1
            
            avg_train_loss = train_loss / max(num_batches, 1)
            self.train_losses.append(avg_train_loss)
            
            # Track minimum loss
            if avg_train_loss < min_train_loss:
                min_train_loss = avg_train_loss
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Print progress every 50 epochs
            if verbose and (epoch + 1) % 50 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}: Train Loss = {avg_train_loss:.6f}, Min Loss = {min_train_loss:.6f}')
        
        # Final evaluation
        final_metrics = self.evaluate_overfitting(train_loader, method)
        
        return {
            'train_losses': self.train_losses,
            'final_train_loss': final_metrics['train_loss'],
            'min_train_loss': min_train_loss,
            'alpha_value': final_metrics['alpha_value'],
            'epochs_trained': num_epochs
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


def run_overfitting_experiments(
    K: int = 1,
    dataset_path: str = '/mnt/home/lserrano/disco-ttg/datasets/combined_equation/E_EULER_train_8192.h5',
    nx: int = 256,
    L: float = 16.0,
    hidden_dim: int = 32,
    n_layers: int = 2,
    num_epochs: int = 500,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    method: str = 'rk4',
    num_steps: int = 1,
    device: str = 'auto',
    save_models: bool = False,
    save_dir: str = './overfitting_models',
    output_csv: str = 'overfitting_test_results.csv',
    verbose: bool = True,
    seed: int = 42) -> pd.DataFrame:
    """
    Run K overfitting experiments on different trajectory blocks.
    
    Args:
        K: Number of independent experiments
        dataset_path: Path to HDF5 dataset
        nx: Number of spatial grid points
        L: Domain length
        hidden_dim: Hidden layer dimension
        n_layers: Number of hidden layers
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        method: ODE solver method
        num_steps: Number of ODE solver steps
        device: Device to use ('auto', 'cpu', 'cuda')
        save_models: Whether to save trained models
        save_dir: Directory to save models
        output_csv: Path to save results CSV
        verbose: Whether to print progress
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with experiment results
    """
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed(seed)
    
    # Calculate total number of blocks
    BLOCK_SIZE = 16
    with h5py.File(dataset_path, 'r') as f:
        print('f', f)
        print('f.keys()', f.keys())
        train_data = f['train']
        total_samples = train_data['pde_250-256'].shape[0]
    
    num_blocks = total_samples // BLOCK_SIZE
    
    if verbose:
        print(f"Running {K} overfitting experiments on {device}")
        print(f"Dataset: {dataset_path}")
        print(f"Total samples: {total_samples}, Blocks: {num_blocks}")
        print(f"Grid size: {nx}, Domain length: {L}")
        print(f"Model: {hidden_dim} hidden dims, {n_layers} layers")
        print(f"Training: {num_epochs} epochs, batch size {batch_size}, LR {learning_rate}")
    
    # Create save directory if needed
    if save_models:
        os.makedirs(save_dir, exist_ok=True)
    
    # Results storage
    results = []
    
    # Sample K random blocks
    block_indices = np.random.choice(num_blocks, size=K, replace=False)
    
    for exp_id, block_idx in enumerate(block_indices):
        if verbose:
            print(f"\n{'='*60}")
            print(f"EXPERIMENT {exp_id + 1}/{K} - Block {block_idx}")
            print(f"{'='*60}")
        
        # Create dataset for this block
        dataset = BlockTrajectoryDataset(dataset_path, block_idx, BLOCK_SIZE)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Create fresh model
        operators = create_neural_operators(nx, L, hidden_dim, n_layers, 
                                          padding_mode="circular", num_steps=num_steps)
        model = operators['advection']  # Using advection operator for Euler equation
        
        # Create trainer
        trainer = NeuralODETrainer(model, device, learning_rate, num_epochs=num_epochs)
        
        # Train for overfitting
        start_time = time.time()
        history = trainer.train_overfitting(dataloader, num_epochs, method, verbose)
        train_time = time.time() - start_time
        
        # Record results
        result = {
            'experiment_id': exp_id,
            'block_idx': block_idx,
            'start_idx': block_idx * BLOCK_SIZE,
            'end_idx': (block_idx + 1) * BLOCK_SIZE - 1,
            'alpha_value': history['alpha_value'],
            'final_train_loss': history['final_train_loss'],
            'min_train_loss': history['min_train_loss'],
            'epochs_trained': history['epochs_trained'],
            'train_time_seconds': train_time
        }
        results.append(result)
        
        # Save model if requested
        if save_models:
            model_path = os.path.join(save_dir, f'model_exp{exp_id}_block{block_idx}.pth')
            torch.save(model.state_dict(), model_path)
            result['model_path'] = model_path
        
        # Plot training curve
        if verbose:
            fig = trainer.plot_training_history(
                os.path.join(save_dir if save_models else '.', 
                           f'training_exp{exp_id}_block{block_idx}.png')
            )
            plt.close(fig)
        
        if verbose:
            print(f"\nExperiment {exp_id + 1} Results:")
            print(f"  Alpha value: {result['alpha_value']:.4f}")
            print(f"  Final train loss: {result['final_train_loss']:.6f}")
            print(f"  Min train loss: {result['min_train_loss']:.6f}")
            print(f"  Training time: {result['train_time_seconds']:.1f}s")
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"EXPERIMENT SUMMARY")
        print(f"{'='*60}")
        print(f"Results saved to: {output_csv}")
        print("\nStatistics:")
        print(f"  Mean final loss: {df['final_train_loss'].mean():.6f} ± {df['final_train_loss'].std():.6f}")
        print(f"  Mean min loss: {df['min_train_loss'].mean():.6f} ± {df['min_train_loss'].std():.6f}")
        print(f"  Alpha range: [{df['alpha_value'].min():.4f}, {df['alpha_value'].max():.4f}]")
    
    return df


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test overfitting quality of neural operators on trajectory blocks')
    
    parser.add_argument('--K', type=int, default=1,
                        help='Number of independent experiments (default: 1)')
    
    parser.add_argument('--dataset_path', type=str, 
                        default='/mnt/home/lserrano/disco-ttg/datasets/combined_equation/E_EULER_train_8192.h5',
                        help='Path to HDF5 dataset')
    
    parser.add_argument('--nx', type=int, default=256,
                        help='Number of spatial grid points (default: 256)')
    
    parser.add_argument('--L', type=float, default=16.0,
                        help='Domain length (default: 16.0)')
    
    parser.add_argument('--hidden_dim', type=int, default=32,
                        help='Hidden layer dimension (default: 32)')
    
    parser.add_argument('--n_layers', type=int, default=2,
                        help='Number of hidden layers (default: 2)')
    
    parser.add_argument('--num_epochs', type=int, default=500,
                        help='Number of training epochs (default: 500)')
    
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size (default: 4)')
    
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    
    parser.add_argument('--method', type=str, default='rk4',
                        choices=['rk4', 'dopri5', 'euler', 'ssp_rk3'],
                        help='ODE solver method (default: rk4)')
    
    parser.add_argument('--save_models', action='store_true',
                        help='Save trained models (default: False)')
    
    parser.add_argument('--save_dir', type=str, 
                        default='./overfitting_models',
                        help='Directory to save models')
    
    parser.add_argument('--output_csv', type=str, 
                        default='overfitting_test_results.csv',
                        help='Output CSV file for results')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    parser.add_argument('--num_steps', type=int, default=1,
                        help='Number of ODE solver steps (default: 1)')
    
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Print progress (default: True)')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    print("Neural Operator Overfitting Test")
    print("================================")
    print(f"Configuration: {vars(args)}")
    
    # Run overfitting experiments
    results_df = run_overfitting_experiments(
        K=args.K,
        dataset_path=args.dataset_path,
        nx=args.nx,
        L=args.L,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        method=args.method,
        num_steps=args.num_steps, # redo 10 steps
        device='auto',
        save_models=args.save_models,
        save_dir=args.save_dir,
        output_csv=args.output_csv,
        verbose=args.verbose,
        seed=args.seed
    )
    
    print("\nExperiments completed!")
    print(f"Results saved to: {args.output_csv}")
    
    # Display results summary
    print("\nResults Summary:")
    print(results_df)
"""
Simple training script for neural ODE operators.
Direct and straightforward implementation without complex abstractions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import os
import re
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt

from neural_ode_operators2d import WrapperNeuralODE


class SimpleDataset(Dataset):
    """Simple dataset for loading HDF5 trajectories."""
    
    def __init__(self, hdf5_file, split='train', train_ratio=0.8):
        with h5py.File(hdf5_file, 'r') as f:
            # Check if split exists as a group
            if split in f:
                group = f[split]
                # Try different possible keys
                if 'euler' in group:
                    all_trajectories = group['euler'][:]
                elif 'diffusion' in group:
                    all_trajectories = group['diffusion'][:]
                elif 'navier_stokes' in group:
                    all_trajectories = group['navier_stokes'][:]
                else:
                    # Fallback to old key if available
                    all_trajectories = group['pde_250-256'][:]
            else:
                # Load from single dataset and split
                # Try different possible keys at root level
                if 'euler' in f:
                    all_trajectories = f['euler'][:]
                elif 'diffusion' in f:
                    all_trajectories = f['diffusion'][:]
                elif 'navier_stokes' in f:
                    all_trajectories = f['navier_stokes'][:]
                else:
                    # Try to find the first dataset
                    keys = list(f.keys())
                    if keys:
                        all_trajectories = f[keys[0]][:]
                    else:
                        raise ValueError(f"No recognized datasets found in {hdf5_file}")
                
                # Split the data
                n_total = len(all_trajectories)
                n_train = int(n_total * train_ratio)
                
                if split == 'train':
                    all_trajectories = all_trajectories[:n_train]
                elif split == 'valid' or split == 'val':
                    all_trajectories = all_trajectories[n_train:]
                else:
                    raise ValueError(f"Unknown split: {split}")
        
        self.trajectories = all_trajectories
        self.n_samples = len(self.trajectories)
        self.time_points = np.linspace(0, 4.0, self.trajectories.shape[1])
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return {
            'trajectory': torch.tensor(self.trajectories[idx], dtype=torch.float32),
            'time_points': torch.tensor(self.time_points, dtype=torch.float32)
        }


def train_step(model, batch, optimizer, device, num_steps=1):
    """Single training step for 2D case."""
    model.train()
    optimizer.zero_grad()
    
    trajectories = batch['trajectory'].to(device)  # Shape: (batch_size, seq_len, height, width)
    time_points = batch['time_points'].to(device)
    
    batch_size, seq_len = trajectories.shape[:2]
    
    # Random time index for training
    start_idx = torch.randint(0, seq_len - num_steps, (1,)).item()
    
    # Initial condition
    u0 = trajectories[:, start_idx]  # Shape: (batch_size, height, width)
    # Time span for prediction
    t_span = time_points[0, start_idx:start_idx+num_steps+1] - time_points[0, start_idx]
    
    # Predict with neural ODE
    _, pred = model(u0, t_span=t_span)
    pred = pred[1:]  # Remove initial condition
    
    # Target
    target = trajectories[:, start_idx+1:start_idx+num_steps+1].transpose(0, 1)  # Shape: (num_steps, batch_size, height, width)
    
    # Compute loss
    diff = pred - target
    l2_diff = torch.norm(diff.flatten(start_dim=1), p=2, dim=1)
    l2_target = torch.norm(target.flatten(start_dim=1), p=2, dim=1)
    loss = torch.mean(l2_diff / (l2_target + 1e-8))
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    return loss.item()


def validate(model, val_loader, device, num_steps=1):
    """Validation step for 2D case."""
    model.eval()
    total_loss = 0
    num_samples = 0
    
    with torch.no_grad():
        for batch in val_loader:
            trajectories = batch['trajectory'].to(device)  # Shape: (batch_size, seq_len, height, width)
            time_points = batch['time_points'].to(device)
            
            # Use beginning of sequence
            u0 = trajectories[:, 0]  # Shape: (batch_size, height, width)
            
            # Time for prediction
            t_span = time_points[0, :num_steps+1] - time_points[0, 0]
            
            # Predict
            _, pred = model(u0, t_span=t_span)
            pred = pred[1:]
            
            # Target
            target = trajectories[:, 1:num_steps+1].transpose(0, 1)  # Shape: (num_steps, batch_size, height, width)
            
            # Loss
            diff = pred - target
            l2_diff = torch.norm(diff.flatten(start_dim=1), p=2, dim=1)
            l2_target = torch.norm(target.flatten(start_dim=1), p=2, dim=1)
            loss = torch.mean(l2_diff / (l2_target + 1e-8))

            total_loss += loss.item() * u0.shape[0]
            num_samples += u0.shape[0]
    
    return total_loss / num_samples


def train_model(train_file=None,
                val_file=None,
                num_epochs=100,
                batch_size=64,
                learning_rate=1e-3,
                device='cuda',
                save_dir='./models',
                model_name='model',
                train_ratio=0.8):
    """Train a neural ODE model."""
    
    # Create save directory based on parameters
    if "euler_trajectories" in train_file:
        param_dir = "euler2d"
    elif "diffusion_trajectories" in train_file:
        param_dir = "diffusion2d"
    else:
        param_dir = "navier_stokes2d"
    
    full_save_dir = os.path.join(save_dir, param_dir)
    os.makedirs(full_save_dir, exist_ok=True)
    
    save_path = os.path.join(full_save_dir, f"model.pth")
    
    print(f"Training model: {param_dir}")
    print(f"Model will be saved to: {save_path}")
    
    # Create model for 2D case
    model = WrapperNeuralODE(input_shape=(128, 128), L=16.0, hidden_dim=128, n_layers=4, padding_mode='circular')
    model = model.to(device)
    
    # Create datasets
    if val_file is None:
        # Use same file for both train and validation with splitting
        train_dataset = SimpleDataset(train_file, 'train', train_ratio=train_ratio)
        val_dataset = SimpleDataset(train_file, 'valid', train_ratio=train_ratio)
        print(f"Splitting {train_file} into train/validation with ratio {train_ratio:.1%}")
    else:
        # Use separate files
        train_dataset = SimpleDataset(train_file, 'train')
        val_dataset = SimpleDataset(val_file, 'valid')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Training history
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    # Checkpoint tracking
    checkpoint_interval = max(1, num_epochs // 10)
    checkpoint_info = []
    
    print(f"Training {model_name}...")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Training loop
    for epoch in range(num_epochs):
        # Train
        train_loss = 0
        num_samples = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            n_samples = batch['trajectory'].shape[0]
            loss = train_step(model, batch, optimizer, device)
            train_loss += loss * n_samples
            num_samples += n_samples
        
        avg_train_loss = train_loss / num_samples
        train_losses.append(avg_train_loss)
        
        # Print training loss every epoch
        print(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}')
        
        # Save checkpoint and validate at checkpoint intervals
        if (epoch + 1) % checkpoint_interval == 0 or epoch == 0:
            val_loss = validate(model, val_loader, device)
            val_losses.append(val_loss)
            
            print(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Val Loss = {val_loss:.6f}')
            
            # Save checkpoint with epoch info
            checkpoint_path = os.path.join(full_save_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss
            }, checkpoint_path)
            
            checkpoint_info.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
                'checkpoint_path': checkpoint_path
            })
            
            print(f'Checkpoint saved: {checkpoint_path}')
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_path)
                print(f'New best model saved! Val Loss: {best_val_loss:.6f}')
        
        scheduler.step()
    
    # Final save
    torch.save(model.state_dict(), save_path)
    
    # Save checkpoint information
    import json
    checkpoint_info_path = os.path.join(full_save_dir, "checkpoint_info.json")
    with open(checkpoint_info_path, 'w') as f:
        json.dump(checkpoint_info, f, indent=2)
    print(f"Checkpoint information saved to: {checkpoint_info_path}")
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train')
    if val_losses:
        val_epochs = [info['epoch']-1 for info in checkpoint_info]  # -1 for 0-based indexing
        plt.plot(val_epochs, val_losses, 'o-', label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title(f'{model_name} Training ({param_dir})')
    plt.tight_layout()
    
    plot_path = os.path.join(full_save_dir, f"{model_name}_training.png")
    plt.savefig(plot_path)
    print(f"Training plot saved to: {plot_path}")
    
    return model, train_losses, val_losses, best_val_loss


def main():
    parser = argparse.ArgumentParser(description='Simple Neural ODE Training')
    parser.add_argument('--train_file', type=str, required=True,
                        help='Path to training HDF5 file')
    parser.add_argument('--val_file', type=str, default=None,
                        help='Path to validation HDF5 file (if not provided, will split train_file)')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--save_dir', type=str, default='./models',
                        help='Base directory to save models')
    parser.add_argument('--model_name', type=str, default='burgers',
                        help='Name for the model (e.g., burgers, heat, dispersion)')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Ratio for train/validation split when using single file')
    
    args = parser.parse_args()
    
    # Train model
    model, train_losses, val_losses, best_val_loss = train_model(
        train_file=args.train_file,
        val_file=args.val_file,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        save_dir=args.save_dir,
        model_name=args.model_name,
        train_ratio=args.train_ratio
    )
    
    print(f"\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
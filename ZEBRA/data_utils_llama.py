import torch
import numpy as np
from torch.utils.data import IterableDataset, Dataset
from einops import rearrange
import random
from .data_utils import TemporalBatchDatasetFly, HDF5TemporalDataset, GrayScottDatasetWrapper


class RawDatasetWithContextFly(IterableDataset):
    """On-the-fly dataset wrapper for advection-diffusion that returns raw trajectories with context"""
    
    def __init__(self, base_dataset: TemporalBatchDatasetFly, 
                 sub_t=1, slice_size=10, num_context_trajectories=1):
        self.base_dataset = base_dataset
        self.sub_t = sub_t
        self.slice_size = slice_size
        self.num_context_trajectories = num_context_trajectories
        
    def __iter__(self):
        for batch in self.base_dataset:
            # batch shape: (batch_size, channels, height, time)
            batch_size = batch.shape[0]
            
            # Temporal slicing
            max_start_index = batch.shape[-1] - self.slice_size
            if max_start_index < 0:
                raise ValueError("Slice size is larger than the sequence length.")
            start_index = np.random.randint(0, max_start_index + 1)

            images = batch[..., start_index:start_index + self.slice_size]
               
            yield images, images.unsqueeze(1)


class RawDatasetWithContext(Dataset):
    """Wrapper for datasets that returns raw trajectories with context"""
    
    def __init__(self, base_dataset, sub_t=1, slice_size=10, 
                 num_context_trajectories=1, trajectories_per_environment=16):
        self.base_dataset = base_dataset
        self.sub_t = sub_t
        self.slice_size = slice_size
        self.num_context_trajectories = num_context_trajectories
        self.trajectories_per_environment = trajectories_per_environment
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # Get raw trajectory data
        data = self.base_dataset[idx]
        
        # Handle different data formats
        if isinstance(data, dict):
            # HDF5TemporalDataset returns dict
            trajectories = data
        else:
            # Direct tensor
            trajectories = data
        
        # Apply temporal subsampling
        trajectories = trajectories[..., ::self.sub_t]
        
        # Temporal slicing
        max_start_index = trajectories.shape[-1] - self.slice_size
        if max_start_index < 0:
            raise ValueError("Slice size is larger than the sequence length.")
        start_index = np.random.randint(0, max_start_index + 1)
        images = trajectories[..., start_index:start_index + self.slice_size]

        # Return empty tensor for context when num_context_trajectories=0
        if self.num_context_trajectories == 0:
            # Create empty context tensor with correct shape for collation
            # Shape: (0, C, H, W, T) for 2D or (0, C, H, T) for 1D
            empty_context = images.unsqueeze(0)[:0]  # Creates (0, ...) tensor
            return images, empty_context

        # Fallback: use the same trajectory as context
        context_images = images.unsqueeze(0)

        return images, context_images
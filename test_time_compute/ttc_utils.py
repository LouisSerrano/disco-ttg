"""Shared utilities for test-time compute: dataset/checkpoint loading, metrics, result saving."""
import torch
import os
from torch.utils.data import DataLoader
from src.utils.database import RelativeL2
from typing import Union, List, Dict
from torch.utils.data import Dataset
import logging
import time
import h5py
import numpy as np
import random
import matplotlib.pyplot as plt

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_INPUT_FRAMES = 10
N_OUTPUT_FRAMES = 10
SUB_X = 1
SUB_T = 1
BATCH_SIZE = 32


# File paths for datasets
VALIDATION_FILES = {
    'advection_diffusion': '/mnt/home/lserrano/disco-ttg/datasets/advection-diffusion/validation.h5',
    'combined_equation': '/mnt/home/lserrano/disco-ttg/datasets/combined_equation/validation.h5',
    'reaction_diffusion': '/mnt/home/lserrano/disco-ttg/datasets/reaction-diffusion/validation.h5'
}

TEST_FILES = {
    'advection_diffusion': '/mnt/home/lserrano/disco-ttg/datasets/advection-diffusion/test.h5',
    'combined_equation': '/mnt/home/lserrano/disco-ttg/datasets/combined_equation/test.h5',
    'reaction_diffusion': '/mnt/home/lserrano/disco-ttg/datasets/reaction-diffusion/test.h5'
}

TRAINING_FILES = {
    'advection_diffusion': '/mnt/home/lserrano/disco-ttg/datasets/advection-diffusion/training.h5',
    'combined_equation': '/mnt/home/lserrano/disco-ttg/datasets/combined_equation/training.h5',
    'reaction_diffusion': '/mnt/home/lserrano/disco-ttg/datasets/reaction-diffusion/training.h5'
}


def create_dataset_for_equation(equation_type, split='val', files_dict=None):
    """Create dataset for a specific equation type"""
    if split == 'val':
        files_dict = VALIDATION_FILES
    elif split == 'test':
        files_dict = TEST_FILES
    elif split == 'train':
        files_dict = TRAINING_FILES
    
    if equation_type not in files_dict:
        print(f"Equation type {equation_type} not found in files dict")
        return None, None
        
    file_path = files_dict[equation_type]
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None, None
    
    dataset = HDF5TemporalDataset(
        hdf5_files=[file_path],
        input_frames=N_INPUT_FRAMES,
        output_frames=N_OUTPUT_FRAMES,
        sub_x=SUB_X,
        sub_t=SUB_T,
        split=split
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"{equation_type} {split} dataset: {len(dataset)} samples")
    return dataloader, dataset


def get_relative_l2_error():
    """Get relative L2 error metric"""
    return RelativeL2()


def save_results(results, filename):
    """Save results to file"""
    import json
    import torch
    
    def convert_to_serializable(obj):
        """Convert PyTorch tensors to serializable format"""
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_to_serializable(v) for v in obj)
        else:
            return obj
    
    serializable_results = convert_to_serializable(results)
    with open(filename, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"Results saved to {filename}")


class GrayScottHDF5Dataset:
    """
    Efficient dataset class for loading Gray-Scott trajectories from HDF5 files.
    Supports multiple files and keeps file handles open for fast access.
    """
    
    def __init__(self, filenames: Union[str, List[str]], group_name: str = 'train', 
                 reshape_to_spatial: bool = True, keep_file_open: bool = True):
        """
        Initialize the dataset.
        
        Args:
            filenames: Path(s) to HDF5 file(s)
            group_name: HDF5 group name
            reshape_to_spatial: Whether to reshape flattened data to spatial dimensions
            keep_file_open: Whether to keep file handle open for faster access
        """
        print(f"DEBUG: GrayScottHDF5Dataset received filenames: {filenames} (type: {type(filenames)})")
        self.filenames = filenames if isinstance(filenames, list) else [filenames]
        print(f"DEBUG: self.filenames after processing: {self.filenames}")
        self.group_name = group_name
        self.reshape_to_spatial = reshape_to_spatial
        self.keep_file_open = keep_file_open
        
        # Build file index for efficient access
        print("Building file index for Gray-Scott dataset...")
        start_time = time.time()
        self.total_trajectories = self._build_file_index()
        index_time = time.time() - start_time
        print(f"Dataset length calculation took {index_time:.2f}s for {self.total_trajectories} trajectories")
        
        # Open file handles if requested
        self.file_handles = {}
        if keep_file_open:
            for file_path in self.filenames:
                if os.path.exists(file_path):
                    self.file_handles[file_path] = h5py.File(file_path, 'r')
    
    def _build_file_index(self):
        """Pre-compute file offsets for efficient __len__ and __getitem__"""
        self.file_offsets = []
        total_trajectories = 0
        self.n_x = None
        self.n_y = None
        self.n_t = None
        
        for file_path in self.filenames:
            if not os.path.exists(file_path):
                print(f"Warning: HDF5 file not found: {file_path}")
                continue
                
            try:
                with h5py.File(file_path, 'r') as f:
                    # Try to find the appropriate group
                    group_to_use = self.group_name
                    if self.group_name not in f:
                        # If requested group doesn't exist, try 'train' as fallback
                        if 'train' in f:
                            group_to_use = 'train'
                            print(f"Warning: Group '{self.group_name}' not found in {file_path}, using 'train' instead")
                        else:
                            print(f"Warning: Neither '{self.group_name}' nor 'train' group found in {file_path}")
                            print(f"Available groups: {list(f.keys())}")
                            continue
                    
                    # Get dataset info for this file
                    info = get_dataset_info(file_path, group_to_use)
                    n_trajectories = info['n_trajectories']
                    
                    # Store spatial dimensions (should be same for all files)
                    if self.n_x is None:
                        self.n_x = info['n_spatial_x']
                        self.n_y = info['n_spatial_y']
                        self.n_t = info['n_timesteps']
                    else:
                        # Verify dimensions match across files
                        if (self.n_x != info['n_spatial_x'] or 
                            self.n_y != info['n_spatial_y'] or 
                            self.n_t != info['n_timesteps']):
                            print(f"Warning: Dimensions mismatch in {file_path}. Skipping.")
                            continue
                    
                    self.file_offsets.append((file_path, total_trajectories, n_trajectories, group_to_use))
                    total_trajectories += n_trajectories
                    print(f"Added {n_trajectories} trajectories from {file_path} (group: {group_to_use})")
                    
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue
                
        if total_trajectories == 0:
            raise ValueError("No valid trajectories found in any HDF5 files!")
            
        return total_trajectories
    
    def _get_file_and_local_idx(self, idx):
        """Convert global index to file path, local index, and group name"""
        for entry in self.file_offsets:
            # Handle both old format (3-tuple) and new format (4-tuple)
            if len(entry) == 3:
                file_path, offset, n_trajectories = entry
                group_name = self.group_name  # Use default group name
            else:
                file_path, offset, n_trajectories, group_name = entry
            
            if idx < offset + n_trajectories:
                local_idx = idx - offset
                return file_path, local_idx, group_name
        raise IndexError(f"Index {idx} out of range for dataset size {self.total_trajectories}")
    
    def __len__(self):
        return self.total_trajectories
    
    def __getitem__(self, idx: Union[int, List[int]]) -> Dict:
        """
        Get trajectory(ies) by index.
        
        Args:
            idx: Single index or list of indices
            
        Returns:
            Dictionary with trajectory data
        """
        single_item = isinstance(idx, int)
        
        if single_item:
            # Single index case
            file_path, local_idx, group_name = self._get_file_and_local_idx(idx)
            
            # Get file handle or open temporarily
            if self.keep_file_open and file_path in self.file_handles:
                f = self.file_handles[file_path]
                close_file = False
            else:
                f = h5py.File(file_path, 'r')
                close_file = True
            
            try:
                group = f[group_name]
                a_data = group['trajectory_a'][local_idx]
                b_data = group['trajectory_b'][local_idx]
                f_values = group['f'][local_idx] if 'f' in group else None
                k_values = group['k'][local_idx] if 'k' in group else None
                init_cond = group['initial_conditions'][local_idx] if 'initial_conditions' in group else None
            finally:
                if close_file:
                    f.close()
        else:
            # Multiple indices case - need to handle indices from different files
            all_a_data = []
            all_b_data = []
            all_f_values = []
            all_k_values = []
            all_init_cond = []
            
            for i in idx:
                file_path, local_idx, group_name = self._get_file_and_local_idx(i)
                
                if self.keep_file_open and file_path in self.file_handles:
                    f = self.file_handles[file_path]
                    close_file = False
                else:
                    f = h5py.File(file_path, 'r')
                    close_file = True
                
                try:
                    group = f[group_name]
                    all_a_data.append(group['trajectory_a'][local_idx])
                    all_b_data.append(group['trajectory_b'][local_idx])
                    all_f_values.append(group['f'][local_idx] if 'f' in group else None)
                    all_k_values.append(group['k'][local_idx] if 'k' in group else None)
                    all_init_cond.append(group['initial_conditions'][local_idx] if 'initial_conditions' in group else None)
                finally:
                    if close_file:
                        f.close()
            
            # Stack the data
            a_data = np.stack(all_a_data)
            b_data = np.stack(all_b_data)
            f_values = np.stack(all_f_values) if all_f_values[0] is not None else None
            k_values = np.stack(all_k_values) if all_k_values[0] is not None else None
            init_cond = np.stack(all_init_cond) if all_init_cond[0] is not None else None
        
        # Reshape if requested
        if self.reshape_to_spatial:
            # Handle different data shapes
            if len(a_data.shape) == 2:
                # Single trajectory: (n_timesteps, n_spatial) -> (n_timesteps, n_x, n_y)
                n_t, n_spatial = a_data.shape
                a_data = a_data.reshape(n_t, self.n_x, self.n_y)
                b_data = b_data.reshape(n_t, self.n_x, self.n_y)
                
                if init_cond is not None:
                    if len(init_cond.shape) == 2:
                        n_spatial, n_channels = init_cond.shape
                        init_cond = init_cond.reshape(self.n_x, self.n_y, n_channels)
            elif len(a_data.shape) == 3:
                if single_item:
                    # Single item with batch dimension: (1, n_timesteps, n_spatial) -> squeeze and reshape
                    a_data = a_data.squeeze(0)
                    b_data = b_data.squeeze(0)
                    if init_cond is not None:
                        init_cond = init_cond.squeeze(0)
                    
                    n_t, n_spatial = a_data.shape
                    a_data = a_data.reshape(n_t, self.n_x, self.n_y)
                    b_data = b_data.reshape(n_t, self.n_x, self.n_y)
                    
                    if init_cond is not None:
                        n_spatial, n_channels = init_cond.shape
                        init_cond = init_cond.reshape(self.n_x, self.n_y, n_channels)
                else:
                    # Multiple trajectories: (n_batch, n_timesteps, n_spatial) -> (n_batch, n_timesteps, n_x, n_y)
                    n_batch, n_t, n_spatial = a_data.shape
                    a_data = a_data.reshape(n_batch, n_t, self.n_x, self.n_y)
                    b_data = b_data.reshape(n_batch, n_t, self.n_x, self.n_y)
                    
                    if init_cond is not None:
                        n_batch, n_spatial, n_channels = init_cond.shape
                        init_cond = init_cond.reshape(n_batch, self.n_x, self.n_y, n_channels)
        
        # Remove batch dimension for single items (parameters only)
        if single_item:
            if f_values is not None:
                f_values = f_values[0] if hasattr(f_values, '__len__') else f_values
                k_values = k_values[0] if hasattr(k_values, '__len__') else k_values
        
        return {
            'a': a_data,
            'b': b_data,
            'f': f_values,
            'k': k_values,
            'initial_conditions': init_cond
        }
    
    def get_batch(self, indices: List[int]) -> Dict:
        """Get multiple trajectories efficiently."""
        return self.__getitem__(indices)
    
    def close(self):
        """Close all file handles if open."""
        for file_path, file_handle in self.file_handles.items():
            if file_handle is not None:
                file_handle.close()
        self.file_handles = {}
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.close()



class GrayScottDatasetWrapper(Dataset):
    """Efficient wrapper using GrayScottHDF5Dataset with optimized file access."""
    
    def __init__(self, hdf5_files, split='train', input_frames=16, output_frames=2, 
                 sub_x=1, sub_t=1, trajectories_per_environment=512):
        logging.info(f"Initializing efficient dataset from {hdf5_files}")
        self.hdf5_files = hdf5_files if isinstance(hdf5_files, list) else [hdf5_files]
        self.split = split
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.sub_x = sub_x
        self.sub_t = sub_t
        self.trajectories_per_environment = trajectories_per_environment
        
        # Use the efficient GrayScottHDF5Dataset
        self.dataset = GrayScottHDF5Dataset(
            self.hdf5_files, 
            group_name=split,
            reshape_to_spatial=True,
            keep_file_open=True
        )
        
        self.n_samples = len(self.dataset)
        
        # Get time information from first file metadata
        first_file = self.hdf5_files[0] if self.hdf5_files else None
        if first_file and os.path.exists(first_file):
            with h5py.File(first_file, 'r') as f:
                if split in f:
                    self.n_timesteps = f[split].attrs.get('n_timesteps', 50)
                    logging.info(f"Found {self.n_samples} trajectories with {self.n_timesteps} timesteps each")
                else:
                    self.n_timesteps = 50
                    logging.info("Single trajectory file detected")
        else:
            self.n_timesteps = 50
            logging.info("Using default timestep count")
        
        # Generate time points based on the number of time steps
        self.time_points = torch.tensor(np.linspace(0, 50.0, self.n_timesteps), dtype=torch.float32)
        
        logging.info(f"Efficient dataset initialized with {self.n_samples} trajectories from {len(self.hdf5_files)} file(s)")
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        # Load using the efficient dataset
        data = self.dataset[idx]
        
        # Extract a and b channels and convert to tensors
        # Skip first timestep: data shape is (n_timesteps, n_x, n_y)
        a_full = data['a'][1:]  # Skip first timestep
        b_full = data['b'][1:]  # Skip first timestep
        
        # Apply temporal subsampling if specified
        if self.sub_t > 1:
            a_full = a_full[::self.sub_t]
            b_full = b_full[::self.sub_t]
        
        # Apply spatial subsampling if specified
        if self.sub_x > 1:
            a_full = a_full[:, ::self.sub_x, ::self.sub_x]
            b_full = b_full[:, ::self.sub_x, ::self.sub_x]
        
        # Convert to tensors
        a_tensor = torch.from_numpy(a_full).float()
        b_tensor = torch.from_numpy(b_full).float()
        
        # Sample temporal window randomly
        total_frames_needed = self.input_frames + self.output_frames
        max_start = a_tensor.shape[0] - total_frames_needed
        
        #print(f"DEBUG: idx={idx}, a_tensor.shape={a_tensor.shape}, total_frames_needed={total_frames_needed}, max_start={max_start}")
        
        if max_start <= 0:
            # If not enough frames, use what we have
            start_idx = 0
            available_frames = a_tensor.shape[0]
            actual_input_frames = min(self.input_frames, available_frames // 2)
            actual_output_frames = available_frames - actual_input_frames
            #print(f"DEBUG: Not enough frames - available={available_frames}, input={actual_input_frames}, output={actual_output_frames}")
        else:
            # Random temporal sampling
            start_idx = np.random.randint(0, max_start + 1)
            actual_input_frames = self.input_frames
            actual_output_frames = self.output_frames
            #print(f"DEBUG: Enough frames - start_idx={start_idx}, input={actual_input_frames}, output={actual_output_frames}")
        
        # Extract input and output sequences
        start_idx = 0
        input_end = start_idx + actual_input_frames
        #output_end = input_end + actual_output_frames
        
        a_input = a_tensor[start_idx:input_end]
        b_input = b_tensor[start_idx:input_end]

        # WARNING: this version does not assumes input and then output (output is after start_idx)
        max_start = a_tensor.shape[0] - actual_output_frames
        output_start = np.random.randint(start_idx, max_start + 1)
        output_end = output_start + actual_output_frames

        a_output = a_tensor[output_start:output_end]
        b_output = b_tensor[output_start:output_end]
        
        # Stack channels: (n_timesteps, n_channels, n_x, n_y)
        input_trajectory = torch.stack([a_input, b_input], dim=1)
        output_trajectory = torch.stack([a_output, b_output], dim=1)
        
        #print(f"DEBUG: Final shapes - input: {input_trajectory.shape}, output: {output_trajectory.shape}")
        
        # For compatibility with existing training code that expects 'trajectory'
        # we'll concatenate input and output
        #full_trajectory = torch.cat([input_trajectory, output_trajectory], dim=0)
        
        # Convert parameters to tensors
        f_val = torch.tensor(data['f'], dtype=torch.float32) if data['f'] is not None else torch.tensor(0.029, dtype=torch.float32)
        k_val = torch.tensor(data['k'], dtype=torch.float32) if data['k'] is not None else torch.tensor(0.057, dtype=torch.float32)
        
        # Calculate environment index based on global index
        # Assuming each environment has a certain number of trajectories
        environment_idx = idx // self.trajectories_per_environment
        
        return {
            #'trajectory': full_trajectory,
            'input': input_trajectory,
            'output': output_trajectory,
            'f': f_val,
            'k': k_val,
            'environment_idx': environment_idx,
            #'time_points': self.time_points[start_idx:output_end]
        }
    
    def close(self):
        """Close the underlying dataset file handle."""
        if hasattr(self.dataset, 'close'):
            self.dataset.close()


class CombinedHDF5TemporalDataset(Dataset):
    """Dataset for loading pre-computed trajectory data from HDF5 files"""
    
    def __init__(self, hdf5_files, input_frames=16, output_frames=16, 
                 sub_x=1, sub_t=1, split='train', trajectories_per_environment=16):
        """
        Args:
            hdf5_files: List of HDF5 file paths to load data from
            input_frames: Number of input time frames
            output_frames: Number of output time frames  
            sub_x: Spatial subsampling factor
            sub_t: Temporal subsampling factor
            split: Dataset split ('train', 'val', 'test')
            trajectories_per_environment: Number of trajectories per environment (default: 16)
        """
        self.hdf5_files = hdf5_files if isinstance(hdf5_files, list) else [hdf5_files]
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.sub_x = sub_x
        self.sub_t = sub_t
        self.split = split
        self.trajectories_per_environment = trajectories_per_environment
        
        # Build file index for efficient access
        print("Building file index...")
        start_time = time.time()
        self.total_samples = self._build_file_index()
        index_time = time.time() - start_time
        print(f"Dataset length calculation took {index_time:.2f}s for {self.total_samples} samples")
        
        # Track loading times for performance assessment
        self.loading_times = []
        
    def _build_file_index(self):
        """Pre-compute file offsets for efficient __len__ and __getitem__"""
        self.file_offsets = []
        total_samples = 0
        
        for file_path in self.hdf5_files:
            if not os.path.exists(file_path):
                print(f"Warning: HDF5 file not found: {file_path}")
                continue
                
            try:
                with h5py.File(file_path, 'r') as f:
                    # Try different group names based on split
                    data_group = None
                    dataset_path = None
                    
                    # Check for split-specific groups first, then fall back to 'train'
                    possible_groups = [self.split, 'train', 'valid', 'test']
                    for group_name in possible_groups:
                        if group_name in f and 'pde_250-256' in f[group_name]:
                            data_group = group_name
                            dataset_path = f'{group_name}/pde_250-256'
                            break
                    
                    if data_group is None:
                        print(f"Warning: No valid dataset structure found in {file_path}. Checked groups: {possible_groups}")
                        continue
                        
                    n_samples = f[dataset_path].shape[0]
                    n_timesteps = f[dataset_path].shape[1]
                    
                    # Verify we have enough timesteps for input + output frames
                    min_timesteps_needed = (self.input_frames + self.output_frames) * self.sub_t
                    if n_timesteps < min_timesteps_needed:
                        print(f"Warning: Not enough timesteps in {file_path}. "
                              f"Need {min_timesteps_needed}, got {n_timesteps}")
                        continue
                    
                    self.file_offsets.append((file_path, total_samples, n_samples, dataset_path))
                    total_samples += n_samples
                    print(f"Added {n_samples} samples from {file_path} (using {dataset_path})")
                    
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue
                
        if total_samples == 0:
            raise ValueError("No valid samples found in any HDF5 files!")
            
        return total_samples
    
    def _get_file_and_local_idx(self, idx):
        """Convert global index to file path and local index"""
        for file_path, offset, n_samples, dataset_path in self.file_offsets:
            if idx < offset + n_samples:
                local_idx = idx - offset
                return file_path, local_idx, dataset_path
        raise IndexError(f"Index {idx} out of range for dataset size {self.total_samples}")
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        start_time = time.time()
        # Find which file and local index
        file_path, local_idx, dataset_path = self._get_file_and_local_idx(idx)
        
        # Calculate environment index (each environment has trajectories_per_environment trajectories)
        environment_idx = idx // self.trajectories_per_environment
        
        min_target_index = (local_idx//self.trajectories_per_environment)*self.trajectories_per_environment
        max_target_index = min_target_index + self.trajectories_per_environment - 1
        target_local_idx = random.randint(min_target_index, max_target_index)
        
        try:
            with h5py.File(file_path, 'r') as f:
                # Get the group containing the data
                group_name = dataset_path.split('/')[0]
                
                # Load trajectory data - shape: (n_timesteps, n_spatial)
                trajectory = f[dataset_path][local_idx].copy()[::self.sub_t]
                target_trajectory = f[dataset_path][target_local_idx].copy()[::self.sub_t]
                
                # Load PDE parameters (alpha, beta, gamma) for this sample
                alpha = f[group_name]['alpha'][local_idx]
                beta = f[group_name]['beta'][local_idx]
                gamma = f[group_name]['gamma'][local_idx]
                
                # Sample temporal window randomly
                total_frames_needed = self.input_frames + self.output_frames
                max_start = trajectory.shape[0] - total_frames_needed
                if max_start <= 0:
                    # If not enough frames, use what we have
                    start_idx = 0
                    available_frames = trajectory.shape[0] 
                    actual_input_frames = min(self.input_frames, available_frames // 2)
                    actual_output_frames = available_frames - actual_input_frames
                else:
                    start_idx = np.random.randint(0, max_start + 1)
                    actual_input_frames = self.input_frames
                    actual_output_frames = self.output_frames
                
                # Apply temporal subsampling and extract sequences
                # we don't start at t=0, too hard
                start_t = 25 
                input_end_t = start_t + actual_input_frames 
                #output_end_t = input_end_t + actual_output_frames
                
                input_seq = trajectory[start_t:input_end_t, ::self.sub_x]

                # WARNING big change
                #output_start_t = np.random.randint(start_idx, trajectory.shape[0] - actual_output_frames + 1)
                output_seq = trajectory[input_end_t:input_end_t+actual_output_frames, ::self.sub_x]

                # for target, the start is after the frames from the context 
                start_t = np.random.randint(start_idx, max_start + 1)
                input_end_t = start_t + actual_input_frames 
                output_end_t = input_end_t + actual_output_frames
                
                target_input_seq = target_trajectory[start_t:input_end_t, ::self.sub_x]
                target_output_seq = target_trajectory[input_end_t:output_end_t, ::self.sub_x]

                target_input_tensor = torch.from_numpy(target_input_seq).unsqueeze(-2).float()
                target_output_tensor = torch.from_numpy(target_output_seq).unsqueeze(-2).float()
                
                # Add channel dimension and convert to torch tensors
                # Expected format: (time, channels, spatial)
                input_tensor = torch.from_numpy(input_seq).unsqueeze(-2).float()
                output_tensor = torch.from_numpy(output_seq).unsqueeze(-2).float()
                
                # Track loading time
                loading_time = time.time() - start_time
                if len(self.loading_times) < 1000:  # Collect first 1000 samples
                    self.loading_times.append(loading_time)
                
                return {
                    'input': input_tensor, 
                    'output': output_tensor,
                    'target_input': target_input_tensor,
                    'target_output': target_output_tensor,
                    'alpha': float(alpha),
                    'beta': float(beta),
                    'gamma': float(gamma),
                    'environment_idx': environment_idx
                }
                
        except Exception as e:
            print(f"Error loading sample {idx} from {file_path}: {e}")
            # Return dummy data to avoid training crash
            dummy_input = torch.zeros(self.input_frames, 1, 256 // self.sub_x)
            dummy_output = torch.zeros(self.output_frames, 1, 256 // self.sub_x)
            return {
                'input': dummy_input, 
                'output': dummy_output,
                'target_input': dummy_input,
                'target_output': dummy_output,
                'alpha': 0.0,
                'beta': 0.0,
                'gamma': 0.0,
                'environment_idx': environment_idx
            }
    
    def get_loading_stats(self):
        """Return loading performance statistics"""
        if not self.loading_times:
            return {}
        
        return {
            'avg_loading_time': np.mean(self.loading_times),
            'min_loading_time': np.min(self.loading_times),
            'max_loading_time': np.max(self.loading_times),
            'samples_per_second': 1.0 / np.mean(self.loading_times),
            'total_samples_timed': len(self.loading_times)
        }


def get_dataset_info(filename: str, group_name: str) -> Dict:
    """Get dataset information from HDF5 file."""
    with h5py.File(filename, 'r') as f:
        if group_name not in f:
            raise ValueError(f"Group '{group_name}' not found in {filename}")
        
        group = f[group_name]
        info = {
            'n_trajectories': group.attrs.get('n_trajectories', len(group['trajectory_a'])),
            'n_spatial_x': group.attrs.get('n_spatial_x', 128),
            'n_spatial_y': group.attrs.get('n_spatial_y', 128),
            'n_timesteps': group.attrs.get('n_timesteps', group['trajectory_a'].shape[1])
        }
    return info



def plot_results(trajectories, predictions, errors, save_path, num_plots=5):
    """Plot prediction results with enhanced 4-panel visualization."""
    num_plots = min(num_plots, len(trajectories))
    
    # Reshape data if needed - trajectories should be (batch, time, space)
    if len(trajectories.shape) == 2:
        # Reshape from (batch, time*space) back to (batch, time, space)
        nt_steps = predictions.shape[1] // trajectories.shape[-1] + 1  # +1 for initial condition
        nx = trajectories.shape[-1] // nt_steps
        trajectories = trajectories.reshape(trajectories.shape[0], nt_steps, nx)
    
    # Reshape predictions similarly if needed
    if len(predictions.shape) == 2:
        predictions = predictions.reshape(predictions.shape[0], -1, predictions.shape[-1] // trajectories.shape[-1])
    
    for idx in range(num_plots):
        # Get ground truth and prediction for this sample
        ground_truth = trajectories[idx]
        result = predictions[idx]
        
        # Enhanced visualization with 4 panels
        plt.figure(figsize=(12, 10))
        
        # Sample every 10 time steps for visualization
        time_indices = np.arange(0, len(result), max(1, len(result)//10))
        colors = plt.cm.viridis(np.linspace(0, 1, len(time_indices)))
        
        # 1. Ground Truth trajectories every 10 time steps
        plt.subplot(2, 2, 1)
        for i, t_idx in enumerate(time_indices):
            if t_idx < len(ground_truth):
                plt.plot(ground_truth[t_idx, :], '-', color=colors[i], 
                        linewidth=1.5, alpha=0.8)
        
        plt.title("Ground Truth Evolution\\n(Sampled Time Steps)")
        plt.xlabel("Space")
        plt.ylabel("u")
        plt.grid(True, alpha=0.3)
        # Create a colorbar for the time evolution
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=len(time_indices)-1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca(), label='Time Index', shrink=0.8)
        
        # 2. Neural predictions every 10 time steps  
        plt.subplot(2, 2, 2)
        for i, t_idx in enumerate(time_indices):
            if t_idx < len(result):
                plt.plot(result[t_idx], '-', color=colors[i], 
                        linewidth=1.5, alpha=0.8)
        
        plt.title("Neural Predictions\\n(Sampled Time Steps)")
        plt.xlabel("Space")
        plt.ylabel("u")
        plt.grid(True, alpha=0.3)
        # Create a colorbar for the time evolution
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=len(time_indices)-1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca(), label='Time Index', shrink=0.8)
        
        # 3. Spatiotemporal evolution comparison
        plt.subplot(2, 2, 3)
        plt.imshow(ground_truth.T, aspect='auto', origin='lower', cmap='plasma')
        plt.title("Ground Truth\\nSpatiotemporal")
        plt.xlabel("Time")
        plt.ylabel("Space")
        plt.colorbar(shrink=0.8)
        
        # 4. Error evolution and final comparison
        plt.subplot(2, 2, 4)
        # Calculate error at each time step
        time_errors = []
        min_len = min(len(ground_truth), len(result))
        for i in range(min_len):
            error = np.linalg.norm(ground_truth[i, :] - result[i]) / (np.linalg.norm(ground_truth[i, :]) + 1e-8)
            time_errors.append(error)
        
        time_errors = np.array(time_errors)
        
        # Plot error on primary axis
        ax1 = plt.gca()
        line1 = ax1.semilogy(time_errors, 'r-', linewidth=2, label='Relative L2 Error')
        ax1.set_xlabel("Time Step")
        ax1.set_ylabel("Relative L2 Error", color='r')
        ax1.tick_params(axis='y', labelcolor='r')
        ax1.grid(True, alpha=0.3)
        
        # Add final comparison as secondary axis
        ax2 = ax1.twinx()
        ax2.plot(ground_truth[-1, :], 'b-', label='GT Final', linewidth=2, alpha=0.7)
        ax2.plot(result[-1], 'r--', label='Pred Final', linewidth=2, alpha=0.7)
        ax2.set_ylabel("Final State", color='b')
        ax2.tick_params(axis='y', labelcolor='b')
        
        ax1.set_title("Error Evolution &\\nFinal Comparison")
        
        # Add legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
        
        plt.tight_layout()
        plt.suptitle(f"Neural Operator Splitting Results - Sample {idx+1}\\nRelative L2 Error: {errors:.6f}", y=1.02)
        
        # Save individual figure
        sample_save_path = save_path.replace('.png', f'_sample_{idx+1:03d}.png')
        plt.savefig(sample_save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Print error statistics for this sample
        final_error = np.linalg.norm(ground_truth[-1, :] - result[-1]) / (np.linalg.norm(ground_truth[-1, :]) + 1e-8)
        print(f"Sample {idx+1} - Final Relative L2 Error: {final_error:.6f}")
        print(f"Sample {idx+1} - Max relative error over time: {time_errors.max():.6f}")
        print(f"Sample {idx+1} - Mean relative error over time: {time_errors.mean():.6f}")
    
    print(f"Enhanced plots saved with pattern: {save_path.replace('.png', '_sample_*.png')}")
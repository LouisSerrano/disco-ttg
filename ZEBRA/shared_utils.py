"""Shared utilities used by both ZEBRA and train modules to avoid circular imports."""

import numpy as np
import h5py
import os
import time
from typing import Union, List, Tuple, Dict


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

def advection_diffusion_analytical(u0, L=16.0, v=0.1, D=0.5, nt=100, T=10.0):
    """
    Compute the analytical solution of the 1D advection-diffusion equation
    with periodic boundary conditions using the Fourier spectral method.

    Parameters:
        u0 (ndarray): Initial condition, array of shape (nx,)
        L (float): Domain length
        v (float): Advection speed
        D (float): Diffusion coefficient
        nt (int): Number of time steps
        T (float): Final time

    Returns:
        u_xt (ndarray): Solution array of shape (nt, nx)
        x (ndarray): Spatial grid of shape (nx,)
        t (ndarray): Time grid of shape (nt,)
    """
    nx = len(u0)  # infer spatial resolution from input
    x = np.linspace(0, L, nx, endpoint=False)
    t = np.linspace(0, T, nt)

    # Fourier wavenumbers
    k = np.fft.fftfreq(nx, d=L / nx) * 2 * np.pi
    k = 1j * k  # complex wavenumber for exponential form

    # FFT of initial condition
    u0_hat = np.fft.fft(u0)

    # Allocate solution array
    u_xt = np.zeros((nt, nx))

    # Time evolution in spectral space
    for i, ti in enumerate(t):
        decay = np.exp(D * (k**2) * ti) * np.exp(-k * v * ti)
        u_hat_t = u0_hat * decay
        u_xt[i] = np.fft.ifft(u_hat_t).real  # keep only real part

    return u_xt, x, t


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
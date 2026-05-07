from baselines.ZEBRA.shared_utils import advection_diffusion_analytical, GrayScottHDF5Dataset
import h5py
import time
import numpy as np
import torch
from torch.utils.data import IterableDataset, Dataset
import random
import os
import logging
from einops import rearrange
from src.utils.advection_diffusion import Fractaloid, FractaloidPhase, AdvectionDiffusionExplicit


class TemporalBatchDatasetFly(IterableDataset):
    def __init__(self, n_batches, batch_size, sub_x, sub_t, split='train', input_frames=16, output_frames=16,
                 L=16.0, nx=256, nt=100, T=10.0,
                 v_range=(0.01, 1.0), D_range=(0.01, 1.0),
                 fractal_degree=8, fractal_power_range=2, seed=None, in_context=True, num_envs=200):
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.sub_x = sub_x
        self.sub_t = sub_t
        self.split = split
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.L = L
        self.nx = nx
        self.nt = nt
        self.T = T
        self.v_range = v_range
        self.D_range = D_range
        self.fractal_degree = fractal_degree
        self.fractal_power_range = fractal_power_range
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.in_context = in_context
        self.num_envs = num_envs

    def __iter__(self):
        for _ in range(self.n_batches):
            #input_frames = self.rng.integers(2, self.input_frames + 1)
            input_frames = self.input_frames
            batch_inputs = []
            batch_targets = []
            batch_context_inputs = []
            batch_context_targets = []
            batch_advection_speed = []
            batch_diffusion = []
            env_indices = []
            for _ in range(self.batch_size):
                # Sample advection speed and viscosity
                if self.split == 'train':
                    if random.random() < 0.5:
                        v = self.rng.uniform(*self.v_range) if isinstance(self.v_range, (tuple, list)) else float(self.v_range)
                        D = 0
                        # Map v to an environment index (0 to num_v_envs-1)
                        v_min, v_max = self.v_range
                        # Normalize v to [0, 1], then scale to number of v environments
                        normalized_v = (v - v_min) / (v_max - v_min)
                        env = int(normalized_v * (self.num_envs // 2))
                        # Ensure env doesn't exceed bounds due to floating point
                        env = min(env, self.num_envs // 2 - 1)
                        env_indices.append(torch.tensor([env]))
                    else:
                        v = 0
                        D = self.rng.uniform(*self.D_range) if isinstance(self.D_range, (tuple, list)) else float(self.D_range)
                        # Map D to an environment index (num_v_envs to total_envs-1)
                        D_min, D_max = self.D_range
                        normalized_D = (D - D_min) / (D_max - D_min)
                        env = int(normalized_D * (self.num_envs // 2))
                        env = min(env, self.num_envs // 2 - 1)
                        env_indices.append(torch.tensor([self.num_envs // 2 + env]))# the envs are split between advection and diffusion
                else:
                    v = self.rng.uniform(*self.v_range) if isinstance(self.v_range, (tuple, list)) else float(self.v_range)
                    D = self.rng.uniform(*self.D_range) if isinstance(self.D_range, (tuple, list)) else float(self.D_range)

                batch_advection_speed.append(v)
                batch_diffusion.append(D)

                # Generate fractaloid initial condition
                fractal_power = self.rng.uniform(*self.fractal_power_range) if isinstance(self.fractal_power_range, (tuple, list)) else float(self.fractal_power_range)
                fractaloid = FractaloidPhase(
                    degree=self.fractal_degree,
                    power=fractal_power,
                    size=self.nx,
                    patch_size=self.nx
                )
                u0 = fractaloid.generate(batch_size=1, seed=None).squeeze(0).numpy()
                u0 = (u0 - u0.mean()) / (u0.std() + 1e-8)
                u_xt, x, t = advection_diffusion_analytical(
                    u0, L=self.L, v=v, D=D, nt=self.nt, T=self.T
                )
                u_xt = u_xt[::self.sub_t, ::self.sub_x]
                max_start_index_input = u_xt.shape[0] - input_frames - self.output_frames
                if max_start_index_input < 0:
                    raise ValueError("Input frames size is larger than the sequence length.")
                start_index_enc = self.rng.integers(0, max_start_index_input + 1)
                input = u_xt[start_index_enc:start_index_enc + input_frames].copy()
                #max_start_index_target = u_xt.shape[0] - self.output_frames
                #if max_start_index_target < 0:
                #    raise ValueError("Output frames size is larger than the sequence length.")
                #start_index_dec = self.rng.integers(0, max_start_index_target + 1)
                #target = u_xt[start_index_dec:start_index_dec + self.output_frames].copy()
                target = u_xt[start_index_enc + input_frames: start_index_enc + input_frames + self.output_frames].copy()
                batch_inputs.append(torch.from_numpy(input).unsqueeze(-2).float())
                batch_targets.append(torch.from_numpy(target).unsqueeze(-2).float())

                # Second trajectory (context)
                fractal_power_ctx = self.rng.uniform(*self.fractal_power_range) if isinstance(self.fractal_power_range, (tuple, list)) else float(self.fractal_power_range)
                fractaloid_ctx = Fractaloid(
                    degree=self.fractal_degree,
                    power=fractal_power_ctx,
                    size=self.nx,
                    patch_size=self.nx
                )
                u0_ctx = fractaloid_ctx.generate(batch_size=1, seed=None).squeeze(0).numpy()
                u0_ctx = (u0_ctx - u0_ctx.mean()) / (u0_ctx.std() + 1e-8)
                u_xt_ctx, x_ctx, t_ctx = advection_diffusion_analytical(
                    u0_ctx, L=self.L, v=v, D=D, nt=self.nt, T=self.T
                )
                u_xt_ctx = u_xt_ctx[::self.sub_t, ::self.sub_x]
                max_start_index_input_ctx = u_xt_ctx.shape[0] - input_frames
                if max_start_index_input_ctx < 0:
                    raise ValueError("Input frames size is larger than the sequence length (context).")
                start_index_enc_ctx = self.rng.integers(0, max_start_index_input_ctx + 1)
                input_ctx = u_xt_ctx[start_index_enc_ctx:start_index_enc_ctx + input_frames].copy()
                #input_ctx = u_xt_ctx[start_index_enc:start_index_enc + input_frames].copy()
                max_start_index_target_ctx = u_xt_ctx.shape[0] - self.output_frames
                if max_start_index_target_ctx < 0:
                    raise ValueError("Output frames size is larger than the sequence length (context).")
                start_index_dec_ctx = self.rng.integers(0, max_start_index_target_ctx + 1)
                target_ctx = u_xt_ctx[start_index_dec_ctx:start_index_dec_ctx + self.output_frames].copy()
                #target_ctx = u_xt_ctx[start_index_enc + input_frames: start_index_enc + input_frames + self.output_frames].copy()
                batch_context_inputs.append(torch.from_numpy(input_ctx).unsqueeze(-2).float())
                batch_context_targets.append(torch.from_numpy(target_ctx).unsqueeze(-2).float())

            batch_inputs = torch.stack(batch_inputs)
            batch_targets = torch.stack(batch_targets)

            trajectories = torch.cat([batch_inputs, batch_targets], axis=1)
            trajectories = rearrange(trajectories, 'b t c h -> b c h t')

            #batch = {
            #    'input': torch.stack(batch_inputs),
            #    'target': torch.stack(batch_targets),
            #    'context_input': torch.stack(batch_context_inputs),
            #    'context_target': torch.stack(batch_context_targets),
            #    'advection_speed': batch_advection_speed,
            #    'diffusion': batch_diffusion,
            #    "env": torch.cat(env_indices)
            #}
            yield trajectories


def get_hdf5_files(cfg, split_name):
        """Get HDF5 files for a specific split (train/val/test)"""
        # Try to get split-specific files first
        split_key = f'{split_name}_hdf5_files'
        if hasattr(cfg.data, split_key):
            files = getattr(cfg.data, split_key)
            if files:
                return list(files) if not isinstance(files, str) else [files]
        
        # Fall back to general hdf5_files for backward compatibility
        if hasattr(cfg.data, 'hdf5_files'):
            files = cfg.data.hdf5_files
            return list(files) if not isinstance(files, str) else [files]
        
        # Default fallback
        #default_files = ["/mnt/home/lserrano/MP-Neural-PDE-Solvers/data/E_EULER_train_1024.h5"]
        print(f"Warning: No HDF5 files specified for {split_name}, using default: {default_files}")


def get_hdf5_files_gs(cfg, split_name):
        """Get HDF5 files for a specific split (train/val/test)"""
        print(f"DEBUG: Getting HDF5 files for split: {split_name}")
        
        # Try to get split-specific files first
        split_key = f'{split_name}_hdf5_files'
        print(f"DEBUG: Looking for config key: {split_key}")
        
        if hasattr(cfg.data, split_key):
            files = getattr(cfg.data, split_key)
            print(f"DEBUG: Found {split_key} = {files} (type: {type(files)})")
            if files:
                result = list(files) if not isinstance(files, str) else [files]
                print(f"DEBUG: Returning split-specific files: {result}")
                return result
        
        # Fall back to general hdf5_files for backward compatibility
        if hasattr(cfg.data, 'hdf5_files'):
            files = cfg.data.hdf5_files
            print(f"DEBUG: Found general hdf5_files = {files} (type: {type(files)})")
            result = list(files) if not isinstance(files, str) else [files]
            print(f"DEBUG: Returning general files: {result}")
            return result
        print(f"Warning: No HDF5 files specified for {split_name}")

class HDF5TemporalDataset(Dataset):
    """Dataset for loading pre-computed trajectory data from HDF5 files"""
    
    def __init__(self, hdf5_files, input_frames=16, output_frames=16, 
                 sub_x=1, sub_t=1, split='train', trajectories_per_environment=16, mode="train"):
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
        self.mode = mode
        
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
                if self.mode == "train":
                    start_t = start_idx 
                elif self.mode == "test":
                    start_t = 25
                else:
                    raise ValueError

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

                
                trajectories = torch.cat([input_tensor, output_tensor], axis=0)
                trajectories = rearrange(trajectories, "t c h -> c h t")
                
                #return {
                #    'input': input_tensor, 
                #    'output': output_tensor,
                #    'target_input': target_input_tensor,
                #    'target_output': target_output_tensor,
                #    'alpha': float(alpha),
                #    'beta': float(beta),
                #    'gamma': float(gamma),
                #    'environment_idx': environment_idx
                #}
                return trajectories
                
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




class GrayScottDatasetWrapper(Dataset):
    """Efficient wrapper using GrayScottHDF5Dataset with optimized file access."""
    
    def __init__(self, hdf5_files, split='train', input_frames=16, output_frames=2, 
                 sub_x=1, sub_t=1, trajectories_per_environment=512, mode="train"):
        logging.info(f"Initializing efficient dataset from {hdf5_files}")
        self.hdf5_files = hdf5_files if isinstance(hdf5_files, list) else [hdf5_files]
        self.split = split
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.sub_x = sub_x
        self.sub_t = sub_t
        self.trajectories_per_environment = trajectories_per_environment
        self.mode = mode
        
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

        environment_idx = idx // self.trajectories_per_environment
        
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

        if self.mode == "test":
            start_idx = 0
        # Extract input and output sequences
        input_end = start_idx + actual_input_frames
        output_end = input_end + actual_output_frames
        
        a_input = a_tensor[start_idx:input_end]
        b_input = b_tensor[start_idx:input_end]

        # WARNING: this version does not assumes input and then output (output is after start_idx)

        a_output = a_tensor[input_end:output_end]
        b_output = b_tensor[input_end:output_end]
        
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

        trajectories = torch.cat([input_trajectory, output_trajectory], axis=0)
        trajectories = rearrange(trajectories, 't c h w -> c h w t')

        return trajectories
        
        #return {
        #    #'trajectory': full_trajectory,
        #    'input': input_trajectory,
        #    'output': output_trajectory,
        #    'f': f_val,
        #    'k': k_val,
        #    'environment_idx': environment_idx
            #'time_points': self.time_points[start_idx:output_end]
        #}
    
    def close(self):
        """Close the underlying dataset file handle."""
        if hasattr(self.dataset, 'close'):
            self.dataset.close()

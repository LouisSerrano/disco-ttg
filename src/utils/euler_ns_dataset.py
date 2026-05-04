"""
Dataset classes for Euler/Diffusion/Navier-Stokes trajectory data.

Loads directly from split GPU files (no merge required).

Training/Validation: Euler + Diffusion trajectories
Testing: Navier-Stokes trajectories
"""
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from pathlib import Path
from typing import List, Union, Optional


class MultiFileHDF5Dataset:
    """
    Base class for loading trajectories from multiple HDF5 files.
    Keeps file handles open for fast access.
    """

    def __init__(self, file_paths: List[str], keep_file_open: bool = True):
        self.file_paths = [str(p) for p in file_paths]
        self.keep_file_open = keep_file_open
        self.file_handles = {}

        if keep_file_open:
            for path in self.file_paths:
                self.file_handles[path] = h5py.File(path, 'r')

    def _get_file(self, path: str):
        """Get file handle, opening if necessary."""
        if self.keep_file_open:
            return self.file_handles[path], False
        return h5py.File(path, 'r'), True

    def close(self):
        """Close all file handles."""
        for f in self.file_handles.values():
            f.close()
        self.file_handles = {}

    def __del__(self):
        self.close()


class EulerDiffusionDataset(MultiFileHDF5Dataset, Dataset):
    """
    Dataset for training/validation on Euler and Diffusion trajectories.
    Loads directly from split GPU files.

    Labels:
        - 0: Euler (nu=0)
        - 1, 2, ..., m_visc: Diffusion with viscosity index

    Args:
        file_dir: Directory containing trajectories_gpu*.h5 files
        num_gpus: Number of GPU files
        split: 'train' or 'val'
        val_fraction: Fraction of data for validation
        seed: Random seed for train/val split
        filter_labels: Optional list of labels to include in the dataset. If None, all labels are included.
    """

    def __init__(self, file_dir: str, num_gpus: int, split: str = 'train',
                 val_fraction: float = 0.1, seed: int = 42, 
                 filter_labels: Optional[List[int]] = None):
        # Build file paths
        file_dir = Path(file_dir)
        file_paths = [file_dir / f"trajectories_gpu{i}.h5" for i in range(num_gpus)]

        super().__init__(file_paths)

        self.split = split
        self.filter_labels = filter_labels

        # Read metadata from first file
        f0, should_close = self._get_file(str(file_paths[0]))
        self.viscosities = f0.attrs['viscosities'][:]
        self.n_snapshots = f0.attrs['n_snapshots']
        self.save_res = f0.attrs['save_res']
        self.m_visc = len(self.viscosities)
        if should_close:
            f0.close()

        # Build index: list of (file_path, dataset_name, local_idx, label)
        all_indices = []

        for path in self.file_paths:
            f, should_close = self._get_file(str(path))

            euler_count = f.attrs['euler_count']
            diff_count = f.attrs['diff_count']
            n_diff_per_visc = diff_count // self.m_visc

            # Euler samples (label=0)
            if filter_labels is None or 0 in filter_labels:
                for i in range(euler_count):
                    all_indices.append((str(path), 'euler', i, 0))

            # Diffusion samples: stored as [visc0][visc1]...[visc_m] in each file
            for v_idx in range(self.m_visc):
                label = v_idx + 1
                if filter_labels is None or label in filter_labels:
                    start = v_idx * n_diff_per_visc
                    for i in range(n_diff_per_visc):
                        all_indices.append((str(path), 'diffusion', start + i, label))

            if should_close:
                f.close()

        # Train/val split
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(all_indices))
        n_val = int(len(all_indices) * val_fraction)

        if split == 'val':
            selected = indices[:n_val]
        else:
            selected = indices[n_val:]

        self.indices = [all_indices[i] for i in selected]
        
        filter_msg = f" (filtered by {filter_labels})" if filter_labels is not None else ""
        print(f"EulerDiffusionDataset [{split}]{filter_msg}: {len(self.indices)} samples "
              f"({len([x for x in self.indices if x[3] == 0])} euler, "
              f"{len([x for x in self.indices if x[3] > 0])} diffusion)")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        file_path, dataset_name, local_idx, label = self.indices[idx]

        f, should_close = self._get_file(file_path)
        try:
            trajectory = f[dataset_name][local_idx]  # (n_snapshots, H, W)
        finally:
            if should_close:
                f.close()

        return torch.from_numpy(trajectory), label

    def get_viscosity(self, label):
        """Convert label to viscosity value. Label 0 (Euler) returns 0.0."""
        if label == 0:
            return 0.0
        return self.viscosities[label - 1]

    @property
    def num_classes(self):
        return 1 + self.m_visc


class EulerDiffusionDatasetWrapper(Dataset):
    """
    Wrapper for EulerDiffusionDataset that returns data in the format
    expected by the DISCO training loop.

    Returns dict with:
        - 'input': (input_frames, 1, H, W) - context frames
        - 'output': (output_frames, 1, H, W) - target frames
        - 'environment_idx': int - environment index for codebook lookup
        - 'viscosity': float - viscosity value (0.0 for Euler)

    Args:
        file_dir: Directory containing trajectories_gpu*.h5 files
        num_gpus: Number of GPU files
        split: 'train' or 'val'
        input_frames: Number of context frames
        output_frames: Number of target frames
        sub_x: Spatial subsampling factor
        sub_t: Temporal subsampling factor
        val_fraction: Fraction of data for validation
        seed: Random seed for train/val split
        vorticity_scale: Scale factor to divide vorticity by (default: 20.0)
        filter_labels: Optional list of labels to include in the dataset.
    """

    def __init__(self, file_dir: str, num_gpus: int, split: str = 'train',
                 input_frames: int = 16, output_frames: int = 2,
                 sub_x: int = 1, sub_t: int = 1,
                 val_fraction: float = 0.1, seed: int = 42,
                 vorticity_scale: float = 20.0,
                 filter_labels: Optional[List[int]] = None,
                 max_samples: Optional[int] = None,
                 T_limit: Optional[int] = None,
                 contiguous_in_time: bool = False):

        self.dataset = EulerDiffusionDataset(
            file_dir=file_dir,
            num_gpus=num_gpus,
            split=split,
            val_fraction=val_fraction,
            seed=seed,
            filter_labels=filter_labels
        )

        self.input_frames = input_frames
        self.output_frames = output_frames
        self.sub_x = sub_x
        self.sub_t = sub_t
        self.vorticity_scale = vorticity_scale
        self.max_samples = max_samples
        self.T_limit = T_limit
        self.contiguous_in_time = contiguous_in_time

        # Get metadata from underlying dataset
        self.n_snapshots = self.dataset.n_snapshots
        self.num_environments = self.dataset.num_classes

        # Limit samples if max_samples is specified (for overfit testing)
        if max_samples is not None and max_samples < len(self.dataset):
            # Use deterministic subset based on seed for reproducibility
            rng = np.random.RandomState(seed)
            self._subset_indices = rng.permutation(len(self.dataset))[:max_samples]
            print(f"EulerDiffusionDatasetWrapper [{split}]: LIMITED to {max_samples} samples "
                  f"(from {len(self.dataset)} total) for overfit testing")
        else:
            self._subset_indices = None

        T_limit_msg = f", T_limit={T_limit}" if T_limit is not None else ""
        print(f"EulerDiffusionDatasetWrapper [{split}]: {len(self)} samples, "
              f"{self.num_environments} environments (1 Euler + {self.dataset.m_visc} diffusion), "
              f"vorticity_scale={vorticity_scale}{T_limit_msg}")

    def __len__(self):
        if self._subset_indices is not None:
            return len(self._subset_indices)
        return len(self.dataset)

    def __getitem__(self, idx):
        # Map to actual index if using subset
        if self._subset_indices is not None:
            idx = self._subset_indices[idx]

        # Get raw trajectory and label from underlying dataset
        trajectory, label = self.dataset[idx]  # trajectory: (n_snapshots, H, W), label: int

        # Apply temporal subsampling
        if self.sub_t > 1:
            trajectory = trajectory[::self.sub_t]

        # Apply T_limit to truncate trajectory (discard later timesteps)
        if self.T_limit is not None:
            trajectory = trajectory[:self.T_limit]

        # Apply spatial subsampling
        if self.sub_x > 1:
            trajectory = trajectory[:, ::self.sub_x, ::self.sub_x]

        # Convert to float tensor and scale vorticity
        trajectory = trajectory.float() / self.vorticity_scale

        # Add channel dimension: (T, H, W) -> (T, 1, H, W)
        trajectory = trajectory.unsqueeze(1)

        # Sample temporal window randomly
        total_frames_needed = self.input_frames + self.output_frames
        n_available = trajectory.shape[0]
        max_start = n_available - total_frames_needed

        if max_start <= 0:
            # Not enough frames - use what we have
            start_idx = 0
            actual_input_frames = min(self.input_frames, n_available // 2)
            actual_output_frames = n_available - actual_input_frames
        else:
            # Random temporal sampling
            start_idx = np.random.randint(0, max_start + 1)
            actual_input_frames = self.input_frames
            actual_output_frames = self.output_frames

        # Extract input sequence
        input_end = start_idx + actual_input_frames
        input_trajectory = trajectory[start_idx:input_end]  # (input_frames, 1, H, W)

        # Sample output start point
        if self.contiguous_in_time:
            # Output starts immediately after input (for baseline models)
            output_start = input_end
        else:
            # Output can be anywhere after input start (for DISCO)
            max_output_start = n_available - actual_output_frames
            output_start = np.random.randint(start_idx, max_output_start + 1)
        output_end = output_start + actual_output_frames
        output_trajectory = trajectory[output_start:output_end]  # (output_frames, 1, H, W)

        # Get viscosity value
        viscosity = self.dataset.get_viscosity(label)

        return {
            'input': input_trajectory,           # (input_frames, 1, H, W)
            'output': output_trajectory,         # (output_frames, 1, H, W)
            'environment_idx': label,            # int: 0 for Euler, 1+ for diffusion
            'viscosity': torch.tensor(viscosity, dtype=torch.float32),
        }

    def close(self):
        """Close underlying dataset file handles."""
        self.dataset.close()


class NavierStokesDataset(MultiFileHDF5Dataset, Dataset):
    """
    Dataset for testing on Navier-Stokes trajectories.
    Loads directly from split GPU files.

    Labels: 0, 1, ..., m_visc-1 corresponding to viscosity indices.

    Args:
        file_dir: Directory containing trajectories_gpu*.h5 files
        num_gpus: Number of GPU files
        N_ns_ics: Number of ICs per viscosity (from generator config)
    """

    def __init__(self, file_dir: str, num_gpus: int, N_ns_ics: int = 512):
        file_dir = Path(file_dir)
        file_paths = [file_dir / f"trajectories_gpu{i}.h5" for i in range(num_gpus)]

        super().__init__(file_paths)

        self.N_ns_ics = N_ns_ics

        # Read metadata
        f0, should_close = self._get_file(str(file_paths[0]))
        self.viscosities = f0.attrs['viscosities'][:]
        self.n_snapshots = f0.attrs['n_snapshots']
        self.save_res = f0.attrs['save_res']
        self.m_visc = len(self.viscosities)
        if should_close:
            f0.close()

        # Build index: (file_path, local_idx, visc_label)
        # NS global structure: v_idx = global_idx // N_ns_ics
        all_indices = []

        for path in self.file_paths:
            f, should_close = self._get_file(str(path))

            ns_start = f.attrs['ns_start']
            ns_count = f.attrs['ns_count']

            for local_idx in range(ns_count):
                global_idx = ns_start + local_idx
                v_idx = global_idx // N_ns_ics
                all_indices.append((str(path), local_idx, v_idx))

            if should_close:
                f.close()

        self.indices = all_indices
        print(f"NavierStokesDataset: {len(self.indices)} samples")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        file_path, local_idx, visc_label = self.indices[idx]

        f, should_close = self._get_file(file_path)
        try:
            trajectory = f['navier_stokes'][local_idx]
        finally:
            if should_close:
                f.close()

        return torch.from_numpy(trajectory), visc_label

    def get_viscosity(self, label):
        return self.viscosities[label]

    @property
    def num_classes(self):
        return self.m_visc


class NavierStokesDatasetWrapper(Dataset):
    """
    Wrapper for NavierStokesDataset that returns data in the format
    expected by the DISCO training/evaluation loop.

    Returns dict with:
        - 'input': (input_frames, 1, H, W) - context frames
        - 'output': (output_frames, 1, H, W) - target frames
        - 'environment_idx': int - viscosity index for codebook lookup
        - 'viscosity': float - viscosity value

    Args:
        file_dir: Directory containing trajectories_gpu*.h5 files
        num_gpus: Number of GPU files
        input_frames: Number of context frames
        output_frames: Number of target frames
        sub_x: Spatial subsampling factor
        sub_t: Temporal subsampling factor
        N_ns_ics: Number of ICs per viscosity
        vorticity_scale: Scale factor to divide vorticity by (default: 20.0)
    """

    def __init__(self, file_dir: str, num_gpus: int,
                 input_frames: int = 16, output_frames: int = 2,
                 sub_x: int = 1, sub_t: int = 1,
                 N_ns_ics: int = 512, vorticity_scale: float = 20.0):

        self.dataset = NavierStokesDataset(
            file_dir=file_dir,
            num_gpus=num_gpus,
            N_ns_ics=N_ns_ics
        )

        self.input_frames = input_frames
        self.output_frames = output_frames
        self.sub_x = sub_x
        self.sub_t = sub_t
        self.vorticity_scale = vorticity_scale

        # Get metadata
        self.n_snapshots = self.dataset.n_snapshots
        self.num_environments = self.dataset.num_classes

        print(f"NavierStokesDatasetWrapper: {len(self)} samples, "
              f"{self.num_environments} viscosity environments, "
              f"vorticity_scale={vorticity_scale}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get raw trajectory and label from underlying dataset
        trajectory, label = self.dataset[idx]  # trajectory: (n_snapshots, H, W), label: int

        # Apply temporal subsampling
        if self.sub_t > 1:
            trajectory = trajectory[::self.sub_t]

        # Apply spatial subsampling
        if self.sub_x > 1:
            trajectory = trajectory[:, ::self.sub_x, ::self.sub_x]

        # Convert to float tensor and scale vorticity
        trajectory = trajectory.float() / self.vorticity_scale

        # Add channel dimension: (T, H, W) -> (T, 1, H, W)
        trajectory = trajectory.unsqueeze(1)

        # Sample temporal window randomly
        total_frames_needed = self.input_frames + self.output_frames
        n_available = trajectory.shape[0]
        max_start = n_available - total_frames_needed

        if max_start <= 0:
            # Not enough frames - use what we have
            start_idx = 0
            actual_input_frames = min(self.input_frames, n_available // 2)
            actual_output_frames = n_available - actual_input_frames
        else:
            # Random temporal sampling
            start_idx = np.random.randint(0, max_start + 1)
            actual_input_frames = self.input_frames
            actual_output_frames = self.output_frames
        
        start_idx = 0 # force it to zero

        # Extract input sequence
        input_end = start_idx + actual_input_frames
        input_trajectory = trajectory[start_idx:input_end]  # (input_frames, 1, H, W)

        # Output starts immediately after input
        output_start = input_end
        output_end = output_start + actual_output_frames
        output_trajectory = trajectory[output_start:output_end]  # (output_frames, 1, H, W)

        # Get viscosity value
        viscosity = self.dataset.get_viscosity(label)

        return {
            'input': input_trajectory,           # (input_frames, 1, H, W)
            'output': output_trajectory,         # (output_frames, 1, H, W)
            'environment_idx': label,            # int: viscosity index
            'viscosity': torch.tensor(viscosity, dtype=torch.float32),
        }

    def close(self):
        """Close underlying dataset file handles."""
        self.dataset.close()


class EulerNSDatasetWrapperMPP(Dataset):
    """
    Wrapper for Euler/NS dataset that returns data in the format
    expected by MPP and GEPS training loops (matches Gray-Scott format).

    Returns dict with:
        - 'input': (input_frames, 1, H, W) - context frames
        - 'output': (output_frames, 1, H, W) - target frames
        - 'environment_idx': int - environment index (0=Euler, 1+=Diffusion)

    Args:
        file_dir: Directory containing trajectories_gpu*.h5 files
        num_gpus: Number of GPU files
        split: 'train' or 'val'
        input_frames: Number of context frames
        output_frames: Number of target frames
        sub_x: Spatial subsampling factor
        sub_t: Temporal subsampling factor
        val_fraction: Fraction of data for validation
        seed: Random seed for train/val split
        vorticity_scale: Scale factor to divide vorticity by (default: 20.0)
        filter_labels: Optional list of labels to include
    """

    def __init__(self, file_dir: str, num_gpus: int, split: str = 'train',
                 input_frames: int = 16, output_frames: int = 1,
                 sub_x: int = 1, sub_t: int = 1,
                 val_fraction: float = 0.1, seed: int = 42,
                 vorticity_scale: float = 10.0,
                 filter_labels: Optional[List[int]] = None):

        self.dataset = EulerDiffusionDatasetWrapper(
            file_dir=file_dir,
            num_gpus=num_gpus,
            split=split,
            input_frames=input_frames,
            output_frames=output_frames,
            sub_x=sub_x,
            sub_t=sub_t,
            val_fraction=val_fraction,
            seed=seed,
            vorticity_scale=vorticity_scale,
            filter_labels=filter_labels,
            contiguous_in_time=True
        )

        self.num_environments = self.dataset.num_environments

        print(f"EulerNSDatasetWrapperMPP [{split}]: {len(self)} samples, "
              f"{self.num_environments} environments")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        return {
            'input': sample['input'],           # (input_frames, 1, H, W)
            'output': sample['output'],         # (output_frames, 1, H, W)
            'environment_idx': sample['environment_idx'],
        }

    def close(self):
        """Close underlying dataset file handles."""
        self.dataset.close()


class EulerNSDatasetWrapperZEBRA(Dataset):
    """
    Wrapper for Euler/NS dataset that returns data in the format
    expected by ZEBRA tokenizer training (VQ-VAE).

    Returns tensor of shape (C, H, W, T) - concatenated input and output frames.

    Args:
        file_dir: Directory containing trajectories_gpu*.h5 files
        num_gpus: Number of GPU files
        split: 'train' or 'val'
        input_frames: Number of context frames
        output_frames: Number of target frames
        sub_x: Spatial subsampling factor
        sub_t: Temporal subsampling factor
        val_fraction: Fraction of data for validation
        seed: Random seed for train/val split
        vorticity_scale: Scale factor to divide vorticity by (default: 20.0)
        filter_labels: Optional list of labels to include
    """

    def __init__(self, file_dir: str, num_gpus: int, split: str = 'train',
                 input_frames: int = 16, output_frames: int = 1,
                 sub_x: int = 1, sub_t: int = 1,
                 val_fraction: float = 0.1, seed: int = 42,
                 vorticity_scale: float = 10.0,
                 filter_labels: Optional[List[int]] = None):

        self.dataset = EulerDiffusionDatasetWrapper(
            file_dir=file_dir,
            num_gpus=num_gpus,
            split=split,
            input_frames=input_frames,
            output_frames=output_frames,
            sub_x=sub_x,
            sub_t=sub_t,
            val_fraction=val_fraction,
            seed=seed,
            vorticity_scale=vorticity_scale,
            filter_labels=filter_labels,
            contiguous_in_time=True
        )

        self.input_frames = input_frames
        self.output_frames = output_frames

        print(f"EulerNSDatasetWrapperZEBRA [{split}]: {len(self)} samples, "
              f"output shape: (1, H, W, {input_frames + output_frames})")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        # sample['input']: (input_frames, 1, H, W)
        # sample['output']: (output_frames, 1, H, W)
        input_traj = sample['input']   # (T_in, 1, H, W)
        output_traj = sample['output'] # (T_out, 1, H, W)

        # Concatenate along time dimension
        trajectory = torch.cat([input_traj, output_traj], dim=0)  # (T, 1, H, W)

        # Rearrange to ZEBRA expected format: (C, H, W, T)
        trajectory = trajectory.permute(1, 2, 3, 0)  # (1, H, W, T)

        return trajectory

    def close(self):
        """Close underlying dataset file handles."""
        self.dataset.close()


class NavierStokesDatasetWrapperZEBRA(Dataset):
    """
    Wrapper for Navier-Stokes dataset that returns data in the format
    expected by ZEBRA testing (VQ-VAE).

    Returns tensor of shape (C, H, W, T) - concatenated input and output frames.

    Args:
        file_dir: Directory containing trajectories_gpu*.h5 files
        num_gpus: Number of GPU files
        input_frames: Number of context frames
        output_frames: Number of target frames
        sub_x: Spatial subsampling factor
        sub_t: Temporal subsampling factor
        N_ns_ics: Number of ICs per viscosity
        vorticity_scale: Scale factor to divide vorticity by (default: 10.0)
    """

    def __init__(self, file_dir: str, num_gpus: int,
                 input_frames: int = 16, output_frames: int = 16,
                 sub_x: int = 1, sub_t: int = 1,
                 N_ns_ics: int = 512, vorticity_scale: float = 10.0):

        self.dataset = NavierStokesDatasetWrapper(
            file_dir=file_dir,
            num_gpus=num_gpus,
            input_frames=input_frames,
            output_frames=output_frames,
            sub_x=sub_x,
            sub_t=sub_t,
            N_ns_ics=N_ns_ics,
            vorticity_scale=vorticity_scale
        )

        self.input_frames = input_frames
        self.output_frames = output_frames

        print(f"NavierStokesDatasetWrapperZEBRA: {len(self)} samples, "
              f"output shape: (1, H, W, {input_frames + output_frames})")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        # sample['input']: (input_frames, 1, H, W)
        # sample['output']: (output_frames, 1, H, W)
        input_traj = sample['input']   # (T_in, 1, H, W)
        output_traj = sample['output'] # (T_out, 1, H, W)

        # Concatenate along time dimension
        trajectory = torch.cat([input_traj, output_traj], dim=0)  # (T, 1, H, W)

        # Rearrange to ZEBRA expected format: (C, H, W, T)
        trajectory = trajectory.permute(1, 2, 3, 0)  # (1, H, W, T)

        return trajectory

    def close(self):
        """Close underlying dataset file handles."""
        self.dataset.close()


def get_dataloaders(file_dir: str, num_gpus: int, batch_size: int = 32,
                    num_workers: int = 4, val_fraction: float = 0.1,
                    seed: int = 42, N_ns_ics: int = 512,
                    input_frames: int = 16, output_frames: int = 2,
                    sub_x: int = 1, sub_t: int = 1,
                    vorticity_scale: float = 20.0):
    """
    Get train/val/test dataloaders from split GPU files.

    Args:
        file_dir: Directory containing trajectories_gpu*.h5 files
        num_gpus: Number of GPU files
        batch_size: Batch size
        num_workers: DataLoader workers
        val_fraction: Fraction for validation
        seed: Random seed
        N_ns_ics: NS ICs per viscosity (from generator)
        input_frames: Number of input context frames
        output_frames: Number of output target frames
        sub_x: Spatial subsampling factor
        sub_t: Temporal subsampling factor
        vorticity_scale: Scale factor to divide vorticity by

    Returns:
        train_loader, val_loader, test_loader
    """
    train_dataset = EulerDiffusionDatasetWrapper(
        file_dir, num_gpus, split='train',
        input_frames=input_frames, output_frames=output_frames,
        sub_x=sub_x, sub_t=sub_t,
        val_fraction=val_fraction, seed=seed,
        vorticity_scale=vorticity_scale
    )
    val_dataset = EulerDiffusionDatasetWrapper(
        file_dir, num_gpus, split='val',
        input_frames=input_frames, output_frames=output_frames,
        sub_x=sub_x, sub_t=sub_t,
        val_fraction=val_fraction, seed=seed,
        vorticity_scale=vorticity_scale
    )
    test_dataset = NavierStokesDatasetWrapper(
        file_dir, num_gpus,
        input_frames=input_frames, output_frames=output_frames,
        sub_x=sub_x, sub_t=sub_t,
        N_ns_ics=N_ns_ics,
        vorticity_scale=vorticity_scale
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers,
                             pin_memory=True)

    return train_loader, val_loader, test_loader

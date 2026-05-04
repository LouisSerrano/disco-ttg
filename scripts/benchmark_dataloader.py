"""
Benchmark script for data loading performance.

Tests:
1. Raw HDF5 access speed
2. Dataset wrapper speed
3. DataLoader speed with different num_workers
"""
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.euler_ns_dataset import (
    EulerDiffusionDataset,
    EulerDiffusionDatasetWrapper,
)


def benchmark_raw_dataset(file_dir: str, num_gpus: int, num_samples: int = 100):
    """Benchmark raw dataset (no wrapper)."""
    print("\n" + "="*60)
    print("Benchmarking RAW EulerDiffusionDataset")
    print("="*60)

    # Time dataset creation
    t0 = time.time()
    ds = EulerDiffusionDataset(file_dir, num_gpus, split='train')
    t_create = time.time() - t0
    print(f"Dataset creation time: {t_create:.2f}s")
    print(f"Dataset size: {len(ds)} samples")

    # Time sequential access
    t0 = time.time()
    for i in range(min(num_samples, len(ds))):
        _ = ds[i]
    t_seq = time.time() - t0
    samples_per_sec = num_samples / t_seq
    print(f"Sequential access ({num_samples} samples): {t_seq:.2f}s ({samples_per_sec:.1f} samples/s)")

    # Time random access
    indices = np.random.randint(0, len(ds), num_samples)
    t0 = time.time()
    for i in indices:
        _ = ds[i]
    t_rand = time.time() - t0
    samples_per_sec = num_samples / t_rand
    print(f"Random access ({num_samples} samples): {t_rand:.2f}s ({samples_per_sec:.1f} samples/s)")

    ds.close()
    return t_seq, t_rand


def benchmark_wrapper(file_dir: str, num_gpus: int, num_samples: int = 100,
                      input_frames: int = 16, output_frames: int = 2):
    """Benchmark dataset wrapper."""
    print("\n" + "="*60)
    print("Benchmarking EulerDiffusionDatasetWrapper")
    print("="*60)

    # Time dataset creation
    t0 = time.time()
    ds = EulerDiffusionDatasetWrapper(
        file_dir, num_gpus, split='train',
        input_frames=input_frames, output_frames=output_frames
    )
    t_create = time.time() - t0
    print(f"Dataset creation time: {t_create:.2f}s")
    print(f"Dataset size: {len(ds)} samples")
    print(f"Num environments: {ds.num_environments}")

    # Time sequential access
    t0 = time.time()
    for i in range(min(num_samples, len(ds))):
        sample = ds[i]
    t_seq = time.time() - t0
    samples_per_sec = num_samples / t_seq
    print(f"Sequential access ({num_samples} samples): {t_seq:.2f}s ({samples_per_sec:.1f} samples/s)")

    # Show sample shapes
    print(f"  Input shape: {sample['input'].shape}")
    print(f"  Output shape: {sample['output'].shape}")

    # Time random access
    indices = np.random.randint(0, len(ds), num_samples)
    t0 = time.time()
    for i in indices:
        _ = ds[i]
    t_rand = time.time() - t0
    samples_per_sec = num_samples / t_rand
    print(f"Random access ({num_samples} samples): {t_rand:.2f}s ({samples_per_sec:.1f} samples/s)")

    ds.close()
    return t_seq, t_rand


def benchmark_dataloader(file_dir: str, num_gpus: int, batch_size: int = 32,
                         num_batches: int = 20, num_workers_list: list = [0, 2, 4, 8]):
    """Benchmark DataLoader with different num_workers."""
    print("\n" + "="*60)
    print("Benchmarking DataLoader")
    print("="*60)

    results = {}

    for num_workers in num_workers_list:
        ds = EulerDiffusionDatasetWrapper(
            file_dir, num_gpus, split='train',
            input_frames=16, output_frames=2
        )

        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
            drop_last=True,
        )

        # Warmup
        loader_iter = iter(loader)
        for _ in range(min(3, num_batches)):
            try:
                _ = next(loader_iter)
            except StopIteration:
                break

        # Benchmark
        t0 = time.time()
        loader_iter = iter(loader)
        batches_loaded = 0
        for _ in range(num_batches):
            try:
                batch = next(loader_iter)
                batches_loaded += 1
            except StopIteration:
                break
        t_total = time.time() - t0

        samples_per_sec = (batches_loaded * batch_size) / t_total
        batches_per_sec = batches_loaded / t_total

        print(f"num_workers={num_workers}: {batches_loaded} batches in {t_total:.2f}s "
              f"({batches_per_sec:.1f} batches/s, {samples_per_sec:.1f} samples/s)")

        results[num_workers] = {
            'time': t_total,
            'batches_per_sec': batches_per_sec,
            'samples_per_sec': samples_per_sec,
        }

        ds.close()
        del loader, ds

    return results


def benchmark_with_gpu_transfer(file_dir: str, num_gpus: int, batch_size: int = 32,
                                 num_batches: int = 20, num_workers: int = 4):
    """Benchmark including GPU transfer time."""
    print("\n" + "="*60)
    print("Benchmarking DataLoader + GPU Transfer")
    print("="*60)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU transfer benchmark")
        return None

    device = torch.device('cuda')

    ds = EulerDiffusionDatasetWrapper(
        file_dir, num_gpus, split='train',
        input_frames=16, output_frames=2
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        drop_last=True,
    )

    # Warmup
    loader_iter = iter(loader)
    for _ in range(3):
        try:
            batch = next(loader_iter)
            _ = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        except StopIteration:
            break

    torch.cuda.synchronize()

    # Benchmark
    t0 = time.time()
    loader_iter = iter(loader)
    batches_loaded = 0
    for _ in range(num_batches):
        try:
            batch = next(loader_iter)
            batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            torch.cuda.synchronize()
            batches_loaded += 1
        except StopIteration:
            break
    t_total = time.time() - t0

    samples_per_sec = (batches_loaded * batch_size) / t_total
    print(f"With GPU transfer: {batches_loaded} batches in {t_total:.2f}s "
          f"({samples_per_sec:.1f} samples/s)")

    ds.close()
    return samples_per_sec


def main():
    parser = argparse.ArgumentParser(description='Benchmark data loading')
    parser.add_argument('--file_dir', type=str,
                        default='/mnt/home/lserrano/ceph/data/euler_ns/',
                        help='Directory with trajectories_gpu*.h5 files')
    parser.add_argument('--num_gpus', type=int, default=8,
                        help='Number of GPU files')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples for single-item benchmarks')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for DataLoader benchmarks')
    parser.add_argument('--num_batches', type=int, default=20,
                        help='Number of batches for DataLoader benchmarks')
    args = parser.parse_args()

    print(f"Benchmarking data loading from: {args.file_dir}")
    print(f"Number of GPU files: {args.num_gpus}")

    # Run benchmarks
    benchmark_raw_dataset(args.file_dir, args.num_gpus, args.num_samples)
    benchmark_wrapper(args.file_dir, args.num_gpus, args.num_samples)
    benchmark_dataloader(args.file_dir, args.num_gpus, args.batch_size, args.num_batches)
    benchmark_with_gpu_transfer(args.file_dir, args.num_gpus, args.batch_size, args.num_batches)

    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    print("""
If data loading is slow, consider these optimizations:

1. **Increase num_workers**: More parallel data loading
   - Start with num_workers = 4-8
   - Don't exceed CPU cores

2. **Use persistent_workers=True**: Keeps workers alive between epochs

3. **Pre-load to memory**: If dataset fits in RAM, load everything upfront
   - Add a 'preload' option to the dataset wrapper

4. **Use faster storage**:
   - SSD > HDD
   - Local disk > Network storage (like Ceph)

5. **Chunked HDF5**: Ensure HDF5 files are chunked properly

6. **Memory-map the files**: Use h5py's mmap mode for large files

7. **Cache trajectories**: Add LRU cache for recently accessed trajectories
""")


if __name__ == '__main__':
    main()

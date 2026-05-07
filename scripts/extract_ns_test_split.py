"""Extract a reproducible test split from the Euler/NS GPU files into one HDF5.

The training pipeline (src/utils/euler_ns_dataset.py) does a 90/10 random
split (seed=42) across all 8 GPU files. This script replicates that split
and writes the val (10%) portion to a single HDF5 in the generic format.

Output format:
    trajectories: (N, T, 1, H, W)   float32
    env_id:       (N,)              int64    # 0=euler, 1+=diffusion-by-viscosity
    env_params/equation_label: (N,) int64    # same as env_id (for clarity)
    env_params/viscosity:      (N,) float64  # NaN for euler samples

Usage:
    python scripts/extract_ns_test_split.py \
        --file_dir ./datasets/euler_ns_short \
        --num_gpus 8 \
        --output /tmp/disco_generic/ns_val.h5 \
        --max_samples 1024
"""
import argparse
import os
from pathlib import Path

import h5py
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--file_dir", required=True)
    p.add_argument("--num_gpus", type=int, default=8)
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True)
    p.add_argument("--max_samples", type=int, default=None,
                   help="Cap the number of samples extracted (useful for keeping the "
                        "uploaded test set small). If unset, extracts the full val split.")
    p.add_argument("--compression", choices=["gzip", "lzf", "none"], default="lzf")
    args = p.parse_args()

    file_dir = Path(args.file_dir)
    file_paths = [file_dir / f"trajectories_gpu{i}.h5" for i in range(args.num_gpus)]
    for fp in file_paths:
        if not fp.exists():
            raise FileNotFoundError(fp)

    # Read metadata from first file
    with h5py.File(file_paths[0], "r") as f0:
        viscosities = list(f0.attrs["viscosities"])
        m_visc = len(viscosities)

    # Build index across all files (replicates EulerDiffusionDataset logic)
    all_indices = []  # list of (file_path, dataset_name, local_idx, label, viscosity)
    for path in file_paths:
        with h5py.File(path, "r") as f:
            euler_count = int(f.attrs["euler_count"])
            diff_count = int(f.attrs["diff_count"])
            n_diff_per_visc = diff_count // m_visc
            for i in range(euler_count):
                all_indices.append((str(path), "euler", i, 0, float("nan")))
            for v_idx in range(m_visc):
                label = v_idx + 1
                start = v_idx * n_diff_per_visc
                for i in range(n_diff_per_visc):
                    all_indices.append((str(path), "diffusion", start + i, label, float(viscosities[v_idx])))

    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(all_indices))
    n_val = int(len(all_indices) * args.val_fraction)
    val_idx = perm[:n_val]
    selected = [all_indices[i] for i in val_idx]
    if args.max_samples is not None:
        selected = selected[: args.max_samples]
    print(f"Extracting {len(selected)} samples from val split (seed={args.seed}).")

    # Pre-allocate output (we know dimensions)
    with h5py.File(file_paths[0], "r") as f0:
        sample_shape = f0["euler"][0].shape  # (T, H, W)
    T, H, W = sample_shape
    N = len(selected)
    trajs = np.empty((N, T, 1, H, W), dtype=np.float32)
    labels = np.empty(N, dtype=np.int64)
    viscs = np.empty(N, dtype=np.float64)

    # Group by file for fewer reopen calls
    by_file = {}
    for i, (path, ds, lidx, label, visc) in enumerate(selected):
        by_file.setdefault(path, []).append((i, ds, lidx, label, visc))

    for path, items in by_file.items():
        print(f"  reading {len(items)} samples from {os.path.basename(path)}")
        with h5py.File(path, "r") as f:
            for i, ds, lidx, label, visc in items:
                trajs[i, :, 0, :, :] = f[ds][lidx]
                labels[i] = label
                viscs[i] = visc

    print(f"Writing {args.output} ...")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    comp_kwargs = (
        {"compression": "gzip", "compression_opts": 4} if args.compression == "gzip"
        else {"compression": "lzf"} if args.compression == "lzf"
        else {}
    )
    with h5py.File(args.output, "w") as fout:
        fout.create_dataset("trajectories", data=trajs, **comp_kwargs)
        fout.create_dataset("env_id", data=labels)
        fout.create_dataset("env_params/equation_label", data=labels)
        fout.create_dataset("env_params/viscosity", data=viscs)
    print("Done.")


if __name__ == "__main__":
    main()

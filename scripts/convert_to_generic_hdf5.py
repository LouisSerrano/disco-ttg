"""Convert existing combined-equation HDF5 files to the generic format.

Input format (paper experiments):
    {split}/pde_250-256: (N, T, X)
    {split}/alpha, beta, gamma: (N,)        # PDE coefficients
    {split}/dt, dx: (N,)
    ...

Output format (consumed by train/train_generic.py and test/test_generic.py):
    trajectories: (N, T, 1, X) float32      # adds a channel dim
    env_id:       (N,)         int64        # one id per unique (alpha,beta,gamma) tuple

Usage:
    python scripts/convert_to_generic_hdf5.py \
        --input  /path/to/E_EULER_train_8192.h5 \
        --output /path/to/E_EULER_train_generic.h5 \
        --split  train
"""
import argparse
import os

import h5py
import numpy as np


def detect_split(h5file):
    """Return the first top-level group that looks like a split."""
    for cand in ("train", "valid", "val", "test"):
        if cand in h5file:
            return cand
    raise ValueError(f"No known split group in {list(h5file.keys())}")


def make_env_ids(alpha, beta, gamma):
    """Map each unique (alpha, beta, gamma) tuple to a contiguous integer id."""
    keys = list(zip(alpha.tolist(), beta.tolist(), gamma.tolist()))
    seen = {}
    ids = np.empty(len(keys), dtype=np.int64)
    for i, k in enumerate(keys):
        if k not in seen:
            seen[k] = len(seen)
        ids[i] = seen[k]
    return ids, seen


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--split", default=None,
                   help="Group name inside the input file (auto-detected if omitted)")
    p.add_argument("--trajectory_key", default="pde_250-256",
                   help="Name of the trajectories dataset inside the split group")
    args = p.parse_args()

    print(f"Reading {args.input} ...")
    with h5py.File(args.input, "r") as fin:
        split = args.split or detect_split(fin)
        print(f"  split: {split}")
        if args.trajectory_key not in fin[split]:
            available = list(fin[split].keys())
            raise KeyError(f"'{args.trajectory_key}' not in {available}")

        trajs = fin[f"{split}/{args.trajectory_key}"][:]
        alpha = fin[f"{split}/alpha"][:]
        beta = fin[f"{split}/beta"][:]
        gamma = fin[f"{split}/gamma"][:]
        # Extra metadata to keep alongside, if present.
        extra = {}
        for key in ("dt", "dx", "t", "x"):
            if key in fin[split]:
                extra[key] = fin[f"{split}/{key}"][:]

    print(f"  trajectories: {trajs.shape} {trajs.dtype}")
    print(f"  alpha: {alpha.shape}, beta: {beta.shape}, gamma: {gamma.shape}")

    # Add channel dimension if absent.
    if trajs.ndim == 3:
        # (N, T, X) -> (N, T, 1, X)
        trajs = trajs[:, :, None, :]
    trajs = trajs.astype(np.float32, copy=False)

    env_id, env_map = make_env_ids(alpha, beta, gamma)
    print(f"  unique environments: {len(env_map)}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    print(f"Writing {args.output} ...")
    with h5py.File(args.output, "w") as fout:
        fout.create_dataset("trajectories", data=trajs, compression="gzip", compression_opts=4)
        fout.create_dataset("env_id", data=env_id)
        # Stash original PDE params keyed by env_id for reproducibility.
        unique_keys = sorted(env_map.items(), key=lambda kv: kv[1])
        params = np.array([k for k, _ in unique_keys], dtype=np.float64)
        fout.create_dataset("env_params/alpha_beta_gamma", data=params)
        for key, val in extra.items():
            fout.create_dataset(f"meta/{key}", data=val)

    print("Done.")


if __name__ == "__main__":
    main()

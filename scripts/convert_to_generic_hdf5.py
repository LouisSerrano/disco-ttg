"""Convert paper HDF5 datasets to the generic (trajectories, env_id) format.

The output format consumed by train/train_generic.py and test_generic.py:
    trajectories: (N, T, C, *spatial)   float32
    env_id:       (N,)                  int64
    env_params/*: optional metadata mapping env_id -> physical parameters

Three source formats are supported via --source_format:

  combined  (default — paper combined-equation data):
        {split}/pde_250-256: (N, T, X)
        {split}/alpha, beta, gamma: (N,)
      env_id is unique (alpha, beta, gamma) tuple.

  rd       (Gray-Scott reaction-diffusion):
        {split}/trajectory_a: (N, T, X)
        {split}/trajectory_b: (N, T, X)
        {split}/alpha, beta, f, gamma, k: (N,)
      Output channels: 2 (a stacked with b).
      env_id is unique (alpha, beta, f, gamma, k) tuple.

  ns       (2D Navier-Stokes / Euler — 3 eq types per file):
        {eq_name}: (N, T, H, W)        # eq_name in {diffusion, euler, navier_stokes}
      Output: concatenates the three equation arrays;
              env_id distinguishes the three equation types
              (0=diffusion, 1=euler, 2=navier_stokes).

Usage:
    python scripts/convert_to_generic_hdf5.py \
        --source_format combined \
        --input  E_EULER_test.h5 \
        --output combined_EULER_test_generic.h5
"""
import argparse
import os

import h5py
import numpy as np


def _detect_split(h5file):
    for cand in ("train", "valid", "val", "test"):
        if cand in h5file:
            return cand
    raise ValueError(f"No known split group in {list(h5file.keys())}")


def _make_env_ids(rows):
    """Map each unique tuple in `rows` to a contiguous int id."""
    seen = {}
    ids = np.empty(len(rows), dtype=np.int64)
    for i, k in enumerate(rows):
        if k not in seen:
            seen[k] = len(seen)
        ids[i] = seen[k]
    return ids, seen


def convert_combined(args):
    """combined-equation paper format: pde_250-256 + alpha/beta/gamma."""
    with h5py.File(args.input, "r") as fin:
        split = args.split or _detect_split(fin)
        if args.trajectory_key not in fin[split]:
            raise KeyError(f"'{args.trajectory_key}' not in {list(fin[split].keys())}")
        trajs = fin[f"{split}/{args.trajectory_key}"][:]
        alpha = fin[f"{split}/alpha"][:]
        beta = fin[f"{split}/beta"][:]
        gamma = fin[f"{split}/gamma"][:]
        extra = {k: fin[f"{split}/{k}"][:] for k in ("dt", "dx", "t", "x") if k in fin[split]}

    print(f"  trajectories: {trajs.shape}, alpha/beta/gamma: {alpha.shape}")
    if trajs.ndim == 3:
        trajs = trajs[:, :, None, :]  # add channel dim
    trajs = trajs.astype(np.float32, copy=False)
    rows = list(zip(alpha.tolist(), beta.tolist(), gamma.tolist()))
    env_id, env_map = _make_env_ids(rows)
    params = np.array(sorted(env_map, key=env_map.get), dtype=np.float64)
    return trajs, env_id, {"alpha_beta_gamma": params}, extra


def convert_rd(args):
    """Gray-Scott format: trajectory_a + trajectory_b + alpha/beta/f/gamma/k."""
    with h5py.File(args.input, "r") as fin:
        split = args.split or _detect_split(fin)
        traj_a = fin[f"{split}/trajectory_a"][:]
        traj_b = fin[f"{split}/trajectory_b"][:]
        params = {p: fin[f"{split}/{p}"][:] for p in ("alpha", "beta", "f", "gamma", "k")
                  if p in fin[split]}

    print(f"  trajectory_a: {traj_a.shape} | trajectory_b: {traj_b.shape}")
    # (N, T, X) + (N, T, X) -> (N, T, 2, X)
    trajs = np.stack([traj_a, traj_b], axis=2).astype(np.float32, copy=False)

    rows = list(zip(*[params[p].tolist() for p in sorted(params)]))
    env_id, env_map = _make_env_ids(rows)
    params_arr = np.array(sorted(env_map, key=env_map.get), dtype=np.float64)
    return trajs, env_id, {"_".join(sorted(params)): params_arr}, {}


def convert_ns(args):
    """2D Navier-Stokes format: 3 equation types per file (diffusion/euler/navier_stokes)."""
    eq_names = ["diffusion", "euler", "navier_stokes"]
    pieces = []
    env_pieces = []
    with h5py.File(args.input, "r") as fin:
        for i, name in enumerate(eq_names):
            if name not in fin:
                print(f"  WARN: '{name}' not found in file, skipping")
                continue
            arr = fin[name][:]
            print(f"  {name}: {arr.shape}")
            pieces.append(arr)
            env_pieces.append(np.full(arr.shape[0], i, dtype=np.int64))

    if not pieces:
        raise ValueError("No NS equation groups found in file.")

    trajs = np.concatenate(pieces, axis=0)  # (N_total, T, H, W)
    if trajs.ndim == 4:
        trajs = trajs[:, :, None, :, :]  # add channel dim
    trajs = trajs.astype(np.float32, copy=False)
    env_id = np.concatenate(env_pieces, axis=0)
    eq_present = np.array([eq_names[i] for i in sorted(set(env_id.tolist()))], dtype="S")
    return trajs, env_id, {"equation_names": eq_present}, {}


CONVERTERS = {
    "combined": convert_combined,
    "rd":       convert_rd,
    "ns":       convert_ns,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--source_format", choices=list(CONVERTERS.keys()), default="combined")
    p.add_argument("--split", default=None,
                   help="Group name inside the input file (auto-detected if omitted; "
                        "ignored for --source_format ns)")
    p.add_argument("--trajectory_key", default="pde_250-256",
                   help="Trajectory dataset name (combined format only)")
    p.add_argument("--compression", choices=["gzip", "lzf", "none"], default="lzf",
                   help="HDF5 compression for the trajectories array (lzf is fast, gzip smaller)")
    args = p.parse_args()

    print(f"Reading {args.input} (format={args.source_format}) ...")
    trajs, env_id, env_params, meta = CONVERTERS[args.source_format](args)
    print(f"  unique environments: {len(set(env_id.tolist()))}  |  trajectories: {trajs.shape}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    print(f"Writing {args.output} ...")
    comp_kwargs = (
        {"compression": "gzip", "compression_opts": 4} if args.compression == "gzip"
        else {"compression": "lzf"} if args.compression == "lzf"
        else {}
    )
    with h5py.File(args.output, "w") as fout:
        fout.create_dataset("trajectories", data=trajs, **comp_kwargs)
        fout.create_dataset("env_id", data=env_id)
        for k, v in env_params.items():
            fout.create_dataset(f"env_params/{k}", data=v)
        for k, v in meta.items():
            fout.create_dataset(f"meta/{k}", data=v)
    print("Done.")


if __name__ == "__main__":
    main()

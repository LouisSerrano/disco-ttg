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
    """2D Navier-Stokes format: 3 equation types per file (diffusion/euler/navier_stokes).

    Returns sentinel values that signal main() to take the streaming write path,
    since reading all three (1024, 50, 256, 256) float32 arrays into memory at
    once peaks around ~40 GB and OOMs on most nodes.
    """
    eq_names = ["diffusion", "euler", "navier_stokes"]
    sections = []  # list of (eq_idx, n_samples, dataset_name)
    with h5py.File(args.input, "r") as fin:
        sample_shape = None
        for i, name in enumerate(eq_names):
            if name not in fin:
                print(f"  WARN: '{name}' not found in file, skipping")
                continue
            shape = fin[name].shape
            print(f"  {name}: {shape}")
            sections.append((i, shape[0], name))
            if sample_shape is None:
                sample_shape = shape[1:]
            elif sample_shape != shape[1:]:
                raise ValueError(f"NS arrays have inconsistent shapes: {sample_shape} vs {shape[1:]}")
    if not sections:
        raise ValueError("No NS equation groups found in file.")

    # Sentinel return: caller will detect this and stream-write per equation.
    return ("__NS_STREAM__", args.input, sections, sample_shape, eq_names)


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
    result = CONVERTERS[args.source_format](args)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    comp_kwargs = (
        {"compression": "gzip", "compression_opts": 4} if args.compression == "gzip"
        else {"compression": "lzf"} if args.compression == "lzf"
        else {}
    )

    # NS streaming path: copy one equation at a time so we never hold ~40 GB at once.
    if isinstance(result, tuple) and result and result[0] == "__NS_STREAM__":
        _, src_path, sections, sample_shape, eq_names = result
        N_total = sum(n for _, n, _ in sections)
        T = sample_shape[0]
        spatial = sample_shape[1:]
        print(f"  streaming write: total {N_total} trajectories, shape (T={T}, *{spatial})")
        print(f"Writing {args.output} ...")
        with h5py.File(args.output, "w") as fout, h5py.File(src_path, "r") as fin:
            traj_ds = fout.create_dataset(
                "trajectories",
                shape=(N_total, T, 1, *spatial),
                dtype=np.float32,
                chunks=(1, T, 1, *spatial),
                **comp_kwargs,
            )
            env_id = np.empty(N_total, dtype=np.int64)
            offset = 0
            CHUNK = 32  # samples per IO transaction
            for eq_idx, n_samples, name in sections:
                print(f"  copying {name} ({n_samples} samples) ...")
                for start in range(0, n_samples, CHUNK):
                    end = min(start + CHUNK, n_samples)
                    block = fin[name][start:end].astype(np.float32, copy=False)
                    traj_ds[offset + start : offset + end, :, 0, :, :] = block
                env_id[offset : offset + n_samples] = eq_idx
                offset += n_samples
            fout.create_dataset("env_id", data=env_id)
            eq_present = np.array(
                [eq_names[i] for i in sorted({s[0] for s in sections})], dtype="S"
            )
            fout.create_dataset("env_params/equation_names", data=eq_present)
        print("Done.")
        return

    # In-memory path (combined / rd)
    trajs, env_id, env_params, meta = result
    print(f"  unique environments: {len(set(env_id.tolist()))}  |  trajectories: {trajs.shape}")
    print(f"Writing {args.output} ...")
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

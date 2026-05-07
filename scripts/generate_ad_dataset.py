"""Generate a fixed advection-diffusion test dataset and save in the generic HDF5 format.

The advection-diffusion experiments don't ship a static dataset — trajectories are
synthesised on the fly via TemporalBatchDatasetFly (in train/train.py) using
Fractaloid initial conditions and the analytical Fourier solution.

This script reproduces a deterministic test set with a fixed seed (default 124,
the same as test_time_compute/test_advection_diffusion.py) so users can download
it from HuggingFace instead of regenerating locally.

Usage:
    python scripts/generate_ad_dataset.py --output /tmp/ad_test.h5 --num_samples 512
"""
import argparse
import os
import random
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.advection_diffusion import Fractaloid


def advection_diffusion_analytical(u0, L=16.0, v=0.1, D=0.5, nt=100, T=10.0):
    """Spectral analytical solution of 1D advection-diffusion (periodic BCs)."""
    nx = len(u0)
    t = np.linspace(0, T, nt)
    k = 1j * np.fft.fftfreq(nx, d=L / nx) * 2 * np.pi
    u0_hat = np.fft.fft(u0)
    u_xt = np.empty((nt, nx))
    for i, ti in enumerate(t):
        decay = np.exp(D * (k**2) * ti) * np.exp(-k * v * ti)
        u_xt[i] = np.fft.ifft(u0_hat * decay).real
    return u_xt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--num_samples", type=int, default=512)
    p.add_argument("--nx", type=int, default=256)
    p.add_argument("--nt", type=int, default=100)
    p.add_argument("--L", type=float, default=16.0)
    p.add_argument("--T", type=float, default=10.0)
    p.add_argument("--v_range", nargs=2, type=float, default=[0.01, 1.0])
    p.add_argument("--d_range", nargs=2, type=float, default=[0.01, 1.0])
    p.add_argument("--fractal_degree", type=int, default=8)
    p.add_argument("--fractal_power_range", type=int, default=2)
    p.add_argument("--seed", type=int, default=124,
                   help="Default 124 matches test_time_compute/test_advection_diffusion.py")
    p.add_argument("--n_env_bins", type=int, default=64,
                   help="Quantize (v, D) into a grid of n_env_bins x n_env_bins for env_id")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    fractaloid = Fractaloid(
        nx=args.nx, L=args.L,
        degree=args.fractal_degree, power_range=args.fractal_power_range,
    )

    trajs = np.empty((args.num_samples, args.nt, 1, args.nx), dtype=np.float32)
    vs = np.empty(args.num_samples, dtype=np.float64)
    ds = np.empty(args.num_samples, dtype=np.float64)
    print(f"Generating {args.num_samples} AD trajectories (seed={args.seed}) ...")

    for i in range(args.num_samples):
        v = random.uniform(*args.v_range)
        D = random.uniform(*args.d_range)
        u0 = fractaloid.generate(batch_size=1, seed=None).squeeze(0).numpy()
        u_xt = advection_diffusion_analytical(u0, L=args.L, v=v, D=D, nt=args.nt, T=args.T)
        trajs[i, :, 0, :] = u_xt.astype(np.float32)
        vs[i] = v
        ds[i] = D
        if (i + 1) % 64 == 0:
            print(f"  {i + 1}/{args.num_samples}")

    # Quantise (v, D) to give meaningful env_id (collisions per cell define an "environment")
    v_lo, v_hi = args.v_range
    d_lo, d_hi = args.d_range
    v_bin = np.clip(((vs - v_lo) / (v_hi - v_lo) * args.n_env_bins).astype(int), 0, args.n_env_bins - 1)
    d_bin = np.clip(((ds - d_lo) / (d_hi - d_lo) * args.n_env_bins).astype(int), 0, args.n_env_bins - 1)
    env_id = (v_bin * args.n_env_bins + d_bin).astype(np.int64)

    print(f"Writing {args.output} ...")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with h5py.File(args.output, "w") as fout:
        fout.create_dataset("trajectories", data=trajs, compression="lzf")
        fout.create_dataset("env_id", data=env_id)
        fout.create_dataset("env_params/advection_speed", data=vs)
        fout.create_dataset("env_params/diffusion", data=ds)
        fout.attrs["seed"] = args.seed
        fout.attrs["L"] = args.L
        fout.attrs["T"] = args.T
        fout.attrs["nx"] = args.nx
        fout.attrs["nt"] = args.nt
    print(f"Done. {args.num_samples} samples × {args.nt} timesteps × {args.nx} grid points")
    print(f"Unique env_id buckets: {len(set(env_id.tolist()))}")


if __name__ == "__main__":
    main()

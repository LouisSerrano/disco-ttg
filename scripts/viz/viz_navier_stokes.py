"""Standalone viz driver for Navier-Stokes. Mirrors
test_navier_stokes_targeted.py without modifying it: codebook dictionary (17),
balanced sampling across viscosity buckets, paper recipe with
refinement_factor=4.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "/mnt/home/lserrano/disco-ball")

from test_time_compute.ttc_utils import DEVICE
from train.train_euler_diffusion_aggregate import DISCOLitModule
from src.utils.euler_ns_dataset import NavierStokesDatasetWrapper
from scripts.viz._common import run_viz_for_trajectory, make_run, finalize_run, git_sha


# Paper recipe for NS (per existing test_navier_stokes_targeted.py + bash launchers).
N_INPUT_FRAMES = 16
N_OUTPUT_FRAMES = 16               # H = 16 for NS (paper §5.1)
RANDOM_TRIALS  = 100               # paper §5.1: T = 100
RANDOM_BATCH   = 16
COMPOSITION_LENGTHS_RANDOM = [2, 3]
BEAM_WIDTH     = 4                 # paper §5.1: B = 4
BEAM_MAX_OPS   = 5
BEAM_THRESHOLD = 5.0
BEAM_BATCH     = 32
REFINEMENT_FACTOR = 4              # paper §5.1: 4 sub-steps per dt for NS
SPLITTING_METHOD  = "strang"
VORTICITY_SCALE   = 10.0


def load_model(model_path: str):
    if not os.path.exists(model_path):
        raise SystemExit(f"checkpoint not found: {model_path}")
    lit = DISCOLitModule.load_from_checkpoint(model_path, map_location=DEVICE)
    lit.eval()
    model = lit.model.to(DEVICE).eval()
    print(f"Loaded model with {sum(p.numel() for p in model.parameters()):,} params")
    return model, lit


def select_samples_balanced_across_viscosity(test_ds, num_samples: int, seed: int = 42):
    """Pick num_samples trajectories spread across viscosity buckets, mirroring
    the curated selection used by test_navier_stokes_targeted.py.

    Reads the dataset's ``.indices`` directly (a list of
    ``(file_path, local_idx, visc_label)`` 3-tuples for NavierStokesDataset),
    avoiding the per-sample HDF5 round-trip that the previous implementation
    paid. Per-viscosity ordering uses ``seed=42`` so this matches the same
    trajectories evaluated for the paper.
    """
    import numpy as np

    ds = test_ds.dataset
    visc_groups: dict[int, list[int]] = {}
    for global_idx, entry in enumerate(ds.indices):
        # NavierStokesDataset: (file_path, local_idx, visc_label) — 3-tuple.
        visc_label = int(entry[2])
        visc_groups.setdefault(visc_label, []).append(global_idx)

    visc_keys = sorted(visc_groups.keys())
    if len(visc_keys) < 2:
        raise RuntimeError(
            f"Expected multiple viscosity environments; got {len(visc_keys)} "
            f"({visc_keys}). Check that test_ds spans all viscosities."
        )

    rng = np.random.RandomState(seed)
    # How many per viscosity? Spread num_samples evenly; remainder fills the
    # earliest buckets.
    base = num_samples // len(visc_keys)
    extra = num_samples - base * len(visc_keys)

    sizes = {}
    selected: list[int] = []
    for i, v in enumerate(visc_keys):
        take = base + (1 if i < extra else 0)
        if take == 0: continue
        pool = visc_groups[v]
        shuffled = rng.permutation(pool)
        picks = shuffled[: min(take, len(shuffled))].tolist()
        selected.extend(int(x) for x in picks)
        sizes[v] = len(picks)

    print(f"  viscosity coverage: {len(visc_keys)} buckets, "
          f"per-bucket counts={sizes}")
    return selected


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_samples", type=int, default=8)
    p.add_argument("--uniform_save_every", type=int, default=5,
                   help="Save a uniform snapshot every N trials. Default 5 (= 20 snapshots).")
    p.add_argument("--file_dir", default="./datasets/euler_ns_short/")
    p.add_argument("--num_gpus", type=int, default=8)
    p.add_argument("--N_ns_ics", type=int, default=512)
    p.add_argument("--scan_mode", action="store_true",
                   help="Skip uniform search; only direct + final beam.")
    args = p.parse_args()

    model, lit = load_model(args.model_path)
    theta = lit.codebook.to(DEVICE)
    print(f"Codebook dictionary: {theta.shape[0]} operators")
    dt = getattr(model, "default_integration_time", 0.08)
    print(f"dt = {dt}")

    test_ds = NavierStokesDatasetWrapper(
        file_dir=args.file_dir, num_gpus=args.num_gpus,
        input_frames=N_INPUT_FRAMES, output_frames=N_OUTPUT_FRAMES,
        sub_x=1, sub_t=1,
        N_ns_ics=args.N_ns_ics, vorticity_scale=VORTICITY_SCALE,
    )
    print(f"Test set: {len(test_ds)} samples across {test_ds.num_environments} viscosities")

    selected = select_samples_balanced_across_viscosity(test_ds, args.num_samples)
    print(f"Selected {len(selected)} sample indices (balanced across viscosities)")

    manifest = {
        "equation_type": "navier_stokes",
        "experiment": "composition",
        "model_path": args.model_path,
        "file_dir": args.file_dir,
        "n_input_frames": N_INPUT_FRAMES,
        "n_output_frames": N_OUTPUT_FRAMES,
        "dt": dt,
        "vorticity_scale": VORTICITY_SCALE,
        "dictionary_source": "codebook",
        "n_dict_operators": int(theta.shape[0]),
        "random_trials": RANDOM_TRIALS,
        "uniform_save_every": args.uniform_save_every,
        "composition_lengths_random": COMPOSITION_LENGTHS_RANDOM,
        "beam_width": BEAM_WIDTH,
        "beam_max_operators": BEAM_MAX_OPS,
        "beam_threshold": BEAM_THRESHOLD,
        "refinement_factor": REFINEMENT_FACTOR,
        "splitting_method": SPLITTING_METHOD,
        "git_sha": git_sha(),
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }
    run_dir = make_run(args.output_dir, manifest)
    print(f"Run dir: {run_dir}")

    for cand_i, ds_idx in enumerate(selected):
        sample = test_ds[ds_idx]
        inp = sample["input"].unsqueeze(0).to(DEVICE)
        tgt = sample["output"].unsqueeze(0).to(DEVICE)
        visc = float(sample.get("viscosity", 0.0)) if "viscosity" in sample else 0.0
        cand_dir = os.path.join(run_dir, f"cand_{cand_i:04d}")
        print(f"\n[{cand_i}/{len(selected)}] ds_idx={ds_idx}  viscosity={visc:.5f}")
        run_viz_for_trajectory(
            cand_dir=cand_dir,
            inp=inp, tgt=tgt,
            model=model, theta_latent_operators=theta,
            dt=dt,
            uniform_trials=RANDOM_TRIALS,
            uniform_save_every=args.uniform_save_every,
            uniform_batch_size=RANDOM_BATCH,
            composition_lengths_random=COMPOSITION_LENGTHS_RANDOM,
            beam_width=BEAM_WIDTH,
            beam_max_operators=BEAM_MAX_OPS,
            beam_threshold=BEAM_THRESHOLD,
            beam_batch_size=BEAM_BATCH,
            splitting_method=SPLITTING_METHOD,
            refinement_factor=REFINEMENT_FACTOR,
            extra_gt_fields={"viscosity": visc, "ds_idx": ds_idx},
            extra_metadata={**manifest,
                             "trajectory_idx": cand_i,
                             "dataset_index": int(ds_idx),
                             "viscosity": visc},
            scan_mode=args.scan_mode,
        )

    finalize_run(args.output_dir, run_dir)
    print(f"\nDone. {len(selected)} candidates → {run_dir}")


if __name__ == "__main__":
    main()

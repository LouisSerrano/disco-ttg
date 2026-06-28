"""Standalone viz driver for combined-equation experiments (E_BG, E_ED, E_HE,
E_ALL, E_EULER_OOD, E_DISP_OOD). Mirrors test_combined_equation.py without
modifying it: codebook-subsampled dictionary, paper recipe per §5.1.
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

from test_time_compute.ttc_utils import DEVICE, CombinedHDF5TemporalDataset
from train.train_combined_aggregate import DISCOLitModule
from scripts.viz._common import run_viz_for_trajectory, make_run, finalize_run, git_sha


TRAINING_FILES = [
    "./datasets/combined_equation/E_EULER_train_gridparam256.h5",
    "./datasets/combined_equation/E_HEAT_train_gridparam8192.h5",
    "./datasets/combined_equation/E_DISP_train_gridparam8192.h5",
]
EXPERIMENT_FILES = {
    "E_BG":         "./datasets/E_BG_train_gridparam512.h5",
    "E_ED":         "./datasets/E_ED_train_gridparam512.h5",
    "E_HE":         "./datasets/E_HE_train_gridparam512.h5",
    "E_ALL":        "./datasets/E_ALL_train_gridparam512.h5",
    "E_EULER_OOD":  "./datasets/E_EULER_OOD_train_envsize16.h5",
    "E_DISP_OOD":   "./datasets/E_DISP_OOD_train_envsize16.h5",
}

# Paper recipe (per bash/combined-equation/test/random.sh + script defaults).
DT             = 4.0 / 250
N_INPUT_FRAMES = 16
N_OUTPUT_FRAMES = 50               # H = 50 for combined-eq (paper §5.1)
RANDOM_TRIALS  = 100               # paper §5.1: T = 100
RANDOM_BATCH   = 32
COMPOSITION_LENGTHS_RANDOM = [2, 3]   # paper recipe matches test_combined_equation.py for most cases
BEAM_WIDTH     = 4                 # paper §5.1: B = 4
BEAM_MAX_OPS   = 5
BEAM_THRESHOLD = 5.0
BEAM_BATCH     = 32
CODEBOOK_SUBSAMPLE = 4             # matches the existing script: codebook[::4] ≈ 32 ops


def load_model(model_path: str):
    if not os.path.exists(model_path):
        raise SystemExit(f"checkpoint not found: {model_path}")
    lit = DISCOLitModule.load_from_checkpoint(model_path, map_location=DEVICE)
    lit.eval()
    model = lit.model.to(DEVICE).eval()
    print(f"Loaded model with {sum(p.numel() for p in model.parameters()):,} params")
    return model, lit


def build_codebook_dictionary(lit_model):
    """Subsample the codebook by 4, matching test_combined_equation.py."""
    theta_latent = lit_model.codebook[::CODEBOOK_SUBSAMPLE]
    print(f"Codebook dictionary: {theta_latent.shape[0]} operators (subsampled by {CODEBOOK_SUBSAMPLE})")
    return theta_latent.to(DEVICE)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--experiment", required=True, choices=list(EXPERIMENT_FILES))
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_samples", type=int, default=8)
    p.add_argument("--uniform_save_every", type=int, default=1)
    p.add_argument("--scan_mode", action="store_true",
                   help="Skip uniform search; only direct + final beam.")
    p.add_argument("--samples_per_param", type=int, default=None,
                   help="If given, sample N ICs per unique (alpha,beta,gamma) param combo. "
                        "Spans the parameter space instead of taking the first --num_samples sequentially.")
    p.add_argument("--max_param_combos", type=int, default=None,
                   help="If given (with --samples_per_param), cap the number of param combos "
                        "to this many, evenly spaced across the unique combos.")
    args = p.parse_args()

    test_file = EXPERIMENT_FILES[args.experiment]
    if not os.path.exists(test_file):
        raise SystemExit(f"test file not found: {test_file}")

    model, lit = load_model(args.model_path)
    theta = build_codebook_dictionary(lit)

    test_ds = CombinedHDF5TemporalDataset(
        hdf5_files=[test_file],
        input_frames=N_INPUT_FRAMES, output_frames=N_OUTPUT_FRAMES,
        sub_x=1, sub_t=1, split="train",
    )
    print(f"Test set: {len(test_ds)} samples in {test_file}")

    manifest = {
        "equation_type": "combined_equation",
        "experiment": args.experiment,
        "test_file": test_file,
        "model_path": args.model_path,
        "n_input_frames": N_INPUT_FRAMES,
        "n_output_frames": N_OUTPUT_FRAMES,
        "dt": DT,
        "dictionary_source": "codebook",
        "codebook_subsample": CODEBOOK_SUBSAMPLE,
        "n_dict_operators": int(theta.shape[0]),
        "random_trials": RANDOM_TRIALS,
        "uniform_save_every": args.uniform_save_every,
        "composition_lengths_random": COMPOSITION_LENGTHS_RANDOM,
        "beam_width": BEAM_WIDTH,
        "beam_max_operators": BEAM_MAX_OPS,
        "beam_threshold": BEAM_THRESHOLD,
        "git_sha": git_sha(),
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }
    run_dir = make_run(args.output_dir, manifest)
    print(f"Run dir: {run_dir}")

    # ---- Select sample indices spanning the parameter space ----
    if args.samples_per_param is not None:
        # Group sample indices by unique (alpha, beta, gamma) — these are stored
        # in the underlying GenericHDF5Dataset and surfaced via metadata in each
        # __getitem__. Walking once is fine here, the test sets are O(500-1600).
        import numpy as np, collections
        by_param: dict[tuple, list[int]] = collections.defaultdict(list)
        for i in range(len(test_ds)):
            s = test_ds[i]
            key = (round(float(s.get("alpha", 0.0)), 4),
                   round(float(s.get("beta",  0.0)), 4),
                   round(float(s.get("gamma", 0.0)), 4))
            by_param[key].append(i)
        keys_sorted = sorted(by_param.keys())
        if args.max_param_combos and len(keys_sorted) > args.max_param_combos:
            step = max(1, len(keys_sorted) // args.max_param_combos)
            keys_sorted = keys_sorted[::step][:args.max_param_combos]
        selected = []
        for k in keys_sorted:
            ics = by_param[k][: args.samples_per_param]
            selected.extend(ics)
        print(f"Selected {len(selected)} cands  ({len(keys_sorted)} param combos × ≤{args.samples_per_param} ICs)")
    else:
        selected = list(range(min(args.num_samples, len(test_ds))))
        print(f"Sequential selection: first {len(selected)} samples (no param spreading)")

    for cand_i, ds_idx in enumerate(selected):
        sample = test_ds[ds_idx]
        inp = sample["input"].unsqueeze(0).to(DEVICE)
        # CombinedHDF5TemporalDataset returns the rollout target under "output"
        tgt = sample["output"].unsqueeze(0).to(DEVICE)
        alpha = float(sample.get("alpha", 0.0))
        beta  = float(sample.get("beta",  0.0))
        gamma = float(sample.get("gamma", 0.0))
        cand_dir = os.path.join(run_dir, f"cand_{cand_i:04d}")
        print(f"\n[{cand_i}/{len(selected)}] ds_idx={ds_idx}  α={alpha:.3f}  β={beta:.3f}  γ={gamma:.3f}")
        run_viz_for_trajectory(
            cand_dir=cand_dir,
            inp=inp, tgt=tgt,
            model=model, theta_latent_operators=theta,
            dt=DT,
            uniform_trials=RANDOM_TRIALS,
            uniform_save_every=args.uniform_save_every,
            uniform_batch_size=RANDOM_BATCH,
            composition_lengths_random=COMPOSITION_LENGTHS_RANDOM,
            beam_width=BEAM_WIDTH,
            beam_max_operators=BEAM_MAX_OPS,
            beam_threshold=BEAM_THRESHOLD,
            beam_batch_size=BEAM_BATCH,
            refinement_factor=1,
            extra_gt_fields={"alpha": alpha, "beta": beta, "gamma": gamma,
                             "dataset_index": int(ds_idx)},
            extra_metadata={**manifest, "trajectory_idx": cand_i,
                             "dataset_index": int(ds_idx),
                             "alpha": alpha, "beta": beta, "gamma": gamma},
            scan_mode=args.scan_mode,
        )

    finalize_run(args.output_dir, run_dir)
    print(f"\nDone. {len(selected)} candidates → {run_dir}")


if __name__ == "__main__":
    main()

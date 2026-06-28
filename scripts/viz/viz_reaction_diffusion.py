"""Standalone viz driver for Gray-Scott reaction-diffusion. Mirrors
test_reaction_diffusion.py without modifying it: full codebook dictionary,
paper recipe (B = 8 for GS) per §5.1.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import torch

sys.path.insert(0, "/mnt/home/lserrano/disco-ball")

from test_time_compute.ttc_utils import DEVICE, GrayScottDatasetWrapper
from train.train_rd_aggregate import DISCOLitModule
from scripts.viz._common import run_viz_for_trajectory, make_run, finalize_run, git_sha


TEST_FILES = ["./datasets/gray-scott/gray_scott_10x10_params_16traj_each.hdf5"]

# Paper recipe (per bash/rd/test*.sh).
N_INPUT_FRAMES = 16
N_OUTPUT_FRAMES = 32              # H = 32 for Gray-Scott (paper §5.1)
RANDOM_TRIALS  = 200              # paper §5.1: T = 200 for GS
RANDOM_BATCH   = 16
COMPOSITION_LENGTHS_RANDOM = [2]
BEAM_WIDTH     = 8                # paper §5.1: B = 8 for GS
BEAM_MAX_OPS   = 5
BEAM_THRESHOLD = 5.0
BEAM_BATCH     = 16


def load_model(model_path: str):
    if not os.path.exists(model_path):
        raise SystemExit(f"checkpoint not found: {model_path}")
    lit = DISCOLitModule.load_from_checkpoint(model_path, map_location=DEVICE)
    lit.eval()
    model = lit.model.to(DEVICE).eval()
    print(f"Loaded model with {sum(p.numel() for p in model.parameters()):,} params")
    return model, lit


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_samples", type=int, default=8)
    p.add_argument("--uniform_save_every", type=int, default=10,
                   help="Save a uniform snapshot every N trials. Default 10 (= 20 snapshots from T=200).")
    p.add_argument("--scan_mode", action="store_true",
                   help="Skip uniform search; only direct + final beam.")
    p.add_argument("--samples_per_param", type=int, default=None,
                   help="Sample N ICs per unique (F, k) param combo.")
    p.add_argument("--max_param_combos", type=int, default=None,
                   help="Cap number of param combos, evenly spaced.")
    args = p.parse_args()

    model, lit = load_model(args.model_path)
    theta = lit.codebook.to(DEVICE)
    print(f"Codebook dictionary: {theta.shape[0]} operators")

    # dt is bound to the model.
    dt = getattr(model, "default_integration_time", 0.16)
    print(f"dt = {dt}")

    test_ds = GrayScottDatasetWrapper(
        hdf5_files=TEST_FILES, split="test",
        input_frames=N_INPUT_FRAMES, output_frames=N_OUTPUT_FRAMES,
        sub_x=1, sub_t=1, trajectories_per_environment=16,
    )
    print(f"Test set: {len(test_ds)} samples")

    manifest = {
        "equation_type": "reaction_diffusion",
        "experiment": "composition",
        "test_files": TEST_FILES,
        "model_path": args.model_path,
        "n_input_frames": N_INPUT_FRAMES,
        "n_output_frames": N_OUTPUT_FRAMES,
        "dt": dt,
        "dictionary_source": "codebook",
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

    if args.samples_per_param is not None:
        import collections
        by_param: dict[tuple, list[int]] = collections.defaultdict(list)
        for i in range(len(test_ds)):
            s = test_ds[i]
            key = (round(float(s.get("f", 0.0)), 5),
                   round(float(s.get("k", 0.0)), 5))
            by_param[key].append(i)
        keys_sorted = sorted(by_param.keys())
        if args.max_param_combos and len(keys_sorted) > args.max_param_combos:
            step = max(1, len(keys_sorted) // args.max_param_combos)
            keys_sorted = keys_sorted[::step][:args.max_param_combos]
        selected = []
        for k in keys_sorted:
            selected.extend(by_param[k][: args.samples_per_param])
        print(f"Selected {len(selected)} cands  ({len(keys_sorted)} (F,k) combos × ≤{args.samples_per_param} ICs)")
    else:
        selected = list(range(min(args.num_samples, len(test_ds))))
        print(f"Sequential selection: first {len(selected)} samples")

    for cand_i, ds_idx in enumerate(selected):
        sample = test_ds[ds_idx]
        inp = sample["input"].unsqueeze(0).to(DEVICE)
        tgt = sample["output"].unsqueeze(0).to(DEVICE)
        f_val = float(sample.get("f", 0.0))
        k_val = float(sample.get("k", 0.0))
        cand_dir = os.path.join(run_dir, f"cand_{cand_i:04d}")
        print(f"\n[{cand_i}/{len(selected)}] ds_idx={ds_idx}  f={f_val:.4f}  k={k_val:.4f}")
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
            refinement_factor=1,
            extra_gt_fields={"f": f_val, "k": k_val, "dataset_index": int(ds_idx)},
            extra_metadata={**manifest, "trajectory_idx": cand_i,
                             "dataset_index": int(ds_idx),
                             "f": f_val, "k": k_val},
            scan_mode=args.scan_mode,
        )

    finalize_run(args.output_dir, run_dir)
    print(f"\nDone. {len(selected)} candidates → {run_dir}")


if __name__ == "__main__":
    main()

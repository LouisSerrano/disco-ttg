"""Standalone viz driver for advection-diffusion experiments (E_AD_ALL,
E_AD_v, E_AD_D). Mirrors the setup in
``test_time_compute/equations/test_advection_diffusion.py`` but never modifies
that paper script; instead this script replicates the model loading, the
encoder-based dictionary, and the experiment parameter table, then runs the
viz path (per-trial uniform + per-depth beam snapshots) on the first
``--num_samples`` test trajectories.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import torch

# Allow `python scripts/viz/viz_advection_diffusion.py` from repo root.
sys.path.insert(0, "/mnt/home/lserrano/disco-ball")

from test_time_compute.ttc_utils import DEVICE
from train.train import DISCOLitModule, TemporalBatchDatasetFly
from scripts.viz._common import run_viz_for_trajectory, make_run, finalize_run, git_sha


# Canonical experiment table — same as test_advection_diffusion.py.
EXPERIMENT_CONFIGS = {
    "E_AD_ALL": {"v_range": (0.01, 1.0), "D_range": (0.01, 1.0),
                  "description": "Both advection and diffusion in [0,1] (in-dist composition, paper Table 1)"},
    "E_AD_v":   {"v_range": (1.0, 3.0),  "D_range": (0.0, 0.0),
                  "description": "High advection [1,3], no diffusion (param extrapolation, paper Table 2)"},
    "E_AD_D":   {"v_range": (0.0, 0.0),  "D_range": (1.0, 3.0),
                  "description": "High diffusion [1,3], no advection (param extrapolation, paper Table 2)"},
}

# Paper recipe (per bash/advection-diffusion/test/random.sh + scaling_law.sh).
DT             = 10.0 / 100        # 0.1 s per frame
N_INPUT_FRAMES = 16
N_OUTPUT_FRAMES = 34               # H = 34 for adv-diff (paper §5.1)
RANDOM_TRIALS  = 100               # paper §5.1: T = 100
RANDOM_BATCH   = 32
COMPOSITION_LENGTHS_RANDOM = [2, 3, 4]
BEAM_WIDTH     = 4                 # paper §5.1: B = 4
BEAM_MAX_OPS   = 5                 # paper §5.1: M = 5
BEAM_THRESHOLD = 5.0               # paper §5.1: 5% improvement
BEAM_BATCH     = 32


def load_model(model_path: str):
    if not os.path.exists(model_path):
        raise SystemExit(f"checkpoint not found: {model_path}")
    lit = DISCOLitModule.load_from_checkpoint(model_path, map_location=DEVICE)
    lit.eval()
    model = lit.model.to(DEVICE).eval()
    print(f"Loaded model with {sum(p.numel() for p in model.parameters()):,} params")
    return model, lit


def build_encoder_dictionary(model, *, n_dict_batches: int = 4):
    """Replicates the test_advection_diffusion.py encoder pass."""
    train_ds = TemporalBatchDatasetFly(
        n_batches=n_dict_batches, batch_size=64,
        sub_x=1, sub_t=1, split="train",
        input_frames=N_INPUT_FRAMES, output_frames=N_OUTPUT_FRAMES,
        L=16.0, nx=256, nt=100, T=10.0,
        v_range=(0.01, 1.0), D_range=(0.001, 1.0),
        fractal_degree=256, fractal_power_range=3, seed=42,
    )
    state_labels = torch.tensor([0], device=DEVICE)
    latents = []
    for batch in train_ds:
        inp = batch["input"].to(DEVICE)
        with torch.no_grad():
            theta_latent, _ = model.encode_theta_latent(inp, state_labels)
        latents.append(theta_latent)
    theta = torch.cat(latents)
    print(f"Encoder dictionary: {theta.shape[0]} operators, dim {theta.shape[1]}")
    return theta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--experiment", required=True, choices=list(EXPERIMENT_CONFIGS))
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_samples", type=int, default=8)
    p.add_argument("--uniform_save_every", type=int, default=1,
                   help="Save a uniform snapshot every N trials (default: every trial = 100 snapshots).")
    p.add_argument("--n_dict_batches", type=int, default=4)
    p.add_argument("--scan_mode", action="store_true",
                   help="Skip uniform search; only direct + final beam. ~5x faster per cand.")
    args = p.parse_args()

    cfg = EXPERIMENT_CONFIGS[args.experiment]
    print(f"Experiment {args.experiment}: {cfg['description']}")

    model, lit = load_model(args.model_path)
    theta = build_encoder_dictionary(model, n_dict_batches=args.n_dict_batches)

    # Test set: deterministic seed=124 (matches test_advection_diffusion.py default).
    test_n_batches = max(1, (args.num_samples + 63) // 64)
    test_ds = TemporalBatchDatasetFly(
        n_batches=test_n_batches, batch_size=64,
        sub_x=1, sub_t=1, split="test",
        input_frames=N_INPUT_FRAMES, output_frames=N_OUTPUT_FRAMES,
        L=16.0, nx=256, nt=100, T=10.0,
        v_range=cfg["v_range"], D_range=cfg["D_range"],
        fractal_degree=256, fractal_power_range=3, seed=124,
    )

    manifest = {
        "equation_type": "advection_diffusion",
        "experiment": args.experiment,
        "experiment_description": cfg["description"],
        "v_range": list(cfg["v_range"]),
        "D_range": list(cfg["D_range"]),
        "model_path": args.model_path,
        "n_input_frames": N_INPUT_FRAMES,
        "n_output_frames": N_OUTPUT_FRAMES,
        "dt": DT,
        "dictionary_source": "encoder",
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

    sample_idx = 0
    for batch in test_ds:
        inp_b = batch["input"].to(DEVICE)
        tgt_b = batch["target"].to(DEVICE)
        v_b   = batch.get("advection_speed", torch.zeros(inp_b.size(0)))
        D_b   = batch.get("diffusion",       torch.zeros(inp_b.size(0)))
        for i in range(inp_b.size(0)):
            if sample_idx >= args.num_samples:
                break
            cand_dir = os.path.join(run_dir, f"cand_{sample_idx:04d}")
            print(f"\n[{sample_idx}/{args.num_samples}] v={float(v_b[i]):.3f}  D={float(D_b[i]):.3f}")
            run_viz_for_trajectory(
                cand_dir=cand_dir,
                inp=inp_b[i:i+1], tgt=tgt_b[i:i+1],
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
                extra_gt_fields={"advection_speed": float(v_b[i]), "diffusion": float(D_b[i])},
                extra_metadata={**manifest,
                                "trajectory_idx": sample_idx,
                                "advection_speed": float(v_b[i]),
                                "diffusion":       float(D_b[i])},
                scan_mode=args.scan_mode,
            )
            sample_idx += 1
        if sample_idx >= args.num_samples:
            break

    finalize_run(args.output_dir, run_dir)
    print(f"\nDone. {sample_idx} candidates → {run_dir}")


if __name__ == "__main__":
    main()

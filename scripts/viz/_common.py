"""Shared helpers used by the per-equation viz drivers.

Each driver builds a run dir, runs DISCO on N test trajectories, and saves
per-candidate ``gt.npz``, ``disco_original.npz`` (direct), ``disco_uniform.npz``
(per-trial snapshots), ``disco_beam.npz`` (per-depth snapshots), and
``metadata.json``. The renderer downstream reads these files uniformly.

Direct prediction uses the canonical ``model(inp, state_labels, n_future_steps=H)``
forward call — this is the bug we found in the modified test_*.py scripts
(now reverted). Uniform + beam use the validated ``*_with_predictions``
TTC functions.

Paper canonical settings per benchmark come from the existing launchers in
bash/<equation>/test/ — see the per-driver constants.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from typing import Optional

import numpy as np
import torch

from test_time_compute import viz_io
from test_time_compute.ttc_utils import DEVICE
from test_time_compute.ttc_methods import (
    get_state_labels,
    random_operator_selection_batch_with_predictions,
    beam_search_operator_selection_batch_with_predictions,
)


def run_viz_for_trajectory(
    *,
    cand_dir: str,
    inp: torch.Tensor,
    tgt: torch.Tensor,
    model,
    theta_latent_operators: torch.Tensor,
    dt: float,
    # uniform args
    uniform_trials: int,
    uniform_save_every: int,
    uniform_batch_size: int,
    composition_lengths_random: list,
    # beam args
    beam_width: int,
    beam_max_operators: int,
    beam_threshold: float,
    beam_batch_size: int,
    # shared
    splitting_method: str = "strang",
    refinement_factor: int = 1,
    extra_gt_fields: Optional[dict] = None,
    extra_metadata: Optional[dict] = None,
    scan_mode: bool = False,
):
    """Run direct + uniform + beam on one trajectory and write all four npz files
    plus metadata.json under cand_dir."""
    os.makedirs(cand_dir, exist_ok=True)
    state_labels = get_state_labels(inp)

    # GT save (input is what the model was given; target is the rollout reference).
    viz_io.save_gt_npz(cand_dir, inp, tgt, **(extra_gt_fields or {}))

    # 1) Direct prediction — canonical forward path.
    with torch.no_grad():
        out = model(inp, state_labels, n_future_steps=tgt.shape[1])
        pred_direct = out[0] if isinstance(out, tuple) else out
    direct_err = (torch.norm(pred_direct - tgt) / torch.norm(tgt)).item()
    viz_io.save_single_prediction(cand_dir, "disco_original.npz",
                                  pred_direct, test_error=direct_err)
    print(f"  direct test_err = {direct_err:.4f}")

    # 2) Uniform random search with per-trial snapshots.
    if scan_mode:
        # Broad-scan: skip uniform entirely. Saves the ~50% per-cand cost of
        # running 100 trials + their full-rollout snapshots. The point of the
        # scan pass is to find visually impressive cases by eye — we refine
        # the picks afterwards with the full feature set.
        uniform_err = float("nan")
        snaps_u = {}
        print(f"  uniform: SKIPPED (scan_mode)")
    else:
        save_at = list(range(uniform_save_every, uniform_trials + 1, uniform_save_every))
        _, uniform_err, _, snaps_u = random_operator_selection_batch_with_predictions(
            model, theta_latent_operators, inp, tgt,
            num_compositions=uniform_trials,
            composition_lengths=composition_lengths_random,
            dt=dt,
            random_batch_size=uniform_batch_size,
            splitting_method=splitting_method,
            refinement_factor=refinement_factor,
            save_at_trials=save_at,
        )
        viz_io.save_snapshots_bundle(cand_dir, "disco_uniform.npz", snaps_u, "trials")
        print(f"  uniform final test_err = {uniform_err:.4f}  ({len(snaps_u)} snapshots)")

    # 3) Beam search with per-depth snapshots.
    _, beam_err, _, snaps_b = beam_search_operator_selection_batch_with_predictions(
        model, theta_latent_operators, inp, tgt,
        beam_width=beam_width,
        max_operators=beam_max_operators,
        min_improvement_threshold=beam_threshold,
        dt=dt,
        batch_size=beam_batch_size,
        splitting_method=splitting_method,
        refinement_factor=refinement_factor,
    )
    viz_io.save_snapshots_bundle(cand_dir, "disco_beam.npz", snaps_b, "depth")
    print(f"  beam final test_err = {beam_err:.4f}  (depth snapshots = {sorted(snaps_b)})")

    # Per-trajectory metadata
    md = {
        **(extra_metadata or {}),
        "direct_test_error": float(direct_err),
        "uniform_test_error": float(uniform_err),
        "beam_test_error":    float(beam_err),
        "uniform_n_snapshots": len(snaps_u),
        "beam_depths":        sorted(snaps_b),
    }
    with open(os.path.join(cand_dir, "metadata.json"), "w") as fh:
        json.dump(md, fh, indent=2, default=str)
    return direct_err, uniform_err, beam_err


def make_run(output_dir: str, manifest: dict) -> str:
    """Create a fresh timestamped run dir, write manifest, update 'latest'."""
    run_dir = viz_io.make_run_dir(output_dir)
    viz_io.write_manifest(run_dir, manifest)
    return run_dir


def finalize_run(output_dir: str, run_dir: str) -> None:
    viz_io.update_latest_symlink(output_dir, run_dir)


def git_sha() -> str:
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "-C", "/mnt/home/lserrano/disco-ball", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return ""

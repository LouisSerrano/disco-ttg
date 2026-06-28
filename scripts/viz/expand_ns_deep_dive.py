"""For each NS pick, replay the beam composition depth-by-depth and write a
multi-depth disco_beam.npz in-place.

The targeted-NS ingest only saved the FINAL beam prediction (1 depth). To
animate "composition over time" we need rollouts for each prefix of the
composition: depth=1 uses [c1], depth=2 uses [c1,c2], etc. Each prefix is a
single forward pass through ``_rollout_for_composition``.

Output schema matches what the viz pipeline expects when full mode was used:
    disco_beam.npz keys:
      keys:               (D,) int — depth labels [1,2,...]
      preds:              (D, T, C, H, W) float32 — per-depth prediction
      fit_errors:         (D,) float32 — NaN here (we didn't fit, just rolled out)
      test_errors:        (D,) float32 — relative-L2 vs target
      compositions_json:  json string — list of compositions, one per depth
      keys_label:         "depth"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/mnt/home/lserrano/disco-ball")

from test_time_compute.ttc_utils import DEVICE
from test_time_compute.ttc_methods import _rollout_for_composition, get_state_labels
from train.train_euler_diffusion_aggregate import DISCOLitModule  # NS-specific


NS_CKPT = "/mnt/home/lserrano/ceph/disco/outputs/DISCO_euler_solverrk4_adjFalse_h128_t4_steps4_initFalse_bs16_lr0.0003_hdf5_noise0_inframes16_outframes2_subx1_subt1_20260124_041037/best-checkpoint.ckpt"

# Paper recipe for NS
DT = 0.08
REFINEMENT_FACTOR = 4
SPLITTING_METHOD = "strang"


def load_ns_model():
    print("Loading NS model + codebook ...")
    lit = DISCOLitModule.load_from_checkpoint(NS_CKPT, map_location=DEVICE)
    lit.eval()
    model = lit.model.to(DEVICE).eval()
    theta = lit.codebook.to(DEVICE)
    print(f"  model has {sum(p.numel() for p in model.parameters()):,} params")
    print(f"  codebook: {theta.shape[0]} operators")
    return model, theta


def expand_one(cand_dir: Path, model, theta):
    md = json.load(open(cand_dir / "metadata.json"))
    composition = md.get("beam_composition") or []
    if not composition:
        print(f"  {cand_dir.name}: no composition in metadata — skipping")
        return

    # Load IC + target
    gt = dict(np.load(cand_dir / "gt.npz", allow_pickle=True))
    inp_np = gt["input"]
    tgt_np = gt["target"]
    inp = torch.from_numpy(inp_np).unsqueeze(0).to(DEVICE).float()
    tgt = torch.from_numpy(tgt_np).unsqueeze(0).to(DEVICE).float()

    dim = inp.ndim - 3                                          # 2 for NS (H,W)
    # NS state_labels come from get_state_labels — not the viscosity index.
    # The viscosity is captured by the chosen operators, not the adapter labels.
    state_labels = get_state_labels(inp)

    preds, errs, comp_history = [], [], []
    for d in range(1, len(composition) + 1):
        prefix = composition[:d]
        pred, err = _rollout_for_composition(
            model, theta, prefix, inp, tgt,
            dt=DT, state_labels=state_labels, dim=dim,
            refinement_factor=REFINEMENT_FACTOR,
            splitting_method=SPLITTING_METHOD,
        )
        preds.append(pred.detach().cpu().numpy()[0].astype(np.float32))
        errs.append(float(err))
        comp_history.append(prefix)
        print(f"    depth {d}: composition={prefix}  test_err={err:.4f}")

    # Stack and write
    preds_arr = np.stack(preds, axis=0)
    np.savez(cand_dir / "disco_beam.npz",
              keys=np.arange(1, len(composition) + 1, dtype=np.int64),
              preds=preds_arr,
              fit_errors=np.array([float("nan")] * len(composition), dtype=np.float32),
              test_errors=np.array(errs, dtype=np.float32),
              compositions_json=np.array(json.dumps(comp_history), dtype=object),
              keys_label=np.array("depth", dtype=object))
    # Annotate metadata so downstream knows this was reconstructed
    md["deep_dive_expanded"] = True
    md["per_depth_errors"] = errs
    md["per_depth_compositions"] = comp_history
    with open(cand_dir / "metadata.json", "w") as f:
        json.dump(md, f, indent=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cand_names", required=True,
                   help="Comma-separated cand names, e.g. cand_0096,cand_0120,cand_0037")
    p.add_argument("--ns_run", default="viz/data/scan/navier-stokes/run_20260628_111801")
    args = p.parse_args()

    cand_dirs = [Path(args.ns_run) / n for n in args.cand_names.split(",")]
    for cd in cand_dirs:
        if not cd.exists():
            raise SystemExit(f"missing: {cd}")

    model, theta = load_ns_model()
    for cd in cand_dirs:
        print(f"\n=== {cd.name} ===")
        expand_one(cd, model, theta)


if __name__ == "__main__":
    main()

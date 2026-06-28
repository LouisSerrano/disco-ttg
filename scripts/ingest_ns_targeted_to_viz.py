"""Ingest the canonical paper-eval per-sample outputs at
``results_targeted/{beam,direct}/visc_*/sample_*/`` into our viz npz schema.

Background
----------
The paper-canonical NS eval (test_navier_stokes_targeted.py) writes one dir
per (viscosity, sample) containing the GT trajectory plus the model's
prediction tensor:

    visc_X_YYYYYY/sample_Y/
        input.pt        (T_in, C, H, W)   — context frames
        target.pt       (T_out, C, H, W)  — ground truth
        prediction.pt   (T_out, C, H, W)  — beam OR direct prediction
        metadata.json   — composition, viscosity, error, ...
        plot.png        — pre-rendered preview

For the website hero we want to show BOTH the easy and the hard viscosities,
so we ingest one sample per viscosity (default ``--select best_per_visc``)
into our viz schema:

    viz/data/navier-stokes_canonical/run_<ts>/cand_<NNNN>/
        gt.npz                 # {input, target, viscosity, visc_idx}
        disco_original.npz     # pred from results_targeted/direct/...
        disco_beam.npz         # 1-snapshot bundle (depth = composition length)
        metadata.json

The renderer reads these like any other cand dir. Scrubber animations are
unavailable from this source (the canonical eval didn't save per-trial
snapshots) — only the static hero + rollout overlay get produced.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from test_time_compute import viz_io


def _load_sample(sample_dir: str) -> dict:
    out = {}
    for k in ("input", "target", "prediction"):
        p = os.path.join(sample_dir, f"{k}.pt")
        out[k] = torch.load(p, map_location="cpu", weights_only=True).numpy().astype(np.float32)
    with open(os.path.join(sample_dir, "metadata.json")) as f:
        out["meta"] = json.load(f)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--beam_root", default="/mnt/home/lserrano/disco-ball/results_targeted/beam")
    p.add_argument("--direct_root", default="/mnt/home/lserrano/disco-ball/results_targeted/direct")
    p.add_argument("--output_dir", default="/mnt/home/lserrano/disco-ball/viz/data/navier-stokes_canonical")
    p.add_argument("--select", choices=["best_per_visc", "worst_per_visc", "first_per_visc", "all"],
                   default="best_per_visc",
                   help="Which sample(s) to ingest per viscosity bucket.")
    p.add_argument("--max_samples", type=int, default=None)
    args = p.parse_args()

    visc_dirs_beam = sorted(glob.glob(os.path.join(args.beam_root, "visc_*")))
    visc_dirs_direct = sorted(glob.glob(os.path.join(args.direct_root, "visc_*")))
    if not visc_dirs_beam:
        raise SystemExit(f"No visc_*/ under {args.beam_root}")
    common = sorted(set(os.path.basename(d) for d in visc_dirs_beam)
                    .intersection(os.path.basename(d) for d in visc_dirs_direct))
    print(f"Viscosities with both beam+direct: {len(common)}")

    # Score each sample per viscosity by its beam error, then pick.
    selected: list[tuple[str, str]] = []
    for visc in common:
        scored = []
        for sd in sorted(glob.glob(os.path.join(args.beam_root, visc, "sample_*"))):
            md = json.load(open(os.path.join(sd, "metadata.json")))
            scored.append((os.path.basename(sd), float(md.get("error", float("inf")))))
        if args.select == "best_per_visc":
            scored.sort(key=lambda x: x[1])
            selected.append((visc, scored[0][0]))
        elif args.select == "worst_per_visc":
            scored.sort(key=lambda x: -x[1])
            selected.append((visc, scored[0][0]))
        elif args.select == "first_per_visc":
            selected.append((visc, scored[0][0]))
        elif args.select == "all":
            for s_name, _ in scored:
                selected.append((visc, s_name))

    if args.max_samples and len(selected) > args.max_samples:
        step = max(1, len(selected) // args.max_samples)
        selected = selected[::step][: args.max_samples]
    print(f"Selected {len(selected)} candidates ({args.select})")

    run_dir = viz_io.make_run_dir(args.output_dir)
    manifest = {
        "equation_type": "navier_stokes",
        "slot_label": f"canonical_{args.select}",
        "source": "results_targeted/{beam,direct}",
        "ingested_from": [args.beam_root, args.direct_root],
        "select": args.select,
        "n_candidates": len(selected),
        "n_input_frames": 16,
        "n_output_frames": 16,
        "timestamp": os.path.basename(run_dir).replace("run_", ""),
    }
    viz_io.write_manifest(run_dir, manifest)

    for idx, (visc, sample) in enumerate(selected):
        beam_data = _load_sample(os.path.join(args.beam_root, visc, sample))
        direct_data = _load_sample(os.path.join(args.direct_root, visc, sample))

        cand_dir = os.path.join(run_dir, f"cand_{idx:04d}")
        os.makedirs(cand_dir, exist_ok=True)

        bm = beam_data["meta"]
        dm = direct_data["meta"]
        nu = bm.get("viscosity", 0.0)

        np.savez_compressed(
            os.path.join(cand_dir, "gt.npz"),
            input=beam_data["input"],
            target=beam_data["target"],
            viscosity=np.float32(nu),
            visc_idx=np.int64(bm.get("visc_idx", -1)),
        )
        np.savez_compressed(
            os.path.join(cand_dir, "disco_original.npz"),
            pred=direct_data["prediction"],
            test_error=np.float32(dm.get("error", float("nan"))),
        )
        beam_comp = bm.get("composition", [])
        np.savez_compressed(
            os.path.join(cand_dir, "disco_beam.npz"),
            keys=np.array([len(beam_comp)], dtype=np.int64),
            preds=beam_data["prediction"][None],
            fit_errors=np.array([float("nan")], dtype=np.float32),
            test_errors=np.array([float(bm.get("error", float("nan")))], dtype=np.float32),
            compositions_json=np.array(json.dumps([beam_comp]), dtype=object),
            keys_label=np.array("depth", dtype=object),
        )
        with open(os.path.join(cand_dir, "metadata.json"), "w") as f:
            json.dump({
                **manifest,
                "trajectory_idx": idx,
                "source_visc": visc,
                "source_sample": sample,
                "viscosity": float(nu),
                "visc_idx": int(bm.get("visc_idx", -1)),
                "direct_test_error": float(dm.get("error", float("nan"))),
                "beam_test_error":   float(bm.get("error", float("nan"))),
                "beam_composition":  list(beam_comp),
            }, f, indent=2)
        print(f"  cand_{idx:04d}: {visc} {sample}  nu={nu:.5f}  direct={dm.get('error', float('nan')):.4f}  beam={bm.get('error', float('nan')):.4f}")

    viz_io.update_latest_symlink(args.output_dir, run_dir)
    print(f"\nDone. Run dir: {run_dir}")


if __name__ == "__main__":
    main()

"""Walk viz/data/scan/*/run_*/cand_* and render a static PNG per candidate.

Outputs land in viz/thumbs/scan/<exp>/<cand>.png. Skips candidates whose
thumbnail already exists and is fresher than the cand's metadata.json
(idempotent — safe to re-run as SLURM jobs land more candidates).

Usage: python scripts/batch_thumbnails.py [--force]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, "/mnt/home/lserrano/disco-ball/scripts")
from viz_thumbnail import render

ROOT_DATA   = "/mnt/home/lserrano/disco-ball/viz/data/scan"
ROOT_THUMBS = "/mnt/home/lserrano/disco-ball/viz/thumbs/scan"


def needs_render(cand_dir, out_path, force):
    if force or not os.path.exists(out_path):
        return True
    md_path = os.path.join(cand_dir, "metadata.json")
    if not os.path.exists(md_path):
        return False  # cand still in progress
    return os.path.getmtime(out_path) < os.path.getmtime(md_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    cands = sorted(glob.glob(f"{ROOT_DATA}/*/run_*/cand_*"))
    rendered = skipped = failed = 0
    for cand_dir in cands:
        if "WRONG" in cand_dir or "FAILED" in cand_dir:
            continue
        # require all three npz files present
        required = ["gt.npz", "disco_original.npz", "disco_beam.npz"]
        if not all(os.path.exists(os.path.join(cand_dir, r)) for r in required):
            continue
        exp = cand_dir.split(f"{ROOT_DATA}/")[1].split("/")[0]
        cand = os.path.basename(cand_dir)
        out_path = f"{ROOT_THUMBS}/{exp}/{cand}.jpg"
        if not needs_render(cand_dir, out_path, args.force):
            skipped += 1
            continue
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        try:
            render(cand_dir, out_path)
            rendered += 1
        except Exception as e:
            print(f"FAIL {cand_dir}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n→ rendered={rendered}  skipped={skipped}  failed={failed}")


if __name__ == "__main__":
    main()

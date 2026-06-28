"""Render a 2D hero as a single PNG: 3 rows (GT / direct / beam) × 3 cols
(t=start, t=mid, t=end), with a locked colormap anchored to GT.

This is the fallback for 2D benchmarks where the mp4 pipeline is unstable.
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load(p): return dict(np.load(p, allow_pickle=True))


def render(cand_dir, output, channel=0):
    gt = _load(os.path.join(cand_dir, "gt.npz"))
    orig = _load(os.path.join(cand_dir, "disco_original.npz"))
    beam = _load(os.path.join(cand_dir, "disco_beam.npz"))

    tgt = gt["target"]                          # (T,C,H,W)
    direct = orig["pred"]                       # (T,C,H,W)
    beam_final = beam["preds"][-1]              # (T,C,H,W)

    T = tgt.shape[0]
    n_ch = tgt.shape[1]
    channels = list(range(min(n_ch, 2)))        # show up to 2 channels stacked

    cols = 3
    t_idx = [0, T // 2, T - 1]
    col_labels = [f"t = {i}" for i in t_idx]

    n_rows = 3 * len(channels)                  # 3 rows per channel: GT / direct / beam
    fig, axes = plt.subplots(n_rows, cols,
                              figsize=(cols * 2.4, n_rows * 2.0),
                              constrained_layout=True)
    if n_rows == 1:
        axes = np.array([axes])

    direct_err = float(orig.get("test_error", float("nan")))
    beam_err = float(beam["test_errors"][-1])

    row_specs = []
    for ch in channels:
        flat = tgt[:, ch].reshape(-1)
        lo, hi = float(flat.min()), float(flat.max())
        # Pick centered cmap if values cross zero
        if lo < -0.05 * max(abs(hi), 1e-6) and hi > 0.05 * max(abs(lo), 1e-6):
            cmap, vmin, vmax = "RdBu_r", -max(abs(lo), abs(hi)), max(abs(lo), abs(hi))
        else:
            cmap, vmin, vmax = "viridis", lo, hi
        ch_label = f"ch {ch}  "
        row_specs += [
            (f"{ch_label}GT",                                  tgt,        ch, cmap, vmin, vmax),
            (f"{ch_label}direct  err={direct_err:.3g}",        direct,     ch, cmap, vmin, vmax),
            (f"{ch_label}beam  err={beam_err:.3g}",            beam_final, ch, cmap, vmin, vmax),
        ]

    for r, (label, src, ch, cmap, vmin, vmax) in enumerate(row_specs):
        for c, ti in enumerate(t_idx):
            ax = axes[r, c]
            ax.imshow(src[ti, ch], cmap=cmap, vmin=vmin, vmax=vmax,
                      origin="lower", interpolation="nearest")
            if r == 0: ax.set_title(col_labels[c], fontsize=9)
            if c == 0: ax.set_ylabel(label, fontsize=9, rotation=0,
                                     ha="right", va="center", labelpad=42)
            ax.set_xticks([]); ax.set_yticks([])

    fig.savefig(output, dpi=96, bbox_inches="tight",
                pil_kwargs={"quality": 85, "optimize": True})
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cand_dir", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    render(args.cand_dir, args.output)
    print(f"wrote {args.output}  ({os.path.getsize(args.output)/1024:.0f} KB)")


if __name__ == "__main__":
    main()

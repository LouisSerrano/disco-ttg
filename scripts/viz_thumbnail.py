"""Render a single static thumbnail per candidate (no animation).
1D: 3-panel spacetime heatmap (GT | direct | beam, time × space).
2D: side-by-side final-frame heatmaps stacked per channel.

Usage: python scripts/viz_thumbnail.py --cand_dir <path> --output <path.png>
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load_npz(p):
    return dict(np.load(p, allow_pickle=True))


def _choose_cmap_anchor(gt_arr, channel):
    if gt_arr.ndim == 4:                # (T, C, H, W) — 2D
        flat = gt_arr[:, channel].reshape(-1)
    elif gt_arr.ndim == 3:              # (T, C, nx) — 1D
        flat = gt_arr[:, channel].reshape(-1)
    else:                                # already-flat
        flat = gt_arr.reshape(-1)
    lo, hi = float(flat.min()), float(flat.max())
    if lo < -0.05 * max(abs(hi), 1e-6) and hi > 0.05 * max(abs(lo), 1e-6):
        m = max(abs(lo), abs(hi))
        return "RdBu_r", -m, m
    return "viridis", lo, hi


def thumbnail_1d(cand_dir, output):
    """Two-row 1D thumbnail:
       row 1: 3-panel spacetime heatmap (GT | direct | beam, locked colormap)
       row 2: line profiles at t_final, all 3 overlaid — GT vs direct vs beam.
    The overlay panel is where differences pop visually.
    """
    gt = _load_npz(os.path.join(cand_dir, "gt.npz"))
    orig = _load_npz(os.path.join(cand_dir, "disco_original.npz"))
    beam = _load_npz(os.path.join(cand_dir, "disco_beam.npz"))

    inp = gt["input"]
    tgt = gt["target"]
    orig_pred = orig["pred"]
    beam_final = beam["preds"][-1]

    gt_full   = np.concatenate([inp, tgt],        axis=0)[:, 0]
    orig_full = np.concatenate([inp, orig_pred],  axis=0)[:, 0]
    beam_full = np.concatenate([inp, beam_final], axis=0)[:, 0]
    T_in = inp.shape[0]
    T_full = gt_full.shape[0]
    nx = gt_full.shape[1]

    cmap, vmin, vmax = _choose_cmap_anchor(tgt, channel=0)

    direct_err = float(orig.get("test_error", float("nan")))
    beam_err   = float(beam["test_errors"][-1])

    fig = plt.figure(figsize=(9, 4.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.05], hspace=0.04, wspace=0.06)

    # ---- Row 1: spacetime heatmaps ----
    titles = ["GT", f"direct  err={direct_err:.3g}", f"beam  err={beam_err:.3g}"]
    arrs = [gt_full, orig_full, beam_full]
    ax_row1 = None
    for col, (title, arr) in enumerate(zip(titles, arrs)):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(arr.T, cmap=cmap, vmin=vmin, vmax=vmax,
                  origin="lower", aspect="auto",
                  extent=[0, T_full, 0, nx])
        ax.set_title(title, fontsize=9, color="#A24216" if "direct" in title else
                                          ("#3F6F58" if "beam" in title else "#1A1714"))
        ax.axvline(T_in - 0.5, color="white", lw=0.8, alpha=0.6, ls="--")
        ax.tick_params(labelsize=7)
        if col == 0: ax.set_ylabel("x", fontsize=8)
        ax.set_xlabel("t", fontsize=8)

    # ---- Row 2: profile-overlay at final time t = T_full - 1 ----
    ax2 = fig.add_subplot(gs[1, :])
    t_final = T_full - 1
    x = np.arange(nx)
    ax2.plot(x, gt_full[t_final],   color="#1A1714", lw=2.4, label="GT",     zorder=3)
    ax2.plot(x, orig_full[t_final], color="#A24216", lw=1.4, label="direct", linestyle="--", zorder=2)
    ax2.plot(x, beam_full[t_final], color="#3F6F58", lw=1.4, label="beam",   zorder=4)
    ax2.set_xlabel("x  (profile at t = t_final)", fontsize=8)
    ax2.set_ylabel("u", fontsize=8)
    ax2.tick_params(labelsize=7)
    ax2.grid(alpha=0.15, lw=0.5)
    leg = ax2.legend(loc="upper right", fontsize=7.5, framealpha=0.92,
                      edgecolor="#D9D3C5", handlelength=1.6, borderpad=0.4)
    leg.get_frame().set_linewidth(0.5)
    ax2.set_xlim(0, nx - 1)

    fig.savefig(output, dpi=84, bbox_inches="tight", pil_kwargs={"quality": 82, "optimize": True})
    plt.close(fig)


def thumbnail_2d(cand_dir, output):
    gt = _load_npz(os.path.join(cand_dir, "gt.npz"))
    orig = _load_npz(os.path.join(cand_dir, "disco_original.npz"))
    beam = _load_npz(os.path.join(cand_dir, "disco_beam.npz"))

    tgt = gt["target"]
    orig_pred = orig["pred"]
    beam_final = beam["preds"][-1]
    n_ch = tgt.shape[1]
    channels = list(range(min(n_ch, 2)))
    t = tgt.shape[0] - 1

    fig, axes = plt.subplots(len(channels), 3,
                              figsize=(7.5, 2.6 * len(channels) + 0.4),
                              constrained_layout=True)
    if len(channels) == 1:
        axes = np.array([axes])

    titles = ["GT",
              f"direct  (err={float(orig.get('test_error', float('nan'))):.3g})",
              f"beam  (err={float(beam['test_errors'][-1]):.3g})"]
    for row, ch in enumerate(channels):
        cmap, vmin, vmax = _choose_cmap_anchor(tgt, channel=ch)
        for col, arr in enumerate([tgt[t], orig_pred[t], beam_final[t]]):
            ax = axes[row, col]
            ax.imshow(arr[ch], cmap=cmap, vmin=vmin, vmax=vmax,
                      origin="lower", interpolation="nearest")
            if row == 0: ax.set_title(titles[col], fontsize=9)
            if col == 0: ax.set_ylabel(f"ch {ch}", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(output, dpi=84, bbox_inches="tight", pil_kwargs={"quality": 82, "optimize": True})
    plt.close(fig)


def render(cand_dir, output):
    gt = _load_npz(os.path.join(cand_dir, "gt.npz"))
    spatial_dim = gt["target"].ndim - 2
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    if spatial_dim == 1:
        thumbnail_1d(cand_dir, output)
    elif spatial_dim == 2:
        thumbnail_2d(cand_dir, output)
    else:
        raise ValueError(f"unexpected spatial dim {spatial_dim}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cand_dir", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    render(args.cand_dir, args.output)
    size = os.path.getsize(args.output) / 1024
    print(f"wrote {args.output}  ({size:.0f} KB)")


if __name__ == "__main__":
    main()

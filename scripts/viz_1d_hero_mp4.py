"""Robust 1D hero MP4 via PNG-frames pipeline.

Layout: 3-row, full-width.
  row 1: spacetime heatmap (3 panels, GT | direct | beam)
  row 2: profile at t = current — GT thick black, direct dashed orange, beam dashed green
  row 3: profile at t = T_final — same legend

Animated over t, fps=6 default.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load(p): return dict(np.load(p, allow_pickle=True))


def _choose_cmap(arr):
    flat = arr.reshape(-1)
    lo, hi = float(flat.min()), float(flat.max())
    if lo < -0.05 * max(abs(hi), 1e-6) and hi > 0.05 * max(abs(lo), 1e-6):
        m = max(abs(lo), abs(hi))
        return "RdBu_r", -m, m
    return "viridis", lo, hi


def render(cand_dir, output, fps=6, dpi=88, ffmpeg_bin="ffmpeg"):
    gt = _load(os.path.join(cand_dir, "gt.npz"))
    orig = _load(os.path.join(cand_dir, "disco_original.npz"))
    beam = _load(os.path.join(cand_dir, "disco_beam.npz"))

    inp = gt["input"]          # (T_in, C, nx)
    tgt = gt["target"]         # (T_out, C, nx)
    direct = orig["pred"]      # (T_out, C, nx)
    # take final beam depth
    beam_pred = beam["preds"][-1] if beam["preds"].ndim == 4 else beam["preds"]  # (T_out, C, nx)

    direct_err = float(orig.get("test_error", float("nan")))
    beam_err   = float(beam["test_errors"][-1]) if beam["test_errors"].ndim else float(beam["test_errors"])

    gt_full   = np.concatenate([inp, tgt],        axis=0)[:, 0]   # (T_full, nx)
    dir_full  = np.concatenate([inp, direct],     axis=0)[:, 0]
    beam_full = np.concatenate([inp, beam_pred],  axis=0)[:, 0]
    T_in = inp.shape[0]
    T_full = gt_full.shape[0]
    nx = gt_full.shape[1]

    cmap, vmin, vmax = _choose_cmap(tgt[:, 0])

    fig = plt.figure(figsize=(11.0, 5.8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1.0], hspace=0.30, wspace=0.10,
                           left=0.06, right=0.985, top=0.92, bottom=0.10)

    # Row 1: spacetime heatmaps
    ax_gt = fig.add_subplot(gs[0, 0])
    ax_d  = fig.add_subplot(gs[0, 1], sharey=ax_gt)
    ax_b  = fig.add_subplot(gs[0, 2], sharey=ax_gt)
    for ax, arr, title, color in [
        (ax_gt, gt_full,   "GT",                                "#1A1714"),
        (ax_d,  dir_full,  f"direct  err = {direct_err:.3g}",   "#A04822"),
        (ax_b,  beam_full, f"beam  err = {beam_err:.3g}",       "#3F6F58"),
    ]:
        ax.imshow(arr.T, cmap=cmap, vmin=vmin, vmax=vmax,
                   origin="lower", aspect="auto",
                   extent=[0, T_full, 0, nx])
        ax.axvline(T_in - 0.5, color="white", lw=0.8, alpha=0.7, ls="--")
        ax.set_title(title, fontsize=10, color=color)
        ax.set_xlabel("t", fontsize=9)
        ax.tick_params(labelsize=7)
    ax_gt.set_ylabel("x", fontsize=9)

    # Row 2: profile overlay at the CURRENT time (animated)
    ax_p = fig.add_subplot(gs[1, :])
    x_axis = np.arange(nx)
    profile_lim = (float(min(gt_full.min(), dir_full.min(), beam_full.min())),
                    float(max(gt_full.max(), dir_full.max(), beam_full.max())))
    span = profile_lim[1] - profile_lim[0]
    ax_p.set_xlim(0, nx - 1)
    ax_p.set_ylim(profile_lim[0] - 0.05 * span, profile_lim[1] + 0.05 * span)
    (ln_gt,)   = ax_p.plot(x_axis, gt_full[0],   color="#1A1714", lw=2.4, label="GT",     zorder=3)
    (ln_dir,)  = ax_p.plot(x_axis, dir_full[0],  color="#A04822", lw=1.4, linestyle="--", label="direct", zorder=2)
    (ln_beam,) = ax_p.plot(x_axis, beam_full[0], color="#3F6F58", lw=1.4, label="beam", zorder=4)
    ax_p.set_xlabel("x  (profile at current t)", fontsize=9)
    ax_p.set_ylabel("u", fontsize=9)
    ax_p.tick_params(labelsize=7)
    ax_p.grid(alpha=0.15, lw=0.5)
    leg = ax_p.legend(loc="upper right", fontsize=8.5, framealpha=0.95,
                       edgecolor="#D9D3C5", handlelength=2.0, borderpad=0.5)
    leg.get_frame().set_linewidth(0.5)

    # Time cursor line on each spacetime panel
    cursor_lines = []
    for ax in (ax_gt, ax_d, ax_b):
        ln = ax.axvline(0, color="white", lw=1.6, alpha=0.95)
        ln_outline = ax.axvline(0, color="#1A1714", lw=2.6, alpha=0.45, zorder=-1)
        cursor_lines.append((ln, ln_outline))

    time_text = fig.text(0.992, 0.985, "", ha="right", va="top",
                          fontsize=10, color="#5B6066",
                          family="monospace")

    with tempfile.TemporaryDirectory() as tmp:
        for ti in range(T_full):
            ln_gt.set_data(x_axis, gt_full[ti])
            ln_dir.set_data(x_axis, dir_full[ti])
            ln_beam.set_data(x_axis, beam_full[ti])
            for ln, ln_o in cursor_lines:
                ln.set_xdata([ti])
                ln_o.set_xdata([ti])
            time_text.set_text(f"t = {ti+1:02d} / {T_full}")
            p = os.path.join(tmp, f"f{ti:04d}.png")
            fig.savefig(p, dpi=dpi, facecolor="white")
        plt.close(fig)

        cmd = [
            ffmpeg_bin, "-y", "-loglevel", "error",
            "-framerate", str(fps),
            "-i", os.path.join(tmp, "f%04d.png"),
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2:0:0:white",
            "-vcodec", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "22", "-preset", "medium",
            output,
        ]
        subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cand_dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--fps", type=int, default=6)
    p.add_argument("--dpi", type=int, default=88)
    p.add_argument("--ffmpeg", default="ffmpeg")
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    render(args.cand_dir, args.output, fps=args.fps, dpi=args.dpi, ffmpeg_bin=args.ffmpeg)
    print(f"wrote {args.output}  ({os.path.getsize(args.output)/1024:.0f} KB)")


if __name__ == "__main__":
    main()

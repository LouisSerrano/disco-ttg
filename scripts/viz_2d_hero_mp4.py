"""Render a 2D animated hero MP4 the robust way: write per-frame PNGs to a
temp dir, then assemble with ffmpeg. Avoids the matplotlib FFMpegWriter pipe
issue that broke 2D rendering for NS / Gray-Scott.

Layout: per channel, 1 row × 3 columns (GT | direct | beam) with a locked
GT-anchored colormap. T frames over time. fps default 6.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load(p): return dict(np.load(p, allow_pickle=True))


def render(cand_dir, output, fps=6, dpi=88, ffmpeg_bin="ffmpeg"):
    gt = _load(os.path.join(cand_dir, "gt.npz"))
    orig = _load(os.path.join(cand_dir, "disco_original.npz"))
    beam = _load(os.path.join(cand_dir, "disco_beam.npz"))

    tgt = gt["target"]              # (T, C, H, W)
    direct = orig["pred"]           # (T, C, H, W)
    beam_final = beam["preds"][-1]  # (T, C, H, W)

    direct_err = float(orig.get("test_error", float("nan")))
    beam_err = float(beam["test_errors"][-1])

    T, C = tgt.shape[:2]
    channels = list(range(min(C, 2)))

    # Per-channel cmap anchored to GT
    specs = []
    for ch in channels:
        flat = tgt[:, ch].reshape(-1)
        lo, hi = float(flat.min()), float(flat.max())
        if lo < -0.05 * max(abs(hi), 1e-6) and hi > 0.05 * max(abs(lo), 1e-6):
            m = max(abs(lo), abs(hi))
            specs.append((ch, "RdBu_r", -m, m))
        else:
            specs.append((ch, "viridis", lo, hi))

    n_rows = len(channels)
    fig_w, fig_h = 7.2, 2.2 * n_rows + 0.4  # ensure even pixels at dpi=88
    fig, axes = plt.subplots(n_rows, 3,
                              figsize=(fig_w, fig_h), constrained_layout=False)
    if n_rows == 1:
        axes = np.array([axes])
    fig.subplots_adjust(left=0.06, right=0.99, top=0.90, bottom=0.04,
                         wspace=0.08, hspace=0.18)

    titles = ["GT", f"direct  err = {direct_err:.3g}", f"beam  err = {beam_err:.3g}"]
    title_colors = ["#1A1714", "#A24216", "#3F6F58"]

    # Prepare imshow handles, share artists for fast updates
    im_handles = [[None] * 3 for _ in channels]
    for r, (ch, cmap, vmin, vmax) in enumerate(specs):
        for c, src_arr in enumerate([tgt, direct, beam_final]):
            ax = axes[r, c]
            im = ax.imshow(src_arr[0, ch], cmap=cmap, vmin=vmin, vmax=vmax,
                            origin="lower", interpolation="nearest", animated=True)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(titles[c], fontsize=10, color=title_colors[c], pad=4)
            if c == 0 and n_rows > 1:
                ax.set_ylabel(f"ch {ch}", fontsize=9, rotation=0, ha="right",
                               va="center", labelpad=14)
            im_handles[r][c] = im

    # Stamp a small time indicator above the first row, right side
    time_text = fig.text(0.99, 0.97, "", ha="right", va="top",
                          fontsize=9, color="#5B6066",
                          family="ui-monospace, monospace",
                          fontvariant="small-caps")

    with tempfile.TemporaryDirectory() as tmp:
        frame_paths = []
        for ti in range(T):
            for r, (ch, _, _, _) in enumerate(specs):
                im_handles[r][0].set_data(tgt[ti, ch])
                im_handles[r][1].set_data(direct[ti, ch])
                im_handles[r][2].set_data(beam_final[ti, ch])
            time_text.set_text(f"t = {ti+1:02d} / {T}")
            p = os.path.join(tmp, f"f{ti:04d}.png")
            fig.savefig(p, dpi=dpi, facecolor="white")
            frame_paths.append(p)
        plt.close(fig)

        # ffmpeg assembly. Pad to even dims defensively.
        cmd = [
            ffmpeg_bin, "-y", "-loglevel", "error",
            "-framerate", str(fps),
            "-i", os.path.join(tmp, "f%04d.png"),
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2:0:0:white",
            "-vcodec", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "23", "-preset", "medium",
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
    render(args.cand_dir, args.output, fps=args.fps, dpi=args.dpi,
            ffmpeg_bin=args.ffmpeg)
    print(f"wrote {args.output}  ({os.path.getsize(args.output)/1024:.0f} KB)")


if __name__ == "__main__":
    main()

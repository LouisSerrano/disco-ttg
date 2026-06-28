"""Deep-dive MP4: per-pick animation showing the beam composition growing
depth-by-depth and the rollout closing the gap to GT.

Layout per candidate:
                  ch0 (or only)    ch1 (if 2D 2-ch)     residual ch0     residual ch1
    GT row        u_gt(t)          v_gt(t)              [empty]          [empty]
    depth-1 row   pred_1(t)        ...                  |pred_1 - GT|    ...
    depth-2 row   pred_2(t)        ...                  |pred_2 - GT|    ...
    ...

Top banner: composition + test-error for the depth, color-coded so the user
can see the composition grow and the error drop in real time.

Uses ffmpeg via the PNG-frames pipeline (robust on 2D, unlike matplotlib's
FFMpegWriter pipe).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load(p): return dict(np.load(p, allow_pickle=True))


def _choose_cmap(gt_arr, channel):
    flat = gt_arr[:, channel].reshape(-1)
    lo, hi = float(flat.min()), float(flat.max())
    if lo < -0.05 * max(abs(hi), 1e-6) and hi > 0.05 * max(abs(lo), 1e-6):
        m = max(abs(lo), abs(hi))
        return "RdBu_r", -m, m
    return "viridis", lo, hi


def render(cand_dir, output, fps=6, dpi=92, ffmpeg_bin="ffmpeg"):
    gt = _load(os.path.join(cand_dir, "gt.npz"))
    beam = _load(os.path.join(cand_dir, "disco_beam.npz"))

    tgt = gt["target"]                  # (T, C, *spatial)
    preds = beam["preds"]               # (D, T, C, *spatial)
    test_errors = beam["test_errors"]   # (D,)
    keys = beam["keys"]                 # (D,)
    comps = json.loads(str(beam["compositions_json"]))

    md = json.load(open(os.path.join(cand_dir, "metadata.json")))
    title = f"{md.get('experiment','')}  ds#{md.get('dataset_index','?')}"
    if "viscosity" in md:
        title += f"  ν={md['viscosity']:.4g}"
    elif "f" in md and "k" in md:
        title += f"  F={md['f']:.4g}  k={md['k']:.4g}"

    T = tgt.shape[0]
    D = preds.shape[0]
    n_ch = tgt.shape[1]
    channels = list(range(min(n_ch, 2)))
    is_2d = (tgt.ndim - 2) == 2

    # Pre-compute residuals
    residuals = np.abs(preds - tgt[None, ...])  # (D, T, C, *sp)
    max_res = float(residuals.max())            # for residual colormap

    # Layout: rows = 1 (GT) + D depths, cols = n_ch*2 (preds + residuals)  for 2D
    #         or n_ch for 1D (residual shown as overlay line, see below)
    if is_2d:
        n_cols = len(channels) * 2  # pred + residual per channel
    else:
        n_cols = 1  # one column for 1D, but we overlay residual

    n_rows = 1 + D
    fig_w = 2.6 * n_cols + 1.2
    fig_h = 2.2 * n_rows + 1.0
    fig, axes = plt.subplots(n_rows, n_cols if n_cols > 1 else 2,
                              figsize=(fig_w, fig_h), constrained_layout=False)
    if n_cols == 1:
        axes = axes.reshape(n_rows, 2)

    fig.subplots_adjust(left=0.06, right=0.99, top=0.88, bottom=0.05,
                         wspace=0.10, hspace=0.18)

    # Per-channel cmap for fields, anchored to GT
    cmap_specs = []
    for ch in channels:
        cmap_specs.append(_choose_cmap(tgt, ch))

    # imshow handles
    im_handles: list[list] = [[None] * (n_cols if n_cols > 1 else 2) for _ in range(n_rows)]

    titles_top = []
    for ch in channels:
        titles_top.append(f"ch {ch}" if len(channels) > 1 else "field")
    if is_2d:
        titles_top += [f"|err| ch {ch}" if len(channels) > 1 else "|err|" for ch in channels]

    if is_2d:
        for col in range(n_cols):
            axes[0, col].set_title(titles_top[col], fontsize=10, color="#2A2D33", pad=4)
    else:
        # 1D: overlay residual as a thin secondary line; just label the col header
        axes[0, 0].set_title("field  (line: GT solid, pred dashed)", fontsize=10, pad=4)
        axes[0, 1].set_title("|pred − GT|", fontsize=10, pad=4)

    row_label_color = {"GT": "#1A1714"}

    def field_to_imshow(ax, arr, cmap, vmin, vmax):
        if is_2d:
            return ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax,
                              origin="lower", interpolation="nearest", animated=True)
        else:
            ax.set_xlim(0, arr.shape[0] - 1)
            ax.set_ylim(vmin - 0.05 * (vmax - vmin), vmax + 0.05 * (vmax - vmin))
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.15, lw=0.5)
            (ln_gt,) = ax.plot([], [], color="#1A1714", lw=2.0, label="GT")
            (ln_pr,) = ax.plot([], [], color="#3F6F58", lw=1.2, linestyle="--", label="pred")
            return (ln_gt, ln_pr)

    # Build initial frames at t=0
    t0 = 0
    for r in range(n_rows):
        is_gt_row = (r == 0)
        for col, ch in enumerate(channels):
            cmap, vmin, vmax = cmap_specs[col]
            ax = axes[r, col]
            if is_gt_row:
                if is_2d:
                    im = field_to_imshow(ax, tgt[t0, ch], cmap, vmin, vmax)
                    im_handles[r][col] = im
                else:
                    ln = field_to_imshow(ax, tgt[t0, ch], cmap, vmin, vmax)
                    im_handles[r][col] = ln
            else:
                d = r - 1
                if is_2d:
                    im = field_to_imshow(ax, preds[d, t0, ch], cmap, vmin, vmax)
                    im_handles[r][col] = im
                else:
                    ln = field_to_imshow(ax, preds[d, t0, ch], cmap, vmin, vmax)
                    im_handles[r][col] = ln
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                if is_gt_row:
                    label_text = "GT"
                    color = "#1A1714"
                else:
                    d = r - 1
                    label_text = f"depth {int(keys[d])}"
                    color = "#3F6F58"
                ax.set_ylabel(label_text, fontsize=10, rotation=0, ha="right",
                                va="center", labelpad=22, color=color, fontweight="bold")

        # Residual columns (only for 2D layout)
        if is_2d:
            for k, ch in enumerate(channels):
                col = len(channels) + k
                ax = axes[r, col]
                if is_gt_row:
                    ax.axis("off")
                else:
                    d = r - 1
                    im = ax.imshow(residuals[d, t0, ch], cmap="magma",
                                    vmin=0, vmax=max_res,
                                    origin="lower", interpolation="nearest", animated=True)
                    im_handles[r][col] = im
                    ax.set_xticks([]); ax.set_yticks([])
        else:
            # 1D residual line plot column
            ax = axes[r, 1]
            if is_gt_row:
                ax.axis("off")
            else:
                d = r - 1
                res = residuals[d, t0, channels[0]]
                ax.set_xlim(0, res.shape[0] - 1)
                ax.set_ylim(0, max_res * 1.05)
                ax.tick_params(labelsize=7)
                ax.grid(alpha=0.15, lw=0.5)
                (ln_res,) = ax.plot(np.arange(res.shape[0]), res, color="#A04822", lw=1.0)
                im_handles[r][1] = ln_res

    # Banner area: composition + error per depth
    banner_text = fig.text(0.50, 0.96, "", ha="center", va="top",
                            fontsize=11, color="#1A1714",
                            family="serif", fontweight="bold")
    sub_banner = fig.text(0.50, 0.93, "", ha="center", va="top",
                           fontsize=10, color="#5B6066",
                           family="monospace")
    time_banner = fig.text(0.99, 0.985, "", ha="right", va="top",
                            fontsize=9, color="#7A736B",
                            family="monospace")

    # Update function
    def update_frame(ti, d_highlight):
        for r in range(n_rows):
            is_gt_row = (r == 0)
            for col, ch in enumerate(channels):
                if is_gt_row:
                    if is_2d:
                        im_handles[r][col].set_data(tgt[ti, ch])
                    else:
                        ln_gt, ln_pr = im_handles[r][col]
                        ln_gt.set_data(np.arange(tgt.shape[-1]), tgt[ti, ch])
                        ln_pr.set_data([], [])
                else:
                    d = r - 1
                    if is_2d:
                        im_handles[r][col].set_data(preds[d, ti, ch])
                    else:
                        ln_gt, ln_pr = im_handles[r][col]
                        ln_gt.set_data(np.arange(tgt.shape[-1]), tgt[ti, ch])
                        ln_pr.set_data(np.arange(preds.shape[-1]), preds[d, ti, ch])

            if is_2d:
                for k, ch in enumerate(channels):
                    col = len(channels) + k
                    if not is_gt_row:
                        d = r - 1
                        im_handles[r][col].set_data(residuals[d, ti, ch])
            else:
                if not is_gt_row:
                    d = r - 1
                    res = residuals[d, ti, channels[0]]
                    im_handles[r][1].set_data(np.arange(res.shape[0]), res)

        # Banner — show the current composition being highlighted (= all depths shown)
        banner_text.set_text(title)
        # Per-depth one-liner
        lines = []
        for d in range(D):
            arrow = "→" if d > 0 else " "
            comp = "·".join(str(c) for c in comps[d])
            lines.append(f"depth {int(keys[d])} {arrow} comp=[{comp}]  err={float(test_errors[d]):.3g}")
        sub_banner.set_text("    ".join(lines))
        time_banner.set_text(f"t = {ti+1:02d} / {T}")

    # Render frames + assemble with ffmpeg
    with tempfile.TemporaryDirectory() as tmp:
        for ti in range(T):
            update_frame(ti, d_highlight=D - 1)
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
    p.add_argument("--dpi", type=int, default=92)
    p.add_argument("--ffmpeg", default="ffmpeg")
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    render(args.cand_dir, args.output, fps=args.fps, dpi=args.dpi, ffmpeg_bin=args.ffmpeg)
    print(f"wrote {args.output}  ({os.path.getsize(args.output)/1024:.0f} KB)")


if __name__ == "__main__":
    main()

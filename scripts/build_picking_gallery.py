"""Build a 'pick favorites' gallery: one section per experiment, each
section shows ALL candidates with their hero MP4 inlined.

Run:
    python scripts/build_picking_gallery.py [--out PATH] [--max_size_mb 12]

The output is a single self-contained HTML file (data URIs for all media).
"""
from __future__ import annotations

import argparse
import base64
import glob
import json
import math
import os

ROOT_MEDIA = "/mnt/home/lserrano/disco-ball/viz/media_v2"
ROOT_DATA  = "/mnt/home/lserrano/disco-ball/viz/data"


# Paper Table 1 + Table 2 → label, ordered for the gallery.
EXPERIMENT_LABELS = {
    # ---- Table 1 (physics composition) ----
    "scan/adv-diff_E_AD_ALL":     ("Table 1 · Adv + Diff (E_AD_ALL)",
                                    "Both v and D nonzero — in-distribution physics composition"),
    "scan/combined-eq_E_HE":      ("Table 1 · Nonlinear Adv + Diffusion (E_HE)",
                                    "burgers-style nonlinearity with diffusion"),
    "scan/combined-eq_E_ED":      ("Table 1 · Nonlinear Adv + Dispersion (E_ED)", ""),
    "scan/combined-eq_E_BG":      ("Table 1 · Diffusion + Dispersion (E_BG)", ""),
    "scan/combined-eq_E_ALL":     ("Table 1 · All three (E_ALL)",
                                    "NL Adv + Diff + Disp simultaneously"),
    "scan/gray-scott":            ("Table 1 · Reaction + Diffusion (Gray-Scott)",
                                    "2D Turing patterns; activator + inhibitor stacked"),
    "scan/navier-stokes":         ("Table 1 · Euler + Diffusion (Navier-Stokes)",
                                    "2D vorticity, refinement_factor=4"),
    # ---- Table 2 (parameter extrapolation) ----
    "scan/adv-diff_E_AD_v":       ("Table 2 · Adv-Diff c (E_AD_v)",
                                    "v ∈ [1, 3], no diffusion (3× training range)"),
    "scan/adv-diff_E_AD_D":       ("Table 2 · Adv-Diff D (E_AD_D)",
                                    "D ∈ [1, 3], no advection — biggest paper recovery"),
    "scan/combined-eq_E_EULER_OOD": ("Table 2 · Combined α (E_EULER_OOD)",
                                    "nonlinear advection α ∈ [1, 2]"),
    "scan/combined-eq_E_DISP_OOD":  ("Table 2 · Combined γ (E_DISP_OOD)",
                                    "dispersion γ ∈ [1, 2]"),
}


def b64(path, mime):
    with open(path, "rb") as f:
        return f"data:{mime};base64,{base64.b64encode(f.read()).decode()}"


def inline_svg(path):
    if not os.path.exists(path): return ""
    with open(path) as f: svg = f.read()
    if svg.startswith("<?xml"): svg = svg.split("?>", 1)[1]
    return svg


def find_metadata(exp: str, cand: str) -> dict:
    """Search viz/data/<exp>/run_*/cand for metadata.json — handle the
    'v2/...' prefix used in the gallery key."""
    short = exp[3:] if exp.startswith("v2/") else exp
    # Try v2 first (newer schema), then plain.
    for root in (f"{ROOT_DATA}/v2/{short}", f"{ROOT_DATA}/{exp}"):
        for p in sorted(glob.glob(f"{root}/run_*/{cand}/metadata.json"), reverse=True):
            if "WRONG" in p or "FAILED" in p: continue
            return json.load(open(p))
    return {}


def collect_candidates():
    """Group rendered candidates by experiment (top-level dir under media_v2)."""
    groups: dict[str, list] = {}
    for media in sorted(glob.glob(f"{ROOT_MEDIA}/*/cand_*")):
        if not os.path.exists(f"{media}/hero.mp4"): continue
        exp = os.path.relpath(os.path.dirname(media), ROOT_MEDIA)
        cand = os.path.basename(media)
        md = find_metadata(exp, cand)
        d = float(md.get("direct_test_error", float("nan")))
        b = float(md.get("beam_test_error",  float("nan")))
        u = float(md.get("uniform_test_error", float("nan")))
        impr = math.log10(max(d, 1e-6) / max(b, 1e-9)) if d > 0 and b > 0 else 0
        groups.setdefault(exp, []).append({
            "media": media,
            "cand": cand,
            "direct": d, "beam": b, "uniform": u,
            "impr": impr,
            "params": " ".join(f"{k.split('_')[0]}={float(md[k]):.3g}" for k in
                                ("advection_speed", "diffusion", "alpha", "beta", "gamma",
                                 "f", "k", "viscosity") if k in md),
            "metadata": md,
        })
    return groups


def card_html(cand: dict, num: int, max_inline_mb: float) -> str:
    media = cand["media"]
    hero_path = f"{media}/hero.mp4"
    size_mb = os.path.getsize(hero_path) / 1024 / 1024
    if size_mb > max_inline_mb:
        # Don't inline very large MP4s — fall back to a static error_overlay.svg.
        body = inline_svg(f"{media}/error_overlay.svg") or inline_svg(f"{media}/rollout_overlay.svg")
        body_html = f'<div class="static-fallback">{body}<p class="too-big">[MP4 {size_mb:.1f}MB — too big to inline; rendered at {hero_path}]</p></div>'
    else:
        hero_uri = b64(hero_path, "video/mp4")
        body_html = f'<video src="{hero_uri}" controls muted loop playsinline preload="metadata"></video>'

    return f"""
      <figure class="cand" id="{cand['cand']}">
        <header class="cand-head">
          <span class="num">#{num:02d}</span>
          <span class="cand-id">{cand['cand']}</span>
          <span class="impr">{10**cand['impr']:.0f}× recovery</span>
        </header>
        {body_html}
        <figcaption class="mono">
          <span class="params">{cand['params']}</span>
          <span class="errs">
            <span class="direct">direct {cand['direct']:.3g}</span>
            <span class="beam">beam {cand['beam']:.3g}</span>
          </span>
        </figcaption>
      </figure>"""


def section_html(exp: str, cands: list, max_inline_mb: float) -> str:
    label, blurb = EXPERIMENT_LABELS.get(exp, (exp, ""))
    # Sort by impressiveness descending so the most dramatic first
    cands = sorted(cands, key=lambda r: -r["impr"])
    cards = "".join(card_html(c, i + 1, max_inline_mb) for i, c in enumerate(cands))
    return f"""
    <section class="bench" id="{exp.replace('/', '_')}">
      <header class="bench-head">
        <h2>{label}</h2>
        <p class="blurb">{blurb}</p>
        <p class="count">{len(cands)} candidate{'s' if len(cands) != 1 else ''}</p>
      </header>
      <div class="cands">{cards}</div>
    </section>"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/mnt/home/lserrano/disco-ball/viz/site/picking.html")
    p.add_argument("--max_size_mb", type=float, default=2.0,
                   help="Heroes bigger than this fall back to a static SVG preview.")
    args = p.parse_args()

    groups = collect_candidates()
    total_cands = sum(len(v) for v in groups.values())
    print(f"Found {total_cands} candidates across {len(groups)} experiments")

    # Order sections per EXPERIMENT_LABELS keys; append any unknown ones at the end.
    ordered = [e for e in EXPERIMENT_LABELS if e in groups]
    ordered += [e for e in groups if e not in EXPERIMENT_LABELS]
    sections = "".join(section_html(e, groups[e], args.max_size_mb) for e in ordered)

    html = f"""<title>Picking gallery — pick favorites per case</title>
<meta name="description" content="All rendered candidates per Table 1 case. Pick which ones to use as website heroes.">
<style>
  :root {{
    --bg:    #E8EBED;
    --bg-2:  #F2F4F5;
    --ink:   #0E1116;
    --rule:  #C8CCD0;
    --muted: #5B6066;
    --accent:#1F4961;
    --warn:  #B7411F;
    --good:  #3D7A60;
  }}
  html {{ background: var(--bg); color: var(--ink); }}
  body {{
    font-family: 'Charter', 'Iowan Old Style', Georgia, serif;
    font-size: 15px; line-height: 1.55;
    max-width: 1280px; margin: 0 auto;
    padding: 40px 24px 80px;
  }}
  h1 {{ font-size: 28px; margin: 0 0 6px; font-weight: 600; }}
  h2 {{ font-size: 20px; margin: 0; font-weight: 600; letter-spacing: -0.01em; }}
  p {{ margin: 0; }}
  .mono {{ font-family: ui-monospace, Menlo, Consolas, monospace; font-variant-numeric: tabular-nums; }}

  header.top {{ border-bottom: 1px solid var(--rule); padding-bottom: 16px; margin-bottom: 32px; }}
  header.top .sub {{ color: var(--muted); margin-top: 6px; font-size: 13.5px; }}

  section.bench {{ margin-bottom: 56px; }}
  section.bench .bench-head {{ display: flex; align-items: baseline;
      gap: 18px; border-bottom: 1px solid var(--rule); padding-bottom: 10px; margin-bottom: 18px; }}
  section.bench .blurb {{ color: var(--muted); font-size: 14px; flex: 1; }}
  section.bench .count {{ color: var(--accent); font-family: ui-monospace, monospace; font-size: 12px; font-weight: 600; }}

  .cands {{ display: grid; gap: 18px; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); }}
  figure.cand {{ margin: 0; background: #fff; border: 1px solid var(--rule);
      border-radius: 4px; padding: 10px; display: flex; flex-direction: column; gap: 6px; }}
  figure.cand .cand-head {{ display: flex; align-items: center; gap: 10px; font-size: 12px;
      font-family: ui-monospace, monospace; color: var(--muted); }}
  figure.cand .num {{ color: var(--accent); font-weight: 600; font-size: 13px; }}
  figure.cand .cand-id {{ flex: 1; color: var(--ink); font-weight: 600; }}
  figure.cand .impr {{ color: var(--good); font-weight: 600; }}
  figure.cand video {{ width: 100%; display: block; background: #000; }}
  figure.cand .static-fallback {{ background: #f9f9f9; padding: 8px; line-height: 0; }}
  figure.cand .static-fallback svg {{ width: 100%; height: auto; line-height: normal; }}
  figure.cand .too-big {{ color: var(--muted); font-size: 11px; padding: 6px 0; }}
  figure.cand figcaption {{ font-size: 11.5px; display: flex; flex-direction: column; gap: 2px; }}
  figure.cand .params {{ color: var(--muted); }}
  figure.cand .errs {{ display: flex; gap: 12px; }}
  figure.cand .direct {{ color: var(--warn); font-weight: 600; }}
  figure.cand .beam {{ color: var(--good); font-weight: 600; }}
</style>

<header class="top">
  <h1>Picking gallery — all candidates per Table 1 case</h1>
  <p class="sub">{total_cands} candidates across {len(groups)} experiments. Sort within each section is most-dramatic first (largest direct→beam ratio). Tell me which numbered cards to use for the website.</p>
</header>

{sections}
"""
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f: f.write(html)
    print(f"wrote {args.out}  size {os.path.getsize(args.out)/1024/1024:.1f} MB")


if __name__ == "__main__":
    main()

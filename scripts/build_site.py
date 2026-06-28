"""Build the public website for the DISCO / Neural Operator Splitting paper.

Self-contained HTML at viz/site/index.html. Inlines hero MP4s, the paper's
main figure, and per-case static panels as data URIs. Content blocks are
plain constants so edits are local.
"""
from __future__ import annotations

import argparse
import base64
import math
import os
from pathlib import Path


PICKS_ROOT     = Path("/mnt/home/lserrano/disco-ball/viz/heroes/picks")
DEEP_DIVE_ROOT = Path("/mnt/home/lserrano/disco-ball/viz/heroes/deep_dive")
PAPER_ROOT     = Path("/mnt/home/lserrano/paper-icml-cr")


# Per-case: title, blurb, ABSOLUTE hero_path, direct→beam numbers
# Order matters — top-to-bottom on the page.
TABLE1_CASES = [
    {"title": "Advection + Diffusion",         "blurb": "v + D both active",
     "hero":   PICKS_ROOT / "adv-diff_E_AD_ALL_cand_0012_v2.mp4",
     "d": 0.273, "b": 0.057, "params": "v = 0.78, D = 0.19"},
    {"title": "Nonlinear Adv. + Diffusion",    "blurb": "Burgers-style nonlinearity + heat",
     "hero":   PICKS_ROOT / "combined-eq_E_HE_cand_0002_v2.mp4",
     "d": 0.034, "b": 0.001, "params": "α = 0.125, β = 0.05"},
    {"title": "Nonlinear Adv. + Dispersion",   "blurb": "α and γ simultaneously",
     "hero":   PICKS_ROOT / "combined-eq_E_ED_cand_0016_v2.mp4",
     "d": 0.403, "b": 0.160, "params": "α = 0.625, γ = 0.125"},
    {"title": "Diffusion + Dispersion",        "blurb": "β + γ — heat plus KdV-style term",
     "hero":   PICKS_ROOT / "combined-eq_E_BG_cand_0002_v2.mp4",
     "d": 0.039, "b": 0.002, "params": "β = 0.05, γ = 0.125"},
    {"title": "Reaction + Diffusion",          "blurb": "Gray-Scott · 2D · activator + inhibitor",
     "hero":   DEEP_DIVE_ROOT / "gs_pick02_cand_0037.mp4",
     "d": 0.432, "b": 0.043, "params": "F = 0.055, k = 0.059",
     "is_deep_dive": True},
    {"title": "Euler + Diffusion → Navier–Stokes", "blurb": "2D vorticity at ν = 0.004",
     "hero":   DEEP_DIVE_ROOT / "ns_pick01_cand_0096.mp4",
     "d": 3.24, "b": 0.047, "params": "ν = 0.004",
     "is_deep_dive": True},
]

# "All three" gets a dedicated 4-card sweep showing how the recovery ratio
# evolves as the nonlinear-advection coefficient α grows.
TABLE1_E_ALL_SWEEP = [
    {"title": "α = 0.25",  "blurb": "lowest nonlinear advection — cleanest recovery",
     "hero":  PICKS_ROOT / "combined-eq_E_ALL_cand_0007_v2.mp4",
     "d": 0.164, "b": 0.007, "params": "α = 0.25, β = 0.30, γ = 0.25"},
    {"title": "α = 0.50",  "blurb": "mid-α — balanced dynamics + recovery",
     "hero":  PICKS_ROOT / "combined-eq_E_ALL_cand_0013_v2.mp4",
     "d": 0.110, "b": 0.015, "params": "α = 0.50, β = 0.30, γ = 0.25"},
    {"title": "α = 0.75",  "blurb": "higher-amplitude traveling pattern",
     "hero":  PICKS_ROOT / "combined-eq_E_ALL_cand_0019_v2.mp4",
     "d": 0.250, "b": 0.075, "params": "α = 0.75, β = 0.10, γ = 0.25"},
    {"title": "α = 1.00",  "blurb": "upper-α boundary — most dynamic",
     "hero":  PICKS_ROOT / "combined-eq_E_ALL_cand_0026_v2.mp4",
     "d": 0.293, "b": 0.115, "params": "α = 1.00, β = 0.10, γ = 0.25"},
]

TABLE2_CASES = [
    {"title": "Advection–diffusion · c",       "blurb": "v ∈ [1, 3], well outside training",
     "hero":  PICKS_ROOT / "adv-diff_E_AD_v_cand_0008_v2.mp4",
     "d": 1.249, "b": 0.105, "params": "v = 2.19, D = 0"},
    {"title": "Advection–diffusion · D",       "blurb": "D ∈ [1, 3], well outside training",
     "hero":  PICKS_ROOT / "adv-diff_E_AD_D_cand_0013_v2.mp4",
     "d": 0.129, "b": 0.001, "params": "v = 0, D = 2.01"},
    {"title": "Combined · α (Euler OOD)",      "blurb": "nonlinear-advection coefficient extrapolated",
     "hero":  PICKS_ROOT / "combined-eq_E_EULER_OOD_cand_0010_v2.mp4",
     "d": 0.109, "b": 0.035, "params": "α = 1.29 (3× train)"},
    {"title": "Combined · γ (dispersion OOD)", "blurb": "dispersion coefficient extrapolated",
     "hero":  PICKS_ROOT / "combined-eq_E_DISP_OOD_cand_0013_v2.mp4",
     "d": 1.371, "b": 0.013, "params": "γ = 1.55 (1.6× train)"},
]

# Deep-dive section: all 5 NS + 5 GS deep-dive MP4s
DEEP_DIVE_NS = [
    {"title": "NS · ν = 0.004",      "hero": DEEP_DIVE_ROOT / "ns_pick01_cand_0096.mp4",
     "d": 3.24,  "b": 0.047, "comp": "[13, 0, 1]"},
    {"title": "NS · ν = 0.010",      "hero": DEEP_DIVE_ROOT / "ns_pick02_cand_0120.mp4",
     "d": 2.58,  "b": 0.063, "comp": "[16, 0, 1]"},
    {"title": "NS · ν = 0.0003",     "hero": DEEP_DIVE_ROOT / "ns_pick03_cand_0037.mp4",
     "d": 5.47,  "b": 0.310, "comp": "(2 ops)"},
    {"title": "NS · ν = 0.001",      "hero": DEEP_DIVE_ROOT / "ns_pick04_cand_0068.mp4",
     "d": 1.63,  "b": 0.095, "comp": "(2 ops)"},
    {"title": "NS · ν = 0.0006",     "hero": DEEP_DIVE_ROOT / "ns_pick05_cand_0055.mp4",
     "d": 1.20,  "b": 0.083, "comp": "(2 ops)"},
]
DEEP_DIVE_GS = [
    {"title": "GS · F = 0.055, k = 0.059",  "hero": DEEP_DIVE_ROOT / "gs_pick02_cand_0037.mp4",
     "d": 0.432, "b": 0.043, "comp": "[22, 10]"},
    {"title": "GS · F = 0.095, k = 0.051",  "hero": DEEP_DIVE_ROOT / "gs_pick08_cand_0062.mp4",
     "d": 0.604, "b": 0.111, "comp": "[20, 17]"},
    {"title": "GS · F = 0.045, k = 0.067",  "hero": DEEP_DIVE_ROOT / "gs_pick34_cand_0034.mp4",
     "d": 0.237, "b": 0.098, "comp": "[27, 9]"},
    {"title": "GS · F = 0.005, k = 0.063",  "hero": DEEP_DIVE_ROOT / "gs_pick44_cand_0007.mp4",
     "d": 0.189, "b": 0.096, "comp": "[2, 30]"},
    {"title": "GS · F = 0.055, k = 0.059",  "hero": DEEP_DIVE_ROOT / "gs_pick46_cand_0038.mp4",
     "d": 0.231, "b": 0.140, "comp": "[12, 29]"},
]


TABLE1 = {
    "headers": ["Method", "Adv + Diff", "NonlinAdv + Diff", "NonlinAdv + Disp",
                "Diff + Disp", "All three", "React + Diff", "Euler + Diff"],
    "rows": [
        ("MPP",              ["0.270", "0.050",  "0.105", "0.091", "0.128", "0.191", "0.273"]),
        ("FNO",              ["0.318", "0.031",  "0.165", "0.038", "0.129", "0.224", "0.241"]),
        ("Zebra",            ["0.893", "0.022*", "0.241", "0.069", "0.193", "0.127", "0.198"]),
        ("GEPS",             ["0.039", "0.039",  "0.249", "0.229", "0.265", "0.128", "0.786"]),
        ("DISCO (Original)", ["0.170", "0.085",  "0.100", "0.120", "0.164", "0.245", "0.572"]),
        ("Ours (Uniform)",   ["0.043", "0.068",  "0.103", "0.043", "0.075", "0.089", "0.209"]),
        ("Ours (Beam)",      ["**0.015**", "0.056", "**0.049**", "**0.007**", "**0.036**", "**0.089**", "**0.066**"]),
    ],
}

TABLE2 = {
    "headers": ["Method", "Adv–Diff c", "Adv–Diff D", "Combined α", "Combined γ"],
    "rows": [
        ("MPP",            ["0.588", "0.409", "0.134", "0.369"]),
        ("FNO",            ["0.492", "0.166", "0.166", "0.317"]),
        ("Zebra",          ["1.070", "1.579", "0.128", "0.448"]),
        ("GEPS",           ["0.848", "0.267", "0.020", "0.782"]),
        ("DISCO",          ["0.768", "0.159", "0.088", "1.007"]),
        ("Ours (Uniform)", ["0.113", "0.055", "0.027", "0.070"]),
        ("Ours (Beam)",    ["**0.052**", "**0.002**", "**0.016**", "**0.022**"]),
    ],
}


AUTHORS = [
    ("Louis Serrano",   ["NYU", "PM", "EMMI"]),
    ("Jiequn Han",      ["FI"]),
    ("Edouard Oyallon", ["SOR"]),
    ("Shirley Ho",      ["NYU", "PM", "FI", "PRIN"]),
    ("Rudy Morel",      ["PM", "FI"], "corresponding"),
]

AFFILIATIONS = [
    ("FI",   "Flatiron Institute"),
    ("NYU",  "New York University"),
    ("PRIN", "Princeton University"),
    ("SOR",  "Sorbonne Université, CNRS, ISIR"),
    ("EMMI", "Emmi AI"),
    ("PM",   "Polymathic AI"),
]


GITHUB_URL = "https://github.com/LouisSerrano/neural-operator-splitting"


def media_data_uri(path: Path):
    ext = path.suffix.lower()
    mime = {".mp4": "video/mp4",
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".svg": "image/svg+xml"}[ext]
    with open(path, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode()


def render_table(t, idclass=""):
    def cell(s):
        if s.startswith("**") and s.endswith("**"):
            return f"<td class='best'>{s[2:-2]}</td>"
        return f"<td>{s}</td>"
    head = "".join(f"<th>{h}</th>" for h in t["headers"])
    body = "".join(
        f"<tr class=\"{'ours' if r[0].startswith('Ours') else ''}\">"
        f"<th scope='row'>{r[0]}</th>" + "".join(cell(v) for v in r[1]) + "</tr>"
        for r in t["rows"])
    return f"<div class='tablewrap'><table class='nrmse {idclass}'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def case_card(case, featured=False):
    """Render one case as a figure. Accepts a dict with absolute hero path."""
    hero_path = Path(case["hero"])
    title  = case["title"]
    blurb  = case.get("blurb", "")
    params = case.get("params", "")
    direct = case["d"]
    beam   = case["b"]
    is_dd  = case.get("is_deep_dive", False)

    if not hero_path.exists() or hero_path.stat().st_size == 0:
        body = f"<div class='missing'>render pending — {hero_path.name}</div>"
    else:
        uri = media_data_uri(hero_path)
        if hero_path.suffix == ".mp4":
            body = (f'<video src="{uri}" autoplay loop muted playsinline preload="metadata"'
                     f' aria-label="rollout for {title}"></video>')
        else:
            body = f'<img src="{uri}" alt="rollout snapshots for {title}">'

    ratio = direct / beam if beam else 0
    ratio_str = f"{ratio:,.0f}×" if ratio >= 10 else f"{ratio:.1f}×"
    dd_badge = '<span class="dd-badge" title="composition over time">deep dive</span>' if is_dd else ""
    params_html = f'<p class="params mono">{params}</p>' if params else ""

    return f"""
    <figure class="case{' featured' if featured else ''}">
      <div class="media">{body}{dd_badge}</div>
      <figcaption>
        <header>
          <h3>{title}</h3>
          <p class="sub">{blurb}</p>
          {params_html}
        </header>
        <dl class="errs">
          <div><dt>direct</dt><dd class="d">{direct:.3g}</dd></div>
          <div><dt>ours, beam</dt><dd class="b">{beam:.3g}</dd></div>
          <div class="ratio"><dt>recovery</dt><dd>{ratio_str}</dd></div>
        </dl>
      </figcaption>
    </figure>"""


def section_results(title, eyebrow, cases, table, table_caption, table_id, body_extra=""):
    cards = "".join(
        case_card(c, featured=(i == 0)) for i, c in enumerate(cases)
    )
    return f"""
  <section class="results" id="{table_id}">
    <header class="sec-head">
      <p class="eyebrow">{eyebrow}</p>
      <h2>{title}</h2>
    </header>
    {body_extra}
    <div class="case-grid">{cards}</div>

    <div class="numbers">
      <p class="eyebrow">Reported NRMSE · lower is better · bold marks the best per column</p>
      {render_table(table, table_id + "-tbl")}
      <p class="cap">{table_caption}</p>
    </div>
  </section>"""


def deep_dive_card(c):
    hero_path = Path(c["hero"])
    if not hero_path.exists() or hero_path.stat().st_size == 0:
        body = "<div class='missing'>pending</div>"
    else:
        uri = media_data_uri(hero_path)
        body = (f'<video src="{uri}" autoplay loop muted playsinline preload="metadata"'
                 f' aria-label="deep dive — {c["title"]}"></video>')
    ratio = c["d"] / c["b"] if c["b"] else 0
    return f"""
    <figure class="dd-case">
      <div class="media">{body}</div>
      <figcaption>
        <h4>{c['title']}</h4>
        <dl class="dd-meta mono">
          <div><dt>final comp</dt><dd>{c['comp']}</dd></div>
          <div><dt>d → b</dt><dd><span class="d">{c['d']:.3g}</span> → <span class="b">{c['b']:.3g}</span></dd></div>
          <div><dt>recovery</dt><dd>{ratio:,.0f}×</dd></div>
        </dl>
      </figcaption>
    </figure>"""


def section_deep_dive():
    ns_cards = "".join(deep_dive_card(c) for c in DEEP_DIVE_NS)
    gs_cards = "".join(deep_dive_card(c) for c in DEEP_DIVE_GS)
    return f"""
  <section class="deep-dive" id="deep-dive">
    <header class="sec-head">
      <p class="eyebrow">Composition over time · watch beam search add operators</p>
      <h2>Deep dive: how the prediction closes the gap, depth by depth</h2>
      <p class="cap">
        Top row is the ground-truth rollout. Each row below is the prediction after
        one more operator has been added to the composition. The right-hand columns
        show the residual <span class="mono">|pred − GT|</span> — it darkens as the
        composition grows. The banner above each video lists the operators chosen
        at every depth.
      </p>
    </header>

    <h3 class="dd-sub">2D Navier–Stokes — spanning ν from 1e-4 to 1e-2</h3>
    <div class="dd-grid">{ns_cards}</div>

    <h3 class="dd-sub">2D Gray–Scott — spanning the (F, k) parameter space</h3>
    <div class="dd-grid">{gs_cards}</div>
  </section>"""


def scaling_section():
    """Compute scaling + parameter identification — the two beyond-accuracy stories."""
    fig_path = PAPER_ROOT / "plots/flop_analysis_combined_alt-2.png"
    fig_html = ""
    if fig_path.exists():
        fig_html = (
            f'<figure class="paper-fig">'
            f'<img src="{media_data_uri(fig_path)}" alt="FLOPs versus NRMSE for uniform and beam search">'
            f'<figcaption>'
            f'<span class="figmark">Figure 5</span> '
            f'Fitting error vs. cumulative FLOPs on three OOD tasks. Beam search proceeds sequentially over '
            f'composition complexity — single operators, then pairs, then triples — producing sharp drops as '
            f'more expressive combinations become available. Uniform search improves smoothly with the same '
            f'budget, but more slowly.'
            f'</figcaption></figure>'
        )
    return f"""
  <section class="results" id="scaling">
    <header class="sec-head">
      <p class="eyebrow">Beyond accuracy · §5.4 of the paper</p>
      <h2>Test-time compute scales gracefully — and the picks identify the physics</h2>
    </header>

    <div class="two-col">
      <div class="prose-col">
        <h3>Spend more search, get less error</h3>
        <p>
          Increasing the number of search trials produces a near-power-law decay in
          both fitting error and rollout error. Beam search dominates uniform sampling
          at every budget by expanding the composition by one operator per round —
          first singletons, then pairs, then triples — so each additional unit of compute
          buys a structurally richer hypothesis, not just another random draw.
        </p>
        <p>
          On a single A100, beam search is competitive with extensive GEPS-style
          gradient fine-tuning on advection–diffusion, and remains stable in far-OOD
          regimes where GEPS diverges. We report the full wall-clock comparison in the
          paper appendix.
        </p>
      </div>
      <div>{fig_html}</div>
    </div>

    <div class="callout">
      <p class="eyebrow">Parameter identification</p>
      <h3>Reading the recovered composition</h3>
      <p>
        Each picked operator <span class="mono">f<sub>i</sub></span> traces back to the
        training trajectory it was hyper-network-encoded from — and that trajectory has
        known PDE coefficients. Summing the coefficient contributions of the picks gives
        a coefficient estimate for the unknown test dynamics. The recovered advection
        speed and diffusion coefficient on far-OOD test trajectories track the true values
        as beam search progresses, even though the coupled system was never seen during
        training. This parallels classical equation-discovery methods such as SINDy,
        but composes over learned operators instead of a handcrafted symbolic library.
      </p>
    </div>
  </section>"""


def author_html(author):
    name = author[0]; affs = author[1]
    is_corr = len(author) > 2 and author[2] == "corresponding"
    aff_html = "".join(f"<sup>{a}</sup>" for a in affs)
    corr_html = '<sup class="corr">✱</sup>' if is_corr else ""
    return f'<span class="author">{name}{aff_html}{corr_html}</span>'


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/mnt/home/lserrano/disco-ball/viz/site/index.html")
    args = p.parse_args()

    authors_html = "<span class='sep'>·</span>".join(author_html(a) for a in AUTHORS)
    aff_html = " &nbsp;·&nbsp; ".join(
        f"<sup>{k}</sup> {v}" for k, v in AFFILIATIONS)

    # Inline the paper's figure 1
    main_fig = PAPER_ROOT / "figures/main_figure_arxiv.png"
    main_fig_html = ""
    if main_fig.exists():
        main_fig_html = (
            f'<figure class="paper-fig">'
            f'<img src="{media_data_uri(main_fig)}" alt="DISCO + neural operator splitting overview">'
            f'<figcaption>'
            f'<span class="figmark">Figure 1</span> '
            f'During pretraining (left), DISCO learns a dictionary of operators '
            f'<span class="mono">f<sub>i</sub></span> for distinct physics — e.g. reaction (green), '
            f'diffusion+kill (red) — with a hypernetwork producing each operator&rsquo;s weights '
            f'<span class="mono">θ<sub>i</sub></span>. At test time (right), faced with OOD '
            f'dynamics like reaction–diffusion, we search over '
            f'<em>compositions</em> of dictionary operators '
            f'(<span class="mono">f<sub>OOD</sub> ≈ f<sub>i<sub>1</sub></sub> + f<sub>i<sub>2</sub></sub></span>) '
            f'and integrate the next step via operator splitting.'
            f'</figcaption></figure>'
        )

    html = f"""<title>Test-time Generalization for Physics through Neural Operator Splitting</title>
<meta name="description" content="ICML 2026 — Zero-shot generalization on out-of-distribution PDEs via test-time operator-composition search over a dictionary of pretrained neural operators.">
<style>{CSS}</style>

<header class="masthead">
  <div class="venue-line">
    <span class="venue">International Conference on Machine Learning 2026</span>
    <span class="sep">·</span>
    <span class="collab">Polymathic AI Collaboration</span>
  </div>
  <h1>
    Test-time generalization for physics through
    <span class="emph">neural operator splitting</span>
  </h1>
  <div class="byline">
    <p class="authors">{authors_html}</p>
    <p class="affiliations">{aff_html}</p>
    <p class="corr-line"><sup class="corr">✱</sup> corresponding author · <a href="mailto:rmorel@flatironinstitute.org">rmorel@flatironinstitute.org</a> &nbsp;·&nbsp; first author <a href="mailto:louis.serrano@mistral.ai">louis.serrano@mistral.ai</a></p>
  </div>
  <div class="quicklinks">
    <a href="{GITHUB_URL}" target="_blank" rel="noopener">Code</a>
    <span class="sep">/</span>
    <a href="https://huggingface.co/sogeeking/disco-models" target="_blank" rel="noopener">Models</a>
    <span class="sep">/</span>
    <a href="https://huggingface.co/sogeeking/disco-ns" target="_blank" rel="noopener">Datasets</a>
    <span class="sep">/</span>
    <a href="#composition">Composition results</a>
    <span class="sep">/</span>
    <a href="#extrapolation">Parameter extrapolation</a>
  </div>
</header>

<section class="abstract" id="abstract">
  <p class="eyebrow">Abstract</p>
  <p class="body">
    Neural operators have shown promise in learning solution maps of partial differential
    equations, but they often struggle to generalize when test inputs lie outside the
    training distribution — novel initial conditions, unseen coefficients, or unseen physics.
    Prior work addresses this by large-scale multi-physics pretraining followed by
    fine-tuning; that still requires examples from the new dynamics, falling short of
    true zero-shot generalization.
  </p>
  <p class="body">
    We propose enhancing generalization <em>at test time</em>, without modifying pretrained
    weights. Building on <strong>DISCO</strong>, which provides a dictionary of neural
    operators trained across different dynamics, we introduce a neural-operator-splitting
    strategy that, at test time, searches over compositions of training operators to
    approximate unseen dynamics. On challenging out-of-distribution tasks — parameter
    extrapolation and novel combinations of physics phenomena — our approach achieves
    state-of-the-art zero-shot results while recovering the underlying PDE parameters.
  </p>
</section>

<section class="method" id="method">
  <header class="sec-head">
    <p class="eyebrow">Method · no retraining, no fine-tuning, no labels</p>
    <h2>Search compositions of frozen operators at test time</h2>
  </header>

  {main_fig_html}

  <div class="method-prose">
    <p>
      We begin with a DISCO-pretrained backbone. A transformer hypernetwork
      <span class="mono">ψ<sub>α</sub></span> maps each training trajectory
      <span class="mono">u<sub>i</sub><sup>1:L</sup></span> to a small U-Net operator
      <span class="mono">f<sub>θ<sub>i</sub></sub></span>, where
      <span class="mono">θ<sub>i</sub> = ψ<sub>α</sub>(u<sub>i</sub><sup>1:L</sup>)</span>.
      After pretraining we freeze everything and treat the encoded set
      <span class="mono">{{f<sub>1</sub>, …, f<sub>N</sub>}}</span> as a fixed dictionary
      of differential operators — one per training environment.
    </p>
    <p>
      Given a new test trajectory under unknown — possibly OOD — dynamics, we look for a
      subset <span class="mono">S ⊆ {{f<sub>1</sub>, …, f<sub>N</sub>}}</span> whose
      <em>sum</em> best fits the observed evolution. The fitting objective is the
      averaged NRMSE between the next-step prediction obtained via operator splitting
      and the observed test data. We compare two search strategies: uniform random
      sampling of subsets, and beam search that greedily grows compositions while
      keeping the top-<span class="mono">B</span> candidates per depth.
    </p>
    <p>
      Once a composition is chosen, the next step
      <span class="mono">u<sup>t</sup> → u<sup>t+1</sup></span> is realised through Strang
      operator splitting: each <span class="mono">f<sub>i</sub></span> is integrated for a
      fractional timestep in sequence, producing a single combined integrator.
      Composition lengths up to <span class="mono">M = 5</span> suffice across all benchmarks.
    </p>
  </div>
</section>

{section_results("Composition of unseen physics",
                  "Table 1 results · seven OOD combinations",
                  TABLE1_CASES, TABLE1,
                  "Each column is a held-out physics combination. Methods see each phenomenon "
                  "individually during pretraining; at test time multiple phenomena act simultaneously. "
                  "<span class='note'>*Zebra wins one column (nonlinear advection + diffusion); we win the remaining six.</span>",
                  "composition")}

<section class="results sweep" id="e-all-sweep">
  <header class="sec-head">
    <p class="eyebrow">All three phenomena · sweep across nonlinear-advection strength α</p>
    <h2>How the recovery scales when the unknown PDE gets harder</h2>
    <p class="cap">
      Same Diff + Disp baseline (β ≈ 0.1–0.3, γ = 0.25); the only thing that changes is α.
      As nonlinearity grows the system gets richer dynamics — and the recovery ratio
      compresses from 22× down to 2.5×. Beam search still wins everywhere.
    </p>
  </header>
  <div class="case-grid sweep-grid">{ "".join(case_card(c) for c in TABLE1_E_ALL_SWEEP) }</div>
</section>

{section_results("Parameter extrapolation",
                  "Table 2 results · coefficients pushed outside the training range",
                  TABLE2_CASES, TABLE2,
                  "Advection–diffusion: train on small c and D, test on values roughly 3× larger. "
                  "Combined-equation: extrapolate α and γ beyond the training distribution. "
                  "Beam search consistently recovers OOD parameters that all baselines fail on — most strikingly "
                  "on Adv-Diff D, where the error drops by nearly two orders of magnitude.",
                  "extrapolation")}

{scaling_section()}

{section_deep_dive()}

<footer>
  <div class="cite">
    <p class="eyebrow">Cite as</p>
    <pre>@inproceedings{{serrano2026ttgnos,
  title     = {{Test-time Generalization for Physics through Neural Operator Splitting}},
  author    = {{Serrano, Louis and Han, Jiequn and Oyallon, Edouard and
               Ho, Shirley and Morel, Rudy}},
  booktitle = {{International Conference on Machine Learning}},
  year      = {{2026}},
}}</pre>
  </div>
  <div class="links">
    <p class="eyebrow">Resources</p>
    <ul>
      <li><a href="{GITHUB_URL}" target="_blank" rel="noopener">Code repository on GitHub</a></li>
      <li><a href="https://huggingface.co/sogeeking/disco-models" target="_blank" rel="noopener">Pretrained checkpoints (Hugging Face)</a></li>
      <li><a href="https://huggingface.co/sogeeking/disco-ns" target="_blank" rel="noopener">Datasets (Hugging Face)</a></li>
      <li>Contact (corresponding): <a href="mailto:rmorel@flatironinstitute.org">rmorel@flatironinstitute.org</a></li>
      <li>Contact (first author): <a href="mailto:louis.serrano@mistral.ai">louis.serrano@mistral.ai</a></li>
    </ul>
  </div>
</footer>
"""
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"wrote {out}  ({out.stat().st_size/1024/1024:.1f} MB)")


CSS = r"""
:root {
  /* Near-white with a fractional warm tint — "considered preprint" */
  --paper:    #FBFAF7;
  --paper-2:  #F2F1EC;
  --card:     #FFFFFF;
  --ink:      #15171A;
  --ink-2:    #2A2D33;
  --rule:     #DAD7CF;
  --rule-2:   #E7E5DE;
  --muted:    #6E6F71;
  /* Single accent — deep journal blue, not terracotta */
  --accent:   #1E4A6E;
  --accent-h: #143855;
  /* Semantic only — never used as design accent */
  --d:        #A04822;   /* direct rollout */
  --b:        #386850;   /* beam (ours) */
}

* { box-sizing: border-box; }
html { background: var(--paper); color: var(--ink); }
body {
  font-family: 'Charter','Iowan Old Style','Source Serif Pro','Sitka Text',Georgia,serif;
  font-feature-settings: 'kern','liga','onum';
  font-size: 17.5px;
  line-height: 1.62;
  max-width: 1080px;
  margin: 0 auto;
  padding: 0 32px 120px;
  color: var(--ink);
}
p { margin: 0; }
h1, h2, h3, h4 { font-weight: 600; letter-spacing: -0.014em; text-wrap: balance; }
sup { font-size: 0.72em; font-variant-numeric: tabular-nums; }

.mono {
  font-family: ui-monospace, 'SF Mono', 'JetBrains Mono', Menlo, Consolas, monospace;
  font-variant-numeric: tabular-nums;
  font-size: 0.94em;
}
.eyebrow {
  font-family: ui-monospace, 'SF Mono', Menlo, monospace;
  font-size: 11.5px; letter-spacing: 0.13em; text-transform: uppercase;
  color: var(--muted);
}
a { color: var(--accent); text-decoration-thickness: 1px; text-underline-offset: 2.5px; }
a:hover { color: var(--accent-h); }
em { font-style: italic; }
strong { font-weight: 600; }
.sep { color: var(--rule); margin: 0 6px; }

/* =========================================================================
   Masthead
   ====================================================================== */
header.masthead {
  padding: 56px 0 36px;
  border-bottom: 1px solid var(--rule);
  display: flex; flex-direction: column; gap: 22px;
}
.venue-line {
  font-family: ui-monospace, monospace;
  font-size: 12px; letter-spacing: 0.09em;
  color: var(--muted);
  text-transform: uppercase;
}
.venue-line .collab { color: var(--ink-2); font-weight: 600; }

header.masthead h1 {
  font-size: clamp(34px, 5.4vw, 60px);
  line-height: 1.03;
  margin: 0;
  max-width: 900px;
}
header.masthead h1 .emph {
  display: block;
  color: var(--accent);
  font-style: italic;
  font-weight: 500;
}

.byline { display: flex; flex-direction: column; gap: 6px; }
.authors {
  font-size: 16.5px; line-height: 1.5;
}
.author { white-space: nowrap; }
.author sup { color: var(--muted); margin-left: 1px; }
.author sup.corr { color: var(--d); font-weight: 700; }
.affiliations {
  color: var(--muted); font-size: 13.5px; line-height: 1.65;
  max-width: 820px;
}
.affiliations sup { color: var(--ink-2); font-weight: 600; }
.corr-line { color: var(--muted); font-size: 13px; }
.corr-line .corr { color: var(--d); font-weight: 700; }

.quicklinks {
  display: flex; flex-wrap: wrap; align-items: baseline;
  gap: 8px; padding-top: 6px;
  font-family: ui-monospace, monospace;
  font-size: 13px;
}

/* =========================================================================
   Section heads
   ====================================================================== */
.sec-head { margin: 0 0 28px; }
.sec-head h2 {
  font-size: clamp(22px, 2.6vw, 28px);
  line-height: 1.18;
  margin: 6px 0 0;
}

section { padding: 56px 0; border-bottom: 1px solid var(--rule); }
section:last-of-type { border-bottom: 0; }

/* =========================================================================
   Abstract — two paragraphs, no eyebrow column
   ====================================================================== */
section.abstract {
  display: grid;
  grid-template-columns: 1fr;
  gap: 18px;
  max-width: 760px;
}
section.abstract .eyebrow { margin-bottom: 6px; }
section.abstract .body {
  font-size: 18.5px; line-height: 1.62;
  color: var(--ink);
}
section.abstract .body em { color: var(--accent); }

/* =========================================================================
   Method — figure + prose
   ====================================================================== */
.paper-fig {
  margin: 0 0 28px;
  padding: 0;
  background: var(--card);
  border: 1px solid var(--rule);
}
.paper-fig img {
  width: 100%; height: auto; display: block;
  padding: 16px 16px 8px;
}
.paper-fig figcaption {
  padding: 12px 18px 16px;
  border-top: 1px solid var(--rule-2);
  font-size: 14px; line-height: 1.55; color: var(--ink-2);
}
.paper-fig .figmark {
  font-family: ui-monospace, monospace;
  font-weight: 700; color: var(--accent); margin-right: 4px;
  letter-spacing: 0.04em;
}

.method-prose { max-width: 780px; }
.method-prose p { margin: 0 0 14px; font-size: 17px; }

/* =========================================================================
   Two-col (prose + figure) used by the scaling section
   ====================================================================== */
.two-col {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1.15fr);
  gap: 32px;
  align-items: start;
}
.two-col h3 { font-size: 17px; margin: 0 0 10px; }
.prose-col p { font-size: 16px; line-height: 1.6; margin: 0 0 12px; }
.prose-col p:last-child { margin-bottom: 0; }
.two-col .paper-fig { margin: 0; }

/* "deep dive" badge in the corner of featured 2D cases */
figure.case .dd-badge {
  position: absolute; top: 8px; right: 8px;
  background: rgba(27, 58, 94, 0.92); color: #FBFAF7;
  font-family: ui-monospace, monospace;
  font-size: 10.5px; font-weight: 600; letter-spacing: 0.05em;
  padding: 3px 8px; border-radius: 2px;
}
figure.case .media { position: relative; }
figure.case .params { color: var(--muted); font-size: 12.5px; margin-top: 4px; }

/* Sweep grid (E_ALL): 4 cards side-by-side, smaller */
.case-grid.sweep-grid { grid-template-columns: repeat(4, minmax(0, 1fr)); }
.case-grid.sweep-grid figure.case .media video { aspect-ratio: 11/6; }
.case-grid.sweep-grid figure.case figcaption { padding: 10px 12px; }
.case-grid.sweep-grid figure.case figcaption header h3 { font-size: 13.5px; }
.case-grid.sweep-grid figure.case figcaption .sub { font-size: 11.5px; }
.case-grid.sweep-grid figure.case figcaption .params { font-size: 11px; }
.case-grid.sweep-grid dl.errs { gap: 10px; }
.case-grid.sweep-grid dl.errs dt { font-size: 9.5px; }
.case-grid.sweep-grid dl.errs dd { font-size: 13px; }
.case-grid.sweep-grid dl.errs .ratio dd { font-size: 14.5px; }
@media (max-width: 980px) { .case-grid.sweep-grid { grid-template-columns: repeat(2, 1fr); } }
@media (max-width: 580px) { .case-grid.sweep-grid { grid-template-columns: 1fr; } }

/* Deep-dive section */
section.deep-dive { padding: 56px 0 8px; border-bottom: 1px solid var(--rule); }
section.deep-dive .sec-head .cap {
  color: var(--muted); font-size: 14px; line-height: 1.55;
  max-width: 780px; margin-top: 10px;
}
.dd-sub {
  font-size: 14px; color: var(--ink-2); font-weight: 600;
  margin: 32px 0 12px;
  letter-spacing: 0.01em;
  border-bottom: 1px dotted var(--rule);
  padding-bottom: 6px;
}
.dd-grid {
  display: grid; gap: 18px;
  grid-template-columns: repeat(auto-fill, minmax(380px, 1fr));
}
figure.dd-case {
  margin: 0; background: var(--card);
  border: 1px solid var(--rule);
  display: flex; flex-direction: column; overflow: hidden;
}
figure.dd-case .media { background: #000; line-height: 0; }
figure.dd-case video { width: 100%; height: auto; display: block; }
figure.dd-case figcaption {
  padding: 12px 14px;
  border-top: 1px solid var(--rule-2);
  display: flex; flex-direction: column; gap: 6px;
}
figure.dd-case h4 { font-size: 14px; font-weight: 600; margin: 0; }
dl.dd-meta {
  margin: 0; display: flex; gap: 14px; flex-wrap: wrap;
  font-size: 12px;
}
dl.dd-meta > div { display: flex; flex-direction: column; }
dl.dd-meta dt { font-size: 10px; color: var(--muted);
                 text-transform: uppercase; letter-spacing: 0.05em; }
dl.dd-meta dd { margin: 0; font-size: 13px; color: var(--ink); }
dl.dd-meta .d { color: var(--d); font-weight: 700; }
dl.dd-meta .b { color: var(--b); font-weight: 700; }

/* Callout for parameter identification */
.callout {
  margin-top: 36px;
  padding: 22px 26px 24px;
  background: var(--card);
  border: 1px solid var(--rule);
  border-left: 3px solid var(--accent);
}
.callout .eyebrow { margin-bottom: 4px; }
.callout h3 { font-size: 19px; margin: 0 0 10px; max-width: 720px; }
.callout p { font-size: 16px; line-height: 1.6; max-width: 800px; }

@media (max-width: 760px) {
  .two-col { grid-template-columns: 1fr; gap: 22px; }
}

/* =========================================================================
   Results — case grid + tables
   ====================================================================== */
.results .case-grid {
  display: grid; gap: 22px;
  grid-template-columns: repeat(2, minmax(0, 1fr));
}
figure.case {
  margin: 0;
  background: var(--card);
  border: 1px solid var(--rule);
  display: flex; flex-direction: column;
  overflow: hidden;
}
figure.case.featured {
  grid-column: 1 / -1;
  background: var(--card);
}
figure.case.featured .media {
  background: #fff;
}
figure.case .media {
  background: #000;
  line-height: 0;
}
figure.case video, figure.case img {
  width: 100%; height: auto; display: block;
}
figure.case figcaption {
  padding: 14px 16px 14px;
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 10px 18px;
  align-items: start;
  border-top: 1px solid var(--rule-2);
}
figure.case figcaption header h3 { font-size: 16px; margin: 0; }
figure.case figcaption header .sub {
  color: var(--muted); font-size: 13px; font-style: italic;
  margin-top: 2px;
}
dl.errs {
  margin: 0;
  display: flex; gap: 16px;
  font-family: ui-monospace, monospace;
  font-variant-numeric: tabular-nums;
}
dl.errs > div { display: flex; flex-direction: column; align-items: flex-end; }
dl.errs dt {
  font-size: 10.5px; color: var(--muted);
  letter-spacing: 0.05em; text-transform: uppercase;
}
dl.errs dd { margin: 0; font-size: 15px; font-weight: 600; }
dl.errs dd.d { color: var(--d); }
dl.errs dd.b { color: var(--b); }
dl.errs .ratio dd {
  color: var(--accent);
  font-weight: 700;
  font-size: 17px;
}
.missing {
  padding: 56px 24px; text-align: center;
  color: var(--muted); font-style: italic;
  background: var(--paper-2);
}

/* =========================================================================
   NRMSE tables
   ====================================================================== */
.numbers { margin-top: 36px; }
.numbers .eyebrow { margin-bottom: 8px; }
.numbers .cap {
  color: var(--muted); font-size: 14px; line-height: 1.6;
  max-width: 780px; margin-top: 10px;
}
.numbers .cap .note { font-style: italic; }
.tablewrap {
  overflow-x: auto;
  border: 1px solid var(--rule);
  background: var(--card);
}
table.nrmse {
  width: 100%;
  border-collapse: collapse;
  font-family: ui-monospace, monospace;
  font-variant-numeric: tabular-nums;
  font-size: 13.5px;
}
table.nrmse thead th {
  text-align: right;
  padding: 12px 10px 11px;
  border-bottom: 1px solid var(--rule);
  font-weight: 600; font-size: 11px;
  letter-spacing: 0.04em; text-transform: uppercase;
  color: var(--muted);
  white-space: nowrap;
  background: var(--paper-2);
}
table.nrmse thead th:first-child { text-align: left; }
table.nrmse tbody th, table.nrmse tbody td {
  padding: 8px 10px;
  border-bottom: 1px solid var(--rule-2);
}
table.nrmse tbody th[scope='row'] {
  text-align: left; font-family: 'Charter', Georgia, serif;
  font-weight: 500; color: var(--ink-2);
}
table.nrmse tbody td { text-align: right; }
table.nrmse tbody td.best {
  color: var(--b); font-weight: 700;
}
table.nrmse tbody tr.ours {
  background: rgba(56, 104, 80, 0.05);
}
table.nrmse tbody tr.ours th {
  color: var(--b); font-weight: 600;
}
table.nrmse tbody tr:last-child td,
table.nrmse tbody tr:last-child th { border-bottom: 0; }

/* =========================================================================
   Footer
   ====================================================================== */
footer {
  display: grid;
  grid-template-columns: 1.4fr 1fr;
  gap: 40px;
  padding: 48px 0 0;
}
footer .eyebrow { margin-bottom: 8px; }
footer pre {
  background: var(--card);
  border: 1px solid var(--rule);
  padding: 14px 16px;
  font-family: ui-monospace, monospace;
  font-size: 12.5px; line-height: 1.55;
  overflow-x: auto;
  white-space: pre;
  margin: 0;
}
footer ul { list-style: none; padding: 0; margin: 0; }
footer li { padding: 4px 0; font-size: 15px; }

/* =========================================================================
   Responsive
   ====================================================================== */
@media (max-width: 760px) {
  body { padding: 0 18px 80px; font-size: 16px; }
  header.masthead { padding: 36px 0 28px; }
  section { padding: 36px 0; }
  .results .case-grid { grid-template-columns: 1fr; }
  figure.case figcaption { grid-template-columns: 1fr; }
  dl.errs { width: 100%; justify-content: space-between; }
  footer { grid-template-columns: 1fr; }
}

@media (prefers-reduced-motion: reduce) {
  figure.case video { /* let video still play because it's the substance */
  }
}
"""


if __name__ == "__main__":
    main()

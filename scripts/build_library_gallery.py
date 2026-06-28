"""Build a 'scan the library' gallery — every candidate shows as a thumbnail
+ metadata card. No MP4 inline. The point is fast scanning to pick favorites,
not full visualization.

Run:  python scripts/build_library_gallery.py [--out PATH]
"""
from __future__ import annotations

import argparse
import base64
import glob
import json
import math
import os

ROOT_DATA  = "/mnt/home/lserrano/disco-ball/viz/data/scan"
ROOT_THUMBS = "/mnt/home/lserrano/disco-ball/viz/thumbs/scan"


SECTION_LABELS = {
    "adv-diff_E_AD_ALL":         ("Adv + Diff",                    "Table 1",  "in-distribution v + D"),
    "combined-eq_E_HE":          ("Nonlinear Adv + Diff (E_HE)",   "Table 1",  ""),
    "combined-eq_E_ED":          ("Nonlinear Adv + Disp (E_ED)",   "Table 1",  ""),
    "combined-eq_E_BG":          ("Diff + Disp (E_BG)",            "Table 1",  ""),
    "combined-eq_E_ALL":         ("All three (E_ALL)",             "Table 1",  "α, β, γ simultaneously"),
    "gray-scott":                ("Reaction + Diffusion",          "Table 1",  "Gray-Scott · activator + inhibitor"),
    "navier-stokes":             ("Euler + Diffusion (NS)",        "Table 1",  "2D vorticity"),
    "adv-diff_E_AD_v":           ("Adv-Diff c (E_AD_v)",           "Table 2",  "v ∈ [1, 3], no diffusion"),
    "adv-diff_E_AD_D":           ("Adv-Diff D (E_AD_D)",           "Table 2",  "D ∈ [1, 3], no advection"),
    "combined-eq_E_EULER_OOD":   ("Combined α (E_EULER_OOD)",      "Table 2",  "α ∈ [1, 2]"),
    "combined-eq_E_DISP_OOD":    ("Combined γ (E_DISP_OOD)",       "Table 2",  "γ ∈ [1, 2]"),
}


def thumb_data_uri(path):
    mime = "image/jpeg" if path.endswith((".jpg", ".jpeg")) else "image/png"
    with open(path, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode()


def collect():
    groups = {}
    for cand_dir in sorted(glob.glob(f"{ROOT_DATA}/*/run_*/cand_*")):
        if "WRONG" in cand_dir or "FAILED" in cand_dir: continue
        exp = cand_dir.split("/scan/")[1].split("/")[0]
        cand = os.path.basename(cand_dir)
        md_path = os.path.join(cand_dir, "metadata.json")
        if not os.path.exists(md_path): continue
        thumb = f"{ROOT_THUMBS}/{exp}/{cand}.jpg"
        if not os.path.exists(thumb): continue
        m = json.load(open(md_path))
        groups.setdefault(exp, []).append({
            "cand": cand,
            "thumb": thumb,
            "direct": float(m.get("direct_test_error", float("nan"))),
            "beam":   float(m.get("beam_test_error",   float("nan"))),
            "ds_idx": int(m.get("dataset_index", -1)),
            "params": [(k, float(m[k])) for k in
                       ("advection_speed", "diffusion", "alpha", "beta", "gamma",
                        "f", "k", "viscosity") if k in m],
        })
    return groups


PARAM_SHORT = {
    "advection_speed": "v", "diffusion": "D",
    "alpha": "α", "beta": "β", "gamma": "γ",
    "f": "F", "k": "k", "viscosity": "ν",
}


def fmt_params(params):
    return "  ".join(f"<span class=k>{PARAM_SHORT.get(k,k)}</span><span class=v>{v:.4g}</span>"
                      for k, v in params)


def card_html(c, num, exp_short):
    impr = (c["direct"] / c["beam"]) if (c["beam"] > 0 and not math.isnan(c["direct"]) and not math.isnan(c["beam"])) else 0
    impr_str = f"{impr:.0f}×" if impr > 1.5 else ("≈" if impr > 0 else "—")
    ds_id = f"ds#{c['ds_idx']}" if c["ds_idx"] >= 0 else ""
    pick_id = f"{exp_short}·{num:02d}"
    # data-pick-id is the call-out ID; data-cand carries the on-disk cand name so
    # we can map a pick back to a directory when we render heroes.
    return (
        f'<figure class="card" data-pick-id="{pick_id}" data-cand="{c["cand"]}" data-ds="{c["ds_idx"]}" tabindex="0" role="button" aria-pressed="false">'
        f'<div class="thumb-wrap">'
        f'<img src="{thumb_data_uri(c["thumb"])}" alt="thumbnail" loading="lazy">'
        f'<span class="num">{num:02d}</span>'
        f'<span class="impr" title="direct/beam ratio">{impr_str}</span>'
        f'<span class="check" aria-hidden="true">✓</span>'
        f'</div>'
        f'<figcaption>'
        f'<div class="params">{fmt_params(c["params"])}</div>'
        f'<div class="meta">'
        f'<span class="errs"><span class="d">d {c["direct"]:.3g}</span><span class="b">b {c["beam"]:.3g}</span></span>'
        f'<span class="id">{pick_id}  <span class="dim">{ds_id}</span></span>'
        f'</div>'
        f'</figcaption>'
        '</figure>'
    )


def section_html(exp, cards, exp_short):
    label, table, blurb = SECTION_LABELS.get(exp, (exp, "", ""))
    cards_sorted = sorted(cards, key=lambda c: -(c["direct"] / max(c["beam"], 1e-9)))
    items = "".join(card_html(c, i + 1, exp_short) for i, c in enumerate(cards_sorted))
    blurb_html = f' <span class="blurb">— {blurb}</span>' if blurb else ""
    return (
        f'<section class="bench" id="{exp}">'
        f'<header class="bench-head">'
        f'<div class="label-row">'
        f'<span class="table-tag">{table}</span>'
        f'<h2>{label}{blurb_html}</h2>'
        f'<span class="count">{len(cards)} cand{"s" if len(cards) != 1 else ""}</span>'
        f'</div>'
        f'</header>'
        f'<div class="cards">{items}</div>'
        '</section>'
    )


# Two-letter shortcut for the chip nav + per-card ID prefix.
EXP_SHORT = {
    "adv-diff_E_AD_ALL":       "AD",
    "combined-eq_E_HE":        "HE",
    "combined-eq_E_ED":        "ED",
    "combined-eq_E_BG":        "BG",
    "combined-eq_E_ALL":       "CE",
    "gray-scott":              "GS",
    "navier-stokes":           "NS",
    "adv-diff_E_AD_v":         "Av",
    "adv-diff_E_AD_D":         "AD'",
    "combined-eq_E_EULER_OOD": "Eα",
    "combined-eq_E_DISP_OOD":  "Eγ",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/mnt/home/lserrano/disco-ball/viz/site/library.html")
    p.add_argument("--only", default=None,
                    help="Comma-separated experiment keys to include (e.g. 'gray-scott,navier-stokes').")
    args = p.parse_args()
    groups = collect()
    if args.only:
        keep = set(s.strip() for s in args.only.split(","))
        groups = {k: v for k, v in groups.items() if k in keep}
    total = sum(len(v) for v in groups.values())
    print(f"Collected {total} candidates across {len(groups)} experiments")
    ordered = [e for e in SECTION_LABELS if e in groups]
    ordered += [e for e in groups if e not in SECTION_LABELS]

    nav_chips = "".join(
        f'<a href="#{e}" class="chip"><span class="s">{EXP_SHORT.get(e,e[:2])}</span>'
        f'<span class="t">{SECTION_LABELS.get(e,(e,"",""))[0]}</span>'
        f'<span class="n">{len(groups[e])}</span></a>'
        for e in ordered
    )
    sections = "".join(section_html(e, groups[e], EXP_SHORT.get(e, e[:2])) for e in ordered)

    html = (
        '<title>Library — pick candidates</title>'
        '<meta name="description" content="All DISCO rollout candidates as static thumbnails, grouped by paper benchmark. Scan and call out card IDs.">'
        f'<style>{CSS}</style>'
        '<header class="top">'
        '<div class="hgroup">'
        '<h1>Library</h1>'
        f'<p class="sub">{total} cand{"s" if total != 1 else ""} · {len(groups)} benchmarks · sorted within each by direct ÷ beam (most-dramatic first)</p>'
        '</div>'
        f'<nav class="nav">{nav_chips}</nav>'
        '</header>'
        '<main>'
        f'{sections}'
        '</main>'
        '<footer><p>Each card: <span class="legend"><span class="d">d</span> direct rollout</span> · <span class="legend"><span class="b">b</span> beam-composed rollout</span> · ratio in the corner pill · ds#N is the test-set index. Click a card to add it to your picks; click again to remove.</p></footer>'
        '<aside class="tray" id="tray" aria-label="Selection tray">'
          '<header><span class="ct" id="trayCount">0 picks</span>'
            '<div class="actions">'
              '<button type="button" id="copyBtn" disabled>Copy IDs</button>'
              '<button type="button" id="clearBtn" disabled>Clear</button>'
            '</div>'
          '</header>'
          '<ul id="trayList"></ul>'
        '</aside>'
        f'<script>{JS}</script>'
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f: f.write(html)
    print(f"wrote {args.out}  ({os.path.getsize(args.out)/1024/1024:.1f} MB)")


CSS = """
:root {
  --paper:   #FAF8F2;
  --paper-2: #F2EFE6;
  --ink:     #1A1714;
  --rule:    #D9D3C5;
  --muted:   #7A736B;
  --accent:  #1B3A5E;
  --warn:    #A24216;
  --good:    #3F6F58;
}
* { box-sizing: border-box; }
html { background: var(--paper); color: var(--ink); }
body {
  font-family: Charter, 'Iowan Old Style', 'Source Serif Pro', Georgia, serif;
  font-size: 14.5px; line-height: 1.5;
  max-width: 1640px; margin: 0 auto; padding: 0 24px 96px;
}
.mono, .num, .errs, .params, .id, .impr, .legend, .nav .s, .nav .n {
  font-family: ui-monospace, 'SF Mono', Menlo, Consolas, 'Liberation Mono', monospace;
  font-variant-numeric: tabular-nums;
}
h1, h2 { font-weight: 600; margin: 0; letter-spacing: -0.01em; text-wrap: balance; }
p { margin: 0; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
a:focus-visible { outline: 2px solid var(--accent); outline-offset: 3px; }

/* ---- Top header + sticky nav -------------------------------------------- */
header.top {
  position: sticky; top: 0; z-index: 10;
  background: var(--paper);
  border-bottom: 1px solid var(--rule);
  padding: 18px 0 0;
  margin-bottom: 24px;
}
header.top .hgroup { display: flex; align-items: baseline; gap: 14px; padding-bottom: 8px; }
header.top h1 { font-size: 22px; }
header.top .sub { color: var(--muted); font-size: 12.5px; }
nav.nav { display: flex; flex-wrap: wrap; gap: 4px; padding: 4px 0 10px; }
nav.nav .chip {
  display: inline-flex; align-items: baseline; gap: 6px;
  padding: 3px 9px 4px;
  background: var(--paper-2);
  border: 1px solid var(--rule);
  border-radius: 999px;
  color: var(--ink);
  font-size: 11.5px;
}
nav.nav .chip:hover { background: #fff; border-color: #BCB2A0; text-decoration: none; }
nav.nav .s { color: var(--accent); font-weight: 600; }
nav.nav .t { font-family: Charter, Georgia, serif; }
nav.nav .n { color: var(--muted); font-size: 10.5px; }

/* ---- Sections ----------------------------------------------------------- */
section.bench { margin: 40px 0 8px; scroll-margin-top: 100px; }
.bench-head { margin-bottom: 14px; }
.label-row {
  display: flex; align-items: baseline; gap: 14px;
  border-bottom: 1px solid var(--rule);
  padding-bottom: 8px;
}
.table-tag {
  display: inline-block;
  background: var(--accent); color: var(--paper);
  font-size: 10px; font-weight: 600; letter-spacing: 0.06em;
  text-transform: uppercase;
  padding: 3px 7px 2px; border-radius: 2px;
  font-family: ui-monospace, monospace;
}
h2 { font-size: 18px; flex: 1; }
h2 .blurb { color: var(--muted); font-weight: 400; font-style: italic; font-size: 14.5px; }
.count {
  color: var(--accent); font-size: 11px;
  font-family: ui-monospace, monospace; font-weight: 600;
}

/* ---- Card grid ---------------------------------------------------------- */
.cards {
  display: grid; gap: 14px;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
}
figure.card {
  margin: 0;
  background: #fff;
  border: 1px solid var(--rule);
  border-radius: 3px;
  overflow: hidden;
  display: flex; flex-direction: column;
  cursor: pointer;
  transition: transform 80ms ease, box-shadow 80ms ease, border-color 80ms ease;
}
figure.card:hover { border-color: #B7AC97; }
figure.card:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
figure.card[aria-pressed="true"] {
  border-color: var(--accent);
  box-shadow: 0 0 0 1px var(--accent);
}
figure.card[aria-pressed="true"] .thumb-wrap::after {
  content: ""; position: absolute; inset: 0;
  background: rgba(27, 58, 94, 0.10); pointer-events: none;
}
figure.card .check {
  position: absolute; bottom: 6px; right: 6px;
  width: 22px; height: 22px;
  background: var(--accent); color: #fff;
  font-weight: 700; font-size: 14px;
  display: flex; align-items: center; justify-content: center;
  border-radius: 50%;
  opacity: 0; transform: scale(0.7);
  transition: opacity 80ms ease, transform 80ms ease;
  font-family: ui-monospace, monospace;
}
figure.card[aria-pressed="true"] .check { opacity: 1; transform: scale(1); }
.thumb-wrap { position: relative; line-height: 0; }
.thumb-wrap img {
  width: 100%; height: auto; display: block;
}
.thumb-wrap .num {
  position: absolute; top: 6px; left: 6px;
  background: rgba(255, 252, 247, 0.94);
  color: var(--accent); font-weight: 700;
  font-size: 15px; padding: 2px 7px 1px;
  border: 1px solid var(--rule); border-radius: 2px;
  letter-spacing: 0.02em;
}
.thumb-wrap .impr {
  position: absolute; top: 6px; right: 6px;
  background: rgba(63, 111, 88, 0.96); color: #fff;
  font-size: 11px; font-weight: 600;
  padding: 2px 6px 1px; border-radius: 2px;
}

figcaption {
  padding: 8px 10px 10px;
  display: flex; flex-direction: column; gap: 4px;
  font-size: 11.5px;
}
.params { display: flex; flex-wrap: wrap; gap: 6px 10px; font-size: 11px; }
.params .k { color: var(--muted); margin-right: 3px; }
.params .v { color: var(--ink); }
.meta {
  display: flex; justify-content: space-between; align-items: center;
  gap: 12px; padding-top: 4px;
  border-top: 1px dotted var(--rule);
}
.errs { display: inline-flex; gap: 10px; }
.errs .d { color: var(--warn); font-weight: 600; }
.errs .b { color: var(--good); font-weight: 600; }
.id { color: var(--ink); font-weight: 600; font-size: 11px; }
.id .dim { color: var(--muted); font-weight: 400; margin-left: 4px; }

/* ---- Footer ------------------------------------------------------------- */
footer { margin-top: 48px; padding-top: 16px; border-top: 1px solid var(--rule); color: var(--muted); font-size: 12px; }
footer .legend { display: inline-flex; gap: 4px; align-items: baseline; margin-right: 8px; }
footer .legend .d { color: var(--warn); font-weight: 700; font-family: ui-monospace, monospace; }
footer .legend .b { color: var(--good); font-weight: 700; font-family: ui-monospace, monospace; }
.mono { font-family: ui-monospace, monospace; }

/* ---- Selection tray ----------------------------------------------------- */
aside.tray {
  position: fixed; right: 18px; bottom: 18px;
  width: 280px; max-height: 60vh;
  background: var(--paper);
  border: 1px solid var(--accent);
  border-radius: 6px;
  box-shadow: 0 12px 28px rgba(26, 23, 20, 0.18);
  display: flex; flex-direction: column;
  z-index: 20;
  font-size: 12px;
  transform: translateY(8px); opacity: 0.96;
  transition: transform 120ms ease, opacity 120ms ease;
}
aside.tray.has-picks { transform: translateY(0); opacity: 1; }
aside.tray header {
  display: flex; align-items: center; justify-content: space-between;
  gap: 8px;
  padding: 8px 10px;
  background: var(--accent); color: var(--paper);
  border-radius: 5px 5px 0 0;
}
aside.tray .ct {
  font-family: ui-monospace, monospace;
  font-variant-numeric: tabular-nums;
  font-weight: 600;
}
aside.tray .actions { display: flex; gap: 6px; }
aside.tray button {
  background: rgba(255, 255, 255, 0.12);
  border: 1px solid rgba(255, 255, 255, 0.35);
  color: var(--paper);
  font: inherit; font-size: 11px;
  padding: 3px 8px;
  border-radius: 3px;
  cursor: pointer;
  font-family: ui-monospace, monospace;
}
aside.tray button:hover:not(:disabled) { background: rgba(255, 255, 255, 0.22); }
aside.tray button:disabled { opacity: 0.45; cursor: default; }
aside.tray button.flash { background: var(--good); border-color: var(--good); }
aside.tray ul {
  list-style: none; margin: 0; padding: 6px 0;
  overflow-y: auto;
  font-family: ui-monospace, monospace;
}
aside.tray li {
  display: flex; justify-content: space-between; align-items: center;
  padding: 3px 10px;
  font-size: 11.5px;
}
aside.tray li:hover { background: var(--paper-2); }
aside.tray li .pid { color: var(--accent); font-weight: 600; }
aside.tray li .x {
  color: var(--muted); cursor: pointer; font-weight: 700;
  padding: 0 4px;
}
aside.tray li .x:hover { color: var(--warn); }
aside.tray .empty {
  padding: 12px 14px; color: var(--muted);
  font-family: Charter, Georgia, serif; font-size: 12px; font-style: italic;
}

@media (max-width: 720px) {
  header.top { position: relative; }
  .label-row { flex-wrap: wrap; }
  aside.tray { width: calc(100% - 24px); right: 12px; bottom: 12px; }
}
"""

JS = r"""
(function () {
  var STORAGE_KEY = 'disco-library-picks-v1';
  var picks = new Map();              // pickId -> {cand, ds, exp}
  try {
    var raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      JSON.parse(raw).forEach(function (p) { picks.set(p.id, p); });
    }
  } catch (e) {}

  var tray      = document.getElementById('tray');
  var trayList  = document.getElementById('trayList');
  var trayCount = document.getElementById('trayCount');
  var copyBtn   = document.getElementById('copyBtn');
  var clearBtn  = document.getElementById('clearBtn');

  function persist() {
    try {
      var arr = [];
      picks.forEach(function (v) { arr.push(v); });
      localStorage.setItem(STORAGE_KEY, JSON.stringify(arr));
    } catch (e) {}
  }

  function paintCards() {
    document.querySelectorAll('figure.card').forEach(function (c) {
      var pid = c.getAttribute('data-pick-id');
      c.setAttribute('aria-pressed', picks.has(pid) ? 'true' : 'false');
    });
  }

  function renderTray() {
    var n = picks.size;
    trayCount.textContent = n + ' pick' + (n === 1 ? '' : 's');
    copyBtn.disabled = clearBtn.disabled = (n === 0);
    tray.classList.toggle('has-picks', n > 0);
    trayList.innerHTML = '';
    if (n === 0) {
      var li = document.createElement('li');
      li.className = 'empty';
      li.textContent = 'Click a thumbnail to add it.';
      trayList.appendChild(li);
      return;
    }
    var arr = []; picks.forEach(function (v) { arr.push(v); });
    arr.sort(function (a, b) { return a.id.localeCompare(b.id); });
    arr.forEach(function (p) {
      var li = document.createElement('li');
      li.innerHTML = '<span><span class="pid">' + p.id + '</span></span>' +
                     '<span class="x" data-pid="' + p.id + '" title="remove">×</span>';
      trayList.appendChild(li);
    });
  }

  function toggle(card) {
    var pid = card.getAttribute('data-pick-id');
    if (picks.has(pid)) {
      picks.delete(pid);
    } else {
      picks.set(pid, {
        id:  pid,
        cand: card.getAttribute('data-cand'),
        ds:   card.getAttribute('data-ds'),
        exp:  card.closest('section.bench').id
      });
    }
    paintCards();
    renderTray();
    persist();
  }

  document.querySelectorAll('figure.card').forEach(function (c) {
    c.addEventListener('click', function (e) {
      if (e.target.tagName === 'A') return;
      toggle(c);
    });
    c.addEventListener('keydown', function (e) {
      if (e.key === ' ' || e.key === 'Enter') { e.preventDefault(); toggle(c); }
    });
  });

  trayList.addEventListener('click', function (e) {
    var x = e.target.closest('.x');
    if (!x) return;
    var pid = x.getAttribute('data-pid');
    picks.delete(pid);
    paintCards();
    renderTray();
    persist();
  });

  copyBtn.addEventListener('click', function () {
    var arr = []; picks.forEach(function (v) { arr.push(v.id); });
    arr.sort();
    var text = arr.join(', ');
    function done() {
      copyBtn.classList.add('flash');
      var orig = copyBtn.textContent;
      copyBtn.textContent = 'Copied ✓';
      setTimeout(function () {
        copyBtn.classList.remove('flash');
        copyBtn.textContent = orig;
      }, 1100);
    }
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(done, function () { fallback(text); done(); });
    } else {
      fallback(text); done();
    }
  });

  function fallback(text) {
    var ta = document.createElement('textarea');
    ta.value = text;
    ta.style.position = 'fixed'; ta.style.left = '-9999px';
    document.body.appendChild(ta);
    ta.select();
    try { document.execCommand('copy'); } catch (e) {}
    document.body.removeChild(ta);
  }

  clearBtn.addEventListener('click', function () {
    if (picks.size === 0) return;
    if (!confirm('Clear all ' + picks.size + ' picks?')) return;
    picks.clear();
    paintCards();
    renderTray();
    persist();
  });

  paintCards();
  renderTray();
})();
"""


if __name__ == "__main__":
    main()

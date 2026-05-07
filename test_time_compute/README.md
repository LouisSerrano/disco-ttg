# Test-time compute

Test-time inference and evaluation. Loads a trained DISCO checkpoint and runs operator-selection methods on held-out test sets.

## Layout

```
test_time_compute/
├── ttc_methods.py / ttc_utils.py / ttc_methods_flops.py   # core utilities (top level for easy import)
├── test_generic.py                                         # ★ HF-friendly entry point
├── equations/         # per-equation evaluation scripts
└── analysis/          # sweeps, scaling-law, plotting, perturbation evals
```

## Core modules (top level)

| File | Purpose |
|---|---|
| `ttc_utils.py` | Dataset/checkpoint loading, metrics, result saving |
| `ttc_methods.py` | Operator-selection methods: greedy, random, beam search, gradient-based |
| `ttc_methods_flops.py` | FLOP-counting variant of `ttc_methods.py` (uses `fvcore`) |
| `test_generic.py` | Generic entry point (operators from codebook OR encoder) |

## `equations/` — per-equation evaluation

| Script | Equation |
|---|---|
| `test_advection_diffusion.py` | Advection-diffusion |
| `test_advection_diffusion_flops.py` | Advection-diffusion (with FLOP counting) |
| `test_combined_equation.py` | Combined equation |
| `test_reaction_diffusion.py` | Reaction-diffusion (Gray-Scott) |
| `test_navier_stokes.py` | Navier-Stokes / Euler 2D |
| `test_navier_stokes_targeted.py` | Navier-Stokes with targeted perturbations |
| `test_template.py` | Template for adding a new equation |

## `analysis/` — sweeps, scaling laws, plotting

| Script | Purpose |
|---|---|
| `sweep_ad_fitting_window.py` | Sweep over fitting window L for adv-diff (rebuttal) |
| `sweep_ad_subsampling.py` | Sweep over subsampling rate for adv-diff (rebuttal) |
| `eval_perturbation.py` | Evaluation under input perturbations (rebuttal) |
| `generate_perturbation_data.py` | Generate perturbed test trajectories |
| `scaling_law_advection_diffusion.py` | Compute scaling-law data points |
| `scaling_law_advection_diffusion_analysis.py` | Plot/analyze scaling-law results |
| `advection_diffusion_heatmap.py` | Generate heatmap visualizations |
| `plot_beam_search_predictions.py`, `visualize_beam_predictions.py` | Visualize beam-search outputs |

## Usage

```bash
# Direct invocation
python test_time_compute/equations/test_advection_diffusion.py \
    --model_path /path/to/best-checkpoint.ckpt \
    --experiment E_AD_v \
    --methods beam --beam_width 3

# Via SLURM
sbatch bash/advection-diffusion/test/beam_test.sh
```

Imports of the form `from test_time_compute.ttc_utils import …` work because
`pip install -e .` registers `test_time_compute` as a Python package, and its
`__init__.py` ensures the repo root is on `sys.path[0]`.

## Hardcoded paths

`--model_path` and `--data_path` arguments in `bash/*/test/*.sh` point to absolute checkpoint paths under `/mnt/home/lserrano/`. See `bash/README.md` for the full list to swap.

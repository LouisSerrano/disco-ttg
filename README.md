**Test-time Generalization for Physics through Neural Operator Splitting**
Louis Serrano, Jiequn Han, Edouard Oyallon, Shirley Ho, Rudy Morel.
*ICML 2026.*

[![arXiv](https://img.shields.io/badge/arXiv-2602.00884-b31b1b.svg)](https://arxiv.org/abs/2602.00884)
[![Project page](https://img.shields.io/badge/Project-page-blue.svg)](https://louisserrano.github.io/neural-operator-splitting/)
[![Poster](https://img.shields.io/badge/ICML-poster-brightgreen.svg)](assets/icml-poster.pdf)

- **Paper (arXiv):** https://arxiv.org/abs/2602.00884
- **Project page:** https://louisserrano.github.io/neural-operator-splitting/
- **ICML 2026 poster:** [`assets/icml-poster.pdf`](assets/icml-poster.pdf)

## Project structure

```
neural-operator-splitting/
├── src/                      # Core library
│   ├── operators/            # DISCO + variants (disco.py, disco_vae.py, ...)
│   ├── modules/              # Neural network modules (UNet, MAE, tokenizer, VQ)
│   ├── utils/                # Dataset generation, metrics, plotting
│   └── torchdiffeq/          # Custom ODE solvers
├── train/                    # Training scripts (Hydra + PyTorch Lightning)
│   ├── train.py / train_combined*.py / train_rd*.py / train_euler_diffusion_aggregate.py
│   ├── train_expert.py / train_operator.py
│   ├── train_generic.py      # ★ HF-friendly entry (codebook + envs, no in-context)
│   └── variants/             # ablations + alternative architectures
├── test_time_compute/        # Inference & test-time methods
│   ├── ttc_methods.py / ttc_utils.py / ttc_methods_flops.py
│   ├── test_generic.py       # ★ HF-friendly entry
│   ├── equations/            # per-equation evaluation
│   └── analysis/             # sweeps + scaling laws + plotting
├── baselines/                # baseline models
│   ├── GEPS/
│   ├── MPP/
│   └── ZEBRA/
├── paper/                    # paper-related explorations
│   ├── neural-operator-splitting/   # classical & neural splitting code
│   ├── operator-splitting/          # classical splitting analysis
│   ├── notebooks/                   # exploration notebooks
│   └── tests/                       # paper experiments / sanity tests
├── configs/                  # Hydra YAML configs
├── bash/                     # SLURM launch scripts
├── scripts/                  # data conversion + HF upload helpers
├── assets/                   # ICML 2026 poster + figures
├── setup.py
└── LICENSE
```

## Setup

```bash
git clone https://github.com/LouisSerrano/neural-operator-splitting.git
cd neural-operator-splitting
pip install -e .
```

## Per-folder docs

- [`train/README.md`](train/README.md) — training scripts ↔ configs mapping
- [`test_time_compute/README.md`](test_time_compute/README.md) — TTC methods and per-equation evaluation
- [`configs/README.md`](configs/README.md) — config-to-script mapping
- [`bash/README.md`](bash/README.md) — SLURM launch scripts
- [`paper/neural-operator-splitting/README.md`](paper/neural-operator-splitting/README.md) — classical/neural splitting code

## Configuring data + checkpoint paths

All paths in configs and bash scripts are parameterized via environment variables with sensible defaults relative to the repo root. Set them once in your shell (or `.envrc` / sbatch wrapper):

| Variable | Default | Used for |
|---|---|---|
| `DISCO_DATA` | `./datasets` | base dataset directory |
| `DISCO_OUTPUTS` | `./outputs` | training output / checkpoints |
| `DISCO_RESULTS` | `./results` | evaluation output |
| `DISCO_NS_DATA` | `./datasets/euler_ns_short` | Navier-Stokes / Euler HDF5 |
| `DISCO_RD_DATA` | `./datasets/gray-scott` | Gray-Scott reaction-diffusion HDF5 |
| `DISCO_MPP_DATA` | `./datasets/mp-neural` | MP-Neural-PDE-Solvers data (combined-equation source) |
| `DISCO_CKPT_DIR` | `./outputs` | alternative checkpoint store |

Hydra configs use `${oc.env:VAR,default}` and bash scripts use `${VAR:-default}`. See [`configs/README.md`](configs/README.md) and [`bash/README.md`](bash/README.md) for details.

## Checkpoints & Datasets

**Pretrained checkpoints**: [`sogeeking/disco-models`](https://huggingface.co/sogeeking/disco-models)

| Equation | Path in repo |
|---|---|
| Advection-diffusion | `advection-diffusion/last.ckpt` |
| Combined equation | `combined-equation/last.ckpt` |
| Reaction-diffusion (Gray-Scott) | `reaction-diffusion/last.ckpt` |
| Navier-Stokes / Euler | `navier-stokes/best-checkpoint.ckpt` |

```python
from huggingface_hub import hf_hub_download
ckpt = hf_hub_download(repo_id="sogeeking/disco-models",
                       filename="advection-diffusion/last.ckpt")

from train.train_generic import DISCOLitModule
model = DISCOLitModule.load_from_checkpoint(ckpt)
```

**Datasets** (HuggingFace Hub, one repo per equation):
- [`sogeeking/disco-ad`](https://huggingface.co/datasets/sogeeking/disco-ad) — advection-diffusion (deterministic seed=124 test set, 512 trajectories; the train pipeline still generates trajectories on the fly)
- [`sogeeking/disco-combined`](https://huggingface.co/datasets/sogeeking/disco-combined) — combined equation (Burgers / heat / dispersion test+val splits)
- [`sogeeking/disco-rd`](https://huggingface.co/datasets/sogeeking/disco-rd) — reaction-diffusion (Gray-Scott; test + train)
- [`sogeeking/disco-ns`](https://huggingface.co/datasets/sogeeking/disco-ns) — Navier-Stokes / Euler 2D (test only; NS is a test-time-generalization target in the paper)

Test files use a uniform layout consumed by `train/train_generic.py` and
`test_time_compute/test_generic.py`:

```
trajectories: (N, T, C, *spatial)   float32
env_id:       (N,)                  int64
env_params/*: physical parameters per environment (metadata)
```

The `train/train_generic.py` and `test_time_compute/test_generic.py` entry points
take a `--hf_repo_id` flag and download HDF5 files on first use.

## Training

Training uses [Hydra](https://hydra.cc/) for configuration and [PyTorch Lightning](https://lightning.ai/) for the training loop.

```bash
# Advection-diffusion (main experiment)
python train/train_combined.py

# With custom config overrides
python train/train_combined.py training.lr=1e-4 training.max_steps=200000
```

See `bash/` for SLURM submission scripts used in the paper.

## Evaluation

```bash
# Per-equation evaluation
python test_time_compute/equations/test_advection_diffusion.py
python test_time_compute/equations/test_navier_stokes.py

# Generic (HDF5 from local or HuggingFace; codebook OR encoder dictionary)
python test_time_compute/test_generic.py \
    --model_path /path/to/best-checkpoint.ckpt \
    --train_files /path/to/train.h5 \
    --test_files  /path/to/test.h5 \
    --operator_source codebook \
    --method beam --beam_width 3
```

## Baselines

`baselines/{GEPS, MPP, ZEBRA}/` — each baseline lives in its own subfolder with its own configs. Launch scripts at `bash/*/baselines/`.

## Citation

```bibtex
@inproceedings{serrano2026disco,
  title     = {Test-time Generalization for Physics through Neural Operator Splitting},
  author    = {Serrano, Louis and Han, Jiequn and Oyallon, Edouard and Ho, Shirley and Morel, Rudy},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```

# DISCO: Test-Time Generalization via Neural Operator Splitting

Official code for the ICML 2026 paper *"Test-Time Generalization via Neural Operator Splitting"*.

## Project Structure

```
disco-ball/
├── src/                          # Core library
│   ├── operators/                # DISCO operator variants (disco.py, disco_vae.py, ...)
│   ├── modules/                  # Neural network modules (UNet, MAE, tokenizer, VQ)
│   ├── utils/                    # Dataset generation, metrics, plotting
│   └── torchdiffeq/             # Custom ODE solvers
├── train/                        # Training scripts (Hydra + PyTorch Lightning)
├── test-time-compute/            # Inference & test-time methods
│   ├── ttc_methods.py            # Core test-time compute methods
│   ├── ttc_utils.py              # Utility functions
│   ├── test_advection_diffusion.py
│   ├── test_combined_equation.py
│   ├── test_reaction_diffusion.py
│   └── test_navier_stokes.py
├── neural-operator-splitting/    # Classical & neural operator splitting baselines
├── GEPS/                         # GEPS baseline
├── MPP/                          # MPP baseline
├── ZEBRA/                        # ZEBRA baseline
├── bash/                         # SLURM launch scripts
│   ├── advection-diffusion/
│   ├── combined-equation/
│   ├── euler/
│   ├── rd/
│   └── rebuttal/
├── configs/                      # Hydra configuration files
└── setup.py
```

## Setup

```bash
git clone https://github.com/LouisSerrano/disco-ball.git
cd disco-ball
pip install -e .
```

## Per-folder docs

- [`train/README.md`](train/README.md) — training scripts ↔ configs mapping
- [`test-time-compute/README.md`](test-time-compute/README.md) — TTC methods and per-equation evaluation
- [`configs/README.md`](configs/README.md) — config-to-script mapping
- [`bash/README.md`](bash/README.md) — SLURM launch scripts
- [`neural-operator-splitting/README.md`](neural-operator-splitting/README.md) — classical/neural splitting baselines

## Hardcoded paths

The configs and bash scripts contain absolute paths from the original development environment (`/mnt/home/lserrano/...`). Before running on a new machine, find and replace these paths:

```bash
grep -rn "/mnt/home/lserrano" configs/ bash/
```

See each subfolder's README for the specific paths to swap.

## Checkpoints & Datasets

<!-- TODO: Add links to pretrained checkpoints and datasets -->

**Checkpoints** will be made available at: `[link]`

**Datasets:**
- Advection-diffusion: generated on-the-fly (see `src/utils/advection_diffusion.py`)
- Navier-Stokes (Euler): `[link]`

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
# Test-time compute evaluation on advection-diffusion
python test-time-compute/test_advection_diffusion.py

# Navier-Stokes evaluation
python test-time-compute/test_navier_stokes.py
```

## Baselines

Baseline models (GEPS, MPP, ZEBRA) are in their respective directories with their own configs and training scripts. See `bash/*/baselines/` for launch scripts.

## Citation

```bibtex
@inproceedings{serrano2026disco,
  title={Test-Time Generalization via Neural Operator Splitting},
  author={Serrano, Louis and others},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026}
}
```

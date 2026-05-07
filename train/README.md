# Training scripts

Hydra + PyTorch Lightning training scripts. Each script is paired with a config in `../configs/`.

## Layout

```
train/
├── train.py                              # ← paper main: AD (synthetic on-the-fly)
├── train_combined.py                     # ← paper main: combined equation
├── train_combined_aggregate.py
├── train_rd.py / train_rd_aggregate.py   # ← paper main: reaction-diffusion
├── train_euler_diffusion_aggregate.py    # ← paper main: Euler/NS + diffusion
├── train_expert.py / train_operator.py   # alternative-architecture baselines
├── train_generic.py                      # ★ HF-friendly entry point (codebook + envs, no in-context)
└── variants/                             # ablations + alternative variants
```

## Main scripts (top level)

| Script | Model | Equation | Config |
|---|---|---|---|
| `train.py` | DISCOHouse | Advection-diffusion | `config.yaml` |
| `train_combined.py` | DISCOHouse | Combined equation (HDF5) | `config_hdf5.yaml` |
| `train_combined_aggregate.py` | DISCOHouse | Combined equation, aggregated | `config_hdf5.yaml` |
| `train_rd.py` | DISCOHouse | Reaction-diffusion | `config_rd.yaml` |
| `train_rd_aggregate.py` | DISCOHouse | Reaction-diffusion, aggregated | `config_rd.yaml` |
| `train_euler_diffusion_aggregate.py` | DISCOHouse | Euler/NS + diffusion | `config_euler.yaml` |
| `train_expert.py` | DISCOExpert | Single-equation specialist | `expert.yaml` |
| `train_operator.py` | DiscoOperator | Operator-only baseline | `operator.yaml` |
| `train_generic.py` | DISCOHouse | **Any HDF5 dataset** (local or HF Hub) | `config_generic.yaml` |

## `variants/` — ablations and alternative variants

| Script | Purpose |
|---|---|
| `train_combined_ablations.py`, `train_ablations.py`, `train_ablations_unet.py` | Architectural ablations |
| `train_combined_curriculum.py` | Curriculum-learning variant |
| `train_combined_coda.py` | CoDA variant |
| `train_combined_vae.py`, `train_combined_vqvae.py` | VAE / VQ-VAE variants |

## Usage

```bash
# Default config
python train/train_combined.py

# Override
python train/train_combined.py training.lr=1e-4 training.max_steps=200000

# Via SLURM
sbatch bash/combined-equation/train.sh
```

Hardcoded paths live in the configs (not the scripts). See `../configs/README.md`.

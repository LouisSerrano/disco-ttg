# Configs

Hydra YAML configs paired with the training scripts in `../train/`.

## Mapping

| Config | Used by | Equation |
|---|---|---|
| `config.yaml` | `train.py` | Advection-diffusion |
| `config_hdf5.yaml` | `train_combined.py`, `train_combined_aggregate.py` | Combined equation (HDF5) |
| `config_rd.yaml` | `train_rd.py` | Reaction-diffusion |
| `config_euler.yaml` | `train_euler_diffusion_aggregate.py` | Euler/NS + diffusion |
| `expert.yaml` | `train_expert.py` | Single-equation expert |
| `operator.yaml` | `train_operator.py` | Operator-only baseline |
| `ablations.yaml`, `ablations_unet.yaml` | `train_ablations*.py` | Ablations |
| `config_vae_hdf5.yaml` | `train_combined_vae.py` | VAE variant |
| `config_vqvae_hdf5.yaml` | `train_combined_vqvae.py` | VQ-VAE variant |
| `plot.yaml` | Notebooks / plotting scripts | — |

## Hardcoded paths to update

Paths in these configs need to be swapped for your environment:

| Field | Default value |
|---|---|
| `data.train_path` / `val_path` / `test_path` | `/mnt/home/lserrano/disco-ball/datasets/...` |
| `data.train_hdf5_files` (list) | `/mnt/home/lserrano/disco-ball/datasets/combined_equation/*.h5` |
| `data.file_dir` (Euler config) | `/mnt/home/lserrano/ceph/data/euler_ns_short/` |
| `data.output_dir` | `/mnt/home/lserrano/disco-ball/outputs/` |
| `data.results_dir` | `/mnt/home/lserrano/disco-ball/results/` |

To find every occurrence: `grep -rn "/mnt/home/lserrano" configs/`.

## Override at the command line

Hydra lets you override any field without editing the YAML:

```bash
python train/train_combined.py \
    data.output_dir=/scratch/$USER/disco/outputs \
    training.lr=1e-4
```

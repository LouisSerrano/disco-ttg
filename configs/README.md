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
| `config_generic.yaml` | `train_generic.py` | Generic HF-friendly entry point |
| `plot.yaml` | Notebooks / plotting scripts | — |

## Path configuration via env vars

Configs use OmegaConf's environment-variable interpolation for all data and output paths. Each variable falls back to a path relative to the repo root, so things "just work" if you put your data under `./datasets/` and run from the repo root. Override any of them as needed.

| Hydra interpolation | Env var | Default |
|---|---|---|
| `${oc.env:DISCO_DATA,datasets}/...` | `DISCO_DATA` | `datasets` |
| `${oc.env:DISCO_OUTPUTS,outputs}/...` | `DISCO_OUTPUTS` | `outputs` |
| `${oc.env:DISCO_RESULTS,results}/...` | `DISCO_RESULTS` | `results` |
| `${oc.env:DISCO_NS_DATA,datasets/euler_ns_short}/...` | `DISCO_NS_DATA` | `datasets/euler_ns_short` |
| `${oc.env:DISCO_RD_DATA,datasets/gray-scott}/...` | `DISCO_RD_DATA` | `datasets/gray-scott` |

To find every interpolation: `grep -rn "oc.env" configs/`.

## Override at the command line

Hydra lets you override any field without editing the YAML or env vars:

```bash
python train/train_combined.py \
    data.output_dir=/scratch/$USER/disco/outputs \
    training.lr=1e-4
```

## Override via env vars

```bash
export DISCO_DATA=/data/disco
export DISCO_OUTPUTS=/scratch/$USER/disco/outputs
python train/train_combined.py
```

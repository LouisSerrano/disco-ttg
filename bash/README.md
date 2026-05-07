# Launch scripts

SLURM scripts that wrap `python train/...` and `python test_time_compute/...` invocations, organized by experiment.

## Layout

| Subdir | Purpose |
|---|---|
| `advection-diffusion/` | 1D advection-diffusion experiments (vanilla, expert, ablations, with/without context) |
| `combined-equation/` | Combined-equation experiments (Burgers, heat, dispersion, Euler) |
| `euler/` | 2D Euler / Navier-Stokes experiments and baselines |
| `rd/` | Reaction-diffusion (Gray-Scott) experiments and baselines |
| `rebuttal/` | ICML 2026 rebuttal experiments (perturbation sweeps, fitting-window sweeps, mixed-physics, no-context) |
| `smoke/` | Tiny smoke tests for the train/test pipelines |
| `*/baselines/` | GEPS, MPP, ZEBRA, tokenizer launches per equation |

## Path configuration via env vars

Bash scripts read paths from environment variables with sensible defaults relative to the repo root. Set them once in your shell (or sbatch wrapper) before submitting:

| Variable | Default | What it points to |
|---|---|---|
| `DISCO_DATA` | `./datasets` | base dataset directory |
| `DISCO_OUTPUTS` | `./outputs` | DISCO training output / checkpoints |
| `DISCO_RESULTS` | `./results` | evaluation output |
| `DISCO_CKPT_DIR` | `./outputs` | alternative checkpoint store (e.g. ceph) |
| `DISCO_NS_DATA` | `./datasets/euler_ns_short` | Navier-Stokes / Euler HDF5 |
| `DISCO_RD_DATA` | `./datasets/gray-scott` | Gray-Scott reaction-diffusion HDF5 |
| `DISCO_MPP_DATA` | `./datasets/mp-neural` | MP-Neural-PDE-Solvers data (combined-equation original source) |
| `DISCO_LPSDA_DATA` | `./datasets/lpsda` | LPSDA Euler data |
| `GEPS_CKPT_DIR` | `./outputs/geps` | GEPS baseline checkpoints |
| `MPP_CKPT_DIR` | `./outputs/mpp` | MPP baseline checkpoints |
| `ZEBRA_CKPT_DIR` | `./outputs/zebra` | ZEBRA baseline checkpoints |
| `ZEBRA_OUTPUTS` | `./outputs` | alternative ZEBRA output dir |

Hydra YAML configs use the equivalent `${oc.env:VAR,default}` interpolation. See [`configs/README.md`](../configs/README.md) for that mapping.

## Submission

```bash
sbatch bash/advection-diffusion/train.sh
```

If your data/checkpoints live somewhere other than the defaults, export the relevant env var first:

```bash
export DISCO_DATA=/path/to/datasets
export DISCO_OUTPUTS=/path/to/outputs
sbatch bash/advection-diffusion/train.sh
```

See the SBATCH header in each file for resource requests.

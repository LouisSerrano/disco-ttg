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
| `*/baselines/` | GEPS, MPP, ZEBRA, tokenizer launches per equation |

## Hardcoded paths to update

These scripts contain absolute paths from the original development environment. Search-and-replace the relevant prefix for your setup:

| Path prefix | What it points to |
|---|---|
| `/mnt/home/lserrano/disco-ball/datasets/combined_equation/` | HDF5 datasets (Burgers/heat/dispersion/Euler) |
| `/mnt/home/lserrano/disco-ball/outputs/` | DISCO training output dir (checkpoints) |
| `/mnt/home/lserrano/ceph/disco/outputs/` | Alternate (ceph) checkpoint location |
| `/mnt/home/lserrano/ceph/geps/` | GEPS baseline checkpoints |
| `/mnt/home/lserrano/ceph/zebra/` | ZEBRA tokenizer + LLaMA checkpoints |
| `/mnt/home/lserrano/ceph/data/euler_ns_short/` | Euler/NS HDF5 data |
| `/mnt/home/lserrano/gray-scott-python/data/` | Gray-Scott (reaction-diffusion) data |
| `/mnt/home/lserrano/MP-Neural-PDE-Solvers/data/` | MP-Neural-PDE-Solvers data (mostly in commented-out lines) |

To find every occurrence: `grep -rn "/mnt/home/lserrano" bash/`.

## Submission

All scripts are SLURM `sbatch` jobs. Submit with:

```bash
sbatch bash/advection-diffusion/train.sh
```

See the SBATCH header in each file for resource requests.

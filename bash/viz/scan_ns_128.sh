#!/bin/bash
#SBATCH --job-name=scan-NS-128
#SBATCH --partition=gpu
#SBATCH --account=ccm
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd /mnt/home/lserrano/disco-ball

export HF_HOME="$HOME/.cache/huggingface"
export XDG_CACHE_HOME="$HOME/.cache"
export PYTHONPATH="/mnt/home/lserrano/disco-ball:${PYTHONPATH:-}"

CKPT="/mnt/home/lserrano/ceph/disco/outputs/DISCO_euler_solverrk4_adjFalse_h128_t4_steps4_initFalse_bs16_lr0.0003_hdf5_noise0_inframes16_outframes2_subx1_subt1_20260124_041037/best-checkpoint.ckpt"

/mnt/home/lserrano/venvs/disco/bin/python scripts/viz/viz_navier_stokes.py \
  --model_path "$CKPT" \
  --output_dir viz/data/scan/navier-stokes \
  --num_samples 128 \
  --scan_mode

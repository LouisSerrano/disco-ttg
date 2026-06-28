#!/bin/bash
#SBATCH --job-name=scan-GS-128
#SBATCH --partition=gpu
#SBATCH --account=ccm
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:30:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd /mnt/home/lserrano/disco-ball

export HF_HOME="$HOME/.cache/huggingface"
export XDG_CACHE_HOME="$HOME/.cache"
export PYTHONPATH="/mnt/home/lserrano/disco-ball:${PYTHONPATH:-}"

CKPT="/mnt/home/lserrano/ceph/disco/outputs/DISCO_rd_solverrk4_adjFalse_h128_t3_steps1_initTrue_bs64_lr0.0003_hdf5_noise0_inframes16_outframes2_subx1_subt1_20250916_205814/last.ckpt"

/mnt/home/lserrano/venvs/disco/bin/python scripts/viz/viz_reaction_diffusion.py \
  --model_path "$CKPT" \
  --output_dir viz/data/scan/gray-scott \
  --samples_per_param 8 \
  --max_param_combos 16 \
  --scan_mode

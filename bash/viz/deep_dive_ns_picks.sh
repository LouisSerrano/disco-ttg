#!/bin/bash
#SBATCH --job-name=deep-dive-NS
#SBATCH --partition=gpu
#SBATCH --account=ccm
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd /mnt/home/lserrano/disco-ball

export HF_HOME="$HOME/.cache/huggingface"
export XDG_CACHE_HOME="$HOME/.cache"
export PYTHONPATH="/mnt/home/lserrano/disco-ball:${PYTHONPATH:-}"

/mnt/home/lserrano/venvs/disco/bin/python scripts/viz/expand_ns_deep_dive.py \
  --cand_names cand_0096,cand_0120,cand_0037,cand_0068,cand_0055

#!/bin/bash -l
# Smoke test: advection-diffusion training, ~50 steps.
# Verifies that train/train.py + configs/config.yaml + dataset paths still work.

#SBATCH -p gpu
#SBATCH -t 00:15:00
#SBATCH -C a100
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -J disco-smoke-ad

module load python
module load cuda
module load cudnn
module load nccl

source $VENVDIR/disco/bin/activate

T=10
python3 train/train.py \
    model.max_steps=1 \
    model.use_adjoint=False \
    model.decoder_use_bias=True \
    training.max_steps=50 \
    training.in_context=True \
    data.T=$T \
    model.principled_initialization=False \
    model.theta_dim=2 \
    training.lr=5e-4 \
    data.n_output_frames=2 \
    training.noise_level=0.0001

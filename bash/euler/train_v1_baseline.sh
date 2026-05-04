#!/bin/bash -l

#SBATCH -p gpu
#SBATCH -t 48:00:00
#SBATCH -C a100
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --tasks-per-node=5
#SBATCH --cpus-per-task=4

module load python
module load cuda
module load cudnn
module load nccl

source $VENVDIR/disco/bin/activate

# Baseline: small model, more ODE steps for stability
python3 train/train_euler_diffusion_aggregate.py \
    training.batch_size=16 \
    training.codebook_prob=0.5 \
    training.warmup_steps=1000 \
    training.max_steps=100000 \
    model.theta_dim=4 \
    model.hpnn_head_hidden_dim=32 \
    model.opnn_channels=16 \
    model.max_steps=8 \
    model.default_integration_time=0.08 \
    model.use_adjoint=False \
    model.decoder_use_bias=True \
    model.principled_initialization=False \
    model.solver="dopri5" \
    model.use_adjoint=True

#!/bin/bash -l
# Smoke test: Navier-Stokes / Euler training, ~50 steps.
# Verifies that train/train_euler_diffusion_aggregate.py + configs/config_euler.yaml
# + Euler/NS dataset paths still work.

#SBATCH -p gpu
#SBATCH -t 00:20:00
#SBATCH -C a100
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -J disco-smoke-ns

module load python
module load cuda
module load cudnn
module load nccl

source $VENVDIR/disco/bin/activate

python3 train/train_euler_diffusion_aggregate.py \
    training.batch_size=4 \
    training.codebook_prob=0.5 \
    training.warmup_steps=10 \
    training.max_steps=50 \
    model.theta_dim=4 \
    model.hpnn_head_hidden_dim=32 \
    model.opnn_channels=16 \
    model.max_steps=8 \
    model.default_integration_time=0.08 \
    model.use_adjoint=True \
    model.decoder_use_bias=True \
    model.principled_initialization=False \
    model.solver="dopri5"

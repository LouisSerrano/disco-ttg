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

# Lower learning rate (1e-4) for stability
python3 train/train_euler_diffusion_aggregate.py \
    training.batch_size=32 \
    training.codebook_prob=0.5 \
    training.warmup_steps=1000 \
    training.max_steps=100000 \
    training.lr=1e-4 \
    model.theta_dim=4 \
    model.hpnn_head_hidden_dim=128 \
    model.opnn_channels=8 \
    model.max_steps=1 \
    model.default_integration_time=0.2 \
    model.use_adjoint=False \
    model.decoder_use_bias=True \
    model.principled_initialization=False

#!/bin/bash -l

#SBATCH -p gpu
#SBATCH -t 96:00:00
#SBATCH -C a100
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --tasks-per-node=5
#SBATCH --cpus-per-task=4
#SBATCH -J euler_no_cb

module load python
module load cuda
module load cudnn
module load nccl

source $VENVDIR/disco/bin/activate

unset WANDB_API_KEY
export WANDB_ENTITY=emmi-ai
export WANDB_PROJECT=neural-operator-splitting

# Rebuttal experiment: Euler/NS WITHOUT codebook (codebook_prob=0.0)
# Matches paper's Euler model config (theta_dim=4, max_steps=4, bs=16)
# but disables the codebook trick entirely
# Ablation for reviewer rJBq Q2 (training recipe ablation)
python3 train/train_euler_diffusion_aggregate.py \
    training.batch_size=16 \
    training.codebook_prob=0.0 \
    training.warmup_steps=1000 \
    training.max_steps=100000 \
    model.theta_dim=4 \
    model.hpnn_head_hidden_dim=128 \
    model.opnn_channels=8 \
    model.max_steps=4 \
    model.default_integration_time=0.2 \
    model.use_adjoint=False \
    model.decoder_use_bias=True \
    model.principled_initialization=False \
    data.output_dir=${DISCO_CKPT_DIR:-./outputs}/

#!/bin/bash -l

#SBATCH -p gpu
#SBATCH -t 24:00:00
#SBATCH -C a100
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --tasks-per-node=5
#SBATCH --cpus-per-task=4
#SBATCH -J ad_no_ctx

module load python
module load cuda
module load cudnn
module load nccl

source $VENVDIR/disco/bin/activate

unset WANDB_API_KEY
export WANDB_ENTITY=emmi-ai
export WANDB_PROJECT=neural-operator-splitting

# Rebuttal experiment: advection-diffusion WITHOUT in-context learning
# Same config as paper (train.sh) but in_context=False
# Ablation for reviewer rJBq Q2 (training recipe ablation)
python3 train/train.py \
    model.max_steps=1 \
    model.use_adjoint=False \
    model.decoder_use_bias=True \
    model.principled_initialization=False \
    model.theta_dim=2 \
    training.max_steps=300000 \
    training.in_context=False \
    training.lr=5e-4 \
    training.noise_level=0.0001 \
    data.T=10 \
    data.n_output_frames=2 \
    data.output_dir=${DISCO_CKPT_DIR:-./outputs}/

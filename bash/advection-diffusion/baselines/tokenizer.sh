#!/bin/bash -l

#SBATCH -p gpu
#SBATCH -t 24:00:00
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


python3 baselines/ZEBRA/train_vq.py \
  data.dataset_name=advection-diffusion \
  model.layers='[residual, compress_space, compress_space, compress_space, residual]' \
  model.codebook_size=256
  model.num_codebooks=4
  training.max_steps=100000

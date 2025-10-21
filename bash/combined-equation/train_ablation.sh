#!/bin/bash -l

#SBATCH -p gpu
#SBATCH -t 8:00:00
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


python3 train/train_combined.py model.max_steps=1 model.use_adjoint=False model.decoder_use_bias=True training.warmup_steps=100 training.max_steps=20000 model.principled_initialization=True model.theta_dim=3 training.lr=5e-4 data.n_output_frames=16 training.batch_size=64 

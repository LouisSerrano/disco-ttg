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


T=10
python3 train/train.py model.max_steps=1 model.use_adjoint=False model.decoder_use_bias=True training.warmup_steps=100 training.max_steps=20000 training.in_context=True data.T=$T model.principled_initialization=True model.theta_dim=2 training.lr=5e-4 data.n_output_frames=16 training.batch_size=64 

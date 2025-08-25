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


T=50
python3 train/train_ablations.py model.max_steps=1 model.use_adjoint=False model.decoder_use_bias=True training.max_steps=200000 training.in_context=True data.T=$T model.principled_initialization=True model.theta_dim=2 model.solver=euler training.lr=1e-4

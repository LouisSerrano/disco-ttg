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

python3 train_expert.py model.max_steps=10 model.use_adjoint=False training.max_steps=100000 training.sparsity_alpha=1 training.lr=5e-4 training.batch_size=64 data.n_output_frames=2 

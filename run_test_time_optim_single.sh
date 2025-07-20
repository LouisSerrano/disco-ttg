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

python3 test_time_optimization_single.py 


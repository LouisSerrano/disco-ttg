#!/bin/bash -l

#SBATCH -p gpu
#SBATCH -t 4:00:00
#SBATCH -C v100
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --tasks-per-node=5
#SBATCH --cpus-per-task=4


module load python

python3 generate_dataset.py 

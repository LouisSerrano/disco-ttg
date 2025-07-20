#!/bin/bash -l

#SBATCH -p gpu
#SBATCH -t 24:00:00
##SBATCH -C v100
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=50


module load python

python3 generate_dataset_explicit.py 

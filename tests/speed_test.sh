#!/bin/bash -l

#SBATCH -p gpu
#SBATCH -t 24:00:00
##SBATCH -C v100
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=50


module load python
module load cuda
module load cudnn

source $VENVDIR/disco/bin/activate

python3 test_vectorized_sampling.py 

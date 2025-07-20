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

python3 test.py hydra.run.dir=. test.setting="dense" test.ckpt_time=2025-06-18/22-51-43 #test.ckpt_path=outputs/2025-06-15/23-46-48/model_final.ckpt

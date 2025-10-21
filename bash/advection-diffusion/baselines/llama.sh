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


python3 ZEBRA/pretrain_llama.py data.tokenizer_path=/mnt/home/lserrano/ceph/zebra/tokenizer/advection-diffusion/efficient-night-90/last.ckpt training.batch_size=32 training.max_steps=20000

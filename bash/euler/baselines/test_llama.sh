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


#python baselines/ZEBRA/test_zebra.py --dataset_name euler-ns --model_path /mnt/home/lserrano/ceph/zebra/llama/euler-ns/earthy-fire-25/last.ckpt --batch_size 16 

python baselines/ZEBRA/test_zebra.py --dataset_name euler-ns --model_path /mnt/home/lserrano/ceph/zebra/llama/euler-ns/quiet-meadow-26/last.ckpt --batch_size 16 


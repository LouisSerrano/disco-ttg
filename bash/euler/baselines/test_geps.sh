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

#python GEPS/train_2d.py --config-name=euler_ns.yaml
python GEPS/test_navier_stokes.py --model_path /mnt/home/lserrano/ceph/geps/euler-ns/decent-dawn-11/last.ckpt --mode optimize --n_optimization_steps 100  --batch_size 4


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

python3 baselines/ZEBRA/pretrain_llama.py data.tokenizer_path=/mnt/home/lserrano/ceph/zebra/tokenizer/combined-equation/sandy-gorge-62/last.ckpt  data.dataset_name=combined-equation training.tokenize_on_the_fly=False data.n_input_frames=16 data.n_output_frames=50 data.slice_size=66

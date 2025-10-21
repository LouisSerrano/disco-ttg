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

python3 ZEBRA/pretrain_llama.py \
  data.tokenizer_path=/mnt/home/lserrano/ceph/zebra/tokenizer/rd/rural-sky-68/last.ckpt \
  data.dataset_name=gray-scott \
  model.max_length=16384 \
  training.tokenize_on_the_fly=False \
  data.n_input_frames=16 \
  data.n_output_frames=16 \
  data.slice_size=32 \
  training.batch_size=16 \
  data.train_hdf5_files='["/mnt/home/lserrano/gray-scott-python/data/feed_20params_512traj_each.hdf5","/mnt/home/lserrano/gray-scott-python/data/kill_20params_512traj_each.hdf5"]' \
  data.val_hdf5_files='["/mnt/home/lserrano/gray-scott-python/data/val_feed_20params_8traj_each.hdf5","/mnt/home/lserrano/gray-scott-python/data/val_kill_20params_8traj_each.hdf5"]' \
  data.test_hdf5_files='["/mnt/home/lserrano/gray-scott-python/data/val_feed_20params_8traj_each.hdf5","/mnt/home/lserrano/gray-scott-python/data/val_kill_20params_8traj_each.hdf5"]'

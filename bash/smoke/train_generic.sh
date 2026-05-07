#!/bin/bash -l
# Smoke test: train_generic.py for 50 steps on the converted combined HEAT data.
# Verifies the GenericHDF5Dataset → DISCOLitModule pipeline end-to-end.

#SBATCH -p gpu
#SBATCH -t 00:20:00
#SBATCH -C a100
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -J disco-smoke-generic-train

module load python
module load cuda
module load cudnn
module load nccl

source $VENVDIR/disco/bin/activate

OUT=/mnt/home/lserrano/ceph/disco/smoke_outputs
mkdir -p $OUT

python3 train/train_generic.py \
    data.train_files=["/mnt/home/lserrano/ceph/disco/datasets_generic/combined_HEAT_test.h5"] \
    data.val_files=["/mnt/home/lserrano/ceph/disco/datasets_generic/combined_HEAT_valid.h5"] \
    data.test_files=["/mnt/home/lserrano/ceph/disco/datasets_generic/combined_HEAT_test.h5"] \
    data.num_environments=32 \
    data.output_dir=$OUT \
    training.batch_size=8 \
    training.max_steps=50 \
    training.warmup_steps=5 \
    training.num_workers=2 \
    model.theta_dim=2 \
    model.ndims=[1] \
    model.patch_size=8

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

# Train only on euler dataset to understand fitting on more challenging dataset
# This script trains from scratch (no checkpoint)

python3 train/train_combined.py \
    model.max_steps=1 \
    model.use_adjoint=False \
    model.decoder_use_bias=True \
    model.principled_initialization=False \
    model.theta_dim=128 \
    training.max_steps=10000 \
    training.progressive_steps=True \
    training.project="disco-ball-euler-only" \
    training.lr=5e-4 \
    training.batch_size=64 \
    data.n_input_frames=16 \
    data.n_output_frames=16 \
    data.sub_t=1 \
    data.train_hdf5_files=["/mnt/home/lserrano/MP-Neural-PDE-Solvers/data/OP_Euler_train_alpha1.0_beta0.0_gamma0.0_1024.h5"] \
    data.val_hdf5_files=["/mnt/home/lserrano/MP-Neural-PDE-Solvers/data/OP_Euler_valid_alpha1.0_beta0.0_gamma0.0.h5"] \
    data.test_hdf5_files=["/mnt/home/lserrano/MP-Neural-PDE-Solvers/data/OP_Euler_test_alpha1.0_beta0.0_gamma0.0.h5"]
    #data.train_hdf5_files=["/mnt/home/lserrano/LPSDA/data/E_EULER_train_8192.h5"] \
    #data.val_hdf5_files=["/mnt/home/lserrano/LPSDA/data/E_EULER_valid.h5"] \
    #data.test_hdf5_files=["/mnt/home/lserrano/LPSDA/data/E_EULER_test.h5"]

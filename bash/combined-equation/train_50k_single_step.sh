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

# Train for 50k steps with default num_steps=1
# Resume from checkpoint to accelerate training
#CHECKPOINT_PATH="outputs/DISCO_combined-physics-hdf5_solverrk4_adjFalse_h128_t3_steps1_initTrue_bs64_lr0.0005_hdf5_noise0_inframes16_outframes16_subx1_subt1/last-v1.ckpt"

python3 train/train_combined.py \
    model.max_steps=1 \
    model.solver=rk4 \
    model.use_adjoint=False \
    model.decoder_use_bias=True \
    model.principled_initialization=True \
    model.theta_dim=3 \
    training.in_context=False \
    training.in_context_progressive=True \
    training.max_steps=50000 \
    training.progressive_steps=False \
    training.project="disco-ttg-50k-single" \
    training.lr=5e-4 \
    training.batch_size=64 \
    data.n_output_frames=16 \
    data.sub_t=1 \
    #training.checkpoint_path="$CHECKPOINT_PATH"

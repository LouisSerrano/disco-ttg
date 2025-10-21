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

# Train for 50k steps with curriculum learning
# Starts with 1 frame and progressively increases to 16 frames over 25k steps (50% of training)

python3 train/train_combined_curriculum.py \
    model.max_steps=1 \
    model.solver=rk4 \
    model.use_adjoint=False \
    model.decoder_use_bias=True \
    model.principled_initialization=True \
    model.theta_dim=3 \
    training.max_steps=50000 \
    training.progressive_steps=False \
    training.curriculum_enabled=True \
    training.curriculum_start_frames=1 \
    training.curriculum_end_frames=32 \
    training.curriculum_warmup_steps=45000 \
    training.project="disco-ball-50k-curriculum" \
    training.lr=5e-4 \
    training.batch_size=64 \
    data.n_output_frames=32 \
    data.n_output_frames=32 \
    data.sub_t=1

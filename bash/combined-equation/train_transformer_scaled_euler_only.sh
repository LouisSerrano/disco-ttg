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

# Scaled Transformer for Euler-only dataset
# Scale up Transformer Encoder parameters:
# Increase hidden_dim: 128 -> 256 (2x)
# Increase num_heads: 4 -> 8 (2x, maintains head_dim=32)  
# Increase processor_blocks: 4 -> 8 (2x)
# Keep neural operator params at baseline
# Progressive max_steps: 1, 2, 4 with ratios 0.9, 0.05, 0.05
# Train only on euler dataset to understand fitting on more challenging dataset

python3 train/train_combined.py \
    model.hidden_dim=256 \
    model.num_heads=8 \
    model.processor_blocks=8 \
    model.hpnn_head_hidden_dim=32 \
    model.opnn_channels=8 \
    model.opnn_bottleneck_multiplier=2 \
    model.max_steps=1 \
    model.use_adjoint=False \
    model.decoder_use_bias=True \
    model.principled_initialization=True \
    model.theta_dim=3 \
    training.max_steps=50000 \
    training.progressive_steps=True \
    training.step_schedule="[1,2,4]" \
    training.step_percentages="[0.9,0.05,0.05]" \
    training.project="disco-ball-euler-only" \
    training.lr=5e-4 \
    training.batch_size=64 \
    data.n_input_frames=16 \
    data.n_output_frames=16 \
    data.sub_t=1 \
    data.train_hdf5_files=["/mnt/home/lserrano/disco-ball/datasets/combined_equation/E_EULER_train_8192.h5"] \
    data.val_hdf5_files=["/mnt/home/lserrano/disco-ball/datasets/combined_equation/E_EULER_valid.h5"] \
    data.test_hdf5_files=["/mnt/home/lserrano/disco-ball/datasets/combined_equation/E_EULER_test.h5"]

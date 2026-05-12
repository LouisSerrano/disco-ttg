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

# Train for 50k steps with progressive step increase: 80% at 1 step, 5% each at 2,4,8,16 steps
# Resume from checkpoint to accelerate training
#CHECKPOINT_PATH=${DISCO_OUTPUTS:-./outputs}/DISCO_combined-physics-hdf5_solverrk4_adjFalse_h128_t3_steps1_initTrue_bs64_lr0.0005_hdf5_noise0_inframes16_outframes16_subx1_subt1/last-v1.ckpt

#"outputs/DISCO_combined-physics-hdf5_solverrk4_adjFalse_h128_t3_steps1_initTrue_bs64_lr0.0005_hdf5_noise0_inframes16_outframes16_subx1_subt1/last-v1.ckpt"

python3 train/train_combined_ablations.py \
    model.max_steps=1 \
    model.use_adjoint=False \
    model.decoder_use_bias=True \
    model.principled_initialization=True \
    model.theta_dim=64 \
    model.processor_blocks=8 \
    model.num_heads=8 \
    training.max_steps=50000 \
    training.progressive_steps=True \
    training.step_schedule="[1,2,4]" \
    training.step_percentages="[0.9,0.05,0.05]" \
    training.project="neural-operator-splitting-50k-progressive" \
    training.lr=5e-4 \
    training.batch_size=64 \
    data.n_output_frames=64 \
    data.sub_t=1 \
    #training.checkpoint_path="$CHECKPOINT_PATH"

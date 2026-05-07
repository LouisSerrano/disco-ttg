#!/bin/bash -l

#SBATCH -p gpu
#SBATCH -t 24:00:00
#SBATCH -C a100
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --tasks-per-node=5
#SBATCH --cpus-per-task=4
#SBATCH -J sub_sweep

module load python
module load cuda
module load cudnn
module load nccl

source $VENVDIR/disco/bin/activate

MODEL_PATH=${DISCO_CKPT_DIR:-./outputs}/DISCO_advection-diffusion_solverrk4_adjFalse_h128_t2_steps1_initTrue_bs64_lr0.0005_ctxTrue_noise0_inframes16_outframes16_T10/last-v1.ckpt

# Dictionary subsampling sweep on E_AD_ALL (beam search)
python3 test_time_compute/analysis/sweep_ad_subsampling.py \
    --model_path $MODEL_PATH \
    --experiment E_AD_ALL \
    --dict_sizes 16 32 64 128 256 \
    --num_samples 512 \
    --beam_width 4 \
    --beam_batch_size 32

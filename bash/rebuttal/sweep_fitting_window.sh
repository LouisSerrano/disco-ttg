#!/bin/bash -l

#SBATCH -p gpu
#SBATCH -t 24:00:00
#SBATCH -C a100
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --tasks-per-node=5
#SBATCH --cpus-per-task=4
#SBATCH -J fit_sweep

module load python
module load cuda
module load cudnn
module load nccl

source $VENVDIR/disco/bin/activate

MODEL_PATH=/mnt/home/lserrano/ceph/disco/outputs/DISCO_advection-diffusion_solverrk4_adjFalse_h128_t2_steps1_initFalse_bs64_lr0.0005_ctxTrue_noise0_inframes16_outframes16_T10/last.ckpt

# Fitting window sweep on E_AD_ALL (beam search)
python3 test_time_compute/sweep_ad_fitting_window.py \
    --model_path $MODEL_PATH \
    --experiment E_AD_ALL \
    --window_sizes 2 4 8 16 32 \
    --num_samples 512 \
    --beam_width 4 \
    --beam_batch_size 32

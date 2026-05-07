#!/bin/bash -l
#SBATCH -p gpu
#SBATCH -t 12:00:00
#SBATCH -C a100
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --tasks-per-node=5
#SBATCH --cpus-per-task=4
#SBATCH -J fit_L4

module load python
module load cuda
module load cudnn
module load nccl
source $VENVDIR/disco/bin/activate

python3 test_time_compute/analysis/sweep_ad_fitting_window.py \
    --model_path ${DISCO_CKPT_DIR:-./outputs}/DISCO_advection-diffusion_solverrk4_adjFalse_h128_t2_steps1_initTrue_bs64_lr0.0005_ctxTrue_noise0_inframes16_outframes16_T10/last-v1.ckpt \
    --experiment E_AD_ALL \
    --window_sizes 4 \
    --num_samples 512 \
    --beam_width 4 --beam_batch_size 32

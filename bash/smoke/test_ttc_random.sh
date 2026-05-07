#!/bin/bash -l
# Smoke test: test_time_compute "random" method on advection-diffusion.
# Verifies that test_time_compute/equations/test_advection_diffusion.py + ttc_methods + ttc_utils
# work end-to-end on an existing checkpoint.

#SBATCH -p gpu
#SBATCH -t 00:15:00
#SBATCH -C a100
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -J disco-smoke-ttc

module load python
module load cuda
module load cudnn
module load nccl

source $VENVDIR/disco/bin/activate

CKPT=/mnt/home/lserrano/ceph/disco/outputs/DISCO_advection-diffusion_solverrk4_adjFalse_h128_t2_steps1_initTrue_bs512_lr0.001_ctxTrue_noise0_inframes16_outframes16_T10/last.ckpt

python3 test_time_compute/equations/test_advection_diffusion.py \
    --model_path $CKPT \
    --experiment E_AD_v \
    --methods random

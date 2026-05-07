#!/bin/bash -l
#SBATCH -p gpu
#SBATCH -t 12:00:00
#SBATCH -C a100
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --tasks-per-node=5
#SBATCH --cpus-per-task=4
#SBATCH -J ext_mixed05_128

module load python
module load cuda
module load cudnn
module load nccl
source $VENVDIR/disco/bin/activate

python3 test_time_compute/equations/test_advection_diffusion.py \
    --model_path /mnt/home/lserrano/ceph/disco/outputs/DISCO_advection-diffusion_solverrk4_adjFalse_h128_t2_steps1_initFalse_bs64_lr0.0005_ctxTrue_noise0.0001_mixed0.5_inframes16_outframes2_T10/best-checkpoint.ckpt \
    --experiment E_AD_EXTRAP_ALL \
    --methods direct beam \
    --beam_width 4 \
    --beam_batch_size 32 \
    --n_dict_batches 2 \
    --dict_mixed_ratio 0.5 \
    --output_dir ./test_time_compute/results/rebuttal/extrap_all_mixed05_n128

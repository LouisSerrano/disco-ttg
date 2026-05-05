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


 # python test_time_compute/test_advection_diffusion.py --model_path /mnt/home/lserrano/disco-ball/outputs/DISCO_advection-diffusion_solverrk4_adjFalse_h128_t2_steps1_initTrue_bs64_lr0.0005_ctxTrue_noise0_inframes16_outframes16_T10/last-v1.ckpt --experiment E_AD_ALL --methods beam --beam_width 3 

#  python test_time_compute/test_advection_diffusion.py --model_path /mnt/home/lserrano/disco-ball/outputs/DISCO_advection-diffusion_solverrk4_adjFalse_h128_t2_steps1_initTrue_bs64_lr0.0005_ctxTrue_noise0_inframes16_outframes16_T10/last-v1.ckpt --experiment E_AD_v --methods beam --beam_width 3 

#  python test_time_compute/test_advection_diffusion.py --model_path /mnt/home/lserrano/disco-ball/outputs/DISCO_advection-diffusion_solverrk4_adjFalse_h128_t2_steps1_initTrue_bs64_lr0.0005_ctxTrue_noise0_inframes16_outframes16_T10/last-v1.ckpt --experiment E_AD_D --methods beam --beam_width 3 


python test_time_compute/test_advection_diffusion.py --model_path /mnt/home/lserrano/disco-ball/outputs/DISCO_advection-diffusion_solverrk4_adjFalse_h128_t2_steps1_initTrue_bs64_lr0.0005_ctxTrue_noise0_inframes16_outframes16_T10/last-v1.ckpt --experiment E_AD_v --methods random 

python test_time_compute/test_advection_diffusion.py --model_path /mnt/home/lserrano/disco-ball/outputs/DISCO_advection-diffusion_solverrk4_adjFalse_h128_t2_steps1_initTrue_bs64_lr0.0005_ctxTrue_noise0_inframes16_outframes16_T10/last-v1.ckpt --experiment E_AD_D --methods random 


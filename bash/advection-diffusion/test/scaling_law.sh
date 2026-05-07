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


#python test_time_compute/analysis/scaling_law_advection_diffusion_analysis.py       --model_path /mnt/home/lserrano/disco-ball/outputs/DISCO_advection-diffusion_solverrk4_adjFalse_h128_t2_steps1_initTrue_bs64_lr0.0005_ctxTrue_noise0_inframes16_outframes16_T10/last-v1.ckpt --experiment E_AD_ALL --output_dir test_time_compute/results_evolution --random_batch_size 128 --checkpoints 10 100 200 500 1000 --num_samples 512


#python test_time_compute/analysis/scaling_law_advection_diffusion_analysis.py       --model_path /mnt/home/lserrano/disco-ball/outputs/DISCO_advection-diffusion_solverrk4_adjFalse_h128_t2_steps1_initTrue_bs64_lr0.0005_ctxTrue_noise0_inframes16_outframes16_T10/last-v1.ckpt --experiment E_AD_v --output_dir test_time_compute/results_evolution --random_batch_size 128 --checkpoints 10 100 200 500 1000 --num_samples 512

python test_time_compute/analysis/scaling_law_advection_diffusion_analysis.py       --model_path /mnt/home/lserrano/disco-ball/outputs/DISCO_advection-diffusion_solverrk4_adjFalse_h128_t2_steps1_initTrue_bs64_lr0.0005_ctxTrue_noise0_inframes16_outframes16_T10/last-v1.ckpt --experiment E_AD_D --output_dir test_time_compute/results_evolution --random_batch_size 128 --checkpoints 10 100 200 500 1000 --num_samples 512



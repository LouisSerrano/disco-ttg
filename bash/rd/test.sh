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

python test_time_compute/equations/test_reaction_diffusion.py --model_path /mnt/home/lserrano/disco-ball/outputs/DISCO_rd_solverrk4_adjFalse_h128_t3_steps1_initTrue_bs64_lr0.0003_hdf5_noise0_inframes16_outframes2_subx1_subt1_20250916_205814/last.ckpt --methods random --random_trials 10


#python test_time_compute/equations/test_reaction_diffusion.py --model_path /mnt/home/lserrano/disco-ball/outputs/DISCO_rd_solverrk4_adjFalse_h128_t3_steps1_initTrue_bs64_lr0.0003_hdf5_noise0_inframes16_outframes2_subx1_subt1_20250916_205814/last.ckpt --methods random --random_trials 200 

#python test_time_compute/equations/test_reaction_diffusion.py --model_path /mnt/home/lserrano/disco-ball/outputs/DISCO_rd_solverrk4_adjFalse_h128_t3_steps1_initTrue_bs64_lr0.0003_hdf5_noise0_inframes16_outframes2_subx1_subt1_20250916_205814/last.ckpt --methods random --random_trials 500 


#python test_time_compute/equations/test_reaction_diffusion.py --model_path /mnt/home/lserrano/disco-ball/outputs/DISCO_rd_solverrk4_adjFalse_h128_t3_steps1_initTrue_bs64_lr0.0003_hdf5_noise0_inframes16_outframes2_subx1_subt1_20250916_205814/last.ckpt --methods gradient 

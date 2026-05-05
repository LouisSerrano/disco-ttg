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


#python test_time_compute/test_combined_equation.py --model_path  /mnt/home/lserrano/disco-ball/outputs/DISCO_combined-physics-hdf5_solverrk4_adjFalse_h128_t3_steps1_initTrue_bs64_lr0.0005_hdf5_noise0_inframes16_outframes2_subx1_subt1_20250918_135244/last.ckpt --methods beam --experiment E_BG --beam_width 4

#python test_time_compute/test_combined_equation.py --model_path  /mnt/home/lserrano/disco-ball/outputs/DISCO_combined-physics-hdf5_solverrk4_adjFalse_h128_t3_steps1_initTrue_bs64_lr0.0005_hdf5_noise0_inframes16_outframes2_subx1_subt1_20250918_135244/last.ckpt --methods beam --experiment E_HE --beam_width 4

#python test_time_compute/test_combined_equation.py --model_path  /mnt/home/lserrano/disco-ball/outputs/DISCO_combined-physics-hdf5_solverrk4_adjFalse_h128_t3_steps1_initTrue_bs64_lr0.0005_hdf5_noise0_inframes16_outframes2_subx1_subt1_20250918_135244/last.ckpt --methods beam --experiment E_ED --beam_width 4

#python test_time_compute/test_combined_equation.py --model_path  /mnt/home/lserrano/disco-ball/outputs/DISCO_combined-physics-hdf5_solverrk4_adjFalse_h128_t3_steps1_initTrue_bs64_lr0.0005_hdf5_noise0_inframes16_outframes2_subx1_subt1_20250918_135244/last.ckpt --methods beam --experiment E_ALL --beam_width 4


ckpt_path=DISCO_combined-physics-hdf5_solverrk4_adjFalse_h128_t3_steps1_initFalse_bs64_lr0.0005_hdf5_noise0_inframes16_outframes2_subx1_subt1_20250915_235633

#python test_time_compute/test_combined_equation.py --model_path  /mnt/home/lserrano/disco-ball/outputs/$ckpt_path/last.ckpt --methods beam --experiment E_BG --beam_width 4

#python test_time_compute/test_combined_equation.py --model_path  /mnt/home/lserrano/disco-ball/outputs/$ckpt_path/last.ckpt --methods beam --experiment E_HE --beam_width 4

python test_time_compute/test_combined_equation.py --model_path  /mnt/home/lserrano/disco-ball/outputs/$ckpt_path/last.ckpt --methods beam --experiment E_ED --beam_width 4

python test_time_compute/test_combined_equation.py --model_path  /mnt/home/lserrano/disco-ball/outputs/$ckpt_path/last.ckpt --methods beam --experiment E_ALL --beam_width 4

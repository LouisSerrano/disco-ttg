#!/bin/bash -l

#SBATCH -p gpu
#SBATCH -t 48:00:00
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


#run_name=DISCO_euler_solverrk4_adjFalse_h128_t4_steps2_initFalse_bs32_lr0.0003_hdf5_noise0_inframes16_outframes2_subx1_subt1_20260124_040907
run_name=DISCO_euler_solverrk4_adjFalse_h128_t4_steps4_initFalse_bs16_lr0.0003_hdf5_noise0_inframes16_outframes2_subx1_subt1_20260124_041037

python test_time_compute/equations/test_navier_stokes.py \
    --model_path ./outputs/${run_name}/last.ckpt \
    --method direct \
    #--output_dir ./results_targeted \

    #--splitting_method strang \
    #--min_improvement 1 \
    #--plot \
    #--num_plots 16 \
    #--random_batch_size 32 \


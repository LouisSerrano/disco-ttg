#!/bin/bash -l

#SBATCH -p gpu
#SBATCH -t 24:00:00
#SBATCH -C a100
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --tasks-per-node=5
#SBATCH --cpus-per-task=4
#SBATCH -J eval_perturb

module load python
module load cuda
module load cudnn
module load nccl

source $VENVDIR/disco/bin/activate

MODEL_PATH=/mnt/home/lserrano/ceph/disco/outputs/DISCO_advection-diffusion_solverrk4_adjFalse_h128_t2_steps1_initFalse_bs64_lr0.0005_ctxTrue_noise0_inframes16_outframes16_T10/last.ckpt

# Evaluate perturbation data (128 samples x 7 epsilon values)
python3 test_time_compute/eval_perturbation.py \
    --model_path $MODEL_PATH \
    --data_dir ./test_time_compute/results/perturbation_data \
    --output_dir ./test_time_compute/results/rebuttal/perturbation_eval \
    --beam_width 4 \
    --beam_batch_size 32 \
    --max_operators 5

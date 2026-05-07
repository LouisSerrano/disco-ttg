#!/bin/bash -l

#SBATCH -p gpu
#SBATCH -t 48:00:00
#SBATCH -C a100
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --tasks-per-node=5
#SBATCH --cpus-per-task=4
#SBATCH -J eval_euler

module load python
module load cuda
module load cudnn
module load nccl

source $VENVDIR/disco/bin/activate

MODEL_PATH=${DISCO_CKPT_DIR:-./outputs}/DISCO_euler_solverrk4_adjFalse_h128_t4_steps4_initFalse_bs16_lr0.0003_hdf5_noise0_inframes16_outframes2_subx1_subt1_20260328_124504/best-checkpoint.ckpt
OUTPUT_DIR=./test_time_compute/results/rebuttal/euler_no_codebook

echo "=========================================="
echo "Evaluating Euler no-codebook model"
echo "=========================================="
python test_time_compute/equations/test_navier_stokes_targeted.py \
    --model_path $MODEL_PATH \
    --method beam \
    --splitting_method strang \
    --min_improvement 1 \
    --beam_width 4 \
    --beam_batch_size 48 \
    --refinement_factor 4 \
    --use_encoder \
    --num_dict_samples 272 \
    --num_samples_per_visc 32 \
    --output_dir $OUTPUT_DIR

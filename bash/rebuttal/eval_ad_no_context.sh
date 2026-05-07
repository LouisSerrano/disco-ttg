#!/bin/bash -l

#SBATCH -p gpu
#SBATCH -t 24:00:00
#SBATCH -C a100
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --tasks-per-node=5
#SBATCH --cpus-per-task=4
#SBATCH -J eval_noctx

module load python
module load cuda
module load cudnn
module load nccl

source $VENVDIR/disco/bin/activate

MODEL_PATH=${DISCO_CKPT_DIR:-./outputs}/DISCO_advection-diffusion_solverrk4_adjFalse_h128_t2_steps1_initFalse_bs64_lr0.0005_ctxFalse_noise0.0001_inframes16_outframes2_T10/best-checkpoint.ckpt
OUTPUT_DIR=./test_time_compute/results/rebuttal/no_context_of2

for EXP in E_AD_ALL E_AD_v E_AD_D; do
    echo "=========================================="
    echo "Running $EXP on no-context model (outframes=2)"
    echo "=========================================="
    python3 test_time_compute/equations/test_advection_diffusion.py \
        --model_path $MODEL_PATH \
        --experiment $EXP \
        --methods direct beam \
        --beam_width 4 \
        --beam_batch_size 32 \
        --output_dir ${OUTPUT_DIR}/${EXP}
done

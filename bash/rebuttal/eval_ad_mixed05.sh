#!/bin/bash -l

#SBATCH -p gpu
#SBATCH -t 24:00:00
#SBATCH -C a100
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --tasks-per-node=5
#SBATCH --cpus-per-task=4
#SBATCH -J eval_mix05

module load python
module load cuda
module load cudnn
module load nccl

source $VENVDIR/disco/bin/activate

MODEL_PATH=/mnt/home/lserrano/ceph/disco/outputs/DISCO_advection-diffusion_solverrk4_adjFalse_h128_t2_steps1_initFalse_bs64_lr0.0005_ctxTrue_noise0.0001_mixed0.5_inframes16_outframes2_T10/best-checkpoint.ckpt

# === Pure physics dictionary (paper default) ===
OUTPUT_DIR=./test_time_compute/results/rebuttal/mixed05_of2_puredict
for EXP in E_AD_ALL E_AD_v E_AD_D; do
    echo "=========================================="
    echo "[$EXP] mixed0.5 model — PURE dict"
    echo "=========================================="
    python3 test_time_compute/equations/test_advection_diffusion.py \
        --model_path $MODEL_PATH \
        --experiment $EXP \
        --methods direct beam \
        --beam_width 4 \
        --beam_batch_size 32 \
        --output_dir ${OUTPUT_DIR}/${EXP}
done

# === Mixed physics dictionary (matching training: ratio=0.5) ===
OUTPUT_DIR=./test_time_compute/results/rebuttal/mixed05_of2_mixeddict
for EXP in E_AD_ALL E_AD_v E_AD_D; do
    echo "=========================================="
    echo "[$EXP] mixed0.5 model — MIXED dict (ratio=0.5)"
    echo "=========================================="
    python3 test_time_compute/equations/test_advection_diffusion.py \
        --model_path $MODEL_PATH \
        --experiment $EXP \
        --methods direct beam \
        --beam_width 4 \
        --beam_batch_size 32 \
        --dict_mixed_ratio 0.5 \
        --output_dir ${OUTPUT_DIR}/${EXP}
done

#!/bin/bash -l
#SBATCH -p gpu
#SBATCH -t 24:00:00
#SBATCH -C a100
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --tasks-per-node=5
#SBATCH --cpus-per-task=4
#SBATCH -J geps_E_AD_v_100

module load python
module load cuda
module load cudnn
module load nccl
source $VENVDIR/disco/bin/activate

export PYTHONPATH=./neural-operator-splitting/GEPS:$PYTHONPATH
python3 baselines/GEPS/test_geps_inference.py \
    --model_path ${GEPS_CKPT_DIR:-./outputs/geps}/advection-diffusion/jumping-shadow-48/last.ckpt \
    --equation_type advection_diffusion \
    --experiment E_AD_v \
    --num_samples 512 \
    --batch_size 64 \
    --n_optimization_steps 100 \
    --lr 0.001 \
    --n_pred 1 \
    --output_dir ./GEPS/results/wallclock_E_AD_v_100steps

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

# Model checkpoint to use (with context)
#MODEL_CHECKPOINT="DISCO_advection-diffusion_adjFalse_h128_t2_steps1_bs64_lr0.0005_ctxTrue_inframes16_outframes2_T10.0"
MODEL_CHECKPOINT="DISCO_advection-diffusion_solvereuler_adjFalse_h128_t2_steps1_bs64_lr0.0005_ctxTrue_inframes16_outframes2_T10"

#"DISCO_advection-diffusion_adjFalse_h128_t2_steps1_bs64_lr0.0005_ctxTrue_inframes16_outframes2_T10.0"

# Run optimizer comparison experiment
echo "Running optimizer comparison experiment..."
python3 tests/expe_test_time_optimization_double.py hydra.run.dir=. test.run_name=$MODEL_CHECKPOINT test.ckpt_path=outputs/$MODEL_CHECKPOINT/best-checkpoint.ckpt test.num_steps=1

echo "Experiment completed!"

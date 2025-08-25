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
MODEL_CHECKPOINT="DISCO_advection-diffusion_adjFalse_h128_t2_steps1_bs64_lr0.0005_ctxTrue_inframes16_outframes2_T10.0"

# Run test_inter_extra
echo "Running test_inter_extra..."
python3 tests/test_inter_extra.py hydra.run.dir=. test.run_name=$MODEL_CHECKPOINT test.ckpt_path=outputs/$MODEL_CHECKPOINT/best-checkpoint.ckpt

# Run test_operator_composition
echo "Running test_operator_composition..."
python3 tests/test_operator_composition.py hydra.run.dir=. test.run_name=$MODEL_CHECKPOINT test.ckpt_path=outputs/$MODEL_CHECKPOINT/best-checkpoint.ckpt

# Run test_time_optimization_double
echo "Running test_time_optimization_double..."
python3 tests/test_time_optimization_double.py hydra.run.dir=. test.run_name=$MODEL_CHECKPOINT test.ckpt_path=outputs/$MODEL_CHECKPOINT/best-checkpoint.ckpt

# Run test_time_optimization_single
echo "Running test_time_optimization_single..."
python3 tests/test_time_optimization_single.py hydra.run.dir=. test.run_name=$MODEL_CHECKPOINT test.ckpt_path=outputs/$MODEL_CHECKPOINT/best-checkpoint.ckpt

echo "All tests completed!"

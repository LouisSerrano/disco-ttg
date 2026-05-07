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

CHECKPOINT="/mnt/home/lserrano/disco-ttg/outputs/DISCO_advection-diffusion_solverrk4_adjFalse_h128_t2_steps1_initFalse_bs64_lr0.0005_ctxTrue_noise0.0001_inframes16_outframes2_T10/last-v1.ckpt"

# Test case 1: advection (0.9,1), diffusion (0.9,1) - baseline case
echo "Testing advection (0.9,1), diffusion (0.9,1)"

# Default settings
#python tests/neural-operator-splitting/main_streamlined.py --checkpoint-path $CHECKPOINT --num-operators 20 --n-trajectories-per-operator 8 --preservation-coeff=1 --num-integration-steps=5 --test-v-min 0.9 --test-v-max 1.0 --test-D-min 0.9 --test-D-max 1.0 --finetune-epochs=300 --num-runs=20 --n-test-trajectories=1

# Vary number of operators
#python tests/neural-operator-splitting/main_streamlined.py --checkpoint-path $CHECKPOINT --num-operators 200 --n-trajectories-per-operator 8 --preservation-coeff=1 --num-integration-steps=5 --test-v-min 0.9 --test-v-max 1.0 --test-D-min 0.9 --test-D-max 1.0 --finetune-epochs=300 --num-runs=20 --n-test-trajectories=1

# Vary trajectories per operator
#python tests/neural-operator-splitting/main_streamlined.py --checkpoint-path $CHECKPOINT --num-operators 20 --n-trajectories-per-operator 1 --preservation-coeff=1 --num-integration-steps=5 --test-v-min 0.9 --test-v-max 1.0 --test-D-min 0.9 --test-D-max 1.0 --finetune-epochs=300 --num-runs=20 --n-test-trajectories=1

# Vary integration steps
#python tests/neural-operator-splitting/main_streamlined.py --checkpoint-path $CHECKPOINT --num-operators 20 --n-trajectories-per-operator 8 --preservation-coeff=1 --num-integration-steps=1 --test-v-min 0.9 --test-v-max 1.0 --test-D-min 0.9 --test-D-max 1.0 --finetune-epochs=300 --num-runs=20 --n-test-trajectories=1

#python tests/neural-operator-splitting/main_streamlined.py --checkpoint-path $CHECKPOINT --num-operators 20 --n-trajectories-per-operator 8 --preservation-coeff=1 --num-integration-steps=10 --test-v-min 0.9 --test-v-max 1.0 --test-D-min 0.9 --test-D-max 1.0 --finetune-epochs=300 --num-runs=20 --n-test-trajectories=1

# Vary test trajectories
#python tests/neural-operator-splitting/main_streamlined.py --checkpoint-path $CHECKPOINT --num-operators 20 --n-trajectories-per-operator 8 --preservation-coeff=1 --num-integration-steps=5 --test-v-min 0.9 --test-v-max 1.0 --test-D-min 0.9 --test-D-max 1.0 --finetune-epochs=300 --num-runs=20 --n-test-trajectories=4

#python tests/neural-operator-splitting/main_streamlined.py --checkpoint-path $CHECKPOINT --num-operators 20 --n-trajectories-per-operator 8 --preservation-coeff=1 --num-integration-steps=5 --test-v-min 0.9 --test-v-max 1.0 --test-D-min 0.9 --test-D-max 1.0 --finetune-epochs=300 --num-runs=20 --n-test-trajectories=8


# Test case 2: advection (0,0), diffusion (3,5) - pure diffusion with high values
echo "Testing advection (0,0), diffusion (3,5)"

python tests/neural-operator-splitting/main_streamlined.py --checkpoint-path $CHECKPOINT --num-operators 20 --n-trajectories-per-operator 8 --preservation-coeff=1 --num-integration-steps=5 --test-v-min 0.0 --test-v-max 0.0 --test-D-min 3.0 --test-D-max 5.0 --finetune-epochs=300 --num-runs=20 --n-test-trajectories=1

python tests/neural-operator-splitting/main_streamlined.py --checkpoint-path $CHECKPOINT --num-operators 200 --n-trajectories-per-operator 8 --preservation-coeff=1 --num-integration-steps=5 --test-v-min 0.0 --test-v-max 0.0 --test-D-min 3.0 --test-D-max 5.0 --finetune-epochs=300 --num-runs=20 --n-test-trajectories=1

#python tests/neural-operator-splitting/main_streamlined.py --checkpoint-path $CHECKPOINT --num-operators 20 --n-trajectories-per-operator 1 --preservation-coeff=1 --num-integration-steps=5 --test-v-min 0.0 --test-v-max 0.0 --test-D-min 3.0 --test-D-max 5.0 --finetune-epochs=300 --num-runs=20 --n-test-trajectories=1


# Test case 3: advection (1.5,3), diffusion (0,0) - pure advection with high values
echo "Testing advection (1.5,3), diffusion (0,0)"

#python tests/neural-operator-splitting/main_streamlined.py --checkpoint-path $CHECKPOINT --num-operators 20 --n-trajectories-per-operator 8 --preservation-coeff=1 --num-integration-steps=5 --test-v-min 1.5 --test-v-max 3.0 --test-D-min 0.0 --test-D-max 0.0 --finetune-epochs=300 --num-runs=20 --n-test-trajectories=1

#python tests/neural-operator-splitting/main_streamlined.py --checkpoint-path $CHECKPOINT --num-operators 200 --n-trajectories-per-operator 8 --preservation-coeff=1 --num-integration-steps=5 --test-v-min 1.5 --test-v-max 3.0 --test-D-min 0.0 --test-D-max 0.0 --finetune-epochs=300 --num-runs=20 --n-test-trajectories=1

python tests/neural-operator-splitting/main_streamlined.py --checkpoint-path $CHECKPOINT --num-operators 20 --n-trajectories-per-operator 1 --preservation-coeff=1 --num-integration-steps=5 --test-v-min 1.5 --test-v-max 3.0 --test-D-min 0.0 --test-D-max 0.0 --finetune-epochs=300 --num-runs=20 --n-test-trajectories=1

echo "All tests completed!"

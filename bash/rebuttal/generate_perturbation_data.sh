#!/bin/bash -l

#SBATCH -p gpu
#SBATCH -t 4:00:00
#SBATCH -C a100
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --tasks-per-node=5
#SBATCH --cpus-per-task=4
#SBATCH -J gen_perturb

module load python
module load cuda
module load cudnn
module load nccl

source $VENVDIR/disco/bin/activate

# Generate perturbation dataset: 128 trajectories per epsilon, fixed v=0.5, D=0.3
python3 test_time_compute/analysis/generate_perturbation_data.py \
    --n_samples 128 \
    --epsilons 0.0 0.01 0.05 0.1 0.25 0.5 1.0 \
    --v 0.5 --D 0.3 \
    --output_dir ./test_time_compute/results/perturbation_data \
    --seed 42

#!/bin/bash -l
# Smoke test: test_generic.py using the ckpt produced by bash/smoke/train_generic.sh.
# Picks up the most-recent generic-smoke run and exercises codebook → random selection.

#SBATCH -p gpu
#SBATCH -t 00:15:00
#SBATCH -C a100
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -J disco-smoke-generic-test

module load python
module load cuda
module load cudnn
module load nccl

source $VENVDIR/disco/bin/activate

DATA=${DISCO_DATA:-./datasets}
OUT=${DISCO_OUTPUTS:-./outputs}/smoke

# Find the most-recent generic-smoke checkpoint
CKPT=$(ls -t $OUT/DISCO_*/last.ckpt 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then
    echo "No checkpoint found under $OUT/. Run train_generic.sh first." >&2
    exit 1
fi
echo "Using checkpoint: $CKPT"

python3 test_time_compute/test_generic.py \
    --model_path "$CKPT" \
    --train_files $DATA/combined_HEAT_test.h5 \
    --test_files  $DATA/combined_HEAT_test.h5 \
    --operator_source codebook \
    --method random \
    --random_trials 16 \
    --random_batch_size 4 \
    --num_samples 16 \
    --batch_size 4 \
    --output_dir $OUT/test_results

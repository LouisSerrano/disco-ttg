#!/bin/bash

# Test script for overfitting experiments

# Load required modules
module load python
module load cuda
module load cudnn

# Activate virtual environment
source $VENVDIR/disco/bin/activate

# Run overfitting test with 5 experiments
python training_burgers_test.py \
    --K 5 \
    --num_epochs 100 \
    --batch_size 4 \
    --learning_rate 1e-3 \
    --hidden_dim 32 \
    --n_layers 2 \
    --output_csv overfitting_test_K5.csv \
    --verbose

echo "Test completed. Check overfitting_test_K5.csv for results."
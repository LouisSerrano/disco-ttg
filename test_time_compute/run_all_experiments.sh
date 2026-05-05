#!/bin/bash

# Configuration
MODEL_PATH="/path/to/your/model.ckpt"  # Update this
OUTPUT_DIR="./results"
NUM_SAMPLES=10
NUM_OPERATORS=20

# Create output directory
mkdir -p $OUTPUT_DIR

# Run experiments for each dataset
echo "Running test-time compute experiments..."
echo "========================================"

echo "1. Testing advection-diffusion..."
python test_advection_diffusion.py \
    --model_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --num_samples $NUM_SAMPLES \
    --num_operators $NUM_OPERATORS

echo ""
echo "2. Testing combined equation..."
python test_combined_equation.py \
    --model_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --num_samples $NUM_SAMPLES \
    --num_operators $NUM_OPERATORS

echo ""
echo "3. Testing reaction-diffusion..."
python test_reaction_diffusion.py \
    --model_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --num_samples $NUM_SAMPLES \
    --num_operators $NUM_OPERATORS

echo ""
echo "All experiments completed!"
echo "Results saved to: $OUTPUT_DIR"
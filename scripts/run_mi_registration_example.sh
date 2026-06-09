#!/bin/bash

# MI Registration Example Script
# This script demonstrates how to run MI registration and compare with NEMAR

set -e

echo "========================================="
echo "MI Registration Example"
echo "========================================="

# Configuration
DATASET="rire"
NAME="mi_registration_example"
NUM_TEST=10

# Create results directory
mkdir -p ./results/${NAME}

echo ""
echo "Step 1: Running MI Registration on ${DATASET} dataset..."
echo ""

# Run MI registration
python test_mi.py \
    --dataroot ./datasets/${DATASET} \
    --name ${NAME} \
    --model mi_registration \
    --dataset_mode rire_2d \
    --gpu_ids 0 \
    --num_test ${NUM_TEST} \
    --mi_num_bins 64 \
    --mi_sigma 2.0 \
    --lambda_mi_smooth 0.1 \
    --optim_lr 1.0 \
    --optim_max_iter 100 \
    --optim_tolerance 1e-5 \
    --optim_history_size 100 \
    --transform_type affine

echo ""
echo "Step 2: Evaluating registration quality..."
echo ""

# Evaluate results
python scripts/eval_mi_registration.py \
    --results_dir ./results/${NAME} \
    --num_samples ${NUM_TEST} \
    --output ./results/${NAME}/evaluation.txt

echo ""
echo "========================================="
echo "MI Registration completed!"
echo "Results saved in: ./results/${NAME}"
echo "Evaluation results: ./results/${NAME}/evaluation.txt"
echo "========================================="

# Display comparison
echo ""
echo "Comparison of MI Registration vs NEMAR:"
echo "========================================="
echo "MI Registration (Traditional)"
echo "  - Uses Mutual Information as similarity metric"
echo "  - Optimizes affine transformation parameters"
echo "  - No training required (iterative optimization)"
echo "  - Computationally expensive per image"
echo ""
echo "NEMAR (Deep Learning)"
echo "  - Uses adversarial learning + contrastive loss"
echo "  - Learns non-rigid deformations"
echo "  - Training required, fast inference"
echo "  - Better for complex, non-linear deformations"
echo "========================================="
#!/bin/bash

# MI Registration Testing Script
# This script runs the MI-based image registration model in test mode

# Configuration
DATASET_NAME=${1:-"RIRE_2d"}        # Dataset: RIRE_2d or L2R_2d
DATASET_MODE=${2:-"rire_2d"}        # Dataset mode: rire_2d or l2r_2d
NUM_TEST=${3:-100}                  # Number of test images
NAME=${4:-"mi_registration_${DATASET_NAME}"}

# Set common parameters
COMMON_PARAMS="
    --model mi_registration
    --dataset_mode ${DATASET_MODE}
    --gpu_ids 0
    --num_test ${NUM_TEST}
"

# MI-specific parameters
MI_PARAMS="
    --mi_num_bins 64
    --mi_sigma 2.0
    --lambda_mi_smooth 0.1
    --optim_lr 0.1
    --optim_max_iter 100
    --optim_tolerance 1e-5
    --optim_history_size 100
    --optim_line_search strong_wolfe
    --transform_type affine
"

echo "========================================="
echo "MI Registration Testing"
echo "========================================="
echo "Dataset: ${DATASET_NAME}"
echo "Dataset Mode: ${DATASET_MODE}"
echo "Num Test: ${NUM_TEST}"
echo "Name: ${NAME}"
echo "========================================="

# Run test
python test_mi.py \
    --dataroot ./datasets/${DATASET_NAME} \
    --name ${NAME} \
    ${COMMON_PARAMS} \
    ${MI_PARAMS}

echo "========================================="
echo "Test completed!"
echo "Results saved in ./results/${NAME}"
echo "========================================="
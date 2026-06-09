#!/bin/bash

# VoxelMorph-MI Training and Testing Script
# Uses Mutual Information loss for cross-modal registration (CT→MR)

# Configuration
DATASET_NAME=${1:-"RIRE_2d_paired"}
DATASET_MODE=${2:-"aligned_2d"}
NUM_TEST=${3:-100}
NAME=${4:-"voxelmorph_rire"}
NITER=${5:-200}

echo "========================================="
echo "VoxelMorph-MI Registration"
echo "========================================="
echo "Dataset: ${DATASET_NAME}"
echo "Dataset Mode: ${DATASET_MODE}"
echo "Num Test: ${NUM_TEST}"
echo "Name: ${NAME}"
echo "Iterations: ${NITER}"
echo "Loss: MI (Mutual Information)"
echo "========================================="

# Train VoxelMorph-MI
python test_voxelmorph.py \
    --dataroot ./datasets/${DATASET_NAME} \
    --name ${NAME} \
    --dataset_mode ${DATASET_MODE} \
    --gpu_ids 0 \
    --num_test ${NUM_TEST} \
    --vm_num_features 32 64 128 256 \
    --vm_loss_type mi \
    --vm_smoothness_weight 1.0 \
    --vm_mi_bins 64 \
    --vm_lr 1e-4 \
    --vm_niter ${NITER} \
    --train

echo ""
echo "========================================="
echo "VoxelMorph-MI completed!"
echo "Results saved in: ./results/${NAME}"
echo "Model saved in: ./checkpoints/${NAME}"
echo "========================================="

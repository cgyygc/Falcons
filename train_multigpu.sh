#!/bin/bash

# Multi-GPU Training Script for NEMAR (Falcon)
# Uses DataParallel for simple multi-GPU training

set -e

echo "========================================="
echo "Multi-GPU Training Configuration"
echo "========================================="
echo "GPU 0: RTX 5090 (32 GB)"
echo "GPU 1: RTX 5090 (32 GB)"
echo "========================================="

# Configuration
DATASET_NAME=${1:-"RIRE_2d"}
DATASET_MODE=${2:-"rire_2d"}
NAME=${3:-"falcon_multigpu"}
GPUS=${4:-"0,1"}
BATCH_SIZE=${5:-8}  # Increase batch size for multi-GPU

echo "Dataset: ${DATASET_NAME}"
echo "Model: ${NAME}"
echo "GPUs: ${GPUS}"
echo "Batch Size: ${BATCH_SIZE}"
echo "========================================="

# Run training with DataParallel
CUDA_VISIBLE_DEVICES=0,1 python -u train.py \n    --dataroot ./datasets/${DATASET_NAME} \n    --name ${NAME} \n    --model nemar \n    --dataset_mode ${DATASET_MODE} \n    --direction AtoB \n    --stn_type ukan_gbcm_contrastive \n    --use_gbcm \n    --use_contrastive \n    --contrastive_weight 0.2 \n    --use_label_smooth \n    --use_disc_noise \n    --gpu_ids ${GPUS} \n    --batch_size ${BATCH_SIZE} \n    --niter 200 \n    --niter_decay 0 \n    --img_height 288 \n    --img_width 384 \n    --save_epoch_freq 5 \n    --display_freq 10 \n    --print_freq 50

echo ""
echo "========================================="
echo "Multi-GPU training completed!"
echo "Results in: ./checkpoints/${NAME}"
echo "========================================="
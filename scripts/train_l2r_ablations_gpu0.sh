#!/bin/bash
# Train L2R ablation experiments - Part 1 (GPU 0)
# Experiments 1-4

GPU=0
DATAROOT="./datasets/L2R_2d/Train"
COMMON="--dataroot $DATAROOT --dataset_mode l2r_2d --direction AtoB --gpu_ids $GPU --batch_size 1 --niter 200 --niter_decay 0 --img_height 512 --img_width 512 --preprocess none --no_flip --save_epoch_freq 10 --print_freq 100 --input_nc 1 --output_nc 1 --no_html"

echo "Starting L2R ablation experiments Part 1 on GPU $GPU"
echo "================================================"

# 1. STN Affine
echo "[1/9] Training: l2r_ablation_stn_affine"
OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 python train.py $COMMON --name l2r_ablation_stn_affine --model nemar --stn_type affine --netG resnet_6blocks_gbcm

# 2. STN UKAN (no contrastive)
echo "[2/9] Training: l2r_ablation_stn_ukan"
OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 python train.py $COMMON --name l2r_ablation_stn_ukan --model nemar --stn_type ukan --netG resnet_6blocks_gbcm

# 3. No GBCM
echo "[3/9] Training: l2r_ablation_no_gbcm"
OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 python train.py $COMMON --name l2r_ablation_no_gbcm --model nemar --stn_type ukan_contrastive --netG resnet_6blocks --use_contrastive

# 4. Only disc noise
echo "[4/9] Training: l2r_ablation_only_disc_noise"
OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 python train.py $COMMON --name l2r_ablation_only_disc_noise --model nemar --stn_type ukan_contrastive --netG resnet_6blocks_gbcm --use_contrastive --disc_noise_std 0.05

echo "================================================"
echo "L2R ablation experiments Part 1 completed!"

#!/bin/bash
# Train L2R ablation experiments - Part 2 (GPU 1)
# Experiments 5-9

GPU=1
DATAROOT="./datasets/L2R_2d/Train"
COMMON="--dataroot $DATAROOT --dataset_mode l2r_2d --direction AtoB --gpu_ids $GPU --batch_size 1 --niter 200 --niter_decay 0 --img_height 512 --img_width 512 --preprocess none --no_flip --save_epoch_freq 10 --print_freq 100 --input_nc 1 --output_nc 1 --no_html"

echo "Starting L2R ablation experiments Part 2 on GPU $GPU"
echo "================================================"

# 5. Only label smoothing (no disc noise)
echo "[5/9] Training: l2r_ablation_only_label_smooth"
OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 python train.py $COMMON --name l2r_ablation_only_label_smooth --model nemar --stn_type ukan_contrastive --netG resnet_6blocks_gbcm --use_contrastive --label_smoothing 0.1

# 6. Contrastive weight = 0.0
echo "[6/9] Training: l2r_ablation_weight_00"
OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 python train.py $COMMON --name l2r_ablation_weight_00 --model nemar --stn_type ukan --netG resnet_6blocks_gbcm

# 7. Contrastive weight = 0.05
echo "[7/9] Training: l2r_ablation_weight_005"
OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 python train.py $COMMON --name l2r_ablation_weight_005 --model nemar --stn_type ukan_contrastive --netG resnet_6blocks_gbcm --use_contrastive --contrastive_weight 0.05

# 8. Contrastive weight = 0.2
echo "[8/9] Training: l2r_ablation_weight_02"
OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 python train.py $COMMON --name l2r_ablation_weight_02 --model nemar --stn_type ukan_contrastive --netG resnet_6blocks_gbcm --use_contrastive --contrastive_weight 0.2 --disc_noise_std 0.05 --label_smoothing 0.1

# 9. Contrastive weight = 0.3
echo "[9/9] Training: l2r_ablation_weight_03"
OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 python train.py $COMMON --name l2r_ablation_weight_03 --model nemar --stn_type ukan_contrastive --netG resnet_6blocks_gbcm --use_contrastive --contrastive_weight 0.3 --disc_noise_std 0.05 --label_smoothing 0.1

echo "================================================"
echo "L2R ablation experiments Part 2 completed!"

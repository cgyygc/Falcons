#!/bin/bash
# Train TransMorph on RIRE and L2R datasets

# RIRE dataset
echo "Training TransMorph on RIRE..."
OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 python train.py \
    --dataroot ./datasets/RIRE_2d_paired \
    --name transmorph_rire \
    --model transmorph \
    --dataset_mode aligned_2d \
    --direction AtoB \
    --gpu_ids 0 \
    --batch_size 1 \
    --niter 200 --niter_decay 0 \
    --img_height 512 --img_width 512 \
    --preprocess none --no_flip \
    --save_epoch_freq 20 --print_freq 50 \
    --input_nc 1 --output_nc 1 \
    --sim_loss mind \
    --lambda_sim 1 --lambda_reg 1 \
    --no_html

echo "TransMorph RIRE training complete!"

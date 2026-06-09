#!/bin/bash
# Sequential training script for all 3D models
# Run after VM3D v2 finishes

set -e

echo "=== Starting TransMorph3D+SVF v2 ==="
python -u train_3d.py \
    --model transmorph3d \
    --name tm3d_l2r_v2 \
    --niter 1000 \
    --use_amp \
    --early_stop_patience 10 \
    --early_stop_min_epochs 200

echo "=== Starting Falcon3D+SVF v2 ==="
python -u train_3d.py \
    --model nemar3d \
    --name falcon3d_l2r_v2 \
    --niter 1000 \
    --use_amp \
    --early_stop_patience 10 \
    --early_stop_min_epochs 200

echo "=== All training complete ==="

echo "=== Starting VoxelMorph3D on IXI ==="
python -u train_ixi.py \
    --model voxelmorph3d \
    --name vm3d_ixi \
    --niter 200 \
    --use_amp \
    --target_shape 128 128 128

echo "=== Starting TransMorph3D on IXI ==="
python -u train_ixi.py \
    --model transmorph3d \
    --name tm3d_ixi \
    --niter 200 \
    --use_amp \
    --target_shape 128 128 128

echo "=== Starting Falcon3D on IXI ==="
python -u train_ixi.py \
    --model nemar3d \
    --name falcon3d_ixi \
    --niter 200 \
    --use_amp \
    --target_shape 128 128 128

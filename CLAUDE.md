# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NEMAR (Neural Multimodal Adversarial Registration) is an unsupervised multi-modal image registration system that learns both registration and translation networks simultaneously without requiring paired training data. It implements the paper "Unsupervised Multi-Modal Image Registration via Geometry Preserving Image-to-Image Translation" (https://arxiv.org/abs/2003.08073).

The system uses a three-network architecture:
- **netT**: Translation network (modality A → modality B)
- **netR**: Registration network (Spatial Transformer Network - STN)
- **netD**: Adversarial discriminator

## Common Commands

### Training

Basic training with RIRE dataset (most common use case):
```bash
# RIRE 2D dataset with UKAN-GBM-Contrastive STN
python train.py --dataroot ./datasets/rire --name rire_experiment --model nemar \
    --dataset_mode unaligned --direction AtoB \
    --stn_type ukan_gbcm_contrastive --img_height 288 --img_width 384 \
    --niter 200 --niter_decay 0

# Other STN types: affine_stn, unet_stn, ukan_stn, contrastive_stn
```

### Traditional Baseline (MI Registration)

Test the traditional Mutual Information registration method:
```bash
# Run MI registration baseline
bash scripts/test_mi_registration.sh rire rire_2d 100 mi_baseline

# Run with custom parameters
python test_mi.py --dataroot ./datasets/rire --name mi_baseline \
    --model mi_registration --dataset_mode rire_2d \
    --num_test 100 \
    --transform_type affine --optim_max_iter 100

# Evaluate MI registration results
python scripts/eval_mi_registration.py \
    --results_dir ./results/mi_baseline --output mi_results.txt
```

See `MI_REGISTRATION_README.md` for detailed documentation on MI registration.

Standard pix2pix training:
```bash
python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix \
    --netG unet_256 --direction BtoA --lambda_L1 100 --dataset_mode aligned --norm batch
```

### Testing and Evaluation

Test a trained model:
```bash
python test_single.py --dataroot ./datasets/rire --name rire_experiment --epoch latest
```

Evaluate registration performance with metrics (MSE, MAE, PSNR, SSIM, NCC):
```bash
python eval_registration.py --dataroot ./datasets/rire --name rire_experiment --num_test 100
```

### Installation

Dependencies:
```bash
pip install torch torchvision
pip install visdom dominate nibabel scikit-image
```

## Architecture

### Three-Path Training Flow

The model trains both TR (Translation→Registration) and RT (Registration→Translation) paths simultaneously:
1. **TR path**: `real_A → netR → registered_real_A → netT → fake_TR_B`
2. **RT path**: `real_A → netT → fake_B → netR → fake_RT_B`

Both paths contribute to the loss to ensure geometry preservation.

### Configuration System

Configuration uses inheritance through the options system:
- `BaseOptions`: Core parameters (data paths, model type, GPU)
- `TrainOptions`: Training-specific parameters (lr, epochs, schedulers)
- Model-specific options: Added via `modify_commandline_options()`
- Dataset-specific options: Added via dataset classes

Key parameters:
- `--stn_type`: Spatial transformer type (`affine_stn`, `unet_stn`, `ukan_stn`, `ukan_gbcm_contrastive`)
- `--use_contrastive`: Enable contrastive learning
- `--lambda_GAN`: GAN loss weight (default 1.0)
- `--lambda_recon`: L1 reconstruction loss weight (default 100.0)
- `--lambda_smooth`: Smoothness regularization weight (default 0.0)

### Loss Function

Total loss (from `models/nemar_model.py`):
```
L_total = L1_TR + L1_RT + GAN_TR + GAN_RT + smoothness + contrastive_loss
```

Where:
- `L1_TR = λ_recon * L1(fake_TR_B, real_B)` - Reconstruction loss for TR path
- `GAN_TR = λ_GAN * GAN_loss(fake_TR_B)` - Adversarial loss for TR path
- `L1_RT = λ_recon * L1(fake_RT_B, real_B)` - Reconstruction loss for RT path
- `GAN_RT = λ_GAN * GAN_loss(fake_RT_B)` - Adversarial loss for RT path
- `smoothness = λ_smooth * stn_reg_term` - Deformation smoothness
- `contrastive_loss` - Optional contrastive learning loss

### STN Variants

Located in `models/stn/`:
- `affine_stn.py`: Simple affine transformations
- `unet_stn.py`: UNET-based spatial transformer
- `ukan_stn.py`: UKAN-based transformer
- `ukan_gbcm_contrastive.py`: UKAN with Gradient-Based Cross-Modality and contrastive learning
- `contrastive_head.py`: Contrastive learning components

### Dataset Support

Medical imaging datasets in `data/`:
- `rire_dataset.py`: 3D NIfTI format
- `rire_2d_dataset.py`: 2D PNG slices (converted from 3D)
- `l2r_2d_dataset.py`: Left-to-right 2D dataset

RIRE dataset structure:
```
datasets/rire/
    trainA/          # Modality A images (e.g., CT)
    trainB/          # Modality B images (e.g., MR)
    # OR
    ct/              # CT slices
    mr_t1/           # MR T1 slices
    mr_t2/           # MR T2 slices
```

## Training Parameters

Default training parameters (from `options/train_options.py`):
- Optimizer: Adam (β1=0.5, β2=0.999)
- Learning rate: 0.0002
- LR policy: Linear decay (can be: linear, step, plateau, cosine)
- Epochs: 100 (niter) + 100 (niter_decay) = 200 total
- Batch size: 1 (default, can be changed)
- Image size: 286x286 (load_size) → 256x256 (crop_size)
- GAN mode: Vanilla (can be: vanilla, lsgan, wgangp)

## Model Checkpoints

Models are saved in `./checkpoints/[name]/`:
- `latest_net_G.pth`, `latest_net_D.pth`, `latest_net_R.pth`: Latest weights
- `[epoch]_net_*.pth`: Checkpoints every `save_epoch_freq` epochs
- `opt.txt`: Configuration used for training

Use `--continue_train` to resume from latest checkpoint.

## Baseline Models for Comparison

### Traditional Methods

| Model | Type | Command |
|-------|------|---------|
| **MI Registration** | Mutual Information (Traditional) | `--model mi_registration` |
| *Note*: MI registration uses L-BFGS optimization to maximize mutual information between registered and fixed images. No training required. |  |  |

### Deep Learning Baselines

| Model | Type | Status |
|-------|------|--------|
| **NEMAR (Falcon)** | Adversarial + Contrastive + Translation | ✅ Implemented |
| **VoxelMorph** | UNet Registration | ✅ Implemented |
| **TransMorph** | Swin Transformer Registration | ✅ Implemented |
| **ConvexAdam** | Optimization-based (MIND) | ✅ Implemented |
| **DINO-Reg** | DINOv2 + ConvexAdam | ✅ Implemented |

### Comparison Commands

```bash
# TransMorph training
python train.py --dataroot ./datasets/RIRE_2d_paired --name transmorph_rire \
    --model transmorph --dataset_mode aligned_2d --direction AtoB \
    --gpu_ids 0 --batch_size 1 --niter 200 --niter_decay 0 \
    --img_height 512 --img_width 512 --preprocess none --no_flip \
    --sim_loss mind --input_nc 1 --output_nc 1

# Evaluate TransMorph
python scripts/eval_baselines.py --method transmorph --name transmorph_rire \
    --dataset rire --gpu 0

# ConvexAdam (no training needed)
python scripts/run_convexadam.py --dataset rire --gpu 0

# DINO-Reg (no training needed)
python scripts/run_dinoreg.py --dataset rire --gpu 0
```

## Visualization

- **Visdom**: Display training progress (port 8097 by default)
- **TensorBoard**: Enable with `--enable_tbvis`
- **HTML**: Saved to `./checkpoints/[name]/web/`

## Important Notes

### RIRE Dataset Experiments

Based on `RIRE_Experiment_Report.md`, 200 epochs training shows:
- **Best configuration**: `stn_type=ukan_gbcm_contrastive` with contrastive weight 0.2
- **Key insights**:
  - UKAN-STN outperforms affine and simple UNET STNs significantly
  - GBCM (Gradient-Based Cross-Modality) provides ~18% improvement
  - Contrastive learning with weight 0.2 is optimal (vs 0.0, 0.05, 0.3)
  - Discriminator regularization (label smoothing + noise) is beneficial

### Multi-Resolution Training

For large images, enable multi-resolution discriminator:
```bash
--multi_resolution 3  # Uses 3 discriminators at different scales
```

### STN-Specific Parameters

Some STN types have additional parameters:
- `--stn_bilateral_alpha`: Bilateral filtering coefficient for smoothness
- `--stn_no_identity_init`: Skip identity initialization
- `--ukan_embed_dims`: UKAN embedding dimensions (default [64, 128, 256])
- `--ukan_depths`: UKAN depths (default [1, 1, 1])
- `--contrastive_weight`: Contrastive loss weight (default 0.1)
- `--contrastive_temperature`: Contrastive temperature (default 0.07)

### Common Issues

1. **GPU memory**: Reduce `img_height`/`img_width` or use `batch_size=1`
2. **STN grid issues**: Ensure image dimensions match expected sizes for chosen STN
3. **NaN losses**: Check learning rate, gradient clipping, or reduce `lambda_GAN`
4. **RIRE data**: Must convert 3D NIfTI to 2D PNG slices first (see `scripts/convert_rire_to_2d.py`)

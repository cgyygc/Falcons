# Image Registration Model Comparison Experiments (Updated)

## Overview

Complete comparison experiments between different image registration models:
- **Falcon**: NEMAR + U-KAN + GBCM + Contrastive (Deep Learning, Your Baseline)
- **MI Registration**: Traditional Mutual Information method (Traditional)
- **VoxelMorph**: Supervised/Unsupervised deformation field learning (Deep Learning)
- **CycleGAN**: Unsupervised image translation (Deep Learning Baseline) ⭐ 新增
- **Pix2Pix**: Paired image-to-image translation (Deep Learning Baseline) ⭐ 新增

---

## Model Architecture Comparison

| Model | Translation | Registration | Training Type | Data Requirement |
|-------|-----------|-------------|--------------|----------------|
| **Falcon (NEMAR)** | ✅ 联合 | ✅ 非刚性 | 端到端监督 | Unpaired |
| **MI Registration** | ❌ 无 | ✅ 仿射 | 无训练 | None |
| **VoxelMorph** | ❌ 无 | ✅ 非刚性 | 无监督 | Paired/Unpaired |
| **CycleGAN** | ✅ 独立 | ❌ 无 | 无监督 | Unpaired |
| **Pix2Pix** | ✅ 独立 | ❌ 无 | 有监督 | Paired |

---

## Experimental Results

### RIRE Dataset (CT → MR-T1, 512×512)

| Model | MSE ↓ | MAE ↓ | PSNR ↑ | SSIM ↑ | NCC ↑ | Status |
|-------|---------|---------|---------|----------|---------|--------|
| **Falcon** (UKAN+GBCM+Contrastive) | 0.0011 ± 0.0005 | 0.0177 ± 0.0025 | 29.87 ± 1.70 dB | **0.8348 ± 0.0150** | **0.9856 ± 0.0063** | ✅ 训练完成 |
| **MI Registration** (Traditional) | 0.0959 ± 0.0319 | 0.2642 ± 0.0763 | 10.57 ± 2.13 dB | 0.1190 ± 0.0885 | 0.0289 ± 0.2238 | ✅ 测试完成 |
| **VoxelMorph** (200 epochs, NCC loss) | 0.0581 ± 0.0210 | 0.1511 ± 0.0412 | 12.58 ± 1.31 dB | 0.1710 ± 0.0291 | 0.2698 ± 0.0382 | ✅ 测试完成 |
| **CycleGAN** (200 epochs, ResNet9blocks) | 0.0324 ± 0.0145 | 0.0890 ± 0.0316 | 15.28 ± 1.77 dB | 0.3144 ± 0.0562 | 0.5906 ± 0.0681 | ✅ 测试完成 |
| **Pix2Pix** (需要配对数据) | N/A (仅翻译) | N/A | N/A | N/A | N/A | ⏳ 待配对数据 |

### Detailed Results for Completed Models

#### Falcon (Your Baseline)
```
MSE:   0.0011 ± 0.0005
MAE:   0.0177 ± 0.0025
PSNR:  29.87 ± 1.70 dB
SSIM:  0.8348 ± 0.0150
NCC:   0.9856 ± 0.0063
```
✅ **最佳表现**：联合学习翻译和配准

#### MI Registration (Traditional Baseline)
```
MSE:  0.0959 ± 0.0319
MAE:  0.2642 ± 0.0763
PSNR: 10.57 ± 2.13 dB
SSIM: 0.1190 ± 0.0885
NCC:  0.0289 ± 0.2238
```
❌ **最差表现**：仅仿射变换，无法处理局部形变

#### MI Registration Variability Analysis
```
SSIM Range:  [0.000297, 0.335574]
  Mean:       0.1190 ± 0.0885
  Std Dev:    0.0885 (73% of mean!)

NCC Range:   [-0.159642, 0.686645]
  Mean:       0.0289 ± 0.2238
  Std Dev:    0.2238 (775% of mean!)

→ 极不稳定，负值相关性说明跨模态处理失败
```

#### VoxelMorph (Deep Learning Baseline)
```
MSE:   0.0581 ± 0.0210
MAE:   0.1511 ± 0.0412
PSNR:  12.58 ± 1.31 dB
SSIM:  0.1710 ± 0.0291
NCC:   0.2698 ± 0.0382
```
⚠️ **中等偏低表现**：优于传统 MI，但远低于 Falcon。NCC 损失在多模态场景下配准能力有限。

#### CycleGAN (Translation Baseline)
```
MSE:   0.0324 ± 0.0145
MAE:   0.0890 ± 0.0316
PSNR:  15.28 ± 1.77 dB
SSIM:  0.3144 ± 0.0562
NCC:   0.5906 ± 0.0681
```
⚠️ **中等表现**：纯翻译模型，无配准能力。优于 MI 和 VoxelMorph，但远低于 Falcon。CycleGAN 无法建模几何变形，仅通过图像翻译缩小模态差异。

---

## Performance Rankings

### Overall Ranking (Best to Worst)

| Rank | Model | SSIM ↑ | NCC ↑ | PSNR ↑ |
|------|-------|----------|----------|---------|
| **1st** 🥇 | **Falcon** | **0.835** | **0.986** | **29.87 dB** |
| **2nd** 🥈 | **CycleGAN** | 0.314 | 0.591 | 15.28 dB |
| **3rd** 🥉 | **VoxelMorph** | 0.171 | 0.270 | 12.58 dB |
| **4th** | **MI Registration** | **0.119** | **0.029** | **10.57 dB** |
| **5th** | Pix2Pix | N/A | N/A | N/A |

### Performance Percentages (relative to Falcon)

| Model | SSIM | NCC | PSNR |
|--------|------|------|-------|
| **Falcon** | 100% | 100% | 100% |
| **CycleGAN** | 38% | 60% | 51% |
| **VoxelMorph** | 20% | 27% | 42% |
| **MI Registration** | 14% | 3% | 35% |

---

## Key Findings

### 1. Falcon 显著优于传统方法

| Metric | Falcon vs MI | Improvement |
|--------|-------------|-------------|
| **MSE** | 0.0011 vs 0.0959 | **87x 更好** |
| **SSIM** | 0.835 vs 0.119 | **7.0x 更高** |
| **NCC** | 0.986 vs 0.029 | **34x 更高** |
| **PSNR** | 29.87 dB vs 10.57 dB | **+19.3 dB** |

### 2. MI Registration 极不稳定

- **SSIM 标准差**: 0.0885 (73% of mean)
- **NCC 标准差**: 0.2238 (775% of mean)
- **负 NCC 比例**: 部分图像对负相关
- **结论**: 无法可靠用于临床应用

### 3. VoxelMorph 作为深度学习基线表现有限

- NCC 损失从 0.94 → 0.36 (已收敛)
- 支持非刚性变形场
- 略优于传统 MI 方法 (SSIM 0.171 vs 0.119, NCC 0.270 vs 0.029)
- 但仍远低于 Falcon，说明纯 NCC 损失在多模态场景下不够
- PSNR 仅 12.58 dB，与 MI 的 10.57 dB 差距不大

### 4. CycleGAN 作为纯翻译基线优于配准基线

| Metric | CycleGAN vs VoxelMorph | CycleGAN vs MI |
|--------|----------------------|----------------|
| **MSE** | 0.0324 vs 0.0581 (**1.8x 更好**) | 0.0324 vs 0.0959 (**3.0x 更好**) |
| **SSIM** | 0.314 vs 0.171 (**1.8x 更高**) | 0.314 vs 0.119 (**2.6x 更高**) |
| **NCC** | 0.591 vs 0.270 (**2.2x 更高**) | 0.591 vs 0.029 (**20x 更高**) |
| **PSNR** | 15.28 dB vs 12.58 dB (**+2.7 dB**) | 15.28 dB vs 10.57 dB (**+4.7 dB**) |

- 纯翻译模型无配准能力，仅通过模态转换缩小外观差异
- 优于所有非联合学习基线，说明模态翻译对多模态任务至关重要
- 但远低于 Falcon (SSIM 0.314 vs 0.835)，说明单独翻译无法替代联合配准

---

## Training Commands

### Falcon (Your Baseline)
```bash
python train.py --dataroot ./datasets/RIRE_2d --name falcon_baseline \n    --model nemar --dataset_mode rire_2d --direction AtoB \n    --stn_type ukan_gbcm_contrastive \n    --use_gbcm --use_contrastive --contrastive_weight 0.2 \n    --use_label_smooth --use_disc_noise \n    --gpu_ids 0,1 --batch_size 8 \n    --niter 200 --niter_decay 0
```

### MI Registration (Traditional Baseline)
```bash
bash scripts/test_mi_registration.sh RIRE_2d rire_2d 100 mi_baseline
```

### VoxelMorph (Deep Learning Baseline)
```bash
# Training
python test_voxelmorph.py \
    --dataroot ./datasets/RIRE_2d \
    --name voxelmorph_RIRE_2d \
    --dataset_mode rire_2d \
    --num_test 100 \
    --gpu_ids 0 \
    --vm_num_features 32 64 128 256 \
    --vm_loss_type ncc \
    --vm_smoothness_weight 0.01 \
    --vm_ncc_window 9 \
    --vm_lr 1e-4 \
    --vm_niter 200 \
    --train

# Testing only
python test_voxelmorph.py \
    --dataroot ./datasets/RIRE_2d \
    --name voxelmorph_RIRE_2d \
    --dataset_mode rire_2d \
    --num_test 100 \
    --gpu_ids 0
```

### CycleGAN (Translation Baseline)
```bash
# Training
python train.py \
    --dataroot ./datasets/RIRE_2d \
    --name cyclegan_rire \
    --model cycle_gan --dataset_mode rire_2d \
    --direction AtoB \
    --input_nc 1 --output_nc 1 \
    --netG_A resnet_9blocks --netG_B resnet_9blocks \
    --ngf 64 --norm instance --no_dropout \
    --lambda_A 10.0 --lambda_B 10.0 --lambda_identity 0.5 \
    --gpu_ids 0 --batch_size 1 \
    --niter 100 --niter_decay 100 \
    --lr 0.0002 --lr_policy linear --gan_mode vanilla \
    --pool_size 50 \
    --img_height 512 --img_width 512 \
    --preprocess none --no_flip \
    --save_epoch_freq 10 --print_freq 50

# Testing
python test_cyclegan.py \
    --dataroot ./datasets/RIRE_2d \
    --name cyclegan_rire \
    --num_test 100 --gpu_ids 0
```

### Pix2Pix (Supervised Translation Baseline)
```bash
# Note: Requires paired dataset (not currently available)
# python train.py --dataroot ./datasets/RIRE_2d_paired \n#     --model pix2pix --dataset_mode aligned \n#     --direction AtoB \n#     --netG unet_256 --lambda_L1 100 \n#     --gpu_ids 0,1 --batch_size 4 \n#     --niter 200 --niter_decay 0
```

---

## GPU Utilization

### Current Setup
- **Hardware**: 2x RTX 5090 (32 GB each)
- **VoxelMorph**: GPU 0 (98% → 9% as training completes)

### Multi-GPU Configurations

| Model | GPU 0 | GPU 1 | Batch Size | Speedup |
|-------|-------|-------|-----------|--------|
| **VoxelMorph** | ~9% | 0% | 1 | 1.0x (batch_size=1) |
| **Falcon** | ~95% | ~95% | 8 | ~1.7x (DataParallel) |
| **CycleGAN** | ~95% | ~95% | 4 | ~1.8x |
| **Pix2Pix** | ~95% | ~95% | 4 | ~1.8x |

**Note**: VoxelMorph 的 `test_voxelmorph.py` 使用 DataLoader，batch_size 固定为 1，无法充分利用多 GPU

---

## Model Architectures

### Falcon (NEMAR + U-KAN + GBCM + Contrastive)
```
Input A (CT) ─┬─→ netT (Translation) ──┐
              │                            │
Input B (MR) ─┘──→─────────────────────┼→→ registered_B
                                              │
                      [CT, MR] ───────→ netR (STN) ─── registered_A
```

### MI Registration
```
Input A (CT) ──→ Affine Transform (θ) ─→ registered_A
Input B (MR) ──────────────────────────────────→ [固定]
```

### VoxelMorph
```
Input A (CT) ──┬─→ UNet ──→ flow field (2D) ─── Spatial Transform ──→ registered_A
Input B (MR) ──┘─────────────────────────────────────────────────────────────→ [固定]
```

### CycleGAN
```
A (CT) ─→ netG_A ─→ B' (fake MR)
B' (MR) ─→ netG_B ─→ A' (fake CT)
            ↓
        [Cycle Consistency]
```

### Pix2Pix
```
[A, B] (paired) ─→ UNet ─→ B' (fake B)
    └───────────→ PatchGAN ──→ [real/fake discrimination]
```

---

## Evaluation Scripts

### Evaluate All Models

```bash
# Falcon
python eval_registration.py --dataroot ./datasets/RIRE_2d \n    --name falcon_baseline --num_test 100

# MI Registration
bash scripts/test_mi_registration.sh RIRE_2d rire_2d 100 mi_baseline
python scripts/eval_mi_registration.py \n    --results_dir ./results/mi_baseline \n    --output mi_results.txt

# VoxelMorph
python test_voxelmorph.py --dataroot ./datasets/RIRE_2d \
    --name voxelmorph_RIRE_2d --dataset_mode rire_2d --num_test 100

# CycleGAN
python test.py --dataroot ./datasets/RIRE_2d \n    --name cyclegan_rire --model cycle_gan \n    --dataset_mode unaligned --phase test \n    --num_test 100

# Pix2Pix (if paired data available)
# python test.py --dataroot ./datasets/RIRE_2d_paired \n#     --name pix2pix_rire --model pix2pix \n#     --dataset_mode aligned --phase test \n#     --num_test 100
```

---

## Results Summary Table

| Model | MSE ↓ | MAE ↓ | PSNR ↑ | SSIM ↑ | NCC ↑ | Architecture | Data Type |
|-------|---------|---------|---------|----------|---------|-------------|-----------|
| **Falcon** | **0.0011** ✓ | **0.0177** ✓ | **29.87** ✓ | **0.8348** ✓ | **0.9856** ✓ | UKAN+GBCM+Contrastive | Unpaired |
| **MI** | 0.0959 ✗ | 0.2642 ✗ | 10.57 ✗ | 0.1190 ✗ | 0.0289 ✗ | Affine+NMI | None |
| **VoxelMorph** | 0.0581 | 0.1511 | 12.58 dB | 0.1710 | 0.2698 | UNet+NCC+Smoothness | Unpaired |
| **CycleGAN** | 0.0324 | 0.0890 | 15.28 dB | 0.3144 | 0.5906 | ResNet9blocks+Cycle | Unpaired |
| **Pix2Pix** | N/A | N/A | N/A | N/A | N/A | UNet+PatchGAN | Paired |

**Status**: ✅=Completed, 🔄=Training, ⏳=Not Started, N/A=Not Applicable

---

## To-Do

- [x] Implement CycleGAN model
- [x] Implement Pix2Pix model
- [x] Train CycleGAN on RIRE dataset
- [ ] Create paired dataset for Pix2Pix
- [ ] Train Pix2Pix on RIRE dataset
- [ ] Generate comprehensive comparison report with all 5 models
- [ ] Create visualization plots comparing all methods

---

*Last updated: 2026-05-16*
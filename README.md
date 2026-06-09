# Falcons: Neural Multimodal Adversarial Registration

Cross-modal image registration using adversarial training with UKAN-based spatial transformer networks.

## Architecture

Three-network design:
- **T**: Modality translation network (ResNet generator)
- **R**: Registration network (UKAN3D-STN)
- **D**: Adversarial discriminator

Dual-path training:
- **TR path**: real_A → R → registered_A → T → fake_TR_B
- **RT path**: real_A → T → fake_B → R → fake_RT_B

## Quick Start

### 2D Registration (RIRE)

```bash
python train.py --dataroot ./datasets/rire --name falcon_rire \
    --model nemar --dataset_mode unaligned --direction AtoB \
    --stn_type ukan_gbcm_contrastive --img_height 288 --img_width 384 \
    --niter 200 --niter_decay 0 --contrastive_weight 0.2
```

### 3D Registration (L2R)

```bash
# Single GPU
python train_3d.py --model nemar3d --name falcon3d_l2r --niter 1000 --use_amp

# Multi-GPU (4 GPUs)
python train_3d.py --model nemar3d --name falcon3d_l2r --niter 1000 --use_amp --gpu_ids 0 1 2 3
```

### 3D Registration (IXI)

```bash
python train_ixi.py --model nemar3d --name falcon3d_ixi --niter 200 --use_amp \
    --gpu_ids 0 1 2 3 --target_shape 128 128 128
```

### Evaluation

```bash
python eval_registration.py --dataroot ./datasets/rire --name falcon_rire --num_test 100
python eval_dsc_3d.py --name falcon3d_l2r
```

## Results

### 2D RIRE (Brain CT↔MR)

| Method | SSIM ↑ | NCC ↑ | PSNR ↑ |
|--------|---------|--------|---------|
| **Falcon (Ours)** | **0.8914** | **0.9902** | **30.98** |
| TransMorph | 0.6986 | 0.6250 | 16.09 |
| ConvexAdam | 0.6518 | 0.6168 | 15.52 |

### 3D L2R (Abdominal CT↔MR)

| Method | DSC ↑ |
|--------|-------|
| **Falcon3D (Ours)** | **0.5403** |
| VoxelMorph3D | 0.3955 |
| TransMorph3D | 0.340 |

## Project Structure

```
├── models/
│   ├── nemar3d_model.py       # Falcon3D (T+R+D)
│   ├── voxelmorph3d_model.py   # VoxelMorph3D baseline
│   ├── transmorph3d_model.py   # TransMorph3D baseline
│   ├── nemar_model.py          # 2D Falcon (legacy)
│   └── stn/
│       ├── ukan3d_stn.py       # UKAN3D spatial transformer
│       ├── ukan_stn.py         # UKAN spatial transformer
│       ├── contrastive_stn.py  # Contrastive STN
│       └── spatial_transformer_3d.py  # 3D transforms and losses
├── data/
│   ├── l2r_3d_dataset.py       # L2R 3D dataset
│   ├── ixi_3d_dataset.py       # IXI 3D dataset
│   └── rire_2d_dataset.py      # RIRE 2D dataset
├── train_3d.py                 # 3D training (multi-GPU)
├── train_ixi.py                # IXI training (multi-GPU)
├── train_ixi_direct.py         # R-only training (no T/D)
├── train.py                    # 2D training
├── generate_registration_vis.py # Paper figure generation
└── eval_registration.py        # Evaluation script
```

## Dependencies

- PyTorch >= 1.12
- nibabel, scikit-image, scipy
- matplotlib, dominate (for visualization)

# MI Registration - Traditional Baseline

This module implements a traditional Mutual Information (MI) based image registration method as a baseline comparison for NEMAR.

## Overview

MI Registration is a classical multi-modal image registration technique that maximizes the mutual information between the registered moving image and fixed image. It serves as a traditional baseline to compare against the deep learning-based NEMAR approach.

## Key Features

- **Mutual Information Loss**: Histogram-based MI estimation with Parzen window
- **Affine Transformation**: Supports rigid, similarity, and affine transformations
- **L-BFGS Optimization**: Second-order optimization for efficient convergence
- **Smoothness Regularization**: Penalizes large transformation parameters
- **No Training Required**: Traditional iterative optimization approach

## Model Architecture

```
Moving Image (A)
       ↓
Affine Transformation (θ = [scale_x, scale_y, shear, rotation, tx, ty])
       ↓
Registered Moving Image (A_reg)
       ↓
MI Loss (A_reg, Fixed Image B) + Smoothness Loss (θ)
       ↓
Optimize θ with L-BFGS
```

## Installation

The MI registration model is included in the main NEMAR repository. No additional dependencies required beyond the standard NEMAR requirements.

## Usage

### Quick Start

```bash
# Test MI registration on RIRE dataset
bash scripts/test_mi_registration.sh rire rire_2d 100 mi_registration_rire

# Run the example script
bash scripts/run_mi_registration_example.sh
```

### Manual Usage

```bash
# Test MI registration
python test_mi.py \
    --dataroot ./datasets/rire \
    --name mi_registration_test \
    --model mi_registration \
    --dataset_mode rire_2d \
    --num_test 100 \
    --mi_num_bins 64 \
    --mi_sigma 2.0 \
    --lambda_mi_smooth 0.1 \
    --optim_lr 1.0 \
    --optim_max_iter 100 \
    --transform_type affine
```

### Evaluate Results

```bash
# Evaluate registration quality metrics
python scripts/eval_mi_registration.py \
    --results_dir ./results/mi_registration_test \
    --output ./results/mi_registration_test/metrics.txt
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--mi_num_bins` | int | 64 | Number of bins for MI histogram |
| `--mi_sigma` | float | 2.0 | Sigma for Parzen window estimation |
| `--lambda_mi_smooth` | float | 0.1 | Smoothness regularization weight |
| `--optim_lr` | float | 1.0 | Learning rate for L-BFGS |
| `--optim_max_iter` | int | 100 | Maximum iterations per image |
| `--optim_tolerance` | float | 1e-5 | Convergence tolerance |
| `--optim_history_size` | int | 100 | History size for L-BFGS |
| `--transform_type` | str | affine | Transformation type: affine, similarity, rigid |

## Transformation Types

1. **Rigid**: Scale fixed (1, 1), shear fixed (0), rotation and translation free
   ```
   θ = [1.0, 1.0, 0.0, rotation, tx, ty]
   ```

2. **Similarity**: Scale uniform (s, s), shear fixed (0), rotation and translation free
   ```
   θ = [s, s, 0.0, rotation, tx, ty]
   ```

3. **Affine**: All parameters free (full affine)
   ```
   θ = [scale_x, scale_y, shear, rotation, tx, ty]
   ```

## Loss Function

The total loss is:

```
L_total = L_MI + λ_smooth * L_smoothness
```

Where:
- `L_MI`: Negative Mutual Information (to be minimized)
- `L_smoothness`: L2 regularization on transformation parameters

## Evaluation Metrics

The evaluation script computes:
- **MSE**: Mean Squared Error
- **MAE**: Mean Absolute Error
- **PSNR**: Peak Signal-to-Noise Ratio (dB)
- **SSIM**: Structural Similarity Index
- **NCC**: Normalized Cross-Correlation

## Comparison with NEMAR

| Aspect | MI Registration | NEMAR |
|--------|----------------|-------|
| **Approach** | Traditional | Deep Learning |
| **Training** | None required | Required (iterative) |
| **Inference** | Slow (iterative optimization) | Fast (single forward pass) |
| **Transformation** | Affine only | Non-rigid deformations |
| **Similarity** | Mutual Information | Adversarial + Contrastive |
| **Scalability** | O(n) per image | O(1) per image |
| **Best For** | Simple global alignments | Complex local deformations |

## Expected Results

### RIRE Dataset (CT → MR)

Based on traditional MI registration literature:
- **MSE**: ~0.002-0.003 (higher than NEMAR)
- **SSIM**: ~0.60-0.70 (lower than NEMAR)
- **NCC**: ~0.90-0.92 (lower than NEMAR)

NEMAR typically outperforms MI registration by 20-30% on non-rigid deformations due to its ability to learn complex, local transformations.

## File Structure

```
models/
└── mi_registration_model.py    # MI registration model implementation

scripts/
├── test_mi_registration.sh      # Testing script
├── eval_mi_registration.py      # Evaluation script
└── run_mi_registration_example.sh  # Example script

results/
└── mi_registration_*/           # Results directory
    ├── test_latest/
    │   └── images/
    │       ├── real_A_*.png
    │       ├── real_B_*.png
    │       └── registered_A_*.png
    └── mi_eval_results.txt     # Evaluation metrics
```

## References

1. Pluim, J. P., Maintz, J. B., & Viergever, M. A. (2001). "Mutual information matching in multiresolution contexts". *Image and Vision Computing*, 19(1-2), 45-52.

2. Viola, P., & Wells III, W. M. (1997). "Alignment by maximization of mutual information". *International Journal of Computer Vision*, 24(2), 137-154.

3. Thevenaz, P., & Unser, M. (1998). "A pyramid approach to subpixel registration based on intensity". *IEEE Transactions on Image Processing*, 7(1), 27-41.

## Notes

- MI registration is computationally expensive per image (100+ iterations)
- Best for global affine alignments, not suitable for non-rigid deformations
- Can be used as initialization for more sophisticated registration methods
- Results may vary based on image quality and initial alignment

## Future Improvements

1. **Multi-resolution Optimization**: Implement pyramid-based optimization for faster convergence
2. **B-spline Deformations**: Add support for non-rigid B-spline transformations
3. **Similarity Metrics**: Add other metrics like NMI (Normalized Mutual Information)
4. **GPU Acceleration**: Optimize histogram computation for GPU
5. **Hybrid Approach**: Combine MI with deep learning for better initialization

"""
Test script to debug Jacobian folding rate computation
"""

import os
import sys
import torch
import numpy as np
from options.test_options import TestOptions
from data import create_dataset
from models import create_model

# Compute Jacobian folding
def compute_jacobian_folding(disp_field):
    """
    Compute folding rate of deformation field using Jacobian determinant.
    """
    if disp_field.dim() == 4:  # (B, C, H, W)
        if disp_field.size(1) >= 2:
            # Gradient in y direction
            dy = disp_field[:, :, 1:, :] - disp_field[:, :, :-1, :]
            dy = torch.nn.functional.pad(dy, (0, 0, 0, 1), 'replicate')

            # Gradient in x direction
            dx = disp_field[:, :, :, 1:] - disp_field[:, :, :, :-1]
            dx = torch.nn.functional.pad(dx, (0, 1, 0, 0), 'replicate')

            # Jacobian determinant
            if disp_field.size(1) == 2:
                dxdy = dx[:, 0:1, :, :]
                dxdx = dx[:, 1:2, :, :]
                dydy = dy[:, 0:1, :, :]
                dydx = dy[:, 1:2, :, :]

                jac_det = (1 + dxdx) * (1 + dydy) - dxdy * dydx
                folding_rate = (jac_det < 0).float().mean().item()
                return folding_rate
    return 0.0

def main():
    # Parse options
    opt = TestOptions().parse()

    # Hard-code some options for evaluation
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1

    # Set model-specific options
    if not hasattr(opt, 'use_contrastive'):
        opt.use_contrastive = False

    if opt.stn_type == 'ukan_contrastive':
        opt.use_contrastive = True
        if not hasattr(opt, 'contrastive_weight'):
            opt.contrastive_weight = 0.1
        if not hasattr(opt, 'contrastive_temperature'):
            opt.contrastive_temperature = 0.07

    if not hasattr(opt, 'stn_bilateral_alpha'):
        opt.stn_bilateral_alpha = 0.0
    if not hasattr(opt, 'stn_no_identity_init'):
        opt.stn_no_identity_init = False
    if not hasattr(opt, 'stn_multires_reg'):
        opt.stn_multires_reg = 1
    if not hasattr(opt, 'ukan_embed_dims'):
        opt.ukan_embed_dims = [64, 128, 256]
    if not hasattr(opt, 'ukan_depths'):
        opt.ukan_depths = [1, 1, 1]

    if isinstance(opt.no_flip, str):
        opt.no_flip = opt.no_flip.lower() == 'true'

    # Create dataset
    dataset = create_dataset(opt)
    print(f"Dataset created: {len(dataset)} images")

    # Create model
    model = create_model(opt)
    model.setup(opt)
    print(f"Model [{model.model_names}] created")

    # Load latest checkpoint
    if opt.load_iter != 0 and opt.load_iter != 'latest' and opt.load_iter != '0':
        model.load_networks(opt.load_iter)

    # Test on first image
    print("\nTesting on first image...")
    data_iter = iter(dataset)
    data = next(data_iter)
    model.set_input(data)
    model.test()

    real_A = model.real_A
    real_B = model.real_B

    # Access STN
    if hasattr(model.netR, 'module'):
        stn_wrapper = model.netR.module
    else:
        stn_wrapper = model.netR

    print(f"\nSTN wrapper type: {type(stn_wrapper)}")

    # Check if this is a ContrastiveSTNWrapper
    if hasattr(stn_wrapper, 'stn'):
        print("  -> This is a ContrastiveSTNWrapper, accessing underlying STN...")
        stn = stn_wrapper.stn
        print(f"  -> Underlying STN type: {type(stn)}")
    else:
        stn = stn_wrapper

    print(f"STN methods: {[m for m in dir(stn) if not m.startswith('_')]}")

    # Try different methods to get deformation field
    print("\n--- Trying to get deformation field ---")

    # Method 1: offset_map
    if hasattr(stn, 'offset_map'):
        print("✓ STN has offset_map method")
        try:
            deformation = stn.offset_map(real_A, real_B)
            print(f"  offset_map output shape: {deformation.shape}")
            print(f"  offset_map output range: [{deformation.min():.4f}, {deformation.max():.4f}]")

            if deformation.size(1) == 2:
                print("  Using as displacement field [B, 2, H, W]")
                disp_field = deformation
            elif deformation.size(1) == 4:
                print("  Full grid detected, extracting displacement [B, 2, H, W]")
                disp_field = deformation[:, 2:, :, :]
                print(f"  Extracted displacement shape: {disp_field.shape}")

            # Compute Jacobian folding rate
            folding_rate = compute_jacobian_folding(disp_field)
            print(f"\n✓ Jacobian Folding Rate: {folding_rate:.4%}")
            print(f"  (Percentage of pixels with negative Jacobian determinant)")

        except Exception as e:
            import traceback
            print(f"✗ Error calling offset_map: {e}")
            print(f"  {traceback.format_exc()}")
    else:
        print("✗ STN does NOT have offset_map method")

    # Method 2: get_grid
    if hasattr(stn, 'get_grid'):
        print("\n✓ STN has get_grid method")
        try:
            # Try with return_offsets_only
            grid = stn.get_grid(real_A, real_B, return_offsets_only=True)
            print(f"  get_grid(output_only=True) shape: {grid.shape}")

            if grid.dim() == 4 and grid.size(1) == 2:
                print("  Using as displacement field [B, 2, H, W]")
                disp_field = grid

                # Compute Jacobian folding rate
                folding_rate = compute_jacobian_folding(disp_field)
                print(f"\n✓ Jacobian Folding Rate: {folding_rate:.4%}")
                print(f"  (Percentage of pixels with negative Jacobian determinant)")
            else:
                print(f"  Unexpected shape: {grid.shape}, dim={grid.dim()}")
        except Exception as e:
            import traceback
            print(f"✗ Error calling get_grid: {e}")
            print(f"  {traceback.format_exc()}")

            # Try without return_offsets_only
            try:
                grid = stn.get_grid(real_A, real_B, return_offsets_only=False)
                print(f"  get_grid(return_offsets_only=False) shape: {grid.shape}")
            except Exception as e2:
                print(f"✗ Error: {e2}")
    else:
        print("✗ STN does NOT have get_grid method")

    print("\n" + "="*70)

if __name__ == '__main__':
    main()

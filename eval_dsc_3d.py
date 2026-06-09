"""
3D DSC Evaluation for volumetric registration models.

Warps 3D segmentation masks using F.grid_sample(mode='nearest') with 5D tensors,
then computes Dice Similarity Coefficient per organ label.
"""
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib


def warp_segmentation_3d(seg, flow, img_size=(160, 192, 192)):
    """Warp 3D segmentation using nearest-neighbor interpolation.

    Args:
        seg: [B, 1, D, H, W] segmentation mask (integer labels)
        flow: [B, 3, D, H, W] deformation field (channels: x/W, y/H, z/D)
        img_size: (H, W, D) spatial dimensions

    Returns:
        warped_seg: [B, 1, D, H, W] warped segmentation
    """
    B, C, D, H, W = seg.shape

    # Build identity grid
    grid_d, grid_h, grid_w = torch.meshgrid(
        torch.linspace(-1, 1, D, device=seg.device, dtype=seg.dtype),
        torch.linspace(-1, 1, H, device=seg.device, dtype=seg.dtype),
        torch.linspace(-1, 1, W, device=seg.device, dtype=seg.dtype),
        indexing='ij'
    )
    grid = torch.stack([grid_w, grid_h, grid_d], dim=-1)  # [D, H, W, 3]
    grid = grid.unsqueeze(0).expand(B, -1, -1, -1, -1)  # [B, D, H, W, 3]

    # Add flow
    new_grid = grid + flow.permute(0, 2, 3, 4, 1)  # [B, D, H, W, 3]

    # Nearest neighbor for discrete labels
    warped = F.grid_sample(seg.float(), new_grid, mode='nearest',
                           padding_mode='border', align_corners=True)
    return warped


def compute_dice(pred_seg, gt_seg, labels=None):
    """Compute DSC per label.

    Args:
        pred_seg: [D, H, W] numpy array of predicted labels
        gt_seg: [D, H, W] numpy array of ground truth labels
        labels: list of labels to evaluate (default: all unique in gt)

    Returns:
        dict: label -> DSC value
    """
    if labels is None:
        labels = sorted(set(np.unique(gt_seg)) - {0})

    dice_scores = {}
    for label in labels:
        pred_mask = (pred_seg == label)
        gt_mask = (gt_seg == label)
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        total = pred_mask.sum() + gt_mask.sum()
        if total > 0:
            dice_scores[label] = 2.0 * intersection / total
        else:
            dice_scores[label] = 0.0
    return dice_scores


def load_volume(path):
    """Load NIfTI volume and normalize to [-1, 1]."""
    img = nib.load(path)
    data = img.get_fdata().astype(np.float32)
    # Normalize per-volume
    vmin, vmax = data.min(), data.max()
    if vmax - vmin > 0:
        data = 2.0 * (data - vmin) / (vmax - vmin) - 1.0
    else:
        data = np.zeros_like(data)
    return data, img.affine, img.header


def load_seg(path):
    """Load segmentation NIfTI as integer labels."""
    img = nib.load(path)
    return img.get_fdata().astype(np.int32)


def get_model_and_flow(model_name, checkpoint_path, device, img_size=(160, 192, 192)):
    """Load a 3D registration model and return it."""
    if model_name == 'voxelmorph3d':
        from models.voxelmorph3d_model import VoxelMorph3DModel
        from types import SimpleNamespace
        opt = SimpleNamespace(
            vm3d_num_features=[32, 64, 128, 256],
            vm3d_use_dropout=False,
            vm3d_loss_type='mi',
            vm3d_smoothness_weight=1.0,
            vm3d_mi_bins=32,
            vm3d_lr=1e-4,
            vm3d_niter=500,
            use_amp=False,
            isTrain=False, gpu_ids=[], device=device,
            checkpoints_dir=os.path.dirname(checkpoint_path),
            name=os.path.basename(os.path.dirname(checkpoint_path)),
            preprocess='none', phase='test', epoch_count=1, verbose=False,
        )
        model = VoxelMorph3DModel(opt)
        model.load_networks('latest')
        return model

    elif model_name == 'transmorph3d':
        from models.transmorph3d_model import Transmorph3DModel
        from types import SimpleNamespace
        opt = SimpleNamespace(
            tm3d_enc_channels=[48, 96, 192, 384],
            tm3d_num_heads=6,
            tm3d_num_transformer_blocks=2,
            tm3d_sim_loss='mind',
            tm3d_reg_weight=1.0,
            tm3d_lr=1e-4,
            use_amp=False,
            isTrain=False, gpu_ids=[], device=device,
            checkpoints_dir=os.path.dirname(checkpoint_path),
            name=os.path.basename(os.path.dirname(checkpoint_path)),
            preprocess='none', phase='test', epoch_count=1, verbose=False,
        )
        model = Transmorph3DModel(opt)
        model.load_networks('latest')
        return model

    elif model_name == 'nemar3d':
        from models.nemar3d_model import NEMAR3DModel
        from types import SimpleNamespace
        H, W, D = img_size
        opt = SimpleNamespace(
            n3d_ngf=16, n3d_ndf=16, n3d_n_blocks=6,
            n3d_kan_embed_dims=[32, 64, 128], n3d_kan_depths=[1, 1, 1],
            n3d_lambda_gan=1.0, n3d_lambda_recon=100.0, n3d_lambda_smooth=1.0,
            n3d_gan_mode='vanilla', n3d_lr=1e-4, n3d_use_dropout=False,
            use_amp=False, vol_depth=D, img_height=H, img_width=W,
            isTrain=False, gpu_ids=[], device=device,
            checkpoints_dir=os.path.dirname(checkpoint_path),
            name=os.path.basename(os.path.dirname(checkpoint_path)),
            preprocess='none', phase='test', epoch_count=1, verbose=False,
        )
        model = NEMAR3DModel(opt)
        model.load_networks('latest')
        return model

    else:
        raise ValueError(f"Unknown model: {model_name}")


def predict_flow(model, model_name, moving, fixed, device):
    """Predict deformation field from a model.

    Args:
        model: loaded model
        model_name: str
        moving: [1, D, H, W] numpy array
        fixed: [1, D, H, W] numpy array
        device: torch device

    Returns:
        flow: [1, 3, D, H, W] torch tensor on device
    """
    moving_t = torch.from_numpy(moving).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,D,H,W]
    fixed_t = torch.from_numpy(fixed).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        if model_name == 'voxelmorph3d':
            x = torch.cat([moving_t, fixed_t], dim=1)
            flow = model.netV(x)
        elif model_name == 'transmorph3d':
            x = torch.cat([moving_t, fixed_t], dim=1)
            flow = model.netTM3D(x)
        elif model_name == 'nemar3d':
            flow = model.netR.get_grid(moving_t, fixed_t, return_offsets_only=False)
            # UKAN3DSTN returns [B, D, H, W, 3] grid - convert to flow format
            flow = flow.permute(0, 4, 1, 2, 3)  # [B, 3, D, H, W]
            # Subtract identity to get deformation only
            D, H, W = flow.shape[2:]
            grid_d, grid_h, grid_w = torch.meshgrid(
                torch.linspace(-1, 1, D, device=device),
                torch.linspace(-1, 1, H, device=device),
                torch.linspace(-1, 1, W, device=device),
                indexing='ij'
            )
            identity = torch.stack([grid_w, grid_h, grid_d], dim=0).unsqueeze(0).to(device)
            flow = flow - identity

    return flow


def main():
    parser = argparse.ArgumentParser(description='3D DSC Evaluation')
    parser.add_argument('--dataroot', type=str, required=True, help='L2R dataset root')
    parser.add_argument('--model', type=str, required=True,
                        choices=['voxelmorph3d', 'transmorph3d', 'nemar3d'])
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Directory containing model checkpoints')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--crop_size', type=int, default=0,
                        help='Crop volumes to this size (0=use full resolution)')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # Discover patient directories
    ct_dir = os.path.join(args.dataroot, 'ct')
    mr_dir = os.path.join(args.dataroot, 'mr')
    ct_seg_dir = os.path.join(args.dataroot, 'ct_seg')
    mr_seg_dir = os.path.join(args.dataroot, 'mr_seg')

    if not os.path.isdir(ct_dir):
        # Try alternative structure
        ct_dir = os.path.join(args.dataroot, 'trainA')
        mr_dir = os.path.join(args.dataroot, 'trainB')

    # Find patient files
    ct_files = sorted([f for f in os.listdir(ct_dir) if f.endswith('.nii.gz') or f.endswith('.nii')])
    mr_files = sorted([f for f in os.listdir(mr_dir) if f.endswith('.nii.gz') or f.endswith('.nii')])

    if len(ct_files) == 0:
        print(f"No NIfTI files found in {ct_dir}")
        return

    print(f"Found {len(ct_files)} CT volumes and {len(mr_files)} MR volumes")

    # Load model
    img_size = (160, 192, 192)  # Default L2R size
    model = get_model_and_flow(args.model, args.checkpoint_dir, device, img_size)
    model.eval()

    all_dice = {}

    for i, (ct_file, mr_file) in enumerate(zip(ct_files, mr_files)):
        patient_id = ct_file.replace('.nii.gz', '').replace('.nii', '')

        # Load volumes
        ct_vol, _, _ = load_volume(os.path.join(ct_dir, ct_file))
        mr_vol, _, _ = load_volume(os.path.join(mr_dir, mr_file))

        # Make same shape (take min across dims)
        min_d = min(ct_vol.shape[0], mr_vol.shape[0])
        min_h = min(ct_vol.shape[1], mr_vol.shape[1])
        min_w = min(ct_vol.shape[2], mr_vol.shape[2])
        ct_vol = ct_vol[:min_d, :min_h, :min_w]
        mr_vol = mr_vol[:min_d, :min_h, :min_w]

        # Optional cropping
        if args.crop_size > 0:
            s = args.crop_size
            ct_vol = ct_vol[:s, :s, :s]
            mr_vol = mr_vol[:s, :s, :s]

        # Predict flow
        flow = predict_flow(model, args.model, ct_vol, mr_vol, device)

        # Load segmentations if available
        ct_seg_path = os.path.join(ct_seg_dir, ct_file) if os.path.isdir(ct_seg_dir) else None
        mr_seg_path = os.path.join(mr_seg_dir, mr_file) if os.path.isdir(mr_seg_dir) else None

        if ct_seg_path and os.path.exists(ct_seg_path):
            ct_seg = load_seg(ct_seg_path)[:min_d, :min_h, :min_w]
            if args.crop_size > 0:
                ct_seg = ct_seg[:s, :s, :s]

            ct_seg_t = torch.from_numpy(ct_seg).unsqueeze(0).unsqueeze(0).float().to(device)
            warped_seg = warp_segmentation_3d(ct_seg_t, flow)
            warped_seg_np = warped_seg.squeeze().cpu().numpy().astype(np.int32)

            if mr_seg_path and os.path.exists(mr_seg_path):
                mr_seg = load_seg(mr_seg_path)[:min_d, :min_h, :min_w]
                if args.crop_size > 0:
                    mr_seg = mr_seg[:s, :s, :s]

                dice = compute_dice(warped_seg_np, mr_seg)
                all_dice[patient_id] = dice
                mean_dsc = np.mean(list(dice.values())) if dice else 0.0
                print(f"Patient {patient_id}: mean DSC = {mean_dsc:.4f} ({len(dice)} labels)")
            else:
                print(f"Patient {patient_id}: no MR segmentation found")
        else:
            print(f"Patient {patient_id}: no CT segmentation found")

    # Summary
    if all_dice:
        # Collect all unique labels
        all_labels = sorted(set().union(*[d.keys() for d in all_dice.values()]))

        print("\n" + "=" * 60)
        print("3D DSC Summary")
        print("=" * 60)

        label_dscs = {}
        for label in all_labels:
            scores = [d[label] for d in all_dice.values() if label in d]
            if scores:
                mean = np.mean(scores)
                label_dscs[label] = mean
                print(f"  Label {label:3d}: DSC = {mean:.4f} (n={len(scores)})")

        overall = np.mean(list(label_dscs.values())) if label_dscs else 0.0
        print(f"\n  Overall mean DSC: {overall:.4f}")
        print(f"  Patients: {len(all_dice)}")
        print(f"  Labels: {len(label_dscs)}")


if __name__ == '__main__':
    main()

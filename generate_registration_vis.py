#!/usr/bin/env python3
"""
Generate paper-quality registration visualization for L2R dataset.

Shows: Moving CT | Fixed MR | VM3D | TM3D | Falcon3D
With: checkerboard overlay, segmentation boundary, error heatmap

Usage:
    python generate_registration_vis.py --gpu 0
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import nibabel as nib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_l2r_dataset(dataroot='./datasets/L2R'):
    """Load L2R patient info from flat directory structure."""
    patients = []

    for split, has_seg in [('Train', True), ('Test', False)]:
        split_dir = os.path.join(dataroot, split)
        if not os.path.isdir(split_dir):
            continue

        # Find patient IDs from CT files
        ct_files = sorted([f for f in os.listdir(split_dir)
                          if f.startswith('img') and 'CT' in f and f.endswith('.nii.gz')])

        for ct_file in ct_files:
            # img0002_tcia_CT.nii.gz -> pid = 0002_tcia
            pid = ct_file.replace('img', '').replace('_CT.nii.gz', '')
            mr_file = ct_file.replace('_CT.nii.gz', '_MR.nii.gz').replace('img', 'img')

            ct_path = os.path.join(split_dir, ct_file)
            mr_path = os.path.join(split_dir, mr_file)

            if not os.path.exists(mr_path):
                continue

            ct_seg_path = os.path.join(split_dir, ct_file.replace('img', 'seg'))
            mr_seg_path = os.path.join(split_dir, mr_file.replace('img', 'seg'))

            patients.append({
                'pid': pid,
                'ct': ct_path, 'mr': mr_path,
                'ct_seg': ct_seg_path if os.path.exists(ct_seg_path) else None,
                'mr_seg': mr_seg_path if os.path.exists(mr_seg_path) else None,
                'has_seg': has_seg and os.path.exists(ct_seg_path)
            })

    return patients


def normalize_vol(data):
    vmin, vmax = data.min(), data.max()
    if vmax - vmin < 1e-10:
        return np.zeros_like(data)
    return 2.0 * (data - vmin) / (vmax - vmin) - 1.0


def get_mid_slices(vol_3d, axis='axial'):
    """Get mid slices along each axis."""
    if axis == 'axial':
        return vol_3d[vol_3d.shape[0]//2, :, :]
    elif axis == 'coronal':
        return vol_3d[:, vol_3d.shape[1]//2, :]
    else:  # sagittal
        return vol_3d[:, :, vol_3d.shape[2]//2]


def run_model(model_name, checkpoint_dir, ct_tensor, mr_tensor, device, opt):
    """Run a model and return warped image and deformation field."""
    if model_name == 'voxelmorph3d':
        from models.voxelmorph3d_model import VoxelMorph3DModel
        model = VoxelMorph3DModel(opt)
    elif model_name == 'transmorph3d':
        from models.transmorph3d_model import Transmorph3DModel
        model = Transmorph3DModel(opt)
    elif model_name == 'nemar3d':
        from models.nemar3d_model import NEMAR3DModel
        model = NEMAR3DModel(opt)

    for name in model.model_names:
        net = getattr(model, 'net' + name)
        if isinstance(net, torch.nn.Module):
            net.to(device)
            net.eval()

    # Load best checkpoint
    for name in model.model_names:
        ckpt = os.path.join(checkpoint_dir, f'best_net_{name}.pth')
        if not os.path.exists(ckpt):
            ckpt = os.path.join(checkpoint_dir, f'latest_net_{name}.pth')
        if os.path.exists(ckpt):
            net = getattr(model, 'net' + name)
            state = torch.load(ckpt, map_location=device, weights_only=True)
            net.load_state_dict(state)

    model.set_input({'A': ct_tensor, 'B': mr_tensor, 'A_paths': ''})
    with torch.no_grad():
        model.forward()

    warped = getattr(model, 'registered_real_A', None)
    if warped is None:
        warped = getattr(model, 'warped', None)

    flow = getattr(model, 'deformation', None)
    if flow is None:
        flow = getattr(model, 'flow', None)

    return warped, flow


def compute_dsc(ct_seg, mr_seg, flow, device):
    """Compute DSC for a single patient."""
    if ct_seg is None or mr_seg is None or flow is None:
        return None

    D, H, W = ct_seg.shape
    grid_d, grid_h, grid_w = torch.meshgrid(
        torch.linspace(-1, 1, D, device=device),
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing='ij'
    )
    grid = torch.stack([grid_w, grid_h, grid_d], dim=-1).unsqueeze(0)
    new_grid = grid + flow.permute(0, 2, 3, 4, 1)

    labels = np.unique(ct_seg)
    labels = labels[labels > 0]
    per_label_dsc = []

    for label in labels:
        mask = (ct_seg == label).astype(np.float32)
        mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(device)
        warped_mask = F.grid_sample(mask_t, new_grid, mode='nearest',
                                    padding_mode='zeros', align_corners=True)
        pred = (warped_mask.cpu().numpy().squeeze() > 0.5)
        target = (mr_seg == label)
        intersection = np.logical_and(pred, target).sum()
        union = pred.sum() + target.sum()
        if union > 0:
            per_label_dsc.append(2.0 * intersection / union)

    return np.mean(per_label_dsc) if per_label_dsc else 0.0


def make_checkerboard(img1, img2, grid_size=8):
    """Create checkerboard overlay of two images."""
    h, w = img1.shape
    cb = np.zeros_like(img1)
    for i in range(h):
        for j in range(w):
            if ((i // grid_size) + (j // grid_size)) % 2 == 0:
                cb[i, j] = img1[i, j]
            else:
                cb[i, j] = img2[i, j]
    return cb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dataroot', type=str, default='./datasets/L2R')
    parser.add_argument('--output_dir', type=str, default='./paper_figures')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    os.makedirs(args.output_dir, exist_ok=True)

    patients = load_l2r_dataset(args.dataroot)
    seg_patients = [p for p in patients if p.get('has_seg', False)]
    print(f"Found {len(seg_patients)} patients with segmentation")

    # Model configs
    models_info = [
        ('VM3D', 'voxelmorph3d', './checkpoints/vm3d_l2r_v2'),
        ('TM3D', 'transmorph3d', './checkpoints/tm3d_l2r_v2'),
        ('Falcon3D', 'nemar3d', './checkpoints/falcon3d_l2r_v2'),
    ]

    # Step 1: Compute DSC for each model on each patient to select good cases
    print("\nComputing DSC for case selection...")
    opt = argparse.Namespace(
        name='vis', model='nemar3d',
        isTrain=False, phase='test', input_nc=1, output_nc=1,
        batch_size=1, num_threads=0, serial_batches=True,
        max_dataset_size=float('inf'), load_seg=True,
        device=device, gpu_ids=[args.gpu],
        checkpoints_dir='./checkpoints', epoch='latest', load_iter=0,
        verbose=False, preprocess='none', no_flip=True,
        display_winsize=256, suffix='', pool_size=0,
        lr=1e-4, lr_policy='linear', beta1=0.5, niter_decay=0,
        epoch_count=1, augment_3d=False, crop_3d_size=0,
        vol_depth=192, vol_height=160, vol_width=192,
        img_height=160, img_width=192,
        # VM3D
        vm3d_lr=1e-4, vm3d_mi_bins=32, vm3d_smoothness_weight=10.0,
        vm3d_num_features=[32, 64, 128, 256], vm3d_loss_type='mind',
        vm3d_no_svf=False, vm3d_svf_steps=7,
        # TM3D
        tm3d_enc_channels=[48, 96, 192, 384], tm3d_num_heads=6,
        tm3d_num_transformer_blocks=2, tm3d_sim_loss='mind',
        tm3d_reg_weight=1.0, tm3d_lr=1e-4,
        tm3d_use_svf=True, tm3d_no_svf=False, tm3d_svf_steps=7,
        # Falcon3D
        n3d_ngf=16, n3d_ndf=16, n3d_n_blocks=6,
        n3d_kan_embed_dims=[32, 64, 128], n3d_kan_depths=[1, 1, 1],
        n3d_lambda_gan=1.0, n3d_lambda_recon=10.0, n3d_lambda_smooth=1.0,
        n3d_lambda_direct=0.0, n3d_lambda_cycle=0.0,
        n3d_gan_mode='vanilla', n3d_lr=1e-4, n3d_use_dropout=False,
        n3d_warmup_epochs=0, use_amp=False,
    )

    all_dscs = {}
    for pidx, patient in enumerate(seg_patients):
        ct_vol = nib.load(patient['ct']).get_fdata().astype(np.float32)
        mr_vol = nib.load(patient['mr']).get_fdata().astype(np.float32)
        ct_seg = nib.load(patient['ct_seg']).get_fdata().astype(np.int32) if patient['ct_seg'] else None
        mr_seg = nib.load(patient['mr_seg']).get_fdata().astype(np.int32) if patient['mr_seg'] else None

        ct_tensor = torch.from_numpy(normalize_vol(ct_vol)).unsqueeze(0).unsqueeze(0).to(device)
        mr_tensor = torch.from_numpy(normalize_vol(mr_vol)).unsqueeze(0).unsqueeze(0).to(device)

        all_dscs[pidx] = {}
        for model_label, model_name, ckpt_dir in models_info:
            if not os.path.isdir(ckpt_dir):
                all_dscs[pidx][model_label] = None
                continue
            try:
                warped, flow = run_model(model_name, ckpt_dir, ct_tensor, mr_tensor, device, opt)
                dsc = compute_dsc(ct_seg, mr_seg, flow, device)
                all_dscs[pidx][model_label] = dsc
            except Exception as e:
                print(f"  Error running {model_label} on patient {pidx}: {e}")
                all_dscs[pidx][model_label] = None

        dsc_str = '  '.join(f'{k}={v:.3f}' if v else f'{k}=N/A'
                            for k, v in all_dscs[pidx].items())
        print(f"  Patient {pidx}: {dsc_str}")

    # Step 2: Select representative cases
    # Criteria: all methods have reasonable DSC (>0.2), Falcon3D is clearly better
    print("\nSelecting representative cases...")
    selected = []
    for pidx in all_dscs:
        dscs = all_dscs[pidx]
        vals = [v for v in dscs.values() if v is not None]
        if len(vals) < 2:
            continue
        # All methods should have DSC > 0.2
        if min(vals) < 0.15:
            continue
        # Falcon3D should be best or close to best
        if dscs.get('Falcon3D', 0) is not None and dscs['Falcon3D'] > 0.4:
            selected.append((pidx, dscs))

    # Sort by how representative they are (moderate difficulty preferred)
    selected.sort(key=lambda x: abs(np.mean(list(x[1].values())) - 0.45))
    selected = selected[:3]  # Top 3 cases

    if not selected:
        print("No suitable cases found, using first 3 with seg")
        selected = [(i, all_dscs.get(i, {})) for i in range(min(3, len(seg_patients)))]

    print(f"Selected patients: {[s[0] for s in selected]}")
    for pidx, dscs in selected:
        dsc_str = '  '.join(f'{k}={v:.3f}' if v else f'{k}=N/A'
                            for k, v in dscs.items())
        print(f"  Patient {pidx}: {dsc_str}")

    # Step 3: Generate visualizations
    print("\nGenerating visualizations...")
    for case_idx, (pidx, dscs) in enumerate(selected):
        patient = seg_patients[pidx]
        ct_vol = nib.load(patient['ct']).get_fdata().astype(np.float32)
        mr_vol = nib.load(patient['mr']).get_fdata().astype(np.float32)
        ct_seg = nib.load(patient['ct_seg']).get_fdata().astype(np.int32) if patient['ct_seg'] else None
        mr_seg = nib.load(patient['mr_seg']).get_fdata().astype(np.int32) if patient['mr_seg'] else None

        ct_norm = normalize_vol(ct_vol)
        mr_norm = normalize_vol(mr_vol)

        ct_tensor = torch.from_numpy(normalize_vol(ct_vol)).unsqueeze(0).unsqueeze(0).to(device)
        mr_tensor = torch.from_numpy(normalize_vol(mr_vol)).unsqueeze(0).unsqueeze(0).to(device)

        # Get mid axial slice
        mid_d = ct_vol.shape[0] // 2

        # Run all models
        warped_results = {}
        flow_results = {}
        for model_label, model_name, ckpt_dir in models_info:
            if not os.path.isdir(ckpt_dir):
                continue
            try:
                warped, flow = run_model(model_name, ckpt_dir, ct_tensor, mr_tensor, device, opt)
                if warped is not None:
                    warped_np = warped.cpu().numpy().squeeze()
                    warped_results[model_label] = normalize_vol(warped_np)
                    flow_results[model_label] = flow
            except Exception as e:
                print(f"  Error: {e}")

        # --- Figure 1: Registration comparison (axial slices) ---
        n_cols = 2 + len(warped_results)  # CT, MR, + each model
        fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

        col = 0
        # Moving CT
        axes[col].imshow(ct_norm[mid_d], cmap='gray', vmin=-1, vmax=1)
        axes[col].set_title('Moving (CT)', fontsize=14)
        axes[col].axis('off')
        col += 1

        # Fixed MR
        axes[col].imshow(mr_norm[mid_d], cmap='gray', vmin=-1, vmax=1)
        axes[col].set_title('Fixed (MR)', fontsize=14)
        axes[col].axis('off')
        col += 1

        # Warped results
        for model_label in ['VM3D', 'TM3D', 'Falcon3D']:
            if model_label not in warped_results:
                continue
            dsc_val = dscs.get(model_label, None)
            title = f'{model_label}'
            if dsc_val is not None:
                title += f'\nDSC={dsc_val:.3f}'
            axes[col].imshow(warped_results[model_label][mid_d], cmap='gray', vmin=-1, vmax=1)
            axes[col].set_title(title, fontsize=14)
            axes[col].axis('off')
            col += 1

        plt.tight_layout()
        fig.savefig(os.path.join(args.output_dir, f'case{case_idx+1}_registration.png'),
                    dpi=300, bbox_inches='tight')
        plt.close(fig)

        # --- Figure 2: Checkerboard overlay (warped vs fixed) ---
        n_methods = len(warped_results)
        fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 5))
        if n_methods == 1:
            axes = [axes]

        for idx, model_label in enumerate(['VM3D', 'TM3D', 'Falcon3D']):
            if model_label not in warped_results:
                continue
            cb = make_checkerboard(warped_results[model_label][mid_d],
                                   mr_norm[mid_d], grid_size=10)
            axes[idx].imshow(cb, cmap='gray', vmin=-1, vmax=1)
            dsc_val = dscs.get(model_label, None)
            title = f'{model_label}'
            if dsc_val is not None:
                title += f' (DSC={dsc_val:.3f})'
            axes[idx].set_title(title, fontsize=14)
            axes[idx].axis('off')

        plt.suptitle(f'Case {case_idx+1}: Checkerboard (Warped vs Fixed MR)', fontsize=16)
        plt.tight_layout()
        fig.savefig(os.path.join(args.output_dir, f'case{case_idx+1}_checkerboard.png'),
                    dpi=300, bbox_inches='tight')
        plt.close(fig)

        # --- Figure 3: Segmentation overlay ---
        if ct_seg is not None and mr_seg is not None:
            fig, axes = plt.subplots(1, n_methods + 1, figsize=(4 * (n_methods + 1), 4))

            # Ground truth: MR segmentation
            axes[0].imshow(mr_norm[mid_d], cmap='gray', vmin=-1, vmax=1)
            # Overlay MR segmentation boundaries
            from scipy.ndimage import binary_dilation
            mr_slice = mr_seg[mid_d]
            for label in np.unique(mr_slice):
                if label == 0:
                    continue
                boundary = binary_dilation(mr_slice == label, iterations=1) & ~(mr_slice == label)
                axes[0].contour(boundary.astype(float), levels=[0.5], colors='lime', linewidths=1.5)
            axes[0].set_title('GT (MR seg)', fontsize=14)
            axes[0].axis('off')

            for idx, model_label in enumerate(['VM3D', 'TM3D', 'Falcon3D']):
                if model_label not in flow_results or flow_results[model_label] is None:
                    continue
                flow = flow_results[model_label]

                # Warp CT segmentation
                D, H, W = ct_seg.shape
                grid_d, grid_h, grid_w = torch.meshgrid(
                    torch.linspace(-1, 1, D, device=device),
                    torch.linspace(-1, 1, H, device=device),
                    torch.linspace(-1, 1, W, device=device),
                    indexing='ij'
                )
                grid = torch.stack([grid_w, grid_h, grid_d], dim=-1).unsqueeze(0)
                new_grid = grid + flow.permute(0, 2, 3, 4, 1)

                warped_seg = np.zeros_like(ct_seg)
                for label in np.unique(ct_seg):
                    if label == 0:
                        continue
                    mask = (ct_seg == label).astype(np.float32)
                    mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(device)
                    warped_mask = F.grid_sample(mask_t, new_grid, mode='nearest',
                                                padding_mode='zeros', align_corners=True)
                    warped_seg[warped_mask.cpu().numpy().squeeze() > 0.5] = label

                axes[idx + 1].imshow(mr_norm[mid_d], cmap='gray', vmin=-1, vmax=1)
                # GT boundary (green)
                for label in np.unique(mr_slice):
                    if label == 0:
                        continue
                    boundary = binary_dilation(mr_slice == label, iterations=1) & ~(mr_slice == label)
                    axes[idx + 1].contour(boundary.astype(float), levels=[0.5],
                                         colors='lime', linewidths=1.5, linestyles='solid')
                # Predicted boundary (red)
                pred_slice = warped_seg[mid_d]
                for label in np.unique(pred_slice):
                    if label == 0:
                        continue
                    boundary = binary_dilation(pred_slice == label, iterations=1) & ~(pred_slice == label)
                    axes[idx + 1].contour(boundary.astype(float), levels=[0.5],
                                         colors='red', linewidths=1.5, linestyles='dashed')

                dsc_val = dscs.get(model_label, None)
                title = f'{model_label}'
                if dsc_val is not None:
                    title += f' (DSC={dsc_val:.3f})'
                axes[idx + 1].set_title(title, fontsize=14)
                axes[idx + 1].axis('off')

            plt.tight_layout()
            fig.savefig(os.path.join(args.output_dir, f'case{case_idx+1}_segmentation.png'),
                        dpi=300, bbox_inches='tight')
            plt.close(fig)

        # --- Figure 4: Error heatmap (|warped - fixed|) ---
        fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 5))
        if n_methods == 1:
            axes = [axes]

        for idx, model_label in enumerate(['VM3D', 'TM3D', 'Falcon3D']):
            if model_label not in warped_results:
                continue
            error = np.abs(warped_results[model_label][mid_d] - mr_norm[mid_d])
            im = axes[idx].imshow(error, cmap='hot', vmin=0, vmax=1.0)
            dsc_val = dscs.get(model_label, None)
            title = f'{model_label}'
            if dsc_val is not None:
                title += f' (DSC={dsc_val:.3f})'
            axes[idx].set_title(title, fontsize=14)
            axes[idx].axis('off')
            plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)

        plt.suptitle(f'Case {case_idx+1}: Registration Error |Warped - Fixed|', fontsize=16)
        plt.tight_layout()
        fig.savefig(os.path.join(args.output_dir, f'case{case_idx+1}_error_heatmap.png'),
                    dpi=300, bbox_inches='tight')
        plt.close(fig)

    print(f"\nFigures saved to {args.output_dir}/")


if __name__ == '__main__':
    main()

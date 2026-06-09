#!/usr/bin/env python3
"""
Generate publication-quality registration visualization figures.

Key insight: Cross-modal (CT→MR) registration visualized by comparing
warped CT to fixed MR looks bad because intensities differ fundamentally.
Proper visualization uses:
1. Segmentation contour overlay on fixed MR (shows spatial alignment)
2. Checkerboard blending (shows alignment at boundaries)
3. Translated+registered result (in MR intensity space)
4. Multi-patient DSC statistics (not single patient)

Usage:
    python generate_registration_figures.py --gpu 0
    python generate_registration_figures.py --gpu 0 --num_patients 8
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
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage
import nibabel as nib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _base_opt(device, name='eval'):
    return argparse.Namespace(
        name=name, isTrain=False, phase='test', input_nc=1, output_nc=1,
        batch_size=1, num_threads=0, serial_batches=True,
        max_dataset_size=float('inf'), load_seg=True,
        device=device, gpu_ids=[0], checkpoints_dir='./checkpoints',
        epoch='latest', load_iter=0, verbose=False,
        preprocess='none', no_flip=True, display_winsize=256,
        suffix='', pool_size=0, lr=1e-4, lr_policy='linear',
        beta1=0.5, niter_decay=0, epoch_count=1, niter=500,
        display_freq=100, display_ncols=4, print_freq=100,
        save_epoch_freq=50, save_latest_freq=5000,
        no_html=True, display_id=-1, display_server='http://localhost',
        display_port=8097, display_env='main', lambda_L1=100.0,
        netG='unet_256', netD='basic', norm='instance',
        dataset_mode='aligned', direction='AtoB', gan_mode='vanilla',
    )


def load_l2r_volume(dataroot, patient_idx=0, split='Train'):
    data_dir = os.path.join(dataroot, split)
    ct_files = sorted([f for f in os.listdir(data_dir) if f.endswith('_CT.nii.gz') and f.startswith('img')])
    if patient_idx >= len(ct_files):
        patient_idx = 0
    ct_file = ct_files[patient_idx]
    mr_file = ct_file.replace('_CT.nii.gz', '_MR.nii.gz')

    ct_vol = nib.load(os.path.join(data_dir, ct_file)).get_fdata().astype(np.float32)
    mr_vol = nib.load(os.path.join(data_dir, mr_file)).get_fdata().astype(np.float32)

    def norm(x):
        v = x.max() - x.min()
        if v < 1e-10: return np.zeros_like(x)
        return 2.0 * (x - x.min()) / v - 1.0

    ct_seg = mr_seg = None
    ct_seg_file = ct_file.replace('img', 'seg')
    mr_seg_file = mr_file.replace('img', 'seg')
    if os.path.exists(os.path.join(data_dir, ct_seg_file)):
        ct_seg = nib.load(os.path.join(data_dir, ct_seg_file)).get_fdata().astype(np.int32)
    if os.path.exists(os.path.join(data_dir, mr_seg_file)):
        mr_seg = nib.load(os.path.join(data_dir, mr_seg_file)).get_fdata().astype(np.int32)

    pid = ct_file.replace('img', '').replace('_tcia_CT.nii.gz', '')
    return {
        'A': torch.from_numpy(norm(ct_vol)).unsqueeze(0).unsqueeze(0),
        'B': torch.from_numpy(norm(mr_vol)).unsqueeze(0).unsqueeze(0),
        'A_seg': torch.from_numpy(ct_seg).unsqueeze(0).unsqueeze(0) if ct_seg is not None else None,
        'B_seg': torch.from_numpy(mr_seg).unsqueeze(0).unsqueeze(0) if mr_seg is not None else None,
        'patient_id': pid,
    }


def load_voxelmorph3d(ckpt_dir, device):
    from models.voxelmorph3d_model import VoxelMorph3DModel
    opt = _base_opt(device, 'vm3d_l2r')
    opt.vm3d_lr = 1e-4; opt.vm3d_mi_bins = 32; opt.vm3d_smoothness_weight = 1.0
    opt.vm3d_num_features = [32, 64, 128, 256]
    model = VoxelMorph3DModel(opt)
    model.netV.load_state_dict(torch.load(os.path.join(ckpt_dir, 'latest_net_V.pth'), map_location=device, weights_only=True))
    model.netV.to(device).eval()
    return model


def load_transmorph3d(ckpt_dir, device):
    from models.transmorph3d_model import Transmorph3DModel
    opt = _base_opt(device, 'tm3d_l2r_svf')
    opt.tm3d_enc_channels = [48, 96, 192, 384]; opt.tm3d_num_heads = 6
    opt.tm3d_num_transformer_blocks = 2; opt.tm3d_sim_loss = 'mind'
    opt.tm3d_reg_weight = 1.0; opt.tm3d_lr = 1e-4
    opt.tm3d_use_svf = True; opt.tm3d_no_svf = False; opt.tm3d_svf_steps = 7
    opt.use_amp = False
    model = Transmorph3DModel(opt)
    model.netTM3D.load_state_dict(torch.load(os.path.join(ckpt_dir, 'latest_net_TM3D.pth'), map_location=device, weights_only=True))
    model.netTM3D.to(device).eval()
    return model


def load_falcon3d(ckpt_dir, device):
    from models.nemar3d_model import NEMAR3DModel
    opt = _base_opt(device, 'falcon3d_l2r_svf')
    opt.n3d_ngf = 16; opt.n3d_ndf = 16; opt.n3d_n_blocks = 6
    opt.n3d_kan_embed_dims = [32, 64, 128]; opt.n3d_kan_depths = [1, 1, 1]
    opt.n3d_lambda_gan = 1.0; opt.n3d_lambda_recon = 100.0; opt.n3d_lambda_smooth = 1.0
    opt.n3d_gan_mode = 'vanilla'; opt.n3d_lr = 1e-4; opt.n3d_use_dropout = False
    opt.use_amp = False; opt.vol_depth = 192; opt.img_height = 160; opt.img_width = 192
    model = NEMAR3DModel(opt)
    for name in ['T', 'R', 'D']:
        net = getattr(model, f'net{name}')
        net.load_state_dict(torch.load(os.path.join(ckpt_dir, f'latest_net_{name}.pth'), map_location=device, weights_only=True))
        net.to(device).eval()
    return model


def warp_segmentation(ct_seg_np, flow, device):
    """Warp CT segmentation using predicted flow field. Returns warped seg as numpy."""
    D, H, W = ct_seg_np.shape
    grid_d, grid_h, grid_w = torch.meshgrid(
        torch.linspace(-1, 1, D), torch.linspace(-1, 1, H), torch.linspace(-1, 1, W),
        indexing='ij'
    )
    grid = torch.stack([grid_w, grid_h, grid_d], dim=-1).unsqueeze(0).float().to(device)
    new_grid = grid + flow.to(device).permute(0, 2, 3, 4, 1)

    warped_segs = {}
    labels = np.unique(ct_seg_np)
    labels = labels[labels > 0]
    for label in labels:
        mask = (ct_seg_np == label).astype(np.float32)
        mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(device)
        warped_mask = F.grid_sample(mask_t, new_grid, mode='nearest',
                                    padding_mode='zeros', align_corners=True)
        warped_segs[label] = (warped_mask.cpu().numpy().squeeze() > 0.5).astype(np.int32)
    return warped_segs


def compute_dsc(ct_seg_np, mr_seg_np, flow, device):
    if ct_seg_np is None or mr_seg_np is None:
        return 0.0, {}
    warped_segs = warp_segmentation(ct_seg_np, flow, device)
    labels = np.unique(ct_seg_np); labels = labels[labels > 0]
    dsc_list, per_label = [], {}
    for label in labels:
        pred = warped_segs.get(label, np.zeros_like(ct_seg_np, dtype=bool))
        target = (mr_seg_np == label)
        inter = np.logical_and(pred, target).sum()
        union = pred.sum() + target.sum()
        dsc = 2.0 * inter / union if union > 0 else 0.0
        dsc_list.append(dsc)
        per_label[label] = dsc
    return np.mean(dsc_list) if dsc_list else 0.0, per_label


def compute_jacobian_determinant(flow):
    flow = flow.float()
    D, H, W = flow.shape[2], flow.shape[3], flow.shape[4]

    dFdx = torch.zeros_like(flow)
    dFdx[:, :, :, :, 1:-1] = (flow[:, :, :, :, 2:] - flow[:, :, :, :, :-2]) / 2.0
    dFdx[:, :, :, :, 0] = flow[:, :, :, :, 1] - flow[:, :, :, :, 0]
    dFdx[:, :, :, :, -1] = flow[:, :, :, :, -1] - flow[:, :, :, :, -2]

    dFdy = torch.zeros_like(flow)
    dFdy[:, :, :, 1:-1, :] = (flow[:, :, :, 2:, :] - flow[:, :, :, :-2, :]) / 2.0
    dFdy[:, :, :, 0, :] = flow[:, :, :, 1, :] - flow[:, :, :, 0, :]
    dFdy[:, :, :, -1, :] = flow[:, :, :, -1, :] - flow[:, :, :, -2, :]

    dFdz = torch.zeros_like(flow)
    dFdz[:, :, 1:-1, :, :] = (flow[:, :, 2:, :, :] - flow[:, :, :-2, :, :]) / 2.0
    dFdz[:, :, 0, :, :] = flow[:, :, 1, :, :] - flow[:, :, 0, :, :]
    dFdz[:, :, -1, :, :] = flow[:, :, -1, :, :] - flow[:, :, -2, :, :]

    J = torch.stack([dFdx, dFdy, dFdz], dim=-1).squeeze(0).permute(1, 2, 3, 0, 4)
    identity = torch.eye(3, device=J.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    det = torch.det(J + identity)
    return det.numpy()


def find_contour(mask_2d):
    """Find boundary pixels of a binary mask."""
    if mask_2d.sum() == 0:
        return np.array([]), np.array([])
    dilated = ndimage.binary_dilation(mask_2d, iterations=1)
    boundary = dilated & ~mask_2d.astype(bool)
    # Also get the outer boundary
    inner = mask_2d.astype(bool) & ~ndimage.binary_erosion(mask_2d.astype(bool), iterations=1)
    boundary = boundary | inner
    ys, xs = np.where(boundary)
    return xs, ys


def make_checkerboard(img1, img2, tile_size=16):
    """Create checkerboard blend of two images."""
    h, w = img1.shape
    cb = np.copy(img1)
    for i in range(0, h, tile_size):
        for j in range(0, w, tile_size):
            if ((i // tile_size) + (j // tile_size)) % 2 == 1:
                cb[i:i+tile_size, j:j+tile_size] = img2[i:i+tile_size, j:j+tile_size]
    return cb


# =============================================================================
# PUBLICATION-QUALITY FIGURES
# =============================================================================

def fig_contour_overlay(mr_np, ct_seg_np, mr_seg_np, all_warped_segs, methods, dsc_scores, save_path):
    """Figure 1: Segmentation contour overlay on MR fixed image.

    This is the MOST IMPORTANT figure for cross-modal registration:
    Shows how well the warped CT segmentation aligns with the MR segmentation,
    overlaid on the MR image so you can visually judge alignment quality.
    """
    D, H, W = mr_np.shape
    mid_d = D // 2
    mid_h = H // 2
    mid_w = W // 2

    views = [
        ('Axial', mid_d, 0),
        ('Coronal', mid_h, 1),
        ('Sagittal', mid_w, 2),
    ]

    n_methods = len(methods)
    n_cols = 2 + n_methods  # Before reg + Fixed MR + each method
    fig, axes = plt.subplots(3, n_cols, figsize=(3.2 * n_cols, 9.5))

    contour_colors = ['#FF6B6B', '#4ECDC4', '#FFE66D']  # per label
    mr_colors = ['#FF6B6B', '#4ECDC4', '#FFE66D']  # same colors but dashed

    for row, (view_name, mid_idx, axis) in enumerate(views):
        # Get MR slice
        if axis == 0:
            mr_sl = mr_np[mid_idx, :, :]
            mr_seg_sl = mr_seg_np[mid_idx, :, :] if mr_seg_np is not None else np.zeros((H, W), dtype=int)
            ct_seg_sl = ct_seg_np[mid_idx, :, :] if ct_seg_np is not None else np.zeros((H, W), dtype=int)
        elif axis == 1:
            mr_sl = mr_np[:, mid_idx, :]
            mr_seg_sl = mr_seg_np[:, mid_idx, :] if mr_seg_np is not None else np.zeros((D, W), dtype=int)
            ct_seg_sl = ct_seg_np[:, mid_idx, :] if ct_seg_np is not None else np.zeros((D, W), dtype=int)
        else:
            mr_sl = mr_np[:, :, mid_idx]
            mr_seg_sl = mr_seg_np[:, :, mid_idx] if mr_seg_np is not None else np.zeros((D, H), dtype=int)
            ct_seg_sl = ct_seg_np[:, :, mid_idx] if ct_seg_np is not None else np.zeros((D, H), dtype=int)

        # Column 0: Before registration (CT seg on MR)
        ax = axes[row, 0]
        ax.imshow(mr_sl, cmap='gray', vmin=-1, vmax=1)
        labels = np.unique(ct_seg_sl); labels = labels[labels > 0]
        for i, label in enumerate(sorted(labels)):
            color = contour_colors[i % len(contour_colors)]
            xs, ys = find_contour(ct_seg_sl == label)
            if len(xs) > 0:
                ax.scatter(xs, ys, s=0.1, c=color, alpha=0.9, linewidths=0)
            xs2, ys2 = find_contour(mr_seg_sl == label)
            if len(xs2) > 0:
                ax.scatter(xs2, ys2, s=0.1, c=color, alpha=0.4, linewidths=0)
        ax.set_title('Before Registration' if row == 0 else '', fontsize=10, fontweight='bold')
        ax.axis('off')

        # Column 1: Fixed MR with its own segmentation (reference)
        ax = axes[row, 1]
        ax.imshow(mr_sl, cmap='gray', vmin=-1, vmax=1)
        for i, label in enumerate(sorted(labels)):
            color = mr_colors[i % len(mr_colors)]
            xs, ys = find_contour(mr_seg_sl == label)
            if len(xs) > 0:
                ax.scatter(xs, ys, s=0.1, c=color, alpha=0.9, linewidths=0)
        ax.set_title('Fixed MR\n(Reference)' if row == 0 else '', fontsize=10, fontweight='bold')
        ax.axis('off')

        # Columns 2+: Each method's warped CT seg on MR
        for col, method_name in enumerate(methods):
            ax = axes[row, 2 + col]
            ax.imshow(mr_sl, cmap='gray', vmin=-1, vmax=1)

            warped_seg_3d = all_warped_segs[method_name]
            if axis == 0:
                wseg_sl = warped_seg_3d[mid_idx, :, :]
            elif axis == 1:
                wseg_sl = warped_seg_3d[:, mid_idx, :]
            else:
                wseg_sl = warped_seg_3d[:, :, mid_idx]

            for i, label in enumerate(sorted(labels)):
                color = contour_colors[i % len(contour_colors)]
                # Warped CT seg (solid)
                xs, ys = find_contour(wseg_sl == label)
                if len(xs) > 0:
                    ax.scatter(xs, ys, s=0.1, c=color, alpha=0.9, linewidths=0)
                # MR seg (semi-transparent)
                xs2, ys2 = find_contour(mr_seg_sl == label)
                if len(xs2) > 0:
                    ax.scatter(xs2, ys2, s=0.1, c=color, alpha=0.4, linewidths=0)

            dsc_val = dsc_scores.get(method_name, 0)
            ax.set_title(f'{method_name}\nDSC={dsc_val:.3f}' if row == 0 else '', fontsize=9, fontweight='bold')
            ax.axis('off')

        # Row label
        axes[row, 0].text(-0.05, 0.5, view_name, transform=axes[row, 0].transAxes,
                          fontsize=11, fontweight='bold', va='center', ha='right', rotation=90)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF6B6B', markersize=8, label='Warped CT seg'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF6B6B', markersize=8, alpha=0.4, label='Fixed MR seg'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=9, frameon=True)

    plt.suptitle('Segmentation Contour Overlay on Fixed MR', fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0.03, 0.04, 1, 0.96])
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def fig_checkerboard(mr_np, all_warped, methods, dsc_scores, save_path):
    """Figure 2: Checkerboard blend of warped CT and fixed MR.

    Shows alignment quality by alternating tiles between warped CT and MR.
    At organ boundaries, misalignment appears as discontinuities.
    """
    D, H, W = mr_np.shape
    mid_d = D // 2

    n_methods = len(methods)
    n_cols = 1 + n_methods  # Before + each method
    fig, axes = plt.subplots(1, n_cols, figsize=(3.5 * n_cols, 3.5))
    if n_cols == 1:
        axes = [axes]

    mr_sl = mr_np[mid_d, :, :]

    # Before registration (original CT)
    ct_3d = all_warped.get('_original_ct', mr_np)
    ct_sl = ct_3d[mid_d, :, :] if ct_3d.ndim == 3 else ct_3d
    cb = make_checkerboard(ct_sl, mr_sl, tile_size=20)
    axes[0].imshow(cb, cmap='gray', vmin=-1, vmax=1)
    axes[0].set_title('Before\nRegistration', fontsize=10, fontweight='bold')
    axes[0].axis('off')

    for col, method_name in enumerate(methods):
        warped_np = all_warped[method_name].squeeze()
        warped_sl = warped_np[mid_d, :, :]
        cb = make_checkerboard(warped_sl, mr_sl, tile_size=20)
        axes[1 + col].imshow(cb, cmap='gray', vmin=-1, vmax=1)
        dsc_val = dsc_scores.get(method_name, 0)
        axes[1 + col].set_title(f'{method_name}\nDSC={dsc_val:.3f}', fontsize=9, fontweight='bold')
        axes[1 + col].axis('off')

    plt.suptitle('Checkerboard: Warped CT (tiles) vs Fixed MR', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def fig_translated_result(mr_np, fake_tr, methods, dsc_scores, save_path):
    """Figure 3: Translation+Registration result for Falcon3D.

    Shows fake_TR_B (registered CT → translated to MR appearance) vs fixed MR.
    This is the proper way to visually assess cross-modal registration quality.
    """
    D = mr_np.shape[0]
    mid_d = D // 2
    mr_sl = mr_np[mid_d, :, :]

    falcon_names = [m for m in methods if 'Falcon' in m]
    if not falcon_names:
        return

    n_cols = 1 + len(falcon_names)
    fig, axes = plt.subplots(2, n_cols, figsize=(3.5 * n_cols, 7))

    # Row 1: Translated+registered results
    axes[0, 0].imshow(mr_sl, cmap='gray', vmin=-1, vmax=1)
    axes[0, 0].set_title('Fixed MR\n(Ground Truth)', fontsize=10, fontweight='bold')
    axes[0, 0].axis('off')

    for col, name in enumerate(falcon_names):
        fake = fake_tr[name].squeeze()
        axes[0, 1 + col].imshow(fake[mid_d, :, :], cmap='gray', vmin=-1, vmax=1)
        dsc_val = dsc_scores.get(name, 0)
        axes[0, 1 + col].set_title(f'{name}\nTR Result (DSC={dsc_val:.3f})', fontsize=9, fontweight='bold')
        axes[0, 1 + col].axis('off')

    # Row 2: Difference maps |fake_TR - fixed MR|
    for col, name in enumerate(falcon_names):
        fake = fake_tr[name].squeeze()
        diff = np.abs(fake[mid_d, :, :] - mr_sl)
        axes[1, 1 + col].imshow(diff, cmap='hot', vmin=0, vmax=0.5)
        mse = np.mean(diff ** 2)
        axes[1, 1 + col].set_title(f'Difference\nMSE={mse:.4f}', fontsize=9, fontweight='bold')
        axes[1, 1 + col].axis('off')

    # Difference for before registration
    axes[1, 0].axis('off')

    plt.suptitle('Translation + Registration Result (MR intensity space)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def fig_jacobian(all_flows, methods, save_path):
    """Figure 4: Jacobian determinant map (folding detection)."""
    D_ref = list(all_flows.values())[0].shape[2]
    mid_d = D_ref // 2
    H_ref = list(all_flows.values())[0].shape[3]
    mid_h = H_ref // 2
    W_ref = list(all_flows.values())[0].shape[4]
    mid_w = W_ref // 2

    n_methods = len(methods)
    fig, axes = plt.subplots(3, n_methods, figsize=(3.5 * n_methods, 9.5))
    if n_methods == 1:
        axes = axes.reshape(3, 1)

    views = [('Axial', mid_d, 0), ('Coronal', mid_h, 1), ('Sagittal', mid_w, 2)]

    for row, (vname, mid_idx, axis) in enumerate(views):
        for col, name in enumerate(methods):
            ax = axes[row, col]
            jac = compute_jacobian_determinant(all_flows[name])

            if axis == 0:
                sl = jac[mid_idx, :, :]
            elif axis == 1:
                sl = jac[:, mid_idx, :]
            else:
                sl = jac[:, :, mid_idx]

            im = ax.imshow(sl, cmap='RdBu_r', vmin=0.5, vmax=1.5)
            fold_pct = 100.0 * (jac < 0).sum() / jac.size
            if row == 0:
                ax.set_title(f'{name}\nFold: {fold_pct:.2f}%', fontsize=9, fontweight='bold')
            ax.axis('off')

        axes[row, 0].text(-0.05, 0.5, vname, transform=axes[row, 0].transAxes,
                          fontsize=10, fontweight='bold', va='center', ha='right', rotation=90)

    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.012, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='det(J)')

    plt.suptitle('Jacobian Determinant (Red=Folding, Blue=Expansion)', fontsize=12, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0.02, 0, 0.92, 0.96])
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def fig_dsc_bar(all_dsc_multi, save_path):
    """Figure 5: DSC bar chart with mean ± std across patients."""
    methods = list(all_dsc_multi.keys())
    means = [np.mean(all_dsc_multi[m]) for m in methods]
    stds = [np.std(all_dsc_multi[m]) for m in methods]

    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(methods))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors[:len(methods)],
                  edgecolor='black', linewidth=0.5, error_kw={'linewidth': 1.5})

    for bar, val, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.005,
                f'{val:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylabel('Dice Similarity Coefficient (DSC)', fontsize=12)
    ax.set_title('3D Registration DSC Comparison (Mean ± Std)', fontsize=13, fontweight='bold')
    ax.set_ylim(0, max(means) * 1.25)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def fig_deformation_field(all_flows, methods, save_path):
    """Figure 6: Deformation field magnitude visualization."""
    D_ref = list(all_flows.values())[0].shape[2]
    mid_d = D_ref // 2

    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_methods, figsize=(3.5 * n_methods, 3.5))
    if n_methods == 1:
        axes = [axes]

    vmax = 0
    mags = {}
    for name in methods:
        flow_np = all_flows[name].squeeze().numpy()
        mag = np.sqrt(np.sum(flow_np ** 2, axis=0))
        mags[name] = mag
        vmax = max(vmax, mag.max())

    for col, name in enumerate(methods):
        sl = mags[name][mid_d, :, :]
        im = axes[col].imshow(sl, cmap='jet', vmin=0, vmax=vmax)
        mean_mag = mags[name].mean()
        axes[col].set_title(f'{name}\nMean |u|={mean_mag:.4f}', fontsize=9, fontweight='bold')
        axes[col].axis('off')

    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.012, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='|Deformation|')

    plt.suptitle('Deformation Field Magnitude (Axial)', fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.92, 1])
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate registration visualization figures')
    parser.add_argument('--dataroot', type=str, default='./datasets/L2R')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='./paper_figures')
    parser.add_argument('--num_patients', type=int, default=8,
                        help='Number of patients to compute DSC statistics')
    parser.add_argument('--vis_patient', type=int, default=0,
                        help='Patient index for visualization')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Step 1: Compute DSC across multiple patients for reliable statistics
    # ------------------------------------------------------------------
    print("\n=== Computing DSC across patients ===")
    model_loaders = {
        'VoxelMorph3D': ('./checkpoints/vm3d_l2r', load_voxelmorph3d),
        'TransMorph3D+SVF': ('./checkpoints/tm3d_l2r_svf', load_transmorph3d),
        'Falcon3D+SVF': ('./checkpoints/falcon3d_l2r_svf', load_falcon3d),
    }
    methods = list(model_loaders.keys())
    all_dsc_multi = {m: [] for m in methods}
    all_per_label_dsc = {m: {} for m in methods}

    # Count available training patients
    train_dir = os.path.join(args.dataroot, 'Train')
    ct_files = sorted([f for f in os.listdir(train_dir) if f.endswith('_CT.nii.gz') and f.startswith('img')])
    n_patients = min(args.num_patients, len(ct_files))

    for pidx in range(n_patients):
        sample = load_l2r_volume(args.dataroot, pidx, split='Train')
        ct = sample['A'].to(device)
        mr = sample['B'].to(device)
        ct_seg_np = sample['A_seg'].squeeze().numpy() if sample['A_seg'] is not None else None
        mr_seg_np = sample['B_seg'].squeeze().numpy() if sample['B_seg'] is not None else None

        if ct_seg_np is None:
            continue

        pid = sample['patient_id']
        print(f"\n  Patient {pid}:")

        for mname, (ckpt_dir, loader_fn) in model_loaders.items():
            model = loader_fn(ckpt_dir, device)
            with torch.no_grad():
                model.set_input({'A': ct, 'B': mr, 'A_paths': ''})
                model.forward()

                if hasattr(model, 'flow'):
                    flow = model.flow.cpu()
                elif hasattr(model, 'deformation'):
                    flow = model.deformation.cpu()
                else:
                    flow = None

            if flow is not None:
                dsc, per_label = compute_dsc(ct_seg_np, mr_seg_np, flow, device)
                all_dsc_multi[mname].append(dsc)
                for label, d in per_label.items():
                    all_per_label_dsc[mname].setdefault(label, []).append(d)
                print(f"    {mname}: DSC={dsc:.4f}")
            else:
                print(f"    {mname}: No flow output")

            del model
            torch.cuda.empty_cache()

    # Print summary
    print("\n=== DSC Summary ===")
    for mname in methods:
        vals = all_dsc_multi[mname]
        if vals:
            print(f"  {mname}: {np.mean(vals):.4f} ± {np.std(vals):.4f} (n={len(vals)})")

    # ------------------------------------------------------------------
    # Step 2: Detailed visualization on one patient
    # ------------------------------------------------------------------
    print(f"\n=== Generating visualization on patient {args.vis_patient} ===")
    sample = load_l2r_volume(args.dataroot, args.vis_patient, split='Train')
    ct = sample['A'].to(device)
    mr = sample['B'].to(device)
    ct_seg_np = sample['A_seg'].squeeze().numpy() if sample['A_seg'] is not None else None
    mr_seg_np = sample['B_seg'].squeeze().numpy() if sample['B_seg'] is not None else None

    mr_np = mr.cpu().squeeze().numpy()
    ct_np = ct.cpu().squeeze().numpy()

    all_warped = {'_original_ct': ct_np}
    all_warped_segs = {}
    all_flows = {}
    dsc_vis = {}
    fake_tr_results = {}

    for mname, (ckpt_dir, loader_fn) in model_loaders.items():
        model = loader_fn(ckpt_dir, device)
        with torch.no_grad():
            model.set_input({'A': ct, 'B': mr, 'A_paths': ''})
            model.forward()

            if hasattr(model, 'flow'):
                flow = model.flow.cpu()
                warped = model.warped.cpu()
            elif hasattr(model, 'deformation'):
                flow = model.deformation.cpu()
                warped = model.registered_real_A.cpu()
            else:
                continue

        all_warped[mname] = warped.squeeze().numpy()
        all_flows[mname] = flow

        # Warp segmentation
        if ct_seg_np is not None:
            warped_segs_dict = warp_segmentation(ct_seg_np, flow, device)
            # Merge into single volume
            merged_seg = np.zeros_like(ct_seg_np)
            for label, wseg in warped_segs_dict.items():
                merged_seg[wseg > 0] = label
            all_warped_segs[mname] = merged_seg

            # DSC for this patient
            dsc, _ = compute_dsc(ct_seg_np, mr_seg_np, flow, device)
            dsc_vis[mname] = dsc

        # Get translated result for Falcon
        if hasattr(model, 'fake_TR_B'):
            fake_tr_results[mname] = model.fake_TR_B.cpu().squeeze().numpy()

        del model
        torch.cuda.empty_cache()

    # Before-registration segmentation
    if ct_seg_np is not None:
        all_warped_segs['_before'] = ct_seg_np

    # Use multi-patient DSC for bar chart
    dsc_for_chart = {}
    for mname in methods:
        if all_dsc_multi[mname]:
            dsc_for_chart[mname] = all_dsc_multi[mname]

    # ------------------------------------------------------------------
    # Generate figures
    # ------------------------------------------------------------------
    print("\nGenerating publication figures...")

    # 1. Segmentation contour overlay (most important for cross-modal)
    if ct_seg_np is not None:
        fig_contour_overlay(
            mr_np, ct_seg_np, mr_seg_np, all_warped_segs, methods, dsc_vis,
            os.path.join(args.output_dir, 'fig1_contour_overlay.png')
        )

    # 2. Checkerboard
    fig_checkerboard(
        mr_np, all_warped, methods, dsc_vis,
        os.path.join(args.output_dir, 'fig2_checkerboard.png')
    )

    # 3. Translation+Registration result (Falcon only)
    if fake_tr_results:
        fig_translated_result(
            mr_np, fake_tr_results, methods, dsc_vis,
            os.path.join(args.output_dir, 'fig3_translated_result.png')
        )

    # 4. Jacobian determinant
    fig_jacobian(
        all_flows, methods,
        os.path.join(args.output_dir, 'fig4_jacobian.png')
    )

    # 5. DSC bar chart (multi-patient statistics)
    if dsc_for_chart:
        fig_dsc_bar(
            dsc_for_chart,
            os.path.join(args.output_dir, 'fig5_dsc_comparison.png')
        )

    # 6. Deformation field magnitude
    fig_deformation_field(
        all_flows, methods,
        os.path.join(args.output_dir, 'fig6_deformation_field.png')
    )

    print(f"\nDone! Figures saved to: {args.output_dir}")


if __name__ == '__main__':
    main()

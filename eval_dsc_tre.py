#!/usr/bin/env python3
"""
Evaluate DSC and TRE for all registration models.

DSC: Computed on L2R dataset using organ segmentation masks.
TRE: Computed on RIRE patient_001 using fiducial marker ground truth.

Flow format conventions:
- FLOW_NORMALIZED: Offsets in [-1,1], added to normalized identity grid (Falcon, VoxelMorph)
  Uses align_corners=False in grid_sample.
- FLOW_PIXEL: Pixel displacement, added to pixel identity grid (TransMorph, ConvexAdam, DINO-Reg)
  Uses align_corners=True in grid_sample.

All models use PULL mapping: flow at position (x,y) gives displacement from output to source.
For TRE: warped_pos = input_pos - pixel_displacement (approximate for small deformations).
"""
import os, sys, json
import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CACHE_DIR = './paper_figures/metrics_cache'

FLOW_NORMALIZED = 'normalized'
FLOW_PIXEL = 'pixel'


# ── Helpers ──────────────────────────────────────────────────────────

def to_tensor(arr):
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0) * 2 - 1


def from_tensor(t):
    return ((t[0, 0].cpu().detach().numpy() + 1) / 2).clip(0, 1)


def load_image(path, size=(512, 512)):
    img = Image.open(path).convert('L')
    if img.size != size:
        img = img.resize(size, Image.BICUBIC)
    return np.array(img).astype(np.float32) / 255.0


def get_pairs(dataroot):
    dA, dB = os.path.join(dataroot, 'trainA'), os.path.join(dataroot, 'trainB')
    a = sorted([os.path.join(dA, f) for f in os.listdir(dA) if f.endswith('.png')])
    b = sorted([os.path.join(dB, f) for f in os.listdir(dB) if f.endswith('.png')])
    bm = {os.path.basename(p): p for p in b}
    return [(p, bm[os.path.basename(p)]) for p in a if os.path.basename(p) in bm]


def parse_opt_txt(path):
    opts = {}
    import ast
    with open(path) as f:
        for line in f:
            line = line.strip()
            if ':' in line and not line.startswith('-'):
                k, v = line.split(':', 1)
                k, v = k.strip(), v.strip().split('\t')[0].strip()
                try:
                    opts[k] = ast.literal_eval(v)
                except:
                    opts[k] = v
    return opts


def bilinear_sample_2d(field, x, y):
    """Bilinear interpolation of a 2D field at (x, y) float positions."""
    H, W = field.shape
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, W - 1)
    y1 = min(y0 + 1, H - 1)
    x0 = max(0, x0)
    y0 = max(0, y0)
    fx = x - x0
    fy = y - y0
    return float(field[y0, x0] * (1 - fx) * (1 - fy) +
                 field[y0, x1] * fx * (1 - fy) +
                 field[y1, x0] * (1 - fx) * fy +
                 field[y1, x1] * fx * fy)


# ── Segmentation Warping ──────────────────────────────────────────

def warp_segmentation_2d(seg, flow, flow_format, img_size=(512, 512)):
    """Warp segmentation mask using deformation field with nearest-neighbor interpolation.

    Uses F.grid_sample(mode='nearest') to preserve label values.

    Args:
        seg: 2D numpy array of label values [H, W]
        flow: tensor [1, 2, H, W] - deformation field (channel 0=x, channel 1=y)
        flow_format: FLOW_NORMALIZED or FLOW_PIXEL
        img_size: spatial dimensions

    Returns:
        Warped segmentation as numpy array
    """
    H, W = img_size

    if seg.shape != (H, W):
        seg = np.array(Image.fromarray(seg.astype(np.uint8)).resize((W, H), Image.NEAREST))

    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, H, dtype=torch.float32, device=flow.device),
        torch.arange(0, W, dtype=torch.float32, device=flow.device),
        indexing='ij'
    )

    if flow_format == FLOW_NORMALIZED:
        grid_y_norm = 2.0 * grid_y / (H - 1) - 1.0
        grid_x_norm = 2.0 * grid_x / (W - 1) - 1.0
        sample_x = grid_x_norm + flow[:, 0, :, :]
        sample_y = grid_y_norm + flow[:, 1, :, :]
        align_corners = False
    else:
        sample_x = grid_x + flow[:, 0, :, :]
        sample_y = grid_y + flow[:, 1, :, :]
        sample_x = 2.0 * sample_x / (W - 1) - 1.0
        sample_y = 2.0 * sample_y / (H - 1) - 1.0
        align_corners = True

    grid = torch.stack([sample_x, sample_y], dim=3)  # [B, H, W, 2]

    labels = np.unique(seg)
    labels = labels[labels > 0]

    warped_seg = np.zeros((H, W), dtype=seg.dtype)
    for label in labels:
        mask = (seg == label).astype(np.float32)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(flow.device)
        warped_mask = F.grid_sample(mask_tensor, grid, mode='nearest',
                                    padding_mode='zeros', align_corners=align_corners)
        warped_seg[warped_mask[0, 0].cpu().detach().numpy() > 0.5] = label

    return warped_seg


# ── DSC Computation (L2R) ──────────────────────────────────────────

def compute_dsc(pred_seg, target_seg):
    """Dice Similarity Coefficient between two binary masks."""
    intersection = np.logical_and(pred_seg, target_seg).sum()
    union = pred_seg.sum() + target_seg.sum()
    if union == 0:
        return 1.0
    return float(2.0 * intersection / union)


# ── TRE Computation (RIRE) ──────────────────────────────────────────

def load_rire_fiducials(gt_path):
    """Load RIRE fiducial points from .standard file.

    Returns:
        ct_points: [N, 3] in mm
        mr_points: [N, 3] in mm
    """
    with open(gt_path) as f:
        lines = f.readlines()
    ct_pts, mr_pts = [], []
    for line in lines[15:23]:
        c = line.split()
        ct_pts.append([float(c[1]), float(c[2]), float(c[3])])
        mr_pts.append([float(c[4]), float(c[5]), float(c[6])])
    return np.array(ct_pts), np.array(mr_pts)


def flow_to_pixel_displacement(flow, flow_format, img_size=(512, 512)):
    """Convert flow to pixel displacement format.

    For normalized flow: pixel_disp = normalized_disp * (size - 1) / 2
    """
    if flow_format == FLOW_NORMALIZED:
        H, W = img_size
        pixel_flow = flow.clone()
        pixel_flow[:, 0] = flow[:, 0] * (W - 1) / 2.0
        pixel_flow[:, 1] = flow[:, 1] * (H - 1) / 2.0
        return pixel_flow
    return flow


def compute_2d_tre(ct_fid_mm, mr_fid_mm, ct_spacing, mr_spacing,
                   flow, flow_format, img_size=(512, 512)):
    """Compute 2D TRE for a single slice.

    All models use PULL mapping: flow at (x,y) gives source = output + displacement.
    For a CT fiducial at input (cx, cy), the warped position is approximately:
        wx = cx - displacement_x(cx, cy)

    Returns:
        mean_tre_mm, list of per-point TREs in mm
    """
    H, W = img_size

    pixel_flow = flow_to_pixel_displacement(flow, flow_format, img_size)
    flow_np = pixel_flow[0].cpu().numpy()  # [2, H, W]

    # CT pixel coords (CT native 512x512, same as img_size)
    ct_px_x = ct_fid_mm[:, 0] / ct_spacing
    ct_px_y = ct_fid_mm[:, 1] / ct_spacing

    # MR pixel coords (MR native 256x256, scaled to 512)
    mr_scale = 512.0 / 256.0
    mr_px_x = mr_fid_mm[:, 0] / mr_spacing * mr_scale
    mr_px_y = mr_fid_mm[:, 1] / mr_spacing * mr_scale

    tres = []
    for i in range(len(ct_px_x)):
        cx, cy = ct_px_x[i], ct_px_y[i]

        # Bilinear interpolate flow at (cx, cy)
        dx = bilinear_sample_2d(flow_np[0], cx, cy)  # x pixel displacement
        dy = bilinear_sample_2d(flow_np[1], cx, cy)  # y pixel displacement

        # PULL mapping: source = output + displacement
        # Inverse: output = source - displacement (approximate for small deformations)
        wx = cx - dx
        wy = cy - dy

        # Distance in pixels at 512x512 MR, convert to mm
        tre_px = np.sqrt((wx - mr_px_x[i])**2 + (wy - mr_px_y[i])**2)
        tre_mm = tre_px * mr_spacing / mr_scale

        tres.append(tre_mm)

    return float(np.mean(tres)), tres


# ── Model runners (with caching) ──────────────────────────────────

_model_cache = {}


def _get_falcon(ckpt_name):
    key = ('falcon', ckpt_name)
    if key not in _model_cache:
        from models.nemar_model import NEMARModel
        import argparse
        opt_path = f'./checkpoints/{ckpt_name}/train_opt.txt'
        if not os.path.exists(opt_path):
            opt_path = f'./checkpoints/{ckpt_name}/test_opt.txt'
        if not os.path.exists(opt_path):
            return None
        opt_dict = parse_opt_txt(opt_path)
        opt = argparse.Namespace()
        for k, v in opt_dict.items():
            setattr(opt, k, v)
        defaults = {
            'isTrain': False, 'gpu_ids': [0], 'phase': 'test', 'device': DEVICE,
            'num_test': 9999, 'serial_batches': True, 'epoch': 'latest', 'load_iter': 0,
            'verbose': False, 'suffix': '', 'batch_size': 1, 'num_threads': 0,
            'max_dataset_size': float('inf'), 'display_winsize': 256, 'pool_size': 50,
            'beta1': 0.5, 'lr': 0.0002, 'lr_policy': 'linear', 'lr_decay_iters': 50,
            'niter': 100, 'niter_decay': 100, 'epoch_count': 1, 'save_epoch_freq': 5,
            'save_by_iter': False, 'save_latest_freq': 5000, 'print_freq': 100,
            'display_freq': 400, 'display_ncols': 4, 'display_id': -1, 'display_port': 8097,
            'display_server': 'http://localhost', 'display_env': 'main', 'no_html': False,
            'update_html_freq': 1000, 'aspect_ratio': 1.0, 'results_dir': './results',
            'ntest': float('inf'), 'continue_train': False,
            'contrastive_proj_dim': 128, 'contrastive_num_stages': None,
            'stn_multires_reg': 1, 'contrastive_loss_type': 'infonce',
            'tbvis_disable_report_offsets': False, 'tbvis_disable_report_weights': False,
            'tbvis_iteration_update_rate': 1000,
        }
        for k, v in defaults.items():
            if not hasattr(opt, k):
                setattr(opt, k, v)
        opt.isTrain = False
        opt.gpu_ids = [0]
        model = NEMARModel(opt)
        model.load_networks('latest')
        model.eval()
        _model_cache[key] = model
    return _model_cache[key]


def run_falcon(moving, fixed, ckpt_name):
    """Returns (warped_img, flow, flow_format)."""
    model = _get_falcon(ckpt_name)
    if model is None:
        return None, None, FLOW_NORMALIZED
    m, f = to_tensor(moving).to(DEVICE), to_tensor(fixed).to(DEVICE)
    with torch.no_grad():
        model.set_input({'A': m, 'B': f, 'A_paths': ''})
        model.forward()
    warped = from_tensor(model.registered_real_A)
    flow = None
    if hasattr(model, 'netR'):
        netR = model.netR.module if hasattr(model.netR, 'module') else model.netR
        # ContrastiveSTNWrapper wraps the inner STN
        inner_stn = netR.stn if hasattr(netR, 'stn') else netR
        if hasattr(inner_stn, 'get_grid'):
            flow = inner_stn.get_grid(m, f, return_offsets_only=True)
            flow = flow.permute(0, 3, 1, 2)  # [B, 2, H, W] channel 0=x, 1=y
    return warped, flow, FLOW_NORMALIZED


def _get_vm(ckpt_name):
    key = ('vm', ckpt_name)
    if key not in _model_cache:
        from models.voxelmorph_model import Unet, SpatialTransformer
        net = Unet(in_channels=2, out_channels=2).to(DEVICE)
        p = f'./checkpoints/{ckpt_name}/latest_net_V.pth'
        if not os.path.exists(p):
            return None, None
        net.load_state_dict(torch.load(p, map_location=DEVICE, weights_only=True))
        net.eval()
        _model_cache[key] = (net, SpatialTransformer())
    return _model_cache[key]


def run_voxelmorph_mi(moving, fixed, ckpt_name):
    """Returns (warped_img, flow, flow_format)."""
    cached = _get_vm(ckpt_name)
    if cached is None:
        return None, None, FLOW_NORMALIZED
    net, st = cached
    m, f = to_tensor(moving).to(DEVICE), to_tensor(fixed).to(DEVICE)
    with torch.no_grad():
        flow = net(torch.cat([m, f], dim=1))
        warped = st(m, flow)
    return from_tensor(warped), flow, FLOW_NORMALIZED


def _get_tm(ckpt_name):
    key = ('tm', ckpt_name)
    if key not in _model_cache:
        from models.transmorph_model import TransMorphNet
        net = TransMorphNet(img_size=(512, 512)).to(DEVICE)
        p = f'./checkpoints/{ckpt_name}/latest_net_TM.pth'
        if not os.path.exists(p):
            return None
        sd = torch.load(p, map_location=DEVICE, weights_only=True)
        sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
        net.load_state_dict(sd)
        net.eval()
        _model_cache[key] = net
    return _model_cache[key]


def run_transmorph(moving, fixed, ckpt_name):
    """Returns (warped_img, flow, flow_format)."""
    net = _get_tm(ckpt_name)
    if net is None:
        return None, None, FLOW_PIXEL
    m, f = to_tensor(moving).to(DEVICE), to_tensor(fixed).to(DEVICE)
    with torch.no_grad():
        warped, flow = net(torch.cat([m, f], dim=1))
    return from_tensor(warped.clamp(-1, 1)), flow, FLOW_PIXEL


def run_convexadam(moving, fixed):
    """Returns (warped_img, flow, flow_format)."""
    from models.convexadam_2d import convex_adam_2d
    m, f = to_tensor(moving).to(DEVICE), to_tensor(fixed).to(DEVICE)
    result = convex_adam_2d(f, m)
    warped = result[0] if isinstance(result, tuple) else result
    flow = result[1] if isinstance(result, tuple) else None
    return from_tensor(warped), flow, FLOW_PIXEL


def run_dinoreg(moving, fixed):
    """Returns (warped_img, flow, flow_format)."""
    from models.dinoreg_2d import dino_reg_2d
    m, f = to_tensor(moving).to(DEVICE), to_tensor(fixed).to(DEVICE)
    result = dino_reg_2d(f, m, feat_size=(36, 36), n_iter_adam=200, lr_adam=3.0)
    warped = result[0] if isinstance(result, tuple) else result
    flow = result[1] if isinstance(result, tuple) else None
    return from_tensor(warped), flow, FLOW_PIXEL


# ── DSC Evaluation (L2R) ──────────────────────────────────────────

def eval_dsc_l2r(methods, max_patients=8):
    """Evaluate DSC on L2R dataset using organ segmentation masks."""
    l2r_train = './datasets/L2R/Train'
    all_dsc = {m[0]: [] for m in methods}
    patient_ids = [2, 4, 6, 8, 10, 12, 14, 16][:max_patients]

    for pid in patient_ids:
        ct_img_path = os.path.join(l2r_train, f'img{pid:04d}_tcia_CT.nii.gz')
        mr_img_path = os.path.join(l2r_train, f'img{pid:04d}_tcia_MR.nii.gz')
        ct_seg_path = os.path.join(l2r_train, f'seg{pid:04d}_tcia_CT.nii.gz')
        mr_seg_path = os.path.join(l2r_train, f'seg{pid:04d}_tcia_MR.nii.gz')

        if not all(os.path.exists(p) for p in [ct_img_path, mr_img_path, ct_seg_path, mr_seg_path]):
            print(f"  Patient {pid:04d}: missing files, skipping")
            continue

        print(f"\n  Patient {pid:04d}:")

        ct_vol = nib.load(ct_img_path).get_fdata()
        mr_vol = nib.load(mr_img_path).get_fdata()
        ct_seg_vol = nib.load(ct_seg_path).get_fdata()
        mr_seg_vol = nib.load(mr_seg_path).get_fdata()

        labels = sorted(set(np.unique(ct_seg_vol).astype(int)) - {0})
        print(f"    Labels: {labels}, CT shape: {ct_vol.shape}")

        mid_slice = ct_vol.shape[2] // 2
        for slice_idx in [mid_slice - 5, mid_slice, mid_slice + 5]:
            if slice_idx < 0 or slice_idx >= ct_vol.shape[2]:
                continue

            ct_slice = ct_vol[:, :, slice_idx]
            mr_slice = mr_vol[:, :, min(slice_idx, mr_vol.shape[2] - 1)]

            ct_2d = ((ct_slice - ct_slice.min()) / (ct_slice.max() - ct_slice.min() + 1e-10) * 255).astype(np.uint8)
            mr_2d = ((mr_slice - mr_slice.min()) / (mr_slice.max() - mr_slice.min() + 1e-10) * 255).astype(np.uint8)

            ct_2d = np.array(Image.fromarray(ct_2d).resize((512, 512), Image.BICUBIC))
            mr_2d = np.array(Image.fromarray(mr_2d).resize((512, 512), Image.BICUBIC))

            ct_seg_2d = np.array(Image.fromarray(
                ct_seg_vol[:, :, slice_idx].astype(np.uint8)).resize((512, 512), Image.NEAREST))
            mr_seg_2d = np.array(Image.fromarray(
                mr_seg_vol[:, :, min(slice_idx, mr_vol.shape[2] - 1)].astype(np.uint8)).resize((512, 512), Image.NEAREST))

            ct_float = ct_2d.astype(np.float32) / 255.0
            mr_float = mr_2d.astype(np.float32) / 255.0

            for mname, mfunc, margs in methods:
                result = mfunc(ct_float, mr_float, *margs) if margs else mfunc(ct_float, mr_float)
                if result[0] is None:
                    continue
                warped, flow, flow_format = result

                if flow is not None:
                    warped_ct_seg = warp_segmentation_2d(ct_seg_2d, flow, flow_format)
                else:
                    warped_ct_seg = ct_seg_2d

                patient_dsc = []
                for label in labels:
                    pred = (warped_ct_seg == label)
                    target = (mr_seg_2d == label)
                    if pred.sum() == 0 and target.sum() == 0:
                        continue
                    dsc = compute_dsc(pred, target)
                    patient_dsc.append(dsc)

                if patient_dsc:
                    mean_dsc = float(np.mean(patient_dsc))
                    all_dsc[mname].append(mean_dsc)
                    print(f"    Slice {slice_idx}, {mname}: DSC={mean_dsc:.4f}")

    return all_dsc


# ── TRE Evaluation (RIRE) ──────────────────────────────────────────

def eval_tre_rire(methods):
    """Evaluate TRE on RIRE patient_001 using fiducial markers."""
    gt_path = './datasets/RIRE_2d/ground_truth/ct_T1.standard'
    ct_dir = './datasets/RIRE_2d/ct'
    mr_dir = './datasets/RIRE_2d/mr_t1'

    if not os.path.exists(gt_path):
        print("  RIRE ground truth not found")
        return {}

    ct_fid_mm, mr_fid_mm = load_rire_fiducials(gt_path)
    ct_spacing = 0.653595
    mr_spacing = 1.25

    ct_paths = sorted([os.path.join(ct_dir, f) for f in os.listdir(ct_dir)
                       if f.startswith('patient_001_ct') and f.endswith('.png')])
    mr_paths = sorted([os.path.join(mr_dir, f) for f in os.listdir(mr_dir)
                       if f.startswith('patient_001_mr') and f.endswith('.png')])

    all_tre = {m[0]: [] for m in methods}

    test_slices = [0, len(ct_paths) - 1]

    for slice_idx in test_slices:
        if slice_idx >= len(ct_paths) or slice_idx >= len(mr_paths):
            continue

        print(f"\n  RIRE Slice {slice_idx}:")

        ct_img = load_image(ct_paths[slice_idx])
        mr_img = load_image(mr_paths[slice_idx])

        if slice_idx == 0:
            ct_fid = ct_fid_mm[:4]
            mr_fid = mr_fid_mm[:4]
        else:
            ct_fid = ct_fid_mm[4:]
            mr_fid = mr_fid_mm[4:]

        for mname, mfunc, margs in methods:
            result = mfunc(ct_img, mr_img, *margs) if margs else mfunc(ct_img, mr_img)
            if result[0] is None:
                continue
            warped, flow, flow_format = result

            if flow is not None:
                tre_mm, tre_list = compute_2d_tre(
                    ct_fid, mr_fid, ct_spacing, mr_spacing, flow, flow_format, img_size=(512, 512)
                )
                all_tre[mname].append(tre_mm)
                print(f"    {mname}: TRE={tre_mm:.2f} mm (points: {[f'{t:.1f}' for t in tre_list]})")
            else:
                print(f"    {mname}: No flow available for TRE")

    return all_tre


# ── Main ────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip_tre', action='store_true', help='Skip TRE evaluation')
    parser.add_argument('--skip_dsc', action='store_true', help='Skip DSC evaluation')
    parser.add_argument('--max_patients', type=int, default=8, help='Max L2R patients for DSC')
    args = parser.parse_args()

    print("=" * 60)
    print("DSC & TRE Evaluation")
    print("=" * 60)

    methods = [
        ('Falcon (Ours)',   run_falcon,       ['l2r2d_ukan_gbcm_contrastive']),
        ('TransMorph',      run_transmorph,   ['transmorph_l2r']),
        ('ConvexAdam',      run_convexadam,   []),
        ('DINO-Reg',        run_dinoreg,      []),
        ('VoxelMorph-MI',   run_voxelmorph_mi,['voxelmorph_l2r']),
    ]

    tre_results = {}
    dsc_results = {}

    # TRE on RIRE (note: requires original image headers for accurate results)
    if not args.skip_tre:
        print("\n[1/2] TRE Evaluation (RIRE patient_001)...")
        tre_results = eval_tre_rire(methods)

    # DSC on L2R
    if not args.skip_dsc:
        print("\n[2/2] DSC Evaluation (L2R)...")
        dsc_results = eval_dsc_l2r(methods, max_patients=args.max_patients)

    # Print summary
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)

    if tre_results:
        print("\nTRE (mm, lower is better):")
        for name, vals in tre_results.items():
            if vals:
                print(f"  {name:20s}: {np.mean(vals):.2f} +/- {np.std(vals):.2f}")

    if dsc_results:
        print("\nDSC (higher is better):")
        for name, vals in dsc_results.items():
            if vals:
                print(f"  {name:20s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

    # Save
    os.makedirs(CACHE_DIR, exist_ok=True)
    results = {'tre': tre_results, 'dsc': dsc_results}
    with open(os.path.join(CACHE_DIR, 'dsc_tre.json'), 'w') as f:
        json.dump(results, f, indent=2, default=float)

    print(f"\nSaved to {CACHE_DIR}/dsc_tre.json")


if __name__ == '__main__':
    main()

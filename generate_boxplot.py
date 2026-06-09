#!/usr/bin/env python3
"""
Collect per-sample metrics from all models and generate boxplot figures.
"""
import os, sys, json
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
OUT_DIR = './paper_figures'
CACHE_DIR = './paper_figures/metrics_cache'


def load_image(path, size=(512, 512)):
    img = Image.open(path).convert('L')
    if img.size != size:
        img = img.resize(size, Image.BICUBIC)
    return np.array(img).astype(np.float32) / 255.0


def to_tensor(arr):
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0) * 2 - 1


def from_tensor(t):
    return ((t[0, 0].cpu().detach().numpy() + 1) / 2).clip(0, 1)


def get_pairs(dataroot):
    dA, dB = os.path.join(dataroot, 'trainA'), os.path.join(dataroot, 'trainB')
    a = sorted([os.path.join(dA, f) for f in os.listdir(dA) if f.endswith('.png')])
    b = sorted([os.path.join(dB, f) for f in os.listdir(dB) if f.endswith('.png')])
    bm = {os.path.basename(p): p for p in b}
    return [(p, bm[os.path.basename(p)]) for p in a if os.path.basename(p) in bm]


def parse_opt_txt(path):
    opts = {}
    with open(path) as f:
        import ast
        for line in f:
            line = line.strip()
            if ':' in line and not line.startswith('-'):
                k, v = line.split(':', 1)
                k, v = k.strip(), v.strip().split('\t')[0].strip()
                try: opts[k] = ast.literal_eval(v)
                except: opts[k] = v
    return opts


# ── Metrics ──────────────────────────────────────────────────────────

def compute_metrics(warped, fixed):
    """Compute metrics on [0,1] numpy arrays."""
    mse = float(np.mean((warped - fixed) ** 2))
    mae = float(np.mean(np.abs(warped - fixed)))
    psnr = float(20 * np.log10(1.0 / (np.sqrt(mse) + 1e-10)))

    from skimage.metrics import structural_similarity as ssim_fn
    s = float(ssim_fn((warped * 255).astype(np.uint8), (fixed * 255).astype(np.uint8), data_range=255))

    w, f = warped.flatten(), fixed.flatten()
    wm, fm = w.mean(), f.mean()
    ncc = float(np.sum((w - wm) * (f - fm)) / (np.sqrt(np.sum((w - wm)**2) * np.sum((f - fm)**2)) + 1e-10))

    return {'mse': mse, 'mae': mae, 'psnr': psnr, 'ssim': s, 'ncc': ncc}


# ── Model runners (same as generate_paper_figures.py) ──────────────

_falcon_cache = {}

def run_falcon(moving, fixed, ckpt_name):
    from models.nemar_model import NEMARModel
    import argparse

    if ckpt_name not in _falcon_cache:
        opt_path = f'./checkpoints/{ckpt_name}/train_opt.txt'
        if not os.path.exists(opt_path): opt_path = f'./checkpoints/{ckpt_name}/test_opt.txt'
        if not os.path.exists(opt_path): return None
        opt_dict = parse_opt_txt(opt_path)
        opt = argparse.Namespace()
        for k, v in opt_dict.items(): setattr(opt, k, v)
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
            if not hasattr(opt, k): setattr(opt, k, v)
        opt.isTrain = False
        opt.gpu_ids = [0]
        try:
            model = NEMARModel(opt)
            model.load_networks('latest')
            model.eval()
            _falcon_cache[ckpt_name] = model
        except Exception as e:
            print(f"  Falcon init error: {e}")
            return None

    model = _falcon_cache[ckpt_name]
    m, f = to_tensor(moving).to(DEVICE), to_tensor(fixed).to(DEVICE)
    with torch.no_grad():
        model.set_input({'A': m, 'B': f, 'A_paths': ''})
        model.forward()
    return from_tensor(model.registered_real_A)


_vm_cache = {}

def run_voxelmorph_mi(moving, fixed, ckpt_name):
    from models.voxelmorph_model import Unet, SpatialTransformer
    if ckpt_name not in _vm_cache:
        net = Unet(in_channels=2, out_channels=2).to(DEVICE)
        p = f'./checkpoints/{ckpt_name}/latest_net_V.pth'
        if not os.path.exists(p): return None
        net.load_state_dict(torch.load(p, map_location=DEVICE, weights_only=True))
        net.eval()
        _vm_cache[ckpt_name] = (net, SpatialTransformer())
    net, st = _vm_cache[ckpt_name]
    m, f = to_tensor(moving).to(DEVICE), to_tensor(fixed).to(DEVICE)
    with torch.no_grad():
        flow = net(torch.cat([m, f], dim=1))
        warped = st(m, flow)
    return from_tensor(warped)


_tm_cache = {}

def run_transmorph(moving, fixed, ckpt_name):
    from models.transmorph_model import TransMorphNet
    if ckpt_name not in _tm_cache:
        net = TransMorphNet(img_size=(512, 512)).to(DEVICE)
        p = f'./checkpoints/{ckpt_name}/latest_net_TM.pth'
        if not os.path.exists(p): return None
        sd = torch.load(p, map_location=DEVICE, weights_only=True)
        sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
        net.load_state_dict(sd)
        net.eval()
        _tm_cache[ckpt_name] = net
    net = _tm_cache[ckpt_name]
    m, f = to_tensor(moving).to(DEVICE), to_tensor(fixed).to(DEVICE)
    with torch.no_grad():
        warped, _ = net(torch.cat([m, f], dim=1))
    return from_tensor(warped.clamp(-1, 1))


def run_convexadam(moving, fixed):
    from models.convexadam_2d import convex_adam_2d
    m, f = to_tensor(moving).to(DEVICE), to_tensor(fixed).to(DEVICE)
    result = convex_adam_2d(f, m)
    warped = result[0] if isinstance(result, tuple) else result
    return from_tensor(warped)


def run_dinoreg(moving, fixed):
    from models.dinoreg_2d import dino_reg_2d
    m, f = to_tensor(moving).to(DEVICE), to_tensor(fixed).to(DEVICE)
    result = dino_reg_2d(f, m, feat_size=(36, 36), n_iter_adam=200, lr_adam=3.0)
    warped = result[0] if isinstance(result, tuple) else result
    return from_tensor(warped)


# ── Collect metrics ──────────────────────────────────────────────────

def collect_metrics(pairs, methods, max_samples=100, cache_name=None):
    """Run all models on paired images and collect per-sample metrics."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f'{cache_name}.json') if cache_name else None

    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            all_metrics = json.load(f)
        print(f"  Loaded cached metrics from {cache_path}")
        return all_metrics

    all_metrics = {m[0]: [] for m in methods}
    n = min(max_samples, len(pairs))

    for i in range(n):
        path_a, path_b = pairs[i]
        mv, fx = load_image(path_a), load_image(path_b)
        print(f"  [{i+1}/{n}]", end='', flush=True)

        for mname, mfunc, margs in methods:
            r = mfunc(mv, fx, *margs) if margs else mfunc(mv, fx)
            if r is not None:
                m = compute_metrics(r, fx)
                all_metrics[mname].append(m)
            else:
                all_metrics[mname].append(None)
        print()

    # Filter out None entries
    for k in all_metrics:
        all_metrics[k] = [x for x in all_metrics[k] if x is not None]

    if cache_path:
        with open(cache_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        print(f"  Cached metrics to {cache_path}")

    return all_metrics


# ── Boxplot ──────────────────────────────────────────────────────────

def fig_boxplot(all_metrics, dataset_tag, metric_key, metric_label, higher_better=True):
    """Generate a single boxplot for one metric."""
    model_names = list(all_metrics.keys())
    data = []
    valid_names = []
    for name in model_names:
        vals = [m[metric_key] for m in all_metrics[name]]
        if vals:
            data.append(vals)
            valid_names.append(name)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Color: highlight Falcon
    colors = ['#E74C3C' if 'Falcon' in n else '#3498DB' for n in valid_names]

    bp = ax.boxplot(data, labels=valid_names, patch_artist=True, widths=0.6,
                    medianprops=dict(color='black', linewidth=1.5),
                    whiskerprops=dict(linewidth=1),
                    capprops=dict(linewidth=1))

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel(metric_label, fontsize=12)
    ax.tick_params(axis='x', rotation=15, labelsize=9)
    ax.grid(axis='y', alpha=0.3)

    if higher_better:
        ax.text(0.98, 0.98, '↑ better', transform=ax.transAxes, ha='right', va='top',
                fontsize=9, color='gray')
    else:
        ax.text(0.98, 0.98, '↓ better', transform=ax.transAxes, ha='right', va='top',
                fontsize=9, color='gray')

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        p = os.path.join(OUT_DIR, f'boxplot_{metric_key}_{dataset_tag}.{ext}')
        fig.savefig(p, dpi=300, bbox_inches='tight')
        print(f"  Saved {p}")
    plt.close(fig)


def fig_combined_boxplot(all_metrics_rire, all_metrics_l2r, metric_key, metric_label,
                          higher_better=True):
    """Generate combined boxplot with RIRE and L2R side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, all_metrics, tag in [(ax1, all_metrics_rire, 'RIRE'), (ax2, all_metrics_l2r, 'L2R')]:
        model_names = list(all_metrics.keys())
        data, valid_names = [], []
        for name in model_names:
            vals = [m[metric_key] for m in all_metrics[name]]
            if vals:
                data.append(vals)
                valid_names.append(name)

        colors = ['#E74C3C' if 'Falcon' in n else '#3498DB' for n in valid_names]
        bp = ax.boxplot(data, labels=valid_names, patch_artist=True, widths=0.6,
                        medianprops=dict(color='black', linewidth=1.5),
                        whiskerprops=dict(linewidth=1), capprops=dict(linewidth=1))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_title(tag, fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=20, labelsize=8)
        ax.grid(axis='y', alpha=0.3)

    fig.text(0.5, 0.0, metric_label, ha='center', fontsize=12)
    if higher_better:
        fig.text(0.98, 0.98, '↑ better', ha='right', va='top', fontsize=9, color='gray')

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        p = os.path.join(OUT_DIR, f'boxplot_{metric_key}_combined.{ext}')
        fig.savefig(p, dpi=300, bbox_inches='tight')
        print(f"  Saved {p}")
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Collecting per-sample metrics & generating boxplots")
    print("=" * 60)

    methods = [
        ('Falcon (Ours)',   run_falcon,       ['rire2d_ukan_gbcm_contrastive']),
        ('TransMorph',      run_transmorph,   ['transmorph_rire']),
        ('ConvexAdam',      run_convexadam,   []),
        ('DINO-Reg',        run_dinoreg,      []),
        ('VoxelMorph-MI',   run_voxelmorph_mi,['voxelmorph_rire']),
    ]

    rire_pairs = get_pairs('./datasets/RIRE_2d_paired')

    # Collect RIRE metrics (100 samples)
    print("\n[1] Collecting RIRE metrics (100 samples)...")
    rire_metrics = collect_metrics(rire_pairs, methods, max_samples=100, cache_name='rire_100')

    # Generate individual boxplots
    print("\n[2] Generating boxplots...")
    metrics_info = [
        ('ssim', 'SSIM', True),
        ('ncc', 'NCC', True),
        ('psnr', 'PSNR (dB)', True),
        ('mse', 'MSE', False),
    ]

    for key, label, higher in metrics_info:
        fig_boxplot(rire_metrics, 'rire', key, label, higher)

    print("\nDone! All boxplots saved to ./paper_figures/")


if __name__ == '__main__':
    main()

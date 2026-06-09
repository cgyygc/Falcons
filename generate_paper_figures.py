#!/usr/bin/env python3
"""Generate publication-quality figures for the paper.

Produces:
1. Multi-method comparison (RIRE)
2. Difference maps
3. Architecture diagram
4. Ablation visualization
5. Performance bar chart
"""
import os, sys, re, ast
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
OUT_DIR = './paper_figures'


# ── Helpers ──────────────────────────────────────────────────────────────

def load_image(path, size=(512, 512)):
    img = Image.open(path).convert('L')
    if img.size != size:
        img = img.resize(size, Image.BICUBIC)
    return np.array(img).astype(np.float32) / 255.0


def to_tensor(arr):
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0) * 2 - 1  # [0,1]→[-1,1]


def from_tensor(t):
    return ((t[0, 0].cpu().detach().numpy() + 1) / 2).clip(0, 1)


def get_pairs(dataroot):
    dA = os.path.join(dataroot, 'trainA')
    dB = os.path.join(dataroot, 'trainB')
    a = sorted([os.path.join(dA, f) for f in os.listdir(dA) if f.endswith('.png')])
    b = sorted([os.path.join(dB, f) for f in os.listdir(dB) if f.endswith('.png')])
    bm = {os.path.basename(p): p for p in b}
    return [(p, bm[os.path.basename(p)]) for p in a if os.path.basename(p) in bm]


def parse_opt_txt(path):
    """Parse opt.txt saved by train.py into a dict."""
    opts = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if ':' in line and not line.startswith('-'):
                k, v = line.split(':', 1)
                k = k.strip()
                v = v.strip().split('\t')[0].strip()  # remove [default: ...]
                try:
                    opts[k] = ast.literal_eval(v)
                except:
                    opts[k] = v
    return opts


# ── Model runners ──────────────────────────────────────────────────────

def run_falcon(moving, fixed, ckpt_name):
    """Run Falcon using saved opt.txt to reconstruct opt."""
    from models.nemar_model import NEMARModel

    opt_path = f'./checkpoints/{ckpt_name}/train_opt.txt'
    if not os.path.exists(opt_path):
        opt_path = f'./checkpoints/{ckpt_name}/test_opt.txt'
    if not os.path.exists(opt_path):
        return None

    opt_dict = parse_opt_txt(opt_path)

    # Build a simple namespace with all needed attributes
    import argparse
    opt = argparse.Namespace()

    # Set all saved options
    for k, v in opt_dict.items():
        setattr(opt, k, v)

    # Ensure required attributes with defaults
    defaults = {
        'isTrain': False, 'gpu_ids': '0', 'phase': 'test', 'device': DEVICE,
        'num_test': 1, 'serial_batches': True, 'epoch': 'latest', 'load_iter': 0,
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
    opt.gpu_ids = [0]  # list of ints for networks.init_net

    try:
        model = NEMARModel(opt)
        model.load_networks('latest')
        model.eval()

        m = to_tensor(moving).to(DEVICE)
        f = to_tensor(fixed).to(DEVICE)
        with torch.no_grad():
            model.set_input({'A': m, 'B': f, 'A_paths': ''})
            model.forward()
        return from_tensor(model.registered_real_A)
    except Exception as e:
        print(f"  Falcon error: {e}")
        import traceback; traceback.print_exc()
        return None


def run_voxelmorph_mi(moving, fixed, ckpt_name):
    from models.voxelmorph_model import Unet, SpatialTransformer
    net = Unet(in_channels=2, out_channels=2).to(DEVICE)
    p = f'./checkpoints/{ckpt_name}/latest_net_V.pth'
    if not os.path.exists(p): return None
    net.load_state_dict(torch.load(p, map_location=DEVICE, weights_only=True))
    net.eval()
    st = SpatialTransformer()
    m, f = to_tensor(moving).to(DEVICE), to_tensor(fixed).to(DEVICE)
    with torch.no_grad():
        flow = net(torch.cat([m, f], dim=1))
        warped = st(m, flow)
    return from_tensor(warped)


def run_transmorph(moving, fixed, ckpt_name):
    from models.transmorph_model import TransMorphNet
    net = TransMorphNet(img_size=(512, 512)).to(DEVICE)
    p = f'./checkpoints/{ckpt_name}/latest_net_TM.pth'
    if not os.path.exists(p): return None
    sd = torch.load(p, map_location=DEVICE, weights_only=True)
    sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
    net.load_state_dict(sd)
    net.eval()
    m, f = to_tensor(moving).to(DEVICE), to_tensor(fixed).to(DEVICE)
    with torch.no_grad():
        warped, _ = net(torch.cat([m, f], dim=1))
    return from_tensor(warped.clamp(-1, 1))


def run_convexadam(moving, fixed):
    from models.convexadam_2d import convex_adam_2d
    m, f = to_tensor(moving).to(DEVICE), to_tensor(fixed).to(DEVICE)
    try:
        result = convex_adam_2d(f, m)
        warped = result[0] if isinstance(result, tuple) else result
        return from_tensor(warped)
    except Exception as e:
        print(f"  ConvexAdam: {e}")
        return None


def run_dinoreg(moving, fixed):
    from models.dinoreg_2d import dino_reg_2d
    m, f = to_tensor(moving).to(DEVICE), to_tensor(fixed).to(DEVICE)
    try:
        result = dino_reg_2d(f, m, feat_size=(36, 36), n_iter_adam=200, lr_adam=3.0)
        warped = result[0] if isinstance(result, tuple) else result
        return from_tensor(warped)
    except Exception as e:
        print(f"  DINO-Reg: {e}")
        return None


def run_mi_reg(moving, fixed):
    from models.mi_registration_model import NormalizedMutualInformationLoss, GridSampler
    from torch.optim import LBFGS
    m, f = to_tensor(moving).to(DEVICE), to_tensor(fixed).to(DEVICE)
    gs = GridSampler()
    theta = torch.zeros(1, 6, device=DEVICE, requires_grad=True)
    mi = NormalizedMutualInformationLoss(num_bins=64)
    opt = LBFGS([theta], lr=0.1, max_iter=100, line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad()
        return mi(gs(m, theta), f)
    try:
        opt.step(closure)
        with torch.no_grad():
            return from_tensor(gs(m, theta))
    except:
        return None


# ── Figure: Multi-method comparison ──────────────────────────────────

def fig_comparison(pairs, methods, sample_ids, dataset_tag):
    """Side-by-side registration comparison."""
    rows = []
    for sid in sample_ids:
        if sid >= len(pairs): continue
        a, b = pairs[sid]
        mv, fx = load_image(a), load_image(b)
        row = [mv, fx]
        for mname, mfunc, margs in methods:
            print(f"    {mname}...", end=' ', flush=True)
            r = mfunc(mv, fx, *margs) if margs else mfunc(mv, fx)
            row.append(r if r is not None else np.zeros_like(mv))
            print("ok" if r is not None else "fail")
        rows.append(row)

    labels = ['CT (Moving)', 'MR (Fixed)'] + [m[0] for m in methods]
    ncols, nrows = len(labels), len(rows)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.2, nrows * 2.2))
    if nrows == 1: axes = axes[np.newaxis, :]
    for r, row in enumerate(rows):
        for c, (img, lab) in enumerate(zip(row, labels)):
            ax = axes[r, c]
            if img is not None and img.max() > 0:
                ax.imshow(img, cmap='gray', vmin=0, vmax=1)
            else:
                ax.text(.5, .5, 'N/A', ha='center', va='center', fontsize=8, color='gray')
            if r == 0: ax.set_title(lab, fontsize=9, fontweight='bold')
            ax.axis('off')
    plt.tight_layout(pad=0.3)
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ['png', 'pdf']:
        p = os.path.join(OUT_DIR, f'comparison_{dataset_tag}.{ext}')
        fig.savefig(p, dpi=300, bbox_inches='tight')
        print(f"  Saved {p}")
    plt.close(fig)


def fig_difference(pairs, methods, sample_ids, dataset_tag):
    """Difference map comparison |registered - fixed|."""
    rows = []
    for sid in sample_ids:
        if sid >= len(pairs): continue
        a, b = pairs[sid]
        mv, fx = load_image(a), load_image(b)
        diffs = []
        for mname, mfunc, margs in methods:
            r = mfunc(mv, fx, *margs) if margs else mfunc(mv, fx)
            d = np.abs(r - fx) if r is not None else np.zeros_like(mv)
            diffs.append(d)
        rows.append(diffs)

    labels = [m[0] for m in methods]
    ncols, nrows = len(labels), len(rows)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.2, nrows * 2.2))
    if nrows == 1: axes = axes[np.newaxis, :]
    for r, diffs in enumerate(rows):
        for c, (diff, lab) in enumerate(zip(diffs, labels)):
            ax = axes[r, c]
            ax.imshow(diff, cmap='hot', vmin=0, vmax=0.5)
            mae = diff.mean()
            ax.text(.5, .02, f'MAE={mae:.3f}', ha='center', va='bottom', fontsize=7,
                    color='white', transform=ax.transAxes,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5))
            if r == 0: ax.set_title(lab, fontsize=9, fontweight='bold')
            ax.axis('off')
    plt.tight_layout(pad=0.3)
    for ext in ['png', 'pdf']:
        p = os.path.join(OUT_DIR, f'difference_{dataset_tag}.{ext}')
        fig.savefig(p, dpi=300, bbox_inches='tight')
        print(f"  Saved {p}")
    plt.close(fig)


# ── Figure: Architecture ─────────────────────────────────────────────

def fig_architecture():
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14); ax.set_ylim(0, 6); ax.axis('off')

    def box(x, y, w, h, txt, col='#4A90D9', fs=9):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=col, ec='black', lw=1.5, alpha=.9, zorder=2))
        ax.text(x+w/2, y+h/2, txt, ha='center', va='center', fontsize=fs, fontweight='bold', color='white', zorder=3)

    def arrow(x1, y1, x2, y2, lab='', c='black'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle='->', color=c, lw=1.5))
        if lab:
            ax.text((x1+x2)/2, (y1+y2)/2+.15, lab, ha='center', fontsize=7, color=c, style='italic')

    ax.text(7, 5.7, 'Falcon: Joint Translation-Registration Framework', ha='center', fontsize=13, fontweight='bold')

    box(.3, 2.0, 1.5, 1.2, 'Input\nCT (A)', '#2ECC71')
    box(.3, 3.8, 1.5, 1.2, 'Input\nMR (B)', '#E74C3C')

    ax.text(4.5, 5.3, 'TR Path: A → R(A) → T(R(A))', ha='center', fontsize=9, color='#2980B9')
    box(2.5, 3.8, 1.8, 1.2, 'netR\n(Registration)', '#8E44AD')
    box(5.0, 3.8, 1.8, 1.2, 'netT\n(Translation)', '#2980B9')
    box(7.8, 3.8, 1.8, 1.2, 'fake_TR_B', '#F39C12')
    arrow(1.8, 4.4, 2.5, 4.4, 'real_A')
    arrow(4.3, 4.4, 5.0, 4.4, 'R(A)')
    arrow(6.8, 4.4, 7.8, 4.4, 'T(R(A))')

    ax.text(4.5, 1.5, 'RT Path: A → T(A) → R(T(A))', ha='center', fontsize=9, color='#E67E22')
    box(2.5, 2.0, 1.8, 1.2, 'netT\n(Translation)', '#2980B9')
    box(5.0, 2.0, 1.8, 1.2, 'netR\n(Registration)', '#8E44AD')
    box(7.8, 2.0, 1.8, 1.2, 'fake_RT_B', '#F39C12')
    arrow(1.8, 2.6, 2.5, 2.6, 'real_A')
    arrow(4.3, 2.6, 5.0, 2.6, 'T(A)')
    arrow(6.8, 2.6, 7.8, 2.6, 'R(T(A))')

    box(10.5, 3.0, 1.8, 1.5, 'netD\n(Discriminator)', '#C0392B')
    arrow(9.6, 4.4, 10.5, 4.0, '', '#F39C12')
    arrow(9.6, 2.6, 10.5, 3.2, '', '#F39C12')
    arrow(1.8, 4.4, 10.5, 3.8, 'real_B', '#E74C3C')

    ax.text(12.5, 5.0, 'Losses:', ha='center', fontsize=9, fontweight='bold')
    for i, l in enumerate(['L1_TR + GAN_TR', 'L1_RT + GAN_RT', 'Smoothness', 'Contrastive']):
        ax.text(12.5, 4.5 - i*.35, l, ha='center', fontsize=7.5, color='#2C3E50')

    for ext in ['png', 'pdf']:
        p = os.path.join(OUT_DIR, f'architecture.{ext}')
        fig.savefig(p, dpi=300, bbox_inches='tight')
        print(f"  Saved {p}")
    plt.close(fig)


# ── Figure: Ablation ─────────────────────────────────────────────────

def fig_ablation(pairs, sample_id, dataset_tag):
    a, b = pairs[sample_id]
    mv, fx = load_image(a), load_image(b)

    if 'RIRE' in dataset_tag:
        configs = [
            ('Falcon (Ours)', 'rire2d_ukan_gbcm_contrastive'),
            ('w/o Contrastive', 'ablation_weight_00'),
            ('w=0.2+Reg', 'ablation_weight_02'),
            ('w/o GBCM', 'ablation_no_gbcm'),
            ('Only Disc Noise', 'ablation_only_disc_noise'),
            ('STN Affine', 'ablation_stn_affine'),
            ('STN UKAN\n(no contr.)', 'ablation_stn_ukan'),
        ]
    else:
        configs = [
            ('Falcon (Ours)', 'l2r2d_ukan_gbcm_contrastive'),
            ('w/o Contrastive', 'l2r_ablation_weight_00'),
            ('w=0.2+Reg', 'l2r_ablation_weight_02'),
            ('w/o GBCM', 'l2r_ablation_no_gbcm'),
            ('Only Disc Noise', 'l2r_ablation_only_disc_noise'),
            ('STN Affine', 'l2r_ablation_stn_affine'),
            ('STN UKAN\n(no contr.)', 'l2r_ablation_stn_ukan'),
        ]

    results = [('Moving\n(CT)', mv), ('Fixed\n(MR)', fx)]
    for name, ckpt in configs:
        print(f"    {name}...", end=' ', flush=True)
        r = run_falcon(mv, fx, ckpt)
        results.append((name, r if r is not None else np.zeros_like(mv)))
        print("ok" if r is not None else "fail")

    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(n * 2.0, 2.5))
    for i, (name, img) in enumerate(results):
        ax = axes[i]
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(name, fontsize=7.5, fontweight='bold')
        ax.axis('off')
    plt.tight_layout(pad=0.3)
    for ext in ['png', 'pdf']:
        p = os.path.join(OUT_DIR, f'ablation_{dataset_tag}.{ext}')
        fig.savefig(p, dpi=300, bbox_inches='tight')
        print(f"  Saved {p}")
    plt.close(fig)


# ── Figure: Bar chart ─────────────────────────────────────────────────

def fig_bar_chart():
    models = ['Falcon\n(Ours)', 'Falcon\n-Stable', 'Trans-\nMorph', 'Convex\nAdam',
              'DINO-\nReg', 'VoxelMorph\n-MI', 'MI\nReg.']
    rire = [0.8498, 0.8666, 0.6986, 0.6518, 0.6499, 0.2987, 0.1820]
    l2r  = [0.8707, 0.9481, 0.3200, 0.3210, 0.3173, 0.3789, 0.3097]

    x = np.arange(len(models))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - w/2, rire, w, label='RIRE', color='#3498DB', ec='black', lw=.5)
    b2 = ax.bar(x + w/2, l2r,  w, label='L2R',  color='#E74C3C', ec='black', lw=.5)
    for b in [b1[0], b1[1], b2[0], b2[1]]:
        b.set_linewidth(1.5)
    ax.set_ylabel('SSIM ↑', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=8)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=.3)
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + .01, f'{h:.2f}',
                    ha='center', va='bottom', fontsize=7)
    plt.tight_layout()
    for ext in ['png', 'pdf']:
        p = os.path.join(OUT_DIR, f'bar_chart_ssim.{ext}')
        fig.savefig(p, dpi=300, bbox_inches='tight')
        print(f"  Saved {p}")
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Generating Paper Figures")
    print("=" * 60)

    rire_pairs = get_pairs('./datasets/RIRE_2d_paired')
    sample_ids = [50, 150, 250]

    methods_rire = [
        ('Falcon (Ours)',   run_falcon,       ['rire2d_ukan_gbcm_contrastive']),
        ('TransMorph',      run_transmorph,   ['transmorph_rire']),
        ('ConvexAdam',      run_convexadam,   []),
        ('DINO-Reg',        run_dinoreg,      []),
        ('VoxelMorph-MI',   run_voxelmorph_mi,['voxelmorph_rire']),
    ]

    # 1. Comparison figure
    print("\n[1/5] Comparison (RIRE)...")
    fig_comparison(rire_pairs, methods_rire, sample_ids, 'rire')

    # 2. Difference map
    print("\n[2/5] Difference maps (RIRE)...")
    fig_difference(rire_pairs, methods_rire, [sample_ids[0]], 'rire')

    # 3. Architecture
    print("\n[3/5] Architecture diagram...")
    fig_architecture()

    # 4. Ablation
    print("\n[4/5] Ablation (RIRE)...")
    fig_ablation(rire_pairs, 80, 'rire')

    # 5. Bar chart
    print("\n[5/5] Bar chart...")
    fig_bar_chart()

    print("\n" + "=" * 60)
    print(f"All figures → {OUT_DIR}/")
    print("=" * 60)


if __name__ == '__main__':
    main()

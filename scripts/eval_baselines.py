"""
Evaluate TransMorph on paired test images.

Usage:
    python scripts/eval_transmorph.py --name transmorph_rire --dataset rire --gpu 0
"""
import os
import argparse
import numpy as np
import torch
from PIL import Image
from collections import defaultdict
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_image(path, target_size=(512, 512)):
    img = Image.open(path).convert('L')
    if img.size != target_size:
        img = img.resize(target_size, Image.BICUBIC)
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    return tensor * 2 - 1


def get_paired_paths(dataroot):
    dir_a = os.path.join(dataroot, 'trainA')
    dir_b = os.path.join(dataroot, 'trainB')
    if not os.path.exists(dir_a):
        dir_a = os.path.join(dataroot, 'testA')
        dir_b = os.path.join(dataroot, 'testB')
    a_paths = sorted([os.path.join(dir_a, f) for f in os.listdir(dir_a)
                       if f.endswith(('.png', '.jpg', '.jpeg'))])
    b_paths = sorted([os.path.join(dir_b, f) for f in os.listdir(dir_b)
                       if f.endswith(('.png', '.jpg', '.jpeg'))])
    b_map = {os.path.basename(p): p for p in b_paths}
    pairs = []
    for a_path in a_paths:
        name = os.path.basename(a_path)
        if name in b_map:
            pairs.append((a_path, b_map[name]))
    return pairs


def mse_metric(pred, target):
    return torch.mean((pred - target) ** 2).item()


def mae_metric(pred, target):
    return torch.mean(torch.abs(pred - target)).item()


def psnr_metric(pred, target):
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return float('inf')
    return 20 * np.log10(2.0 / np.sqrt(mse))


def ssim_metric(pred, target):
    from torch.nn.functional import avg_pool2d
    C1 = (0.01 * 2.0) ** 2
    C2 = (0.03 * 2.0) ** 2
    mu1 = avg_pool2d(pred, 11, stride=1, padding=5)
    mu2 = avg_pool2d(target, 11, stride=1, padding=5)
    sigma1 = avg_pool2d(pred ** 2, 11, stride=1, padding=5) - mu1 ** 2
    sigma2 = avg_pool2d(target ** 2, 11, stride=1, padding=5) - mu2 ** 2
    sigma12 = avg_pool2d(pred * target, 11, stride=1, padding=5) - mu1 * mu2
    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2))
    return ssim_map.mean().item()


def ncc_metric(pred, target):
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    pred_norm = (pred_flat - pred_flat.mean()) / (pred_flat.std() + 1e-8)
    target_norm = (target_flat - target_flat.mean()) / (target_flat.std() + 1e-8)
    return torch.dot(pred_norm, target_norm).item() / pred_norm.numel()


def eval_transmorph(name, dataset, gpu, num_test=0):
    from models.transmorph_model import TransMorphNet
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')

    if dataset == 'rire':
        dataroot = './datasets/RIRE_2d_paired'
    else:
        dataroot = './datasets/L2R_2d/Train'

    pairs = get_paired_paths(dataroot)
    if num_test > 0:
        pairs = pairs[:num_test]

    # Load model
    ckpt_path = f'./checkpoints/{name}/latest_net_TM.pth'
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return

    net = TransMorphNet(img_size=(512, 512))
    state_dict = torch.load(ckpt_path, map_location=device)
    # Handle DataParallel state dict
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    net.load_state_dict(new_state_dict)
    net = net.to(device).eval()

    metrics = defaultdict(list)
    with torch.no_grad():
        for idx, (path_a, path_b) in enumerate(pairs):
            moving = load_image(path_a).to(device)
            fixed = load_image(path_b).to(device)
            x_in = torch.cat([moving, fixed], dim=1)
            warped, flow = net(x_in)
            warped = warped.clamp(-1, 1)

            m = {
                'mse': mse_metric(warped, fixed),
                'mae': mae_metric(warped, fixed),
                'psnr': psnr_metric(warped, fixed),
                'ssim': ssim_metric(warped, fixed),
                'ncc': ncc_metric(warped, fixed),
            }
            for k, v in m.items():
                metrics[k].append(v)

            if (idx + 1) % 20 == 0 or idx == 0:
                print(f"[{idx+1}/{len(pairs)}] PSNR={m['psnr']:.2f} SSIM={m['ssim']:.4f} NCC={m['ncc']:.4f}")

    print(f"\n{'='*60}")
    print(f"TransMorph Results on {dataset}")
    print(f"{'='*60}")
    for k in ['mse', 'mae', 'psnr', 'ssim', 'ncc']:
        vals = metrics[k]
        print(f"{k.upper():6s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    print(f"{'='*60}")
    return {k: (np.mean(v), np.std(v)) for k, v in metrics.items()}


def eval_voxelmorph(dataset, gpu, num_test=0):
    """Evaluate existing VoxelMorph-MI model."""
    from models.voxelmorph_model import Unet, SpatialTransformer

    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')

    if dataset == 'rire':
        dataroot = './datasets/RIRE_2d_paired'
        name = 'voxelmorph_rire'
    else:
        dataroot = './datasets/L2R_2d/Train'
        name = 'voxelmorph_l2r'

    pairs = get_paired_paths(dataroot)
    if num_test > 0:
        pairs = pairs[:num_test]

    # Load model directly
    netV = Unet(in_channels=2, out_channels=2).to(device)
    ckpt_path = f'./checkpoints/{name}/latest_net_V.pth'
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return

    state_dict = torch.load(ckpt_path, map_location=device)
    netV.load_state_dict(state_dict)
    netV = netV.eval()

    spatial_transform = SpatialTransformer()

    metrics = defaultdict(list)
    with torch.no_grad():
        for idx, (path_a, path_b) in enumerate(pairs):
            moving = load_image(path_a).to(device)
            fixed = load_image(path_b).to(device)
            x = torch.cat([moving, fixed], dim=1)
            flow = netV(x)
            warped = spatial_transform(moving, flow)
            warped = warped.clamp(-1, 1)

            m = {
                'mse': mse_metric(warped, fixed),
                'mae': mae_metric(warped, fixed),
                'psnr': psnr_metric(warped, fixed),
                'ssim': ssim_metric(warped, fixed),
                'ncc': ncc_metric(warped, fixed),
            }
            for k, v in m.items():
                metrics[k].append(v)

            if (idx + 1) % 20 == 0 or idx == 0:
                print(f"[{idx+1}/{len(pairs)}] PSNR={m['psnr']:.2f} SSIM={m['ssim']:.4f} NCC={m['ncc']:.4f}")

    print(f"\n{'='*60}")
    print(f"VoxelMorph-MI Results on {dataset}")
    print(f"{'='*60}")
    for k in ['mse', 'mae', 'psnr', 'ssim', 'ncc']:
        vals = metrics[k]
        print(f"{k.upper():6s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    print(f"{'='*60}")
    return {k: (np.mean(v), np.std(v)) for k, v in metrics.items()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, required=True, choices=['transmorph', 'voxelmorph'])
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--dataset', type=str, required=True, choices=['rire', 'l2r'])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_test', type=int, default=0)
    args = parser.parse_args()

    if args.method == 'transmorph':
        eval_transmorph(args.name or f'transmorph_{args.dataset}', args.dataset, args.gpu, args.num_test)
    elif args.method == 'voxelmorph':
        eval_voxelmorph(args.dataset, args.gpu, args.num_test)

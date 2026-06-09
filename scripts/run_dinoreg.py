"""
Run DINO-Reg 2D registration on paired test images and evaluate.

Usage:
    python scripts/run_dinoreg.py --dataset rire --gpu 0
    python scripts/run_dinoreg.py --dataset l2r --gpu 0
"""
import os
import argparse
import numpy as np
import torch
from PIL import Image
from collections import defaultdict
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.dinoreg_2d import dino_reg_2d, DINOv2FeatureExtractor


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['rire', 'l2r'])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--grid_sp', type=int, default=2)
    parser.add_argument('--disp_hw', type=int, default=3)
    parser.add_argument('--lambda_weight', type=float, default=2.0)
    parser.add_argument('--n_iter_adam', type=int, default=200)
    parser.add_argument('--lr_adam', type=float, default=3.0)
    parser.add_argument('--reg_feature_dim', type=int, default=24)
    parser.add_argument('--feat_size_h', type=int, default=36)
    parser.add_argument('--feat_size_w', type=int, default=36)
    parser.add_argument('--num_test', type=int, default=0)
    parser.add_argument('--save_results', action='store_true')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    if args.dataset == 'rire':
        dataroot = './datasets/RIRE_2d_paired'
    else:
        dataroot = './datasets/L2R_2d/Train'

    pairs = get_paired_paths(dataroot)
    if args.num_test > 0:
        pairs = pairs[:args.num_test]

    print(f"DINO-Reg 2D on {args.dataset}: {len(pairs)} pairs")

    # Pre-load DINOv2 extractor
    extractor = DINOv2FeatureExtractor(
        feat_size=(args.feat_size_h, args.feat_size_w),
        device=device
    )

    results_dir = f'./results/dinoreg_{args.dataset}'
    if args.save_results:
        os.makedirs(results_dir, exist_ok=True)

    metrics = defaultdict(list)

    for idx, (path_a, path_b) in enumerate(pairs):
        moving = load_image(path_a).to(device)
        fixed = load_image(path_b).to(device)

        warped, disp = dino_reg_2d(
            fixed, moving,
            feat_size=(args.feat_size_h, args.feat_size_w),
            reg_feature_dim=args.reg_feature_dim,
            grid_sp=args.grid_sp,
            disp_hw=args.disp_hw,
            lambda_weight=args.lambda_weight,
            n_iter_adam=args.n_iter_adam,
            lr_adam=args.lr_adam,
            extractor=extractor
        )

        warped = warped.detach().clamp(-1, 1)

        m = {
            'mse': mse_metric(warped, fixed),
            'mae': mae_metric(warped, fixed),
            'psnr': psnr_metric(warped, fixed),
            'ssim': ssim_metric(warped, fixed),
            'ncc': ncc_metric(warped, fixed),
        }
        for k, v in m.items():
            metrics[k].append(v)

        if (idx + 1) % 5 == 0 or idx == 0:
            print(f"[{idx+1}/{len(pairs)}] MSE={m['mse']:.6f} PSNR={m['psnr']:.2f} SSIM={m['ssim']:.4f} NCC={m['ncc']:.4f}")

        if args.save_results:
            from torchvision.utils import save_image
            save_image((warped + 1) / 2, os.path.join(results_dir, f'warped_{idx:04d}.png'))

    print(f"\n{'='*60}")
    print(f"DINO-Reg 2D Results on {args.dataset}")
    print(f"{'='*60}")
    for k in ['mse', 'mae', 'psnr', 'ssim', 'ncc']:
        vals = metrics[k]
        print(f"{k.upper():6s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

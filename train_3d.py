#!/usr/bin/env python3
"""
Standalone 3D registration model training script.

Supports: voxelmorph3d, nemar3d, transmorph3d
Multi-GPU via DataParallel.

Usage:
    python train_3d.py --model voxelmorph3d --name vm3d_l2r_v2 --niter 1000 --use_amp
    python train_3d.py --model nemar3d --name falcon3d_l2r_v2 --niter 1000 --use_amp --gpu_ids 0 1 2 3
"""
import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    parser = argparse.ArgumentParser(description='3D Registration Training')
    parser.add_argument('--model', type=str, required=True,
                        choices=['voxelmorph3d', 'nemar3d', 'transmorph3d'])
    parser.add_argument('--name', type=str, required=True, help='Experiment name')
    parser.add_argument('--dataroot', type=str, default='./datasets/L2R')
    parser.add_argument('--niter', type=int, default=1000)
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0],
                        help='GPU IDs for training (multi-GPU with DataParallel)')
    parser.add_argument('--use_amp', action='store_true', help='Mixed precision')
    parser.add_argument('--crop_3d_size', type=int, default=0, help='Random crop (0=full volume)')
    parser.add_argument('--save_epoch_freq', type=int, default=50)
    parser.add_argument('--print_freq', type=int, default=1)
    parser.add_argument('--eval_freq', type=int, default=50)
    parser.add_argument('--continue_train', action='store_true')
    parser.add_argument('--early_stop_patience', type=int, default=10,
                        help='Stop if DSC does not improve for N evals (0=disabled)')
    parser.add_argument('--early_stop_min_epochs', type=int, default=200,
                        help='Minimum epochs before early stopping kicks in')
    parser.add_argument('--no_augment_3d', action='store_true',
                        help='Disable aggressive 3D augmentation')
    parser.add_argument('--lr_scheduler', type=str, default='cosine',
                        choices=['cosine', 'linear', 'none'])

    # Model-specific defaults
    parser.add_argument('--vm3d_lr', type=float, default=1e-4)
    parser.add_argument('--vm3d_mi_bins', type=int, default=32)
    parser.add_argument('--vm3d_smoothness_weight', type=float, default=10.0)
    parser.add_argument('--vm3d_num_features', type=int, nargs='+', default=[32, 64, 128, 256])
    parser.add_argument('--vm3d_loss_type', type=str, default='mind', choices=['mi', 'mse', 'mind'])
    parser.add_argument('--vm3d_no_svf', action='store_true')
    parser.add_argument('--vm3d_svf_steps', type=int, default=7)

    # TransMorph3D defaults
    parser.add_argument('--tm3d_enc_channels', type=int, nargs='+', default=[48, 96, 192, 384])
    parser.add_argument('--tm3d_num_heads', type=int, default=6)
    parser.add_argument('--tm3d_num_transformer_blocks', type=int, default=2)
    parser.add_argument('--tm3d_sim_loss', type=str, default='mind', choices=['mind', 'mse'])
    parser.add_argument('--tm3d_reg_weight', type=float, default=1.0)
    parser.add_argument('--tm3d_lr', type=float, default=1e-4)
    parser.add_argument('--tm3d_use_svf', action='store_true', default=True)
    parser.add_argument('--tm3d_no_svf', action='store_true')
    parser.add_argument('--tm3d_svf_steps', type=int, default=7)

    # Falcon3D defaults
    parser.add_argument('--n3d_ngf', type=int, default=16)
    parser.add_argument('--n3d_ndf', type=int, default=16)
    parser.add_argument('--n3d_n_blocks', type=int, default=6)
    parser.add_argument('--n3d_kan_embed_dims', type=int, nargs='+', default=[32, 64, 128])
    parser.add_argument('--n3d_kan_depths', type=int, nargs='+', default=[1, 1, 1])
    parser.add_argument('--n3d_lambda_direct', type=float, default=2.0)
    parser.add_argument('--n3d_lambda_cycle', type=float, default=0.0)
    parser.add_argument('--n3d_warmup_epochs', type=int, default=0)
    parser.add_argument('--n3d_lambda_recon', type=float, default=10.0)
    parser.add_argument('--n3d_lambda_gan', type=float, default=1.0)
    parser.add_argument('--n3d_lambda_smooth', type=float, default=1.0)

    return parser.parse_args()


def create_model(opt):
    if opt.model == 'voxelmorph3d':
        from models.voxelmorph3d_model import VoxelMorph3DModel
        return VoxelMorph3DModel(opt)
    elif opt.model == 'nemar3d':
        from models.nemar3d_model import NEMAR3DModel
        return NEMAR3DModel(opt)
    elif opt.model == 'transmorph3d':
        from models.transmorph3d_model import Transmorph3DModel
        return Transmorph3DModel(opt)
    else:
        raise ValueError(f"Unknown model: {opt.model}")


def create_dataset(opt):
    from data.l2r_3d_dataset import L2R3DDataset
    return L2R3DDataset(opt)


def evaluate_dsc(model, dataset, device):
    dsc_values = []

    for i in range(len(dataset)):
        sample = dataset[i]
        if not sample.get('has_seg', False) or 'A_seg' not in sample:
            continue

        ct = sample['A'].unsqueeze(0).to(device) if sample['A'].dim() == 4 else sample['A'].to(device)
        mr = sample['B'].unsqueeze(0).to(device) if sample['B'].dim() == 4 else sample['B'].to(device)
        if ct.dim() == 4: ct = ct.unsqueeze(0)
        if mr.dim() == 4: mr = mr.unsqueeze(0)
        ct_seg = sample['A_seg'].numpy().squeeze()
        mr_seg = sample['B_seg'].numpy().squeeze()

        model.set_input({'A': ct, 'B': mr, 'A_paths': ''})
        with torch.no_grad():
            model.forward()

        flow = getattr(model, 'flow', None)
        if flow is None:
            flow = getattr(model, 'deformation', None)
        if flow is None:
            continue

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

        if per_label_dsc:
            dsc_values.append(np.mean(per_label_dsc))

    return np.mean(dsc_values) if dsc_values else 0.0, dsc_values


def main():
    opt = parse_args()
    opt.isTrain = True
    opt.phase = 'test'
    opt.input_nc = 1
    opt.output_nc = 1
    opt.num_threads = 0
    opt.serial_batches = False
    opt.max_dataset_size = float('inf')
    opt.load_seg = True
    opt.checkpoints_dir = './checkpoints'
    opt.epoch = 'latest'
    opt.load_iter = 0
    opt.verbose = False
    opt.preprocess = 'none'
    opt.no_flip = True
    opt.display_winsize = 256
    opt.suffix = ''
    opt.pool_size = 0
    opt.lr = 1e-4
    opt.lr_policy = 'linear'
    opt.beta1 = 0.5
    opt.niter_decay = 0
    opt.epoch_count = 1
    opt.augment_3d = not opt.no_augment_3d

    n_gpus = len(opt.gpu_ids)
    primary_gpu = opt.gpu_ids[0]
    opt.device = torch.device(f'cuda:{primary_gpu}' if torch.cuda.is_available() else 'cpu')
    opt.batch_size = n_gpus  # 1 sample per GPU

    print(f"Using GPUs: {opt.gpu_ids} ({n_gpus} GPUs)")
    print(f"Model: {opt.model}")
    print(f"AMP: {opt.use_amp}")
    print(f"Augmentation: {opt.augment_3d}")
    print(f"LR scheduler: {opt.lr_scheduler}")
    print(f"Batch size: {opt.batch_size} ({n_gpus} GPUs x 1 sample)")

    dataset = create_dataset(opt)
    model = create_model(opt)

    for name in model.model_names:
        net = getattr(model, 'net' + name)
        if isinstance(net, nn.Module):
            net.to(opt.device)

    # Wrap with DataParallel for multi-GPU
    if n_gpus > 1:
        for name in model.model_names:
            net = getattr(model, 'net' + name)
            if isinstance(net, nn.Module):
                dp_net = nn.DataParallel(net, device_ids=opt.gpu_ids)
                setattr(model, 'net' + name, dp_net)
        print(f"DataParallel enabled on {n_gpus} GPUs")

    # Create DataLoader for efficient batched data loading
    dataloader = DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=4 * n_gpus, pin_memory=True, drop_last=True
    )

    checkpoint_dir = f'./checkpoints/{opt.name}'
    os.makedirs(checkpoint_dir, exist_ok=True)

    with open(os.path.join(checkpoint_dir, 'train_opt.txt'), 'w') as f:
        for k, v in sorted(vars(opt).items()):
            f.write(f'{k}: {v}\n')

    start_epoch = 1
    if opt.continue_train:
        for name in model.model_names:
            net = getattr(model, 'net' + name)
            ckpt = os.path.join(checkpoint_dir, f'latest_net_{name}.pth')
            if os.path.exists(ckpt):
                state = torch.load(ckpt, map_location=opt.device, weights_only=True)
                raw_net = net.module if isinstance(net, nn.DataParallel) else net
                raw_net.load_state_dict(state)
                print(f"Resumed {name} from {ckpt}")

    # Collect optimizers from model
    optimizers = []
    for attr in dir(model):
        if attr.startswith('optimizer') and attr != 'optimizers':
            opt_obj = getattr(model, attr)
            if isinstance(opt_obj, torch.optim.Optimizer):
                optimizers.append(opt_obj)

    if not optimizers:
        all_params = []
        for name in model.model_names:
            net = getattr(model, 'net' + name)
            all_params.extend(list(net.parameters()))
        optimizer = torch.optim.Adam(all_params, lr=opt.lr)
        optimizers = [optimizer]

    # Create LR schedulers
    schedulers = []
    for opt_obj in optimizers:
        if opt.lr_scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt_obj, T_max=opt.niter, eta_min=1e-6)
        elif opt.lr_scheduler == 'linear':
            scheduler = torch.optim.lr_scheduler.LinearLR(
                opt_obj, start_factor=1.0, end_factor=0.01, total_iters=opt.niter)
        else:
            scheduler = torch.optim.lr_scheduler.ConstantLR(opt_obj)
        schedulers.append(scheduler)

    best_dsc = 0.0
    patience_counter = 0

    n_seg_patients = sum(1 for i in range(len(dataset))
                         if dataset.patients[i].get('has_seg', False))

    print(f"\nTraining {opt.model} for {opt.niter} epochs...")
    print(f"Dataset: {len(dataset)} patients ({n_seg_patients} with segmentation)")
    print(f"DataLoader: {len(dataloader)} batches/epoch")
    if opt.early_stop_patience > 0:
        print(f"Early stopping: patience={opt.early_stop_patience}, min_epochs={opt.early_stop_min_epochs}")

    for epoch in range(start_epoch, opt.niter + 1):
        if hasattr(model, 'current_epoch'):
            model.current_epoch = epoch

        epoch_losses = {}
        epoch_start = time.time()

        for batch in dataloader:
            model.set_input(batch)
            model.optimize_parameters()

            for name in model.model_names:
                net = getattr(model, 'net' + name)
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)

            losses = model.get_current_losses()
            for k, v in losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0) + v

        n = len(dataloader)
        for k in epoch_losses:
            epoch_losses[k] /= n

        for scheduler in schedulers:
            scheduler.step()

        elapsed = time.time() - epoch_start

        if epoch % opt.print_freq == 0 or epoch == 1:
            loss_str = '  '.join(f'{k}: {v:.6f}' for k, v in epoch_losses.items())
            lr = schedulers[0].get_last_lr()[0] if schedulers else opt.lr
            print(f'  Epoch [{epoch}/{opt.niter}] lr={lr:.2e} {loss_str}  ({elapsed:.1f}s)')

        # Save checkpoint
        if epoch % opt.save_epoch_freq == 0 or epoch == opt.niter:
            for name in model.model_names:
                net = getattr(model, 'net' + name)
                raw_net = net.module if isinstance(net, nn.DataParallel) else net
                state = raw_net.state_dict()
                save_path = os.path.join(checkpoint_dir, f'{epoch}_net_{name}.pth')
                torch.save(state, save_path)
                latest_path = os.path.join(checkpoint_dir, f'latest_net_{name}.pth')
                torch.save(state, latest_path)
            print(f'  Saved checkpoints to {checkpoint_dir}')

        # Evaluate DSC on patients with segmentation
        if epoch % opt.eval_freq == 0 and epoch > 0:
            for name in model.model_names:
                getattr(model, 'net' + name).eval()
            mean_dsc, all_dsc = evaluate_dsc(model, dataset, opt.device)
            for name in model.model_names:
                getattr(model, 'net' + name).train()
            print(f'  Eval DSC: {mean_dsc:.4f} (n={len(all_dsc)}, '
                  f'per-patient: {[f"{d:.3f}" for d in all_dsc]})')

            if opt.early_stop_patience > 0 and epoch >= opt.early_stop_min_epochs:
                if mean_dsc > best_dsc:
                    best_dsc = mean_dsc
                    patience_counter = 0
                    for name in model.model_names:
                        net = getattr(model, 'net' + name)
                        raw_net = net.module if isinstance(net, nn.DataParallel) else net
                        best_path = os.path.join(checkpoint_dir, f'best_net_{name}.pth')
                        torch.save(raw_net.state_dict(), best_path)
                    print(f'  New best DSC: {best_dsc:.4f}')
                else:
                    patience_counter += 1
                    print(f'  DSC not improved ({patience_counter}/{opt.early_stop_patience}), best: {best_dsc:.4f}')
                    if patience_counter >= opt.early_stop_patience:
                        print(f'  Early stopping at epoch {epoch}, best DSC: {best_dsc:.4f}')
                        break

    print("\nTraining complete!")
    if best_dsc > 0:
        print(f"Best DSC: {best_dsc:.4f}")


if __name__ == '__main__':
    main()

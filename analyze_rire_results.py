#!/usr/bin/env python3
"""
RIRE Dataset Experiment Results Analysis
分析RIRE数据集的实验结果并进行对比
"""

import re
import os
from pathlib import Path
from collections import defaultdict
import json


def parse_loss_log(log_file):
    """解析loss日志文件，提取训练损失数据"""
    if not os.path.exists(log_file):
        print(f"Warning: {log_file} not found")
        return None

    data = {
        'epochs': defaultdict(list),
        'final_losses': {},
        'total_epochs': 0
    }

    with open(log_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        # 匹配带contrastive的loss行
        match = re.search(
            r'\(epoch: (\d+), ite?rs?: \d+, time: [\d.]+, data: [\d.]+\) '
            r'L1_TR: ([\d.]+) GAN_TR: ([\d.]+) L1_RT: ([\d.]+) GAN_RT: ([\d.]+) '
            r'smoothness: ([\d.]+) D_fake_TR: ([\d.]+) D_fake_RT: ([\d.]+) D: ([\d.]+) '
            r'contrastive: ([\d.]+)',
            line
        )

        if match:
            epoch = int(match.group(1))
            losses = {
                'L1_TR': float(match.group(2)),
                'GAN_TR': float(match.group(3)),
                'L1_RT': float(match.group(4)),
                'GAN_RT': float(match.group(5)),
                'smoothness': float(match.group(6)),
                'D_fake_TR': float(match.group(7)),
                'D_fake_RT': float(match.group(8)),
                'D': float(match.group(9)),
                'contrastive': float(match.group(10))
            }
            data['epochs'][epoch].append(losses)
        else:
            # 匹配不带contrastive的loss行
            match2 = re.search(
                r'\(epoch: (\d+), ite?rs?: \d+, time: [\d.]+, data: [\d.]+\) '
                r'L1_TR: ([\d.]+) GAN_TR: ([\d.]+) L1_RT: ([\d.]+) GAN_RT: ([\d.]+) '
                r'smoothness: ([\d.]+) D_fake_TR: ([\d.]+) D_fake_RT: ([\d.]+) D: ([\d.]+)',
                line
            )

            if match2:
                epoch = int(match2.group(1))
                losses = {
                    'L1_TR': float(match2.group(2)),
                    'GAN_TR': float(match2.group(3)),
                    'L1_RT': float(match2.group(4)),
                    'GAN_RT': float(match2.group(5)),
                    'smoothness': float(match2.group(6)),
                    'D_fake_TR': float(match2.group(7)),
                    'D_fake_RT': float(match2.group(8)),
                    'D': float(match2.group(9)),
                    'contrastive': 0.0  # 默认值
                }
                data['epochs'][epoch].append(losses)

    if data['epochs']:
        data['total_epochs'] = max(data['epochs'].keys())

        # 计算每个epoch的平均损失
        epoch_avg = {}
        for epoch, losses_list in sorted(data['epochs'].items()):
            avg_losses = {}
            for key in losses_list[0].keys():
                avg_losses[key] = sum(l[key] for l in losses_list) / len(losses_list)
            epoch_avg[epoch] = avg_losses

        # 获取最后一个epoch的损失
        if epoch_avg:
            data['final_losses'] = epoch_avg[data['total_epochs']]
            data['epoch_losses'] = epoch_avg

    return data


def analyze_experiments(base_dir):
    """分析所有实验"""
    experiments = {
        'Baseline': 'rire2d_ukan_gbcm_contrastive',
        'Stable': 'rire2d_ukan_gbcm_contrastive_stable',
        'High': 'rire2d_ukan_gbcm_contrastive_high',
        'Ablation_Weight_00': 'ablation_weight_00',
        'Ablation_Weight_005': 'ablation_weight_005',
        'Ablation_Weight_02': 'ablation_weight_02',
        'Ablation_Weight_03': 'ablation_weight_03',
        'Ablation_No_GBCM': 'ablation_no_gbcm',
        'Ablation_Only_Disc_Noise': 'ablation_only_disc_noise',
        'Ablation_Only_Label_Smooth': 'ablation_only_label_smooth',
        'Ablation_STN_Affine': 'ablation_stn_affine',
        'Ablation_STN_UKAN': 'ablation_stn_ukan',
    }

    results = {}

    for name, exp_dir in experiments.items():
        log_file = os.path.join(base_dir, exp_dir, 'loss_log.txt')
        data = parse_loss_log(log_file)
        if data:
            results[name] = data
            print(f"✓ {name}: {data['total_epochs']} epochs loaded")
        else:
            print(f"✗ {name}: failed to load")

    return results


def generate_comparison_table(results):
    """生成对比表格"""
    print("\n" + "="*100)
    print("RIRE Dataset Experiment Comparison")
    print("="*100)

    # 表头
    header = f"{'Experiment':<25}"
    for metric in ['L1_TR', 'GAN_TR', 'L1_RT', 'GAN_RT', 'D', 'contrastive']:
        header += f"{metric:>10}"
    print(header)
    print("-"*100)

    # 每个实验的数据
    for exp_name, data in sorted(results.items()):
        final = data['final_losses']
        row = f"{exp_name:<25}"
        for metric in ['L1_TR', 'GAN_TR', 'L1_RT', 'GAN_RT', 'D', 'contrastive']:
            value = final.get(metric, 0.0)
            row += f"{value:>10.3f}"
        print(row)

    print("="*100)


def generate_summary(results):
    """生成实验总结"""
    print("\n" + "="*100)
    print("Summary Statistics")
    print("="*100)

    # 找出最好的实验
    best_l1_tr = min(results.items(), key=lambda x: x[1]['final_losses'].get('L1_TR', float('inf')))
    best_l1_rt = min(results.items(), key=lambda x: x[1]['final_losses'].get('L1_RT', float('inf')))

    print(f"Best L1_TR (Translation): {best_l1_tr[0]} = {best_l1_tr[1]['final_losses']['L1_TR']:.3f}")
    print(f"Best L1_RT (Rotation): {best_l1_rt[0]} = {best_l1_rt[1]['final_losses']['L1_RT']:.3f}")

    # Baseline对比
    if 'Baseline' in results:
        baseline = results['Baseline']['final_losses']
        print("\nBaseline Performance:")
        print(f"  L1_TR: {baseline['L1_TR']:.3f}")
        print(f"  L1_RT: {baseline['L1_RT']:.3f}")
        print(f"  Total Loss (TR+RT): {baseline['L1_TR'] + baseline['L1_RT']:.3f}")
        print(f"  GAN Loss: {baseline['GAN_TR'] + baseline['GAN_RT']:.3f}")
        print(f"  Contrastive: {baseline['contrastive']:.3f}")

    print("\n" + "="*100)


def save_results_to_json(results, output_file):
    """保存结果到JSON文件"""
    output = {}
    for exp_name, data in results.items():
        exp_data = {
            'total_epochs': data['total_epochs'],
            'final_losses': data['final_losses'],
        }
        # 只保存部分epoch的数据以减小文件大小（如果存在）
        if 'epoch_losses' in data:
            exp_data['sample_epochs'] = {k: v for k, v in list(data['epoch_losses'].items())[::10]}
        output[exp_name] = exp_data

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_file}")


def compare_ablation_studies(results):
    """对比消融实验"""
    print("\n" + "="*100)
    print("Ablation Study Comparison")
    print("="*100)

    ablation_experiments = {
        'Baseline': 'Baseline',
        'Weight_0.0': 'Ablation_Weight_00',
        'Weight_0.05': 'Ablation_Weight_005',
        'Weight_0.2': 'Ablation_Weight_02',
        'Weight_0.3': 'Ablation_Weight_03',
        'No GBCM': 'Ablation_No_GBCM',
        'Only Disc Noise': 'Ablation_Only_Disc_Noise',
        'Only Label Smooth': 'Ablation_Only_Label_Smooth',
        'STN Affine': 'Ablation_STN_Affine',
        'STN UKAN': 'Ablation_STN_UKAN',
    }

    header = f"{'Component':<20}"
    header += f"{'L1_TR':>10}{'L1_RT':>10}{'Total':>10}{'Contrastive':>12}"
    print(header)
    print("-"*70)

    for comp_name, exp_name in ablation_experiments.items():
        if exp_name in results:
            data = results[exp_name]['final_losses']
            l1_tr = data.get('L1_TR', 0.0)
            l1_rt = data.get('L1_RT', 0.0)
            contrastive = data.get('contrastive', 0.0)
            total = l1_tr + l1_rt

            row = f"{comp_name:<20}"
            row += f"{l1_tr:>10.3f}{l1_rt:>10.3f}{total:>10.3f}{contrastive:>12.3f}"
            print(row)

    print("="*100)


def main():
    checkpoints_dir = '/root/autodl-tmp/nemar/checkpoints'

    print("Analyzing RIRE dataset experiment results...")
    print(f"Checkpoints directory: {checkpoints_dir}\n")

    # 分析所有实验
    results = analyze_experiments(checkpoints_dir)

    if not results:
        print("No results found!")
        return

    # 生成对比表格
    generate_comparison_table(results)

    # 生成总结
    generate_summary(results)

    # 对比消融实验
    compare_ablation_studies(results)

    # 保存结果
    output_file = '/root/autodl-tmp/nemar/rire_results_summary.json'
    save_results_to_json(results, output_file)

    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Visualize RIRE Dataset Experiment Results
可视化RIRE数据集实验结果
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os


# Set font for Chinese support
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


def load_results(json_file):
    """加载JSON格式的实验结果"""
    with open(json_file, 'r') as f:
        return json.load(f)


def plot_main_comparison(results):
    """绘制主要模型对比图"""
    main_experiments = ['Baseline', 'Stable', 'High']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: L1 Losses
    exp_names = main_experiments
    l1_tr = [results[exp]['final_losses']['L1_TR'] for exp in exp_names]
    l1_rt = [results[exp]['final_losses']['L1_RT'] for exp in exp_names]
    total_l1 = [a + b for a, b in zip(l1_tr, l1_rt)]

    x = np.arange(len(exp_names))
    width = 0.25

    axes[0].bar(x - width, l1_tr, width, label='L1_TR (Translation)', color='#3498db')
    axes[0].bar(x, l1_rt, width, label='L1_RT (Rotation)', color='#e74c3c')
    axes[0].bar(x + width, total_l1, width, label='Total L1', color='#2ecc71')

    axes[0].set_xlabel('Experiment')
    axes[0].set_ylabel('L1 Loss')
    axes[0].set_title('L1 Loss Comparison')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(exp_names)
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    # Plot 2: GAN and Contrastive Losses
    gan_tr = [results[exp]['final_losses']['GAN_TR'] for exp in exp_names]
    gan_rt = [results[exp]['final_losses']['GAN_RT'] for exp in exp_names]
    contrastive = [results[exp]['final_losses']['contrastive'] for exp in exp_names]

    x = np.arange(len(exp_names))
    width = 0.25

    axes[1].bar(x - width, gan_tr, width, label='GAN_TR', color='#9b59b6')
    axes[1].bar(x, gan_rt, width, label='GAN_RT', color='#f39c12')
    axes[1].bar(x + width, contrastive, width, label='Contrastive', color='#1abc9c')

    axes[1].set_xlabel('Experiment')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('GAN and Contrastive Loss Comparison')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(exp_names)
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('/root/autodl-tmp/nemar/rire_main_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved: rire_main_comparison.png")


def plot_ablation_study(results):
    """绘制消融实验对比图"""
    ablation_exp = {
        'Baseline': 'Baseline',
        'Weight_0.0': 'Ablation_Weight_00',
        'Weight_0.05': 'Ablation_Weight_005',
        'Weight_0.2': 'Ablation_Weight_02',
        'Weight_0.3': 'Ablation_Weight_03',
        'No GBCM': 'Ablation_No_GBCM',
        'Only Disc Noise': 'Ablation_Only_Disc_Noise',
        'Only Label Smooth': 'Ablation_Only_Label_Smooth',
        'STN UKAN': 'Ablation_STN_UKAN',
        'STN Affine': 'Ablation_STN_Affine',
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Contrastive Weight Impact
    weight_exp = ['Weight_0.0', 'Weight_0.05', 'Weight_0.2 (Baseline)', 'Weight_0.3']
    weight_total = [
        results['Ablation_Weight_00']['final_losses']['L1_TR'] + results['Ablation_Weight_00']['final_losses']['L1_RT'],
        results['Ablation_Weight_005']['final_losses']['L1_TR'] + results['Ablation_Weight_005']['final_losses']['L1_RT'],
        results['Baseline']['final_losses']['L1_TR'] + results['Baseline']['final_losses']['L1_RT'],
        results['Ablation_Weight_03']['final_losses']['L1_TR'] + results['Ablation_Weight_03']['final_losses']['L1_RT'],
    ]

    bars = axes[0, 0].bar(weight_exp, weight_total, color=['#e74c3c', '#e74c3c', '#2ecc71', '#e74c3c'])
    axes[0, 0].set_ylabel('Total L1 Loss')
    axes[0, 0].set_title('Impact of Contrastive Learning Weight')
    axes[0, 0].grid(axis='y', alpha=0.3)
    # Highlight baseline
    bars[2].set_color('#2ecc71')

    # Plot 2: Component Importance (Total L1)
    comp_names = ['Baseline', 'No GBCM', 'Only Disc\nNoise', 'Only Label\nSmooth', 'STN UKAN']
    comp_values = [
        results['Baseline']['final_losses']['L1_TR'] + results['Baseline']['final_losses']['L1_RT'],
        results['Ablation_No_GBCM']['final_losses']['L1_TR'] + results['Ablation_No_GBCM']['final_losses']['L1_RT'],
        results['Ablation_Only_Disc_Noise']['final_losses']['L1_TR'] + results['Ablation_Only_Disc_Noise']['final_losses']['L1_RT'],
        results['Ablation_Only_Label_Smooth']['final_losses']['L1_TR'] + results['Ablation_Only_Label_Smooth']['final_losses']['L1_RT'],
        results['Ablation_STN_UKAN']['final_losses']['L1_TR'] + results['Ablation_STN_UKAN']['final_losses']['L1_RT'],
    ]

    axes[0, 1].barh(comp_names, comp_values, color=['#2ecc71', '#e74c3c', '#e74c3c', '#e74c3c', '#e74c3c'])
    axes[0, 1].set_xlabel('Total L1 Loss')
    axes[0, 1].set_title('Component Importance Analysis')
    axes[0, 1].grid(axis='x', alpha=0.3)

    # Plot 3: STN Architecture Comparison
    stn_names = ['UKAN (Baseline)', 'STN UKAN', 'STN Affine']
    stn_l1_tr = [
        results['Baseline']['final_losses']['L1_TR'],
        results['Ablation_STN_UKAN']['final_losses']['L1_TR'],
        results['Ablation_STN_Affine']['final_losses']['L1_TR'],
    ]
    stn_l1_rt = [
        results['Baseline']['final_losses']['L1_RT'],
        results['Ablation_STN_UKAN']['final_losses']['L1_RT'],
        results['Ablation_STN_Affine']['final_losses']['L1_RT'],
    ]

    x = np.arange(len(stn_names))
    width = 0.35

    axes[1, 0].bar(x - width/2, stn_l1_tr, width, label='L1_TR', color='#3498db')
    axes[1, 0].bar(x + width/2, stn_l1_rt, width, label='L1_RT', color='#e74c3c')

    axes[1, 0].set_ylabel('L1 Loss')
    axes[1, 0].set_title('STN Architecture Comparison')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(stn_names)
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', alpha=0.3)

    # Plot 4: Overall Performance Ranking
    ranking_names = [
        'Baseline', 'Stable', 'Weight_0.2', 'STN UKAN', 'Only Label\nSmooth',
        'Weight_0.3', 'Only Disc\nNoise', 'No GBCM', 'Weight_0.05', 'Weight_0.0'
    ]
    ranking_keys = [
        'Baseline', 'Stable', 'Ablation_Weight_02', 'Ablation_STN_UKAN', 'Ablation_Only_Label_Smooth',
        'Ablation_Weight_03', 'Ablation_Only_Disc_Noise', 'Ablation_No_GBCM', 'Ablation_Weight_005', 'Ablation_Weight_00'
    ]

    ranking_total = []
    ranking_l1_tr = []
    ranking_l1_rt = []

    for key in ranking_keys:
        data = results[key]['final_losses']
        ranking_total.append(data['L1_TR'] + data['L1_RT'])
        ranking_l1_tr.append(data['L1_TR'])
        ranking_l1_rt.append(data['L1_RT'])

    x = np.arange(len(ranking_names))
    width = 0.25

    axes[1, 1].bar(x - width, ranking_l1_tr, width, label='L1_TR', color='#3498db')
    axes[1, 1].bar(x, ranking_l1_rt, width, label='L1_RT', color='#e74c3c')
    axes[1, 1].bar(x + width, ranking_total, width, label='Total', color='#2ecc71')

    axes[1, 1].set_xlabel('Experiment')
    axes[1, 1].set_ylabel('L1 Loss')
    axes[1, 1].set_title('Overall Performance Ranking (Lower is Better)')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(ranking_names, rotation=45, ha='right')
    axes[1, 1].legend()
    axes[1, 1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('/root/autodl-tmp/nemar/rire_ablation_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved: rire_ablation_comparison.png")


def plot_training_curves(results):
    """绘制训练曲线（如果可用）"""
    if 'sample_epochs' not in results['Baseline']:
        print("✗ No training curve data available")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    main_experiments = ['Baseline', 'Stable']
    colors = ['#2ecc71', '#3498db']

    # Plot 1: L1_TR curve
    for exp, color in zip(main_experiments, colors):
        if 'sample_epochs' in results[exp]:
            epochs = sorted([int(e) for e in results[exp]['sample_epochs'].keys()])
            l1_tr = [results[exp]['sample_epochs'][str(e)]['L1_TR'] for e in epochs]
            axes[0, 0].plot(epochs, l1_tr, marker='o', label=exp, color=color, linewidth=2)

    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('L1_TR Loss')
    axes[0, 0].set_title('L1_TR Training Curve')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # Plot 2: L1_RT curve
    for exp, color in zip(main_experiments, colors):
        if 'sample_epochs' in results[exp]:
            epochs = sorted([int(e) for e in results[exp]['sample_epochs'].keys()])
            l1_rt = [results[exp]['sample_epochs'][str(e)]['L1_RT'] for e in epochs]
            axes[0, 1].plot(epochs, l1_rt, marker='o', label=exp, color=color, linewidth=2)

    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('L1_RT Loss')
    axes[0, 1].set_title('L1_RT Training Curve')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # Plot 3: GAN losses
    for exp, color in zip(main_experiments, colors):
        if 'sample_epochs' in results[exp]:
            epochs = sorted([int(e) for e in results[exp]['sample_epochs'].keys()])
            gan_tr = [results[exp]['sample_epochs'][str(e)]['GAN_TR'] for e in epochs]
            gan_rt = [results[exp]['sample_epochs'][str(e)]['GAN_RT'] for e in epochs]
            axes[1, 0].plot(epochs, gan_tr, marker='s', label=f'{exp} GAN_TR', color=color, linestyle='--', linewidth=2)
            axes[1, 0].plot(epochs, gan_rt, marker='^', label=f'{exp} GAN_RT', color=color, linestyle=':', linewidth=2)

    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('GAN Loss')
    axes[1, 0].set_title('GAN Losses Training Curve')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # Plot 4: Discriminator and Contrastive
    for exp, color in zip(main_experiments, colors):
        if 'sample_epochs' in results[exp]:
            epochs = sorted([int(e) for e in results[exp]['sample_epochs'].keys()])
            d_loss = [results[exp]['sample_epochs'][str(e)]['D'] for e in epochs]
            axes[1, 1].plot(epochs, d_loss, marker='x', label=f'{exp} D', color=color, linewidth=2)

    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Discriminator Loss Training Curve')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('/root/autodl-tmp/nemar/rire_training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved: rire_training_curves.png")


def main():
    json_file = '/root/autodl-tmp/nemar/rire_results_summary.json'

    print("Loading RIRE experiment results...")
    results = load_results(json_file)

    print("\nGenerating visualizations...")
    plot_main_comparison(results)
    plot_ablation_study(results)
    plot_training_curves(results)

    print("\nAll visualizations generated successfully!")
    print("\nGenerated files:")
    print("  - rire_main_comparison.png")
    print("  - rire_ablation_comparison.png")
    print("  - rire_training_curves.png")


if __name__ == '__main__':
    main()

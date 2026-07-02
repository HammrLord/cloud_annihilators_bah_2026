#!/usr/bin/env python
"""
Evaluation script for cloud removal.

Computes metrics stratified by cloud density (thin/medium/thick).
Produces visual comparisons, error maps, NDVI maps.

Usage:
    python scripts/evaluate.py --checkpoint experiments/stage2/best.ckpt \
                               --data-root data/test \
                               --output-dir results/
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.psprnet import SPRNet
from src.data.liss4_dataset import LISS4Dataset
from src.metrics.evaluation import compute_all_metrics
from src.utils.visualization import (plot_comparison, plot_error_map,
                                      plot_ndvi_comparison)


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def evaluate(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = SPRNet({
        'mskt_in_channels': 5,
        'mskt_out_channels': 13,
        'mskt_embed_dim': 64,
        'mskt_num_heads': 8,
        'mskt_encoder_layers': 6,
        'mskt_decoder_layers': 6,
        'ppe_in_channels': 3,
        'ppe_hidden_channels': 16,
        'backbone_in_channels': 15,
        'backbone_out_channels': 3,
        'features_start': 48,
        'num_blocks': (2, 3, 2),
        'num_refine': 4,
    }).to(device)

    ckpt = torch.load(config['model']['checkpoint'], map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Dataset
    dataset = LISS4Dataset(
        root=config['data']['root'],
        crop_size=config['data']['crop_size'],
        use_sar=config['data'].get('use_sar', True),
        use_dem=config['data'].get('use_dem', True),
        augment=False,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Evaluate
    all_metrics = {'psnr': [], 'ssim': [], 'sam': [], 'rmse': [], 'mae': []}
    output_dir = config.get('output_dir', 'results')
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            cloudy = batch['cloudy'].to(device)
            clear = batch['clear'].to(device)
            sar = batch.get('sar', torch.zeros(1, 2, *cloudy.shape[2:])).to(device)
            dem = batch.get('dem', torch.zeros(1, 3, *cloudy.shape[2:])).to(device)

            outputs = model(cloudy, sar, dem)
            pred = outputs['cloud_free']

            metrics = compute_all_metrics(pred, clear)
            for k in all_metrics:
                all_metrics[k].append(metrics[k])

            # Save visualizations for first 10 samples
            if i < 10:
                pred_np = pred.squeeze(0).cpu().numpy()
                clear_np = clear.squeeze(0).cpu().numpy()
                cloudy_np = cloudy.squeeze(0).cpu().numpy()

                plot_comparison(
                    cloudy_np, pred_np, clear_np,
                    title=f"Sample {i+1} — PSNR: {metrics['psnr']:.2f} dB",
                    save_path=os.path.join(output_dir, f"comparison_{i+1}.png")
                )
                plot_error_map(
                    pred_np, clear_np,
                    title=f"Sample {i+1} Error Map",
                    save_path=os.path.join(output_dir, f"error_{i+1}.png")
                )

    # Print summary
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    for k, vals in all_metrics.items():
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        print(f"{k.upper():>8}: {mean_val:.4f} ± {std_val:.4f}")

    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/inference.yaml')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--data-root', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default='results')
    args = parser.parse_args()

    config = load_config(args.config)
    if args.checkpoint:
        config['model']['checkpoint'] = args.checkpoint
    if args.data_root:
        config['data']['root'] = args.data_root
    config['output_dir'] = args.output_dir

    evaluate(config)

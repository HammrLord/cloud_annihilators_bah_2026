#!/usr/bin/env python
"""
Inference script for cloud removal.

Usage:
    python scripts/inference.py --checkpoint experiments/stage2/best.ckpt \
                                --input data/test/cloudy.tif \
                                --sar data/test/sar.tif \
                                --output data/test/output.tif
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
import rasterio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.psprnet import SPRNet
from src.metrics.evaluation import compute_all_metrics


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def read_tif(path):
    with rasterio.open(path) as src:
        return src.read().astype(np.float32), src.transform, src.crs


def write_tif(array, path, transform, crs, dtype='float32'):
    C, H, W = array.shape
    with rasterio.open(
        path, 'w', driver='GTiff',
        height=H, width=W, count=C,
        dtype=dtype, crs=crs, transform=transform,
    ) as dst:
        for i in range(C):
            dst.write(array[i], i + 1)


def preprocess_sar(sar):
    sar = np.clip(sar, -25, 0)
    return (sar + 25) / 25.0


def preprocess_optical(optical):
    optical = np.clip(optical, 0, 10000)
    return optical / 10000.0


def run_inference(config):
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
    print(f"Loaded model from {config['model']['checkpoint']}")

    # Read input
    optical, transform, crs = read_tif(config['inference']['input_dir'])
    optical = preprocess_optical(optical)

    sar = None
    sar_path = config['inference'].get('sar_path')
    if sar_path and os.path.exists(sar_path):
        sar, _, _ = read_tif(sar_path)
        sar = preprocess_sar(sar)

    # To tensor
    optical_t = torch.from_numpy(optical).unsqueeze(0).to(device)
    if sar is not None:
        sar_t = torch.from_numpy(sar).unsqueeze(0).to(device)
    else:
        sar_t = torch.zeros(optical_t.shape[0], 2,
                            optical_t.shape[2], optical_t.shape[3]).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(optical_t, sar_t)

    result = outputs['cloud_free'].squeeze(0).cpu().numpy()
    result = np.clip(result, 0, 1)

    # Save output
    output_path = config['inference']['output_dir']
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    write_tif(result, output_path, transform, crs)
    print(f"Saved output to {output_path}")

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/inference.yaml')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--sar', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.checkpoint:
        config['model']['checkpoint'] = args.checkpoint
    if args.input:
        config['inference']['input_dir'] = args.input
    if args.sar:
        config['inference']['sar_path'] = args.sar
    if args.output:
        config['inference']['output_dir'] = args.output

    run_inference(config)

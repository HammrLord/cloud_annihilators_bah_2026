#!/usr/bin/env python
"""
Stage 2: PSPRNet Training — Physics-guided Cloud Removal.

Loads pretrained MSKT, trains PPE + AdaLN + ECRFormer.
Losses: L1 + SSIM + NDVI + SAM + Physics consistency.

Usage:
    python scripts/train_stage2.py --config configs/stage2.yaml --gpu 0
"""

import os
import sys
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.psprnet import SPRNet
from src.data.liss4_dataset import LISS4Dataset
from src.data.synthetic_cloud import SyntheticCloudGenerator
from src.losses.total import TotalLoss
from src.metrics.evaluation import compute_all_metrics


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def train(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Model
    model_config = config['model']
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

    # Load pretrained MSKT
    mskt_ckpt = model_config.get('mskt_checkpoint')
    if mskt_ckpt and os.path.exists(mskt_ckpt):
        ckpt = torch.load(mskt_ckpt, map_location='cpu')
        model.mskt.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded MSKT from {mskt_ckpt}")

    # Freeze MSKT
    if model_config.get('freeze_mskt', True):
        model.freeze_mskt()
        print("MSKT frozen")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}, Trainable: {trainable_params:,}")

    # Dataset
    dataset = LISS4Dataset(
        root=config['data']['root'],
        crop_size=config['data']['crop_size'],
        use_sar=config['data'].get('use_sar', True),
        use_dem=config['data'].get('use_dem', True),
    )
    dataloader = DataLoader(
        dataset, batch_size=config['data']['batch_size'],
        shuffle=True, num_workers=config['data']['num_workers'],
        pin_memory=True, drop_last=True,
    )

    # Loss
    loss_fn = TotalLoss(
        l1_weight=config['loss']['l1_weight'],
        ssim_weight=config['loss']['ssim_weight'],
        ndvi_weight=config['loss']['ndvi_weight'],
        sam_weight=config['loss']['sam_global_weight'],
        physics_weight=config['loss']['physics_weight'],
    )

    # Optimizer (only trainable parameters)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay'],
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['training']['max_epoch']
    )

    # Synthetic cloud generator for augmenting training data
    cloud_gen = SyntheticCloudGenerator(seed=42)

    # Training loop
    save_dir = config['training']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    best_psnr = 0

    for epoch in range(config['training']['max_epoch']):
        model.train()
        epoch_losses = {}
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            cloudy = batch['cloudy'].to(device)  # [B, 3, H, W]
            clear = batch['clear'].to(device)    # [B, 3, H, W]

            sar = batch.get('sar', torch.zeros(cloudy.shape[0], 2,
                                                cloudy.shape[2],
                                                cloudy.shape[3])).to(device)
            dem = batch.get('dem', torch.zeros(cloudy.shape[0], 3,
                                                cloudy.shape[2],
                                                cloudy.shape[3])).to(device)

            # Forward pass
            outputs = model(cloudy, sar, dem)

            # Loss computation
            targets = {'clear': clear, 'cloudy': cloudy}
            loss, loss_dict = loss_fn(outputs, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                config['training']['gradient_clip']
            )
            optimizer.step()

            for k, v in loss_dict.items():
                epoch_losses[k] = epoch_losses.get(k, 0) + v
            num_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        scheduler.step()

        # Print epoch summary
        avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}
        loss_str = ', '.join(f'{k}: {v:.4f}' for k, v in avg_losses.items())
        print(f"Epoch {epoch+1} — {loss_str}")

        # Validation
        model.eval()
        val_metrics = {'psnr': 0, 'ssim': 0, 'sam': 0}
        num_val = 0
        with torch.no_grad():
            for batch in dataloader:
                cloudy = batch['cloudy'].to(device)
                clear = batch['clear'].to(device)
                sar = batch.get('sar', torch.zeros(cloudy.shape[0], 2,
                                                    cloudy.shape[2],
                                                    cloudy.shape[3])).to(device)
                dem = batch.get('dem', torch.zeros(cloudy.shape[0], 3,
                                                    cloudy.shape[2],
                                                    cloudy.shape[3])).to(device)
                outputs = model(cloudy, sar, dem)
                metrics = compute_all_metrics(outputs['cloud_free'], clear)
                for k in val_metrics:
                    val_metrics[k] += metrics[k]
                num_val += 1
                if num_val >= 50:  # Limit validation for speed
                    break

        for k in val_metrics:
            val_metrics[k] /= max(num_val, 1)

        print(f"  Val — PSNR: {val_metrics['psnr']:.2f} dB, "
              f"SSIM: {val_metrics['ssim']:.4f}")

        if val_metrics['psnr'] > best_psnr:
            best_psnr = val_metrics['psnr']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'psnr': best_psnr,
            }, os.path.join(save_dir, 'best.ckpt'))
            print(f"  Saved best (PSNR: {best_psnr:.2f} dB)")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
        }, os.path.join(save_dir, 'latest.ckpt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/stage2.yaml')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    train(config)

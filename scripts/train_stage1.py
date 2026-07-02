#!/usr/bin/env python
"""
Stage 1: MSKT Pretraining on SEN12MS-CR.

Reconstruct 13-band Sentinel-2 from LISS-IV (3 bands) + SAR (2 bands).
Losses: L1 + SSIM + SAM.

Usage:
    python scripts/train_stage1.py --config configs/stage1_mskt.yaml --gpu 0
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.mskt import MSKT
from src.data.sen12mscr_dataset import SEN12MSCRDataset
from src.losses.reconstruction import L1Loss, SSIMLoss
from src.losses.spectral import SAMLoss
from src.metrics.evaluation import compute_all_metrics


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def train(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Model
    model = MSKT(
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels'],
        embed_dim=config['model']['embed_dim'],
        num_heads=config['model']['num_heads'],
        num_encoder_layers=config['model']['num_encoder_layers'],
        num_decoder_layers=config['model']['num_decoder_layers'],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"MSKT parameters: {total_params:,}")

    # Dataset
    dataset = SEN12MSCRDataset(
        root=config['data']['root'],
        split='train',
        crop_size=config['data']['crop_size'],
    )
    dataloader = DataLoader(
        dataset, batch_size=config['data']['batch_size'],
        shuffle=True, num_workers=config['data']['num_workers'],
        pin_memory=True, drop_last=True,
    )

    val_dataset = SEN12MSCRDataset(
        root=config['data']['root'],
        split='val',
        crop_size=config['data']['crop_size'],
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['data']['batch_size'],
        shuffle=False, num_workers=config['data']['num_workers'],
    )

    # Losses
    l1_loss = L1Loss()
    ssim_loss = SSIMLoss()
    sam_loss = SAMLoss()

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay'],
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['training']['max_epoch']
    )

    # Training loop
    save_dir = config['training']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    best_psnr = 0

    for epoch in range(config['training']['max_epoch']):
        model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            inputs = batch['input'].to(device)    # [B, 5, H, W]
            targets = batch['target'].to(device)  # [B, 13, H, W]

            # Forward pass
            pred = model(inputs)  # [B, 13, H, W]

            # Compute losses
            loss_l1 = l1_loss(pred, targets)
            loss_ssim = ssim_loss(pred, targets)
            loss_sam = sam_loss(pred, targets)

            loss = config['loss']['l1_weight'] * loss_l1 + \
                   config['loss']['ssim_weight'] * loss_ssim + \
                   config['loss']['sam_weight'] * loss_sam

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                            config['training']['gradient_clip'])
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        scheduler.step()
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{config['training']['max_epoch']} — "
              f"Loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        val_metrics = {'psnr': 0, 'ssim': 0, 'sam': 0}
        num_val = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input'].to(device)
                targets = batch['target'].to(device)
                pred = model(inputs)
                metrics = compute_all_metrics(pred, targets)
                for k in val_metrics:
                    val_metrics[k] += metrics[k]
                num_val += 1

        for k in val_metrics:
            val_metrics[k] /= max(num_val, 1)

        print(f"  Val PSNR: {val_metrics['psnr']:.2f} dB, "
              f"SSIM: {val_metrics['ssim']:.4f}, "
              f"SAM: {val_metrics['sam']:.2f}°")

        # Save best model
        if val_metrics['psnr'] > best_psnr:
            best_psnr = val_metrics['psnr']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'psnr': best_psnr,
            }, os.path.join(save_dir, 'best.ckpt'))
            print(f"  Saved best model (PSNR: {best_psnr:.2f} dB)")

        # Save latest
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(save_dir, 'latest.ckpt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/stage1_mskt.yaml')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    train(config)

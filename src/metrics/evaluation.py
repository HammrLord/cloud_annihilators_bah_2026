"""Evaluation metrics for cloud removal."""

import torch
import numpy as np


def compute_psnr(pred, target, data_range=1.0):
    """Peak Signal-to-Noise Ratio."""
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * torch.log10(data_range ** 2 / mse)


def compute_ssim(pred, target, window_size=11):
    """Structural Similarity Index (simplified)."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_pred = torch.nn.functional.avg_pool2d(pred, window_size, stride=1,
                                              padding=window_size // 2)
    mu_target = torch.nn.functional.avg_pool2d(target, window_size, stride=1,
                                                padding=window_size // 2)

    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_cross = mu_pred * mu_target

    sigma_pred_sq = torch.nn.functional.avg_pool2d(pred ** 2, window_size,
                                                     stride=1,
                                                     padding=window_size // 2) - mu_pred_sq
    sigma_target_sq = torch.nn.functional.avg_pool2d(target ** 2, window_size,
                                                       stride=1,
                                                       padding=window_size // 2) - mu_target_sq
    sigma_cross = torch.nn.functional.avg_pool2d(pred * target, window_size,
                                                   stride=1,
                                                   padding=window_size // 2) - mu_cross

    ssim_map = ((2 * mu_cross + C1) * (2 * sigma_cross + C2)) / \
               ((mu_pred_sq + mu_target_sq + C1) *
                (sigma_pred_sq + sigma_target_sq + C2))
    return ssim_map.mean()


def compute_sam(pred, target, eps=1e-7):
    """Spectral Angle Mapper (in degrees)."""
    B, C, H, W = pred.shape
    pred_flat = pred.view(B, C, -1).permute(0, 2, 1)  # [B, H*W, C]
    target_flat = target.view(B, C, -1).permute(0, 2, 1)

    cos_sim = torch.nn.functional.cosine_similarity(pred_flat, target_flat,
                                                     dim=-1)
    cos_sim = torch.clamp(cos_sim, -1 + eps, 1 - eps)
    sam = torch.acos(cos_sim)
    return sam.mean() * (180.0 / np.pi)  # Convert to degrees


def compute_rmse(pred, target):
    """Root Mean Square Error."""
    return torch.sqrt(torch.mean((pred - target) ** 2))


def compute_mae(pred, target):
    """Mean Absolute Error."""
    return torch.mean(torch.abs(pred - target))


def compute_ergas(pred, target, ratio=1.0):
    """ERGAS (Erreur Relative Globale Adimensionnelle de Synthese)."""
    B, C, H, W = pred.shape
    mse_per_band = torch.mean((pred - target) ** 2, dim=[2, 3])  # [B, C]
    ergas = 100 * ratio * torch.sqrt(mse_per_band.mean())
    return ergas


def compute_all_metrics(pred, target, data_range=1.0):
    """Compute all metrics at once.
    
    Args:
        pred: [B, C, H, W] — predicted image
        target: [B, C, H, W] — ground truth
        data_range: max pixel value
        
    Returns:
        dict of metric values
    """
    return {
        'psnr': compute_psnr(pred, target, data_range).item(),
        'ssim': compute_ssim(pred, target).item(),
        'sam': compute_sam(pred, target).item(),
        'rmse': compute_rmse(pred, target).item(),
        'mae': compute_mae(pred, target).item(),
        'ergas': compute_ergas(pred, target).item(),
    }

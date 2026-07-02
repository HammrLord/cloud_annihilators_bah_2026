"""Reconstruction losses: L1 and SSIM."""

import torch
import torch.nn as nn


class L1Loss(nn.Module):
    """L1 pixel reconstruction loss.
    
    L1 preferred over L2: L2's quadratic penalty encourages blurry mean
    predictions under uncertainty. L1 is more robust to outlier errors
    from thick cloud regions.
    """

    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(self, pred, target):
        return self.loss(pred, target)


class SSIMLoss(nn.Module):
    """Structural Similarity Index loss.
    
    Captures luminance, contrast, and spatial structure simultaneously.
    Sensitive to edge-level errors that L1 underweights.
    """

    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def _gaussian_window(self, window_size, sigma):
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        return g.unsqueeze(0) * g.unsqueeze(1)

    def forward(self, pred, target):
        """Compute 1 - SSIM."""
        C = pred.shape[1]
        device = pred.device

        window = self._gaussian_window(self.window_size, 1.5).to(device)
        window = window.unsqueeze(0).unsqueeze(0)
        window = window.expand(C, 1, self.window_size, self.window_size)
        window = window.reshape(C * self.window_size * self.window_size, 1, self.window_size, self.window_size)

        # Pad inputs
        pad = self.window_size // 2
        pred_pad = torch.nn.functional.pad(pred, (pad, pad, pad, pad), mode='reflect')
        target_pad = torch.nn.functional.pad(target, (pad, pad, pad, pad), mode='reflect')

        # Compute local means
        mu_pred = torch.nn.functional.conv2d(pred_pad, window, groups=C, padding=0)
        mu_target = torch.nn.functional.conv2d(target_pad, window, groups=C, padding=0)

        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_cross = mu_pred * mu_target

        sigma_pred_sq = torch.nn.functional.conv2d(pred_pad ** 2, window, groups=C, padding=0) - mu_pred_sq
        sigma_target_sq = torch.nn.functional.conv2d(target_pad ** 2, window, groups=C, padding=0) - mu_target_sq
        sigma_cross = torch.nn.functional.conv2d(pred_pad * target_pad, window, groups=C, padding=0) - mu_cross

        ssim_map = ((2 * mu_cross + self.C1) * (2 * sigma_cross + self.C2)) / \
                   ((mu_pred_sq + mu_target_sq + self.C1) * (sigma_pred_sq + sigma_target_sq + self.C2))

        return 1 - ssim_map.mean()

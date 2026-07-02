"""NDVI consistency loss for vegetation preservation."""

import torch
import torch.nn as nn


class NDVIConsistencyLoss(nn.Module):
    """NDVI consistency loss.
    
    Directly penalizes NDVI error to ensure the reconstruction
    preserves vegetation indices critical for downstream applications
    (crop assessment, forest monitoring, change detection).
    
    NDVI = (NIR - Red) / (NIR + Red + eps)
    
    Applied only to cloudy pixels where reconstruction matters most.
    """

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def compute_ndvi(self, x):
        """Compute NDVI from 3-band image (G, R, NIR)."""
        nir = x[:, 2:3]  # Band 3 (NIR)
        red = x[:, 1:2]  # Band 2 (Red)
        ndvi = (nir - red) / (nir + red + self.eps)
        return ndvi

    def forward(self, pred, target, cloud_mask=None):
        """
        Args:
            pred: [B, 3, H, W] — predicted (G, R, NIR)
            target: [B, 3, H, W] — ground truth (G, R, NIR)
            cloud_mask: [B, 1, H, W] — optional cloud mask (1=cloudy)
        Returns:
            Scalar NDVI loss
        """
        ndvi_pred = self.compute_ndvi(pred)
        ndvi_target = self.compute_ndvi(target)

        loss = torch.abs(ndvi_pred - ndvi_target)

        if cloud_mask is not None:
            # Apply only to cloudy pixels
            loss = (loss * cloud_mask).sum() / (cloud_mask.sum() + 1e-6)
        else:
            loss = loss.mean()

        return loss

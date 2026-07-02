"""Spectral Angle Mapper (SAM) loss."""

import torch
import torch.nn as nn


class SAMLoss(nn.Module):
    """Spectral Angle Mapper loss.
    
    Measures angular distance between predicted and ground truth
    spectral vectors. Illumination-invariant — penalizes only
    spectral direction mismatch, not brightness.
    
    L_SAM = arccos( (pred · target) / (||pred|| * ||target||) )
    """

    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        """
        Args:
            pred: [B, C, H, W]
            target: [B, C, H, W]
        Returns:
            Scalar SAM loss (mean angle in radians)
        """
        B, C, H, W = pred.shape

        pred_flat = pred.view(B, C, -1)     # [B, C, H*W]
        target_flat = target.view(B, C, -1) # [B, C, H*W]

        # Cosine similarity
        dot = (pred_flat * target_flat).sum(dim=1)  # [B, H*W]
        norm_pred = torch.norm(pred_flat, dim=1)    # [B, H*W]
        norm_target = torch.norm(target_flat, dim=1) # [B, H*W]

        cos_sim = dot / (norm_pred * norm_target + self.eps)
        cos_sim = torch.clamp(cos_sim, -1 + self.eps, 1 - self.eps)

        sam = torch.acos(cos_sim)  # [B, H*W]
        return sam.mean()

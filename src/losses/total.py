"""Combined loss function for Stage 2 training."""

import torch
import torch.nn as nn

from .reconstruction import L1Loss, SSIMLoss
from .spectral import SAMLoss
from .ndvi import NDVIConsistencyLoss
from .physics import PhysicsConsistencyLoss


class TotalLoss(nn.Module):
    """Combined loss for PSPRNet Stage 2 training.
    
    L_total = λ1·L1 + λ2·L_SSIM + λ3·L_NDVI + λ4·L_SAM_global + λ5·L_physics
    
    | Loss           | λ    | Supervised? | Works without GT? |
    |----------------|------|-------------|-------------------|
    | L1             | 1.0  | Yes         | No                |
    | L_SSIM         | 0.5  | Yes         | No                |
    | L_NDVI         | 0.2  | Yes         | No                |
    | L_SAM_global   | 0.1  | Yes         | No                |
    | L_physics      | 0.1  | No          | Yes               |
    """

    def __init__(self, l1_weight=1.0, ssim_weight=0.5, ndvi_weight=0.2,
                 sam_weight=0.1, physics_weight=0.1):
        super().__init__()
        self.l1_loss = L1Loss()
        self.ssim_loss = SSIMLoss()
        self.ndvi_loss = NDVIConsistencyLoss()
        self.sam_loss = SAMLoss()
        self.physics_loss = PhysicsConsistencyLoss()

        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.ndvi_weight = ndvi_weight
        self.sam_weight = sam_weight
        self.physics_weight = physics_weight

    def forward(self, outputs, targets, ppe_outputs=None):
        """
        Args:
            outputs: dict from SPRNet.forward()
                - cloud_free: [B, 3, H, W]
                - R_init: [B, 3, H, W]
                - ppe_outputs: dict with T, A, etc.
            targets: dict with:
                - clear: [B, 3, H, W] — ground truth
                - cloudy: [B, 3, H, W] — original cloudy image
            ppe_outputs: dict from PPE (optional, can also come from outputs)
        Returns:
            total_loss, loss_dict
        """
        pred = outputs['cloud_free']
        gt = targets['clear']

        if ppe_outputs is None:
            ppe_outputs = outputs.get('ppe_outputs', {})

        # Supervised losses (require ground truth)
        l1 = self.l1_loss(pred, gt)
        ssim = self.ssim_loss(pred, gt)
        ndvi = self.ndvi_loss(pred, gt)
        sam = self.sam_loss(pred, gt)

        # Physics consistency loss (no ground truth needed)
        physics = torch.tensor(0.0, device=pred.device)
        if ppe_outputs and 'T' in ppe_outputs and 'A' in ppe_outputs:
            cloudy = targets.get('cloudy', None)
            if cloudy is not None:
                physics = self.physics_loss(
                    pred, cloudy, ppe_outputs['T'], ppe_outputs['A']
                )

        # Total loss
        total = (self.l1_weight * l1 +
                 self.ssim_weight * ssim +
                 self.ndvi_weight * ndvi +
                 self.sam_weight * sam +
                 self.physics_weight * physics)

        loss_dict = {
            'total': total.item(),
            'l1': l1.item(),
            'ssim': ssim.item(),
            'ndvi': ndvi.item(),
            'sam': sam.item(),
            'physics': physics.item(),
        }

        return total, loss_dict

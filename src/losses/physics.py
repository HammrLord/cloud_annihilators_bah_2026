"""Physics consistency loss based on Atmospheric Scattering Model."""

import torch
import torch.nn as nn


class PhysicsConsistencyLoss(nn.Module):
    """Physics consistency loss (ASM-based).
    
    Self-consistency constraint: if R_final is correct, applying the
    cloud physics model forward should recover the original cloudy
    observation.
    
    L_physics = ||I_recon - I_cloudy||_1
    where I_recon = A * (1 - T) + R_final * T
    
    Does not require cloud-free ground truth — active on unpaired
    LISS-IV scenes.
    
    Reference: GUPI-Net (Gong et al., Neural Networks 2026)
    """

    def __init__(self):
        super().__init__()

    def forward(self, R_final, cloudy, T, A):
        """
        Args:
            R_final: [B, 3, H, W] — reconstructed cloud-free reflectance
            cloudy: [B, 3, H, W] — original cloudy observation
            T: [B, 1, H, W] — transmission map from PPE
            A: [B, 3] — atmospheric light from PPE
        Returns:
            Scalar physics consistency loss
        """
        # Reconstruct cloudy image from R_final using ASM
        A_expanded = A.unsqueeze(-1).unsqueeze(-1)  # [B, 3, 1, 1]
        I_recon = T * R_final + (1 - T) * A_expanded

        # L1 distance to actual cloudy observation
        loss = torch.abs(I_recon - cloudy).mean()
        return loss

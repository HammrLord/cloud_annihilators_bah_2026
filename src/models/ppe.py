"""
Physical Property Estimator (PPE).

Predicts cloud density, atmospheric light A, and spatially varying transmission T.
Outputs serve two roles:
  1. Compute R_init (physics-guided initial reflectance)
  2. Form AdaLN conditioning vector
"""

import torch
import torch.nn as nn


class PPE(nn.Module):
    """Physical Property Estimator.

    Input: Cloudy optical [B, 3, H, W] (Green, Red, NIR)
    Outputs:
        cloud_density [B, 1]          - global scalar
        A [B, 3]                       - atmospheric light per band
        T [B, 1, H, W]                - spatially varying transmission
        R_init [B, 3, H, W]           - physics-guided initial reflectance
        cond [B, 2]                    - AdaLN conditioning [mean(T), cloud_density]
    """

    def __init__(self, in_channels=3, hidden_channels=16):
        super().__init__()
        eps = 1e-6

        # Shared encoder
        self.shared_encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Cloud density (global scalar)
        self.density_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

        # Atmospheric light A (per-band global)
        self.atm_light_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 3),
            nn.Sigmoid(),
        )

        # Transmission map T (spatially varying)
        self.transmission_head = nn.Sequential(
            nn.Conv2d(hidden_channels, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 1),
            nn.Sigmoid(),
        )

    def compute_r_init(self, I, T, A):
        """Compute physics-guided initial reflectance.
        
        R_init(x,y,lambda) = [I(x,y,lambda) - (1 - T(x,y)) * A(lambda)] / (T(x,y) + eps)
        
        Under thin cloud, R_init is a reasonable estimate.
        Under thick cloud, it is corrupted — the network corrects via residual.
        """
        eps = 1e-6
        # A: [B, 3] -> [B, 3, 1, 1] for broadcasting
        A_expanded = A.unsqueeze(-1).unsqueeze(-1)
        R_init = (I - (1 - T) * A_expanded) / (T + eps)
        return R_init

    def forward(self, x):
        """
        Args:
            x: Cloudy optical [B, 3, H, W]
        Returns:
            dict with cloud_density, A, T, R_init, cond
        """
        feat = self.shared_encoder(x)

        cloud_density = self.density_head(feat)        # [B, 1]
        A = self.atm_light_head(feat)                  # [B, 3]
        T = self.transmission_head(feat)               # [B, 1, H, W]

        R_init = self.compute_r_init(x, T, A)          # [B, 3, H, W]

        # AdaLN conditioning: [mean_transmission, cloud_density]
        mean_T = T.mean(dim=[2, 3], keepdim=False)     # [B, 1]
        cond = torch.cat([mean_T, cloud_density], dim=1)  # [B, 2]

        return {
            'cloud_density': cloud_density,
            'A': A,
            'T': T,
            'R_init': R_init,
            'cond': cond,
        }

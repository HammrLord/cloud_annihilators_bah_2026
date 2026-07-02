"""
Multi-Modal Stem for fusing optical, SAR, and physics features.
"""

import torch
import torch.nn as nn


class MultiModalStem(nn.Module):
    """Fuses optical, SAR, and physics features before ECRFormer.

    Optical input: [B, 9, H, W] (G, R, NIR, NDVI, NDWI, DVI, R_init_G, R_init_R, R_init_NIR)
    SAR input: [B, 3, H, W] (log_VV, log_VH, log_ratio)
    DEM input: [B, 3, H, W] (elevation, slope, aspect)

    Output: [B, 48, H, W]
    """

    def __init__(self, optical_channels=9, sar_channels=3, dem_channels=3,
                 out_channels=48):
        super().__init__()
        self.optical_conv = nn.Sequential(
            nn.Conv2d(optical_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.sar_conv = nn.Sequential(
            nn.Conv2d(sar_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.dem_conv = nn.Sequential(
            nn.Conv2d(dem_channels, 16, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        # Fusion: 32 (optical) + 32 (sar) + 16 (dem) = 80 -> out_channels
        self.fusion = nn.Sequential(
            nn.Conv2d(80, out_channels, 1),
            nn.ReLU(inplace=True),
        )

    def forward(self, optical, sar, dem):
        """
        Args:
            optical: [B, 9, H, W]
            sar: [B, 3, H, W]
            dem: [B, 3, H, W]
        Returns:
            [B, 48, H, W]
        """
        opt_feat = self.optical_conv(optical)
        sar_feat = self.sar_conv(sar)
        dem_feat = self.dem_conv(dem)
        combined = torch.cat([opt_feat, sar_feat, dem_feat], dim=1)
        return self.fusion(combined)


class SimpleStem(nn.Module):
    """Simple stem without DEM (for Stage 1 MSKT)."""

    def __init__(self, in_channels=5, out_channels=48):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 7, padding=3),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

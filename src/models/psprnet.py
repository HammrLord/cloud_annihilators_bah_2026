"""
Full Stage 2 model: MSKT + PPE + AdaLN + ECRFormer → cloud-free output.

Architecture:
    1. MSKT (frozen/pretrained) → augmented spectral knowledge
    2. PPE → physics-guided initialization + AdaLN conditioning
    3. Multi-Modal Stem → fuse optical, SAR, physics features
    4. ECRFormer backbone with AdaLN → predict residual ΔR
    5. R_final = clamp(R_init + ΔR, 0, 1)
"""

import torch
import torch.nn as nn

from .mskt import MSKT
from .ppe import PPE
from .ecrformer import ECRFormerBackbone
from .stems import MultiModalStem


class SPRNet(nn.Module):
    """Physics-guided Spectral Prior Network (Stage 2).

    Orchestrates all components for cloud removal on LISS-IV imagery.
    """

    def __init__(self, config):
        super().__init__()
        # MSKT (pretrained, optionally frozen)
        self.mskt = MSKT(
            in_channels=config.get('mskt_in_channels', 5),
            out_channels=config.get('mskt_out_channels', 13),
            embed_dim=config.get('mskt_embed_dim', 64),
            num_heads=config.get('mskt_num_heads', 8),
            num_encoder_layers=config.get('mskt_encoder_layers', 6),
            num_decoder_layers=config.get('mskt_decoder_layers', 6),
        )

        # PPE
        self.ppe = PPE(
            in_channels=config.get('ppe_in_channels', 3),
            hidden_channels=config.get('ppe_hidden_channels', 16),
        )

        # ECRFormer backbone
        self.backbone = ECRFormerBackbone(
            in_channels=config.get('backbone_in_channels', 15),
            out_channels=config.get('backbone_out_channels', 3),
            features_start=config.get('features_start', 48),
            num_blocks=config.get('num_blocks', (2, 3, 2)),
            num_refine=config.get('num_refine', 4),
            drop_path_rate=config.get('drop_path_rate', 0.0),
        )

        # Stem: fuse all modalities
        self.stem = MultiModalStem(
            optical_channels=9,   # G, R, NIR, NDVI, NDWI, DVI, R_init x3
            sar_channels=3,       # log_VV, log_VH, log_ratio
            dem_channels=3,       # elevation, slope, aspect
            out_channels=config.get('features_start', 48),
        )

    def compute_indices(self, x):
        """Compute spectral indices from optical input.

        Args:
            x: [B, 3, H, W] — Green, Red, NIR
        Returns:
            [B, 6, H, W] — NDVI, NDWI, DVI, R_init_G, R_init_R, R_init_NIR
        """
        eps = 1e-6
        green, red, nir = x[:, 0:1], x[:, 1:2], x[:, 2:3]

        ndvi = (nir - red) / (nir + red + eps)
        ndwi = (green - nir) / (green + nir + eps)
        dvi = nir - red

        return torch.cat([ndvi, ndwi, dvi], dim=1)

    def prepare_sar(self, sar):
        """Convert SAR to log-space representation.

        Args:
            sar: [B, 2, H, W] — VV, VH (raw)
        Returns:
            [B, 3, H, W] — log_VV, log_VH, log_ratio
        """
        eps = 1e-6
        vv = sar[:, 0:1]
        vh = sar[:, 1:2]

        log_vv = torch.log(torch.abs(vv) + eps)
        log_vh = torch.log(torch.abs(vh) + eps)
        log_ratio = torch.log(torch.abs(vv) / (torch.abs(vh) + eps) + eps)

        return torch.cat([log_vv, log_vh, log_ratio], dim=1)

    def compute_dem_features(self, dem):
        """Compute terrain features from DEM.

        Args:
            dem: [B, 3, H, W] — elevation, slope, aspect
        Returns:
            [B, 3, H, W] — same (features are pre-computed)
        """
        return dem

    def freeze_mskt(self):
        """Freeze MSKT parameters."""
        for param in self.mskt.parameters():
            param.requires_grad = False

    def unfreeze_mskt(self):
        """Unfreeze MSKT parameters."""
        for param in self.mskt.parameters():
            param.requires_grad = True

    def forward(self, optical, sar, dem=None):
        """
        Args:
            optical: [B, 3, H, W] — cloudy LISS-IV (G, R, NIR)
            sar: [B, 2, H, W] — Sentinel-1 SAR (VV, VH)
            dem: [B, 3, H, W] — DEM (elevation, slope, aspect), optional

        Returns:
            dict with:
                cloud_free: [B, 3, H, W] — reconstructed cloud-free image
                R_init: [B, 3, H, W] — physics initialization
                ppe_outputs: dict — PPE intermediate outputs
                mskt_output: [B, 13, H, W] — MSKT spectral augmentation
        """
        B, C, H, W = optical.shape

        # Step 1: MSKT spectral augmentation
        mskt_input = torch.cat([optical, sar], dim=1)  # [B, 5, H, W]
        mskt_output = self.mskt(mskt_input)  # [B, 13, H, W]

        # Step 2: PPE — physics-guided initialization
        ppe_outputs = self.ppe(optical)
        R_init = ppe_outputs['R_init']  # [B, 3, H, W]
        cond = ppe_outputs['cond']      # [B, 2]

        # Step 3: Compute spectral indices
        indices = self.compute_indices(optical)  # [B, 3, H, W]

        # Step 4: Prepare SAR
        sar_log = self.prepare_sar(sar)  # [B, 3, H, W]

        # Step 5: DEM features
        if dem is None:
            dem = torch.zeros(B, 3, H, W, device=optical.device)
        dem_feat = self.compute_dem_features(dem)

        # Step 6: Multi-modal stem fusion
        optical_input = torch.cat([
            optical,           # 3 channels (G, R, NIR)
            indices,           # 3 channels (NDVI, NDWI, DVI)
            R_init,            # 3 channels (physics initialization)
        ], dim=1)  # [B, 9, H, W]

        stem_output = self.stem(optical_input, sar_log, dem_feat)  # [B, 48, H, W]

        # Step 7: ECRFormer backbone → residual prediction
        delta_R = self.backbone(stem_output)  # [B, 3, H, W]

        # Step 8: Final reconstruction
        cloud_free = torch.clamp(R_init + delta_R, 0, 1)

        return {
            'cloud_free': cloud_free,
            'R_init': R_init,
            'ppe_outputs': ppe_outputs,
            'mskt_output': mskt_output,
            'cond': cond,
        }

"""
Adaptive Layer Normalization (AdaLN) conditioning for ECRFormer blocks.

Wraps each ECRFormer block with AdaLN that modulates features based on
physics properties (cloud density, transmission).

Conditioning vector: cond [B, 2] = [mean_transmission, cloud_density]
"""

import torch
import torch.nn as nn

from .ecrformer_blocks import ECRFormerBlock


class AdaLN_ECRFormerBlock(nn.Module):
    """ECRFormer block with AdaLN conditioning.

    The conditioning vector generates dynamic scale and shift parameters
    that modulate normalized features before attention. Zero-initialized
    so model starts as standard ECRFormer.
    """

    def __init__(self, dim, cond_dim=2, num_heads=8, window_size=8,
                 mlp_ratio=2.66, drop_path=0.0):
        super().__init__()
        # ECRFormer internals (unchanged from base block)
        self.block = ECRFormerBlock(dim, num_heads=num_heads,
                                     window_size=window_size,
                                     mlp_ratio=mlp_ratio,
                                     drop_path=drop_path)

        # AdaLN parameter generators
        self.adaLN_scale1 = nn.Linear(cond_dim, dim)
        self.adaLN_shift1 = nn.Linear(cond_dim, dim)
        self.adaLN_scale2 = nn.Linear(cond_dim, dim)
        self.adaLN_shift2 = nn.Linear(cond_dim, dim)

        # Zero initialization — at epoch 0, model = standard ECRFormer
        for layer in [self.adaLN_scale1, self.adaLN_shift1,
                      self.adaLN_scale2, self.adaLN_shift2]:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x, cond):
        """
        Args:
            x: [B, C, H, W]
            cond: [B, 2] — [mean_transmission, cloud_density]
        """
        # Get scale/shift from conditioning
        scale1 = self.adaLN_scale1(cond).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        shift1 = self.adaLN_shift1(cond).unsqueeze(-1).unsqueeze(-1)
        scale2 = self.adaLN_scale2(cond).unsqueeze(-1).unsqueeze(-1)
        shift2 = self.adaLN_shift2(cond).unsqueeze(-1).unsqueeze(-1)

        # The base block applies: x + TSA(x) + MDWA(x) + FFN(x)
        # We apply AdaLN before each sub-block by modulating the input
        # For simplicity, apply AdaLN at block level (modulates overall features)
        x_norm = x * (1 + scale1) + shift1
        x = self.block(x_norm)

        return x

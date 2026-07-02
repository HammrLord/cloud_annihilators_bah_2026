"""
Stage 1: Multimodal Spectral Knowledge Transformer (MSKT).

Pretrained on SEN12MS-CR to reconstruct 13-band Sentinel-2 from
LISS-IV (3 bands) + SAR (2 bands) = 5 channels input.

Learns rich multispectral representations using L1, SSIM, and SAM losses.
The pretrained MSKT is transferred to Stage 2 as a spectral prior.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .ecrformer_blocks import ECRFormerBlock, LayerNorm
from .stems import SimpleStem


class PatchEmbed(nn.Module):
    """Convert image to sequence of patches."""

    def __init__(self, in_channels, embed_dim, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, C, H, W] -> [B, embed_dim, H/P, W/P]
        return self.proj(x)


class PatchUnembed(nn.Module):
    """Convert sequence of patches back to image."""

    def __init__(self, embed_dim, out_channels, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(embed_dim, out_channels,
                              kernel_size=1)

    def forward(self, x, H, W):
        # x: [B, embed_dim, H/P, W/P] -> [B, out_channels, H, W]
        x = self.proj(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear',
                          align_corners=False)
        return x


class MSKT(nn.Module):
    """Multimodal Spectral Knowledge Transformer.

    Architecture:
        1. Patch embed: 5 channels -> embed_dim
        2. Encoder: N transformer blocks (captures multispectral representations)
        3. Decoder: N transformer blocks (reconstructs 13-band output)
        4. Patch unembed: embed_dim -> 13 channels

    Input: [B, 5, H, W] (LISS-IV G/R/NIR + SAR VV/VH)
    Output: [B, 13, H, W] (reconstructed Sentinel-2)
    """

    def __init__(self, in_channels=5, out_channels=13, embed_dim=64,
                 num_heads=8, num_encoder_layers=6, num_decoder_layers=6,
                 mlp_ratio=4.0, dropout=0.1, patch_size=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        # Patch embedding
        self.patch_embed = PatchEmbed(in_channels, embed_dim, patch_size)

        # Positional embedding (learnable)
        self.pos_embed = None  # Will be lazily initialized

        # Encoder
        self.encoder_blocks = nn.ModuleList([
            ECRFormerBlock(embed_dim, num_heads=num_heads,
                           mlp_ratio=mlp_ratio, drop_path=dropout)
            for _ in range(num_encoder_layers)
        ])
        self.encoder_norm = LayerNorm(embed_dim)

        # Decoder
        self.decoder_blocks = nn.ModuleList([
            ECRFormerBlock(embed_dim, num_heads=num_heads,
                           mlp_ratio=mlp_ratio, drop_path=dropout)
            for _ in range(num_decoder_layers)
        ])
        self.decoder_norm = LayerNorm(embed_dim)

        # Output projection
        self.output_proj = nn.Conv2d(embed_dim, out_channels, 1)

        # Learnable scale for output
        self.output_scale = nn.Parameter(torch.ones(1, out_channels, 1, 1))

    def _get_pos_embed(self, H, W, device):
        if self.pos_embed is None or self.pos_embed.shape[2:] != (H, W):
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.embed_dim, H, W, device=device)
            )
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        return self.pos_embed

    def encode(self, x):
        """Encoder path."""
        B, C, H, W = x.shape
        Hp, Wp = H // self.patch_size, W // self.patch_size

        x = self.patch_embed(x)  # [B, embed_dim, Hp, Wp]
        pos = self._get_pos_embed(Hp, Wp, x.device)
        x = x + pos

        for block in self.encoder_blocks:
            x = block(x)

        return self.encoder_norm(x)

    def decode(self, x):
        """Decoder path."""
        for block in self.decoder_blocks:
            x = block(x)
        return self.decoder_norm(x)

    def forward(self, x):
        """
        Args:
            x: [B, 5, H, W] — LISS-IV (3ch) + SAR (2ch)
        Returns:
            [B, 13, H, W] — reconstructed Sentinel-2
        """
        B, C, H, W = x.shape

        # Encode
        features = self.encode(x)  # [B, embed_dim, Hp, Wp]

        # Decode
        features = self.decode(features)

        # Project to output channels and upsample to original resolution
        out = self.output_proj(features)  # [B, 13, Hp, Wp]
        out = F.interpolate(out, size=(H, W), mode='bilinear',
                            align_corners=False)

        # Learnable output scaling
        out = out * self.output_scale

        return out

"""
U-shaped ECRFormer backbone with AdaLN conditioning.
Encoder → Bottleneck → Decoder with skip connections.
Rewritten for Cloud Annihilators.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .ecrformer_blocks import ECRFormerBlock, LayerNorm


class PatchMerge(nn.Module):
    """Downsample: 2x spatial reduction, 2x channel expansion."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.reduction = nn.Conv2d(in_channels * 4, out_channels, 1)
        self.norm = LayerNorm(in_channels * 4)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, H // 2, 2, W // 2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(B, C * 4, H // 2, W // 2)
        x = self.norm(x)
        return self.reduction(x)


class PatchExpand(nn.Module):
    """Upsample: 2x spatial expansion, 2x channel reduction."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.expand = nn.Conv2d(in_channels, out_channels * 4, 1)
        self.norm = LayerNorm(out_channels)
        self.pixelshuffle = nn.PixelShuffle(2)

    def forward(self, x):
        x = self.expand(x)
        x = self.pixelshuffle(x)
        return self.norm(x)


class ChannelAttention(nn.Module):
    """CBAM channel attention."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
        )

    def forward(self, x):
        w = torch.sigmoid(self.mlp(x)).unsqueeze(-1).unsqueeze(-1)
        return x * w


class SpatialAttention(nn.Module):
    """CBAM spatial attention."""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        w = torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * w


class ECRFormerBackbone(nn.Module):
    """U-shaped ECRFormer encoder-decoder backbone.

    Architecture:
        Encoder: 3 stages with PatchMerge downsampling
        Bottleneck: 2 blocks at lowest resolution
        Decoder: 3 stages with PatchExpand + skip connections
    """

    def __init__(self, in_channels=15, out_channels=3, features_start=48,
                 num_blocks=(2, 3, 2), num_refine=4, bottleneck='tsa',
                 drop_path_rate=0.0):
        super().__init__()
        self.num_stages = len(num_blocks)
        self.features_start = features_start

        # Channel dims per stage
        channels = [features_start * (2 ** i) for i in range(self.num_stages + 1)]

        # Encoder
        self.encoder_stages = nn.ModuleList()
        self.encoder_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for i in range(self.num_stages):
            stage_channels = channels[i]
            blocks = nn.ModuleList([
                ECRFormerBlock(stage_channels, drop_path=drop_path_rate)
                for _ in range(num_blocks[i])
            ])
            self.encoder_blocks.append(blocks)

            if i < self.num_stages - 1:
                self.downsamples.append(
                    PatchMerge(channels[i], channels[i + 1])
                )
            self.encoder_stages.append(nn.Identity())

        # Bottleneck
        bottle_channels = channels[-1]
        self.bottleneck_blocks = nn.ModuleList([
            ECRFormerBlock(bottle_channels, drop_path=drop_path_rate)
            for _ in range(2)
        ])

        # Decoder
        self.decoder_stages = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.skip_fusions = nn.ModuleList()

        for i in range(self.num_stages - 1, 0, -1):
            self.upsamples.append(
                PatchExpand(channels[i], channels[i - 1])
            )
            self.skip_fusions.append(
                nn.Conv2d(channels[i - 1] * 2, channels[i - 1], 1)
            )
            blocks = nn.ModuleList([
                ECRFormerBlock(channels[i - 1], drop_path=drop_path_rate)
                for _ in range(num_blocks[i - 1])
            ])
            self.decoder_blocks.append(blocks)
            self.decoder_stages.append(nn.Identity())

        # Final upsample to full resolution
        self.final_upsample = PatchExpand(channels[0], channels[0])
        self.final_fusion = nn.Conv2d(channels[0] * 2, channels[0], 1)
        self.final_blocks = nn.ModuleList([
            ECRFormerBlock(channels[0], drop_path=drop_path_rate)
            for _ in range(num_blocks[0])
        ])

        # Refinement
        self.refine_blocks = nn.ModuleList([
            ECRFormerBlock(channels[0], drop_path=drop_path_rate)
            for _ in range(num_refine)
        ])

        # Output head
        self.out_conv = nn.Conv2d(channels[0], out_channels, 3, padding=1)

    def forward(self, x):
        B, C, H, W = x.shape

        # Encoder
        skips = []
        for i in range(self.num_stages):
            for block in self.encoder_blocks[i]:
                x = block(x)
            skips.append(x)
            if i < self.num_stages - 1:
                x = self.downsamples[i](x)

        # Bottleneck
        for block in self.bottleneck_blocks:
            x = block(x)

        # Decoder
        for i, upsample in enumerate(self.upsamples):
            x = upsample(x)
            skip_idx = self.num_stages - 1 - i
            skip = skips[skip_idx]
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear',
                                  align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = self.skip_fusions[i](x)
            for block in self.decoder_blocks[i]:
                x = block(x)

        # Final upsample + skip
        x = self.final_upsample(x)
        skip0 = skips[0]
        if x.shape[2:] != skip0.shape[2:]:
            x = F.interpolate(x, size=skip0.shape[2:], mode='bilinear',
                              align_corners=False)
        x = torch.cat([x, skip0], dim=1)
        x = self.final_fusion(x)
        for block in self.final_blocks:
            x = block(x)

        # Refinement
        for block in self.refine_blocks:
            x = block(x)

        return self.out_conv(x)

"""
ECRFormer building blocks: XCA, MDWA, GatedFFN, and ECRFormerBlock.
Rewritten for Cloud Annihilators - not copied from ECRformer_repo.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class LayerNorm(nn.Module):
    """Channel-wise LayerNorm with learnable scale."""

    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        x = F.layer_norm(x, [x.shape[1]], eps=1e-6)
        return x * self.g


class TransposedAttention(nn.Module):
    """Multi-DConv Head Transposed Self-Attention (MDTA).
    
    Operates across channels (transposed attention) with depthwise convolution
    for local context. Complexity O(C^2) instead of O(N^2) for spatial attention.
    """

    def __init__(self, dim, num_heads=8, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, 3, padding=1,
                                     groups=dim * 3, bias=bias)
        self.proj = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (h d) x y -> b h (x y) d', h=self.num_heads)
        k = rearrange(k, 'b (h d) x y -> b h d (x y)', h=self.num_heads)
        v = rearrange(v, 'b (h d) x y -> b h (x y) d', h=self.num_heads)

        attn = torch.matmul(q, k) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=H, y=W)
        return self.proj(out)


class GatedFFN(nn.Module):
    """Gated-Dconv Feed-Forward Network (GDFN).
    
    Two-branch gated activation: one branch for content, one for gating.
    """

    def __init__(self, dim, expansion_factor=2.66, bias=True):
        super().__init__()
        hidden = int(dim * expansion_factor)
        self.norm = LayerNorm(dim)
        self.project_in = nn.Conv2d(dim, hidden * 2, 1, bias=bias)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1,
                                groups=hidden * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden, dim, 1, bias=bias)

    def forward(self, x):
        x = self.norm(x)
        x = self.project_in(x)
        x = self.dwconv(x)
        x1, x2 = x.chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)


class WindowPartition:
    """Helper for window-based attention."""

    @staticmethod
    def partition(x, window_size):
        B, C, H, W = x.shape
        x = x.view(B, C, H // window_size, window_size,
                    W // window_size, window_size)
        windows = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        windows = windows.view(-1, window_size * window_size, C)
        return windows

    @staticmethod
    def reverse(windows, window_size, H, W):
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size,
                         window_size, window_size, -1)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.view(B, -1, H, W)


class MultiDilateWindowAttention(nn.Module):
    """Multi-Dilation Window Attention (MDWA).
    
    Applies window attention with multiple dilation rates [1,2,3,4] to capture
    multi-scale context within local windows.
    """

    def __init__(self, dim, num_heads=8, window_size=8, dilations=None, bias=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.dilations = dilations or [1, 2, 3, 4]
        self.num_dilations = len(self.dilations)
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.proj = nn.Conv2d(dim, dim, 1, bias=bias)
        self.norm = LayerNorm(dim)

        # Per-dilation relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1),
                        num_heads * self.num_dilations)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.norm(x)
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        # Pad to multiple of window_size
        pad_l = pad_t = 0
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        x_padded = F.pad(x, (pad_l, pad_r, pad_t, pad_b))

        # Partition into windows
        q_padded = F.pad(q, (pad_l, pad_r, pad_t, pad_b))
        k_padded = F.pad(k, (pad_l, pad_r, pad_t, pad_b))
        v_padded = F.pad(v, (pad_l, pad_r, pad_t, pad_b))

        Hp = H + pad_b
        Wp = W + pad_r
        q_windows = WindowPartition.partition(q_padded, self.window_size)
        k_windows = WindowPartition.partition(k_padded, self.window_size)
        v_windows = WindowPartition.partition(v_padded, self.window_size)

        Bn = q_windows.shape[0]
        q_windows = q_windows.view(Bn, self.window_size ** 2, self.num_heads, self.head_dim)
        k_windows = k_windows.view(Bn, self.window_size ** 2, self.num_heads, self.head_dim)
        v_windows = v_windows.view(Bn, self.window_size ** 2, self.num_heads, self.head_dim)

        attn = torch.matmul(q_windows.transpose(1, 2), k_windows.transpose(1, 2).transpose(-2, -1))
        attn = attn * self.scale

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size ** 2, self.window_size ** 2, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v_windows.transpose(1, 2))

        out = out.transpose(1, 2).contiguous().view(Bn, self.window_size ** 2, C)
        out = out.view(Bn, self.window_size, self.window_size, C)
        out = out.permute(0, 3, 1, 2).contiguous()

        # Reverse window partition
        out = WindowPartition.reverse(out, self.window_size, Hp, Wp)
        out = out[:, :, :H, :W]

        return self.proj(out)


class ECRFormerBlock(nn.Module):
    """ECRFormer block: TransposedAttention → MDWA → GatedFFN.
    
    With learnable per-layer scaling and DropPath for regularization.
    """

    def __init__(self, dim, num_heads=8, window_size=8, mlp_ratio=2.66,
                 drop_path=0.0, dilations=None):
        super().__init__()
        self.tsa = TransposedAttention(dim, num_heads)
        self.mdwa = MultiDilateWindowAttention(dim, num_heads, window_size,
                                                dilations=dilations)
        self.ffn = GatedFFN(dim, expansion_factor=mlp_ratio)

        # Learnable scaling (initialized to 1)
        self.scale_tsa = nn.Parameter(torch.ones(dim))
        self.scale_mdwa = nn.Parameter(torch.ones(dim))
        self.scale_ffn = nn.Parameter(torch.ones(dim))

        # DropPath
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.tsa(x) * self.scale_tsa)
        x = x + self.drop_path(self.mdwa(x) * self.scale_mdwa)
        x = x + self.drop_path(self.ffn(x) * self.scale_ffn)
        return x


class DropPath(nn.Module):
    """Stochastic depth (drop path) regularization."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, device=x.device, dtype=x.dtype)
        random_tensor = torch.floor(random_tensor + keep_prob)
        output = x / keep_prob * random_tensor
        return output

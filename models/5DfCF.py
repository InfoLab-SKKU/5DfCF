#!/usr/bin/env python3
"""
5DfCF.py

5DfCF: CrossFormer-based model for 5D fMRI time-series data.

Features:
  - Handles input shape (B, C, T, D, H, W)
  - Collapses time dimension into channels for patch embedding
  - Positional embedding on spatial tokens
  - Swin-inspired 3D attention blocks
  - Stages with optional downsampling
  - Classifier head
Usage:
    model = 5DfCF(
        img_size=(96,96,96),
        seq_len=120,       # number of time frames
        patch_size=(4,4,4),
        in_chans=1,
        num_classes=2,
        embed_dim=24,
        depths=[2,2,6,2],
        num_heads=[3,6,12,24],
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1
    )
    x = torch.randn(2, 1, 120, 96, 96, 96)
    out = model(x)  # [2, num_classes]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_


def _to_3d(x):
    # collapse time into channel
    B, C, T, D, H, W = x.shape
    return x.view(B, C * T, D, H, W)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class PositionalEmbedding3D(nn.Module):
    def __init__(self, dim, vol_dim):
        super().__init__()
        d, h, w = vol_dim
        self.pos_embed = nn.Parameter(torch.zeros(1, d*h*w, dim))
        trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # x: [B, N, C]
        return x + self.pos_embed


class Attention3D(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C//self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        q = q * self.scale
        attn = self.softmax(q @ k.transpose(-2, -1))
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(self.proj(x))
        return x


class SwiFTBlock3D(nn.Module):
    def __init__(self, dim, input_resolution, num_heads,
                 mlp_ratio=4., drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention3D(dim, num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = Mlp(dim, hidden_features=mlp_hidden, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging3D(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        D, H, W = input_resolution
        self.norm = norm_layer(dim)
        self.reduction = nn.Conv3d(dim, dim*2, kernel_size=2, stride=2)

    def forward(self, x):
        B, N, C = x.shape
        D = H = W = int(N ** (1/3))
        x = x.view(B, D, H, W, C).permute(0,4,1,2,3)
        x = self.norm(x.flatten(2).transpose(1,2)).transpose(1,2)
        x = self.reduction(x)
        D2, H2, W2 = D//2, H//2, W//2
        x = x.view(B, 2*C, D2*H2*W2).permute(0,2,1)
        return x


class Stage3D(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, downsample=None,
                 drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwiFTBlock3D(dim, input_resolution, num_heads,
                         drop_path=drop_path, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.downsample = downsample(input_resolution, dim, norm_layer) if downsample else None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample:
            x = self.downsample(x)
        return x


class PatchEmbed3D(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim, norm_layer=None):
        super().__init__()
        D, H, W = img_size
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        x = self.proj(x)
        B, C, Dp, Hp, Wp = x.shape
        x = x.view(B, C, Dp*Hp*Wp).permute(0,2,1)
        if self.norm: x = self.norm(x)
        return x


class 5DfCF(nn.Module):
    """
    5DfCF: CrossFormer for 5D fMRI time-series.
    """
    def __init__(self,
                 img_size=(96,96,96),
                 seq_len=120,
                 patch_size=(4,4,4),
                 in_chans=1,
                 num_classes=2,
                 embed_dim=24,
                 depths=[2,2,6,2],
                 num_heads=[3,6,12,24],
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.seq_len = seq_len
        self.patch_embed = PatchEmbed3D(
            img_size, patch_size,
            in_chans * seq_len,
            embed_dim,
            norm_layer if True else None
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        total_blocks = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        self.stages = nn.ModuleList()
        prev_dim = embed_dim
        Dp, Hp, Wp = [s//p for s,p in zip(img_size, patch_size)]
        cur = 0
        for i, depth in enumerate(depths):
            stage = Stage3D(prev_dim, (Dp, Hp, Wp), depth,
                            num_heads[i],
                            downsample=PatchMerging3D if i < len(depths)-1 else None,
                            drop_path=dpr[cur], norm_layer=norm_layer)
            self.stages.append(stage)
            cur += depth
            if i < len(depths)-1:
                Dp, Hp, Wp = Dp//2, Hp//2, Wp//2
                prev_dim *= 2

        self.norm = norm_layer(prev_dim)
        self.head = nn.Linear(prev_dim, num_classes)

    def forward(self, x):
        # x: [B, C, T, D, H, W]
        x = _to_3d(x)                    # [B, C*T, D, H, W]
        x = self.patch_embed(x)         # [B, N, C]
        x = self.pos_drop(x)
        for stage in self.stages:
            x = stage(x)
        x = self.norm(x)
        x = x.mean(1)
        x = self.head(x)
        return x

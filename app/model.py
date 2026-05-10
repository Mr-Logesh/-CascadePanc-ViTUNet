"""
CascadePanc-ViTUNet Model Architecture
ViT Bottleneck + 3D U-Net for Pancreatic Tumor Segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.residual = (
            nn.Identity() if in_ch == out_ch
            else nn.Conv3d(in_ch, out_ch, 1, bias=False)
        )

    def forward(self, x):
        return self.conv(x) + self.residual(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.down = nn.Conv3d(in_ch, out_ch, 2, stride=2, bias=False)
        self.conv = ConvBlock(out_ch, out_ch, dropout)

    def forward(self, x):
        return self.conv(self.down(x))


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, dropout=0.0):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch, dropout)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class PatchEmbedding3D(nn.Module):
    def __init__(self, in_ch, embed_dim, patch_size=2):
        super().__init__()
        self.proj = nn.Conv3d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        B, C, D, H, W = x.shape
        x = rearrange(x, 'b c d h w -> b (d h w) c')
        return self.norm(x), (D, H, W)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=6, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h)[0]
        return x + self.mlp(self.norm2(x))


class ViTBottleneck(nn.Module):
    def __init__(self, in_ch, embed_dim=384, heads=6, depth=3, patch_size=2, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding3D(in_ch, embed_dim, patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, 27, embed_dim) * 0.02)
        self.blocks = nn.Sequential(
            *[TransformerBlock(embed_dim, heads, dropout=dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.proj_back = nn.Linear(embed_dim, in_ch)

    def forward(self, x):
        B, C, D, H, W = x.shape
        tokens, (Dp, Hp, Wp) = self.patch_embed(x)
        N = tokens.shape[1]
        if N != 27:
            pos = F.interpolate(
                self.pos_embed.transpose(1, 2), size=N,
                mode='linear', align_corners=False
            ).transpose(1, 2)
        else:
            pos = self.pos_embed
        tokens = self.blocks(tokens + pos)
        return rearrange(
            self.proj_back(self.norm(tokens)),
            'b (d h w) c -> b c d h w', d=Dp, h=Hp, w=Wp
        )


class ViTUNet(nn.Module):
    """
    ViT-enhanced U-Net for 3D Medical Image Segmentation.
    
    Architecture: 3D U-Net with Vision Transformer bottleneck
    - Encoder: 4 levels with strided convolution downsampling
    - Bottleneck: ViT with patch embedding + self-attention
    - Decoder: 4 levels with transposed convolution upsampling
    - Deep supervision at intermediate decoder levels (training only)
    """

    def __init__(self, in_ch=1, num_classes=2, base=24,
                 vit_dim=384, vit_depth=3, vit_heads=6, deep_sup=True):
        super().__init__()
        self.deep_sup = deep_sup

        # Encoder
        self.enc1 = ConvBlock(in_ch, base)
        self.enc2 = DownBlock(base, base * 2)
        self.enc3 = DownBlock(base * 2, base * 4, 0.1)
        self.enc4 = DownBlock(base * 4, base * 8, 0.1)

        # ViT Bottleneck
        self.down_bot = nn.Conv3d(base * 8, base * 8, 2, stride=2, bias=False)
        self.vit = ViTBottleneck(base * 8, vit_dim, vit_heads, vit_depth, 2, 0.1)

        # Decoder
        self.dec4 = UpBlock(base * 8, base * 8, base * 8, 0.1)
        self.dec3 = UpBlock(base * 8, base * 4, base * 4, 0.1)
        self.dec2 = UpBlock(base * 4, base * 2, base * 2)
        self.dec1 = UpBlock(base * 2, base, base)

        # Output
        self.final = nn.Conv3d(base, num_classes, 1)

        # Deep supervision heads
        if deep_sup:
            self.ds3 = nn.Conv3d(base * 4, num_classes, 1)
            self.ds2 = nn.Conv3d(base * 2, num_classes, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)

        # Bottleneck
        b = self.vit(self.down_bot(s4))

        # Decoder
        d4 = self.dec4(b, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        out = self.final(d1)

        if self.deep_sup and self.training:
            return out, self.ds3(d3), self.ds2(d2)
        return out

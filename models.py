"""
models.py
---------
3-D U-Net architectures for brain tumour segmentation.

    SimpleUNet    : standard encoder-decoder with skip connections
    AttentionUNet : soft attention gates on skip connections

Architecture (Section IV-C, Section VI-H of the paper):
    - 4 encoder / decoder levels
    - Channels: 32 -> 64 -> 128 -> 256 -> 512 (bottleneck)
    - 3x3x3 Conv + BN + ReLU blocks
    - 2x2x2 MaxPool downsampling
    - 2x2x2 TransposedConv upsampling
    - Input : 3 channels (T1ce, T2, FLAIR)
    - Output: 4 classes (BG, NCR/NET, ED, ET)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = DoubleConv3D(in_ch, out_ch)
        self.pool = nn.MaxPool3d(2, 2)
    def forward(self, x):
        skip = self.conv(x)
        return self.pool(skip), skip


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2)
        self.conv = DoubleConv3D(out_ch * 2, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:],
                              mode="trilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class AttentionGate(nn.Module):
    def __init__(self, g_ch: int, x_ch: int, inter_ch: int):
        super().__init__()
        self.W_g  = nn.Sequential(
            nn.Conv3d(g_ch,    inter_ch, 1, bias=False),
            nn.BatchNorm3d(inter_ch))
        self.W_x  = nn.Sequential(
            nn.Conv3d(x_ch,    inter_ch, 1, bias=False),
            nn.BatchNorm3d(inter_ch))
        self.psi  = nn.Sequential(
            nn.Conv3d(inter_ch, 1,        1, bias=False),
            nn.BatchNorm3d(1),
            nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:],
                               mode="trilinear", align_corners=False)
        return x * self.psi(self.relu(g1 + x1))


class AttentionDecoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2)
        self.att  = AttentionGate(out_ch, out_ch, out_ch // 2)
        self.conv = DoubleConv3D(out_ch * 2, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:],
                              mode="trilinear", align_corners=False)
        return self.conv(torch.cat([x, self.att(g=x, x=skip)], dim=1))


class SimpleUNet(nn.Module):
    """Standard 3-D U-Net with 4 encoder/decoder levels."""

    def __init__(self, in_ch: int = 3, num_classes: int = 4):
        super().__init__()
        self.enc1       = EncoderBlock(in_ch,  32)
        self.enc2       = EncoderBlock(32,     64)
        self.enc3       = EncoderBlock(64,    128)
        self.enc4       = EncoderBlock(128,   256)
        self.bottleneck = DoubleConv3D(256,   512)
        self.dec4       = DecoderBlock(512,   256)
        self.dec3       = DecoderBlock(256,   128)
        self.dec2       = DecoderBlock(128,    64)
        self.dec1       = DecoderBlock( 64,    32)
        self.out        = nn.Conv3d(32, num_classes, 1)

    def forward(self, x):
        x, s1 = self.enc1(x)
        x, s2 = self.enc2(x)
        x, s3 = self.enc3(x)
        x, s4 = self.enc4(x)
        x     = self.bottleneck(x)
        x     = self.dec4(x, s4)
        x     = self.dec3(x, s3)
        x     = self.dec2(x, s2)
        x     = self.dec1(x, s1)
        return self.out(x)


class AttentionUNet(nn.Module):
    """3-D U-Net with soft attention gates on skip connections."""

    def __init__(self, in_ch: int = 3, num_classes: int = 4):
        super().__init__()
        self.enc1       = EncoderBlock(in_ch,  32)
        self.enc2       = EncoderBlock(32,     64)
        self.enc3       = EncoderBlock(64,    128)
        self.enc4       = EncoderBlock(128,   256)
        self.bottleneck = DoubleConv3D(256,   512)
        self.dec4       = AttentionDecoderBlock(512, 256)
        self.dec3       = AttentionDecoderBlock(256, 128)
        self.dec2       = AttentionDecoderBlock(128,  64)
        self.dec1       = AttentionDecoderBlock( 64,  32)
        self.out        = nn.Conv3d(32, num_classes, 1)

    def forward(self, x):
        x, s1 = self.enc1(x)
        x, s2 = self.enc2(x)
        x, s3 = self.enc3(x)
        x, s4 = self.enc4(x)
        x     = self.bottleneck(x)
        x     = self.dec4(x, s4)
        x     = self.dec3(x, s3)
        x     = self.dec2(x, s2)
        x     = self.dec1(x, s1)
        return self.out(x)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x      = torch.randn(1, 3, 128, 128, 128).to(device)
    for M in [SimpleUNet, AttentionUNet]:
        m      = M().to(device)
        out    = m(x)
        params = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"{M.__name__}: {tuple(x.shape)} -> "
              f"{tuple(out.shape)} | params: {params:,}")

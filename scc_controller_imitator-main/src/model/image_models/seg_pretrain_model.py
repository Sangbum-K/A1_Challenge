"""Encoder + lightweight FPN decoder for SAM seg pretraining.

Encoder = same timm backbone used by LateralStripEncoder (features_only) —
state_dict 가 그대로 downstream LateralStripEncoder 에 load 가능.

Decoder = U-Net style: top feature 부터 ConvT 로 upsample 하며 skip concat.
출력: (B, n_classes, 250, 1280) logits.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class _UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        # skip 와 spatial size 다르면 align (stride rounding 보정)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class SegPretrainModel(nn.Module):
    """efficientnet_b0 등 timm features_only backbone + FPN-lite decoder.

    out_indices=[0, 1, 2, 3, 4] → strides [2, 4, 8, 16, 32] 5 scale.
    decoder: top (32x) → 16x → 8x → 4x → 2x → bilinear up 2x → input 해상도.
    """

    def __init__(self, backbone: str = "efficientnet_b0", n_classes: int = 4, d_mid: int = 64):
        super().__init__()
        self.backbone = timm.create_model(
            backbone, pretrained=True, features_only=True, out_indices=[0, 1, 2, 3, 4]
        )
        chs = self.backbone.feature_info.channels()    # e.g. [16, 24, 40, 112, 320] for b0
        assert len(chs) == 5
        c1, c2, c3, c4, c5 = chs

        # decoder: top first
        self.up4 = _UpBlock(c5, c4, d_mid * 2)        # 32 → 16
        self.up3 = _UpBlock(d_mid * 2, c3, d_mid)     # 16 → 8
        self.up2 = _UpBlock(d_mid, c2, d_mid)         # 8  → 4
        self.up1 = _UpBlock(d_mid, c1, d_mid)         # 4  → 2

        self.head = nn.Sequential(
            nn.Conv2d(d_mid, d_mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(d_mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_mid, n_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x (B, 3, H, W) → logits (B, n_classes, H, W)."""
        H, W = x.shape[-2:]
        f1, f2, f3, f4, f5 = self.backbone(x)
        d4 = self.up4(f5, f4)
        d3 = self.up3(d4, f3)
        d2 = self.up2(d3, f2)
        d1 = self.up1(d2, f1)
        logits = self.head(d1)
        # input 해상도까지 1 step upsample (stride 2)
        if logits.shape[-2:] != (H, W):
            logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        return logits

    def backbone_state_dict(self) -> dict:
        """downstream LateralStripEncoder.backbone 에 그대로 load 가능한 state_dict."""
        return {k: v for k, v in self.state_dict().items() if k.startswith("backbone.")}

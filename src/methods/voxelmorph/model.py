from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VoxelMorph2D(nn.Module):
    def __init__(self, in_channels: int = 2, base_channels: int = 16):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.bottleneck = ConvBlock(base_channels * 2, base_channels * 2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)
        self.dec1 = ConvBlock(base_channels * 3, base_channels)
        self.flow = nn.Conv2d(base_channels, 2, kernel_size=3, padding=1)
        nn.init.normal_(self.flow.weight, mean=0.0, std=1e-5)
        nn.init.zeros_(self.flow.bias)

    def forward(self, fixed: torch.Tensor, moving: torch.Tensor) -> torch.Tensor:
        x = torch.cat([fixed, moving], dim=1)
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.avg_pool2d(enc1, kernel_size=2))
        bottleneck = self.bottleneck(F.avg_pool2d(enc2, kernel_size=2))
        up2 = F.interpolate(bottleneck, size=enc2.shape[-2:], mode="bilinear", align_corners=False)
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))
        up1 = F.interpolate(dec2, size=enc1.shape[-2:], mode="bilinear", align_corners=False)
        dec1 = self.dec1(torch.cat([up1, enc1], dim=1))
        return self.flow(dec1)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "VoxelMorph2D":
        model_cfg = config.get("model", {})
        return cls(
            in_channels=int(model_cfg.get("in_channels", 2)),
            base_channels=int(model_cfg.get("base_channels", 16)),
        )


class VoxelMorph3D(nn.Module):
    def __init__(self, in_channels: int = 2, base_channels: int = 8):
        super().__init__()
        self.enc1 = ConvBlock3D(in_channels, base_channels)
        self.enc2 = ConvBlock3D(base_channels, base_channels * 2)
        self.bottleneck = ConvBlock3D(base_channels * 2, base_channels * 2)
        self.dec2 = ConvBlock3D(base_channels * 4, base_channels * 2)
        self.dec1 = ConvBlock3D(base_channels * 3, base_channels)
        self.flow = nn.Conv3d(base_channels, 3, kernel_size=3, padding=1)
        nn.init.normal_(self.flow.weight, mean=0.0, std=1e-5)
        nn.init.zeros_(self.flow.bias)

    def forward(self, fixed: torch.Tensor, moving: torch.Tensor) -> torch.Tensor:
        x = torch.cat([fixed, moving], dim=1)
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.avg_pool3d(enc1, kernel_size=2))
        bottleneck = self.bottleneck(F.avg_pool3d(enc2, kernel_size=2))
        up2 = F.interpolate(
            bottleneck, size=enc2.shape[-3:], mode="trilinear", align_corners=False
        )
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))
        up1 = F.interpolate(
            dec2, size=enc1.shape[-3:], mode="trilinear", align_corners=False
        )
        dec1 = self.dec1(torch.cat([up1, enc1], dim=1))
        return self.flow(dec1)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "VoxelMorph3D":
        model_cfg = config.get("model", {})
        return cls(
            in_channels=int(model_cfg.get("in_channels", 2)),
            base_channels=int(model_cfg.get("base_channels", 8)),
        )


def build_voxelmorph_model(config: dict[str, Any]) -> nn.Module:
    model_cfg = config.get("model", {})
    dim = int(model_cfg.get("dim", 2))
    if dim == 2:
        return VoxelMorph2D.from_config(config)
    if dim == 3:
        return VoxelMorph3D.from_config(config)
    raise ValueError("VoxelMorph model.dim must be 2 or 3.")

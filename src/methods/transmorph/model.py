from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _positional_encoding_2d(
    height: int, width: int, channels: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    quarter = max(channels // 4, 1)
    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    frequencies = torch.exp(
        torch.linspace(0.0, math.log(10000.0), quarter, device=device, dtype=dtype)
    )
    pieces = [
        torch.sin(yy[..., None] * frequencies),
        torch.cos(yy[..., None] * frequencies),
        torch.sin(xx[..., None] * frequencies),
        torch.cos(xx[..., None] * frequencies),
    ]
    encoding = torch.cat(pieces, dim=-1)
    if encoding.shape[-1] < channels:
        pad = channels - encoding.shape[-1]
        encoding = F.pad(encoding, (0, pad))
    encoding = encoding[..., :channels]
    return encoding.reshape(1, height * width, channels)


def _positional_encoding_3d(
    depth: int,
    height: int,
    width: int,
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    sixth = max(channels // 6, 1)
    z = torch.linspace(-1.0, 1.0, depth, device=device, dtype=dtype)
    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    frequencies = torch.exp(
        torch.linspace(0.0, math.log(10000.0), sixth, device=device, dtype=dtype)
    )
    pieces = [
        torch.sin(zz[..., None] * frequencies),
        torch.cos(zz[..., None] * frequencies),
        torch.sin(yy[..., None] * frequencies),
        torch.cos(yy[..., None] * frequencies),
        torch.sin(xx[..., None] * frequencies),
        torch.cos(xx[..., None] * frequencies),
    ]
    encoding = torch.cat(pieces, dim=-1)
    if encoding.shape[-1] < channels:
        encoding = F.pad(encoding, (0, channels - encoding.shape[-1]))
    encoding = encoding[..., :channels]
    return encoding.reshape(1, depth * height * width, channels)


class TransMorph2D(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        embed_dim: int = 48,
        patch_size: int = 8,
        depth: int = 2,
        num_heads: int = 4,
        decoder_channels: int = 32,
    ):
        super().__init__()
        if patch_size < 2 or patch_size & (patch_size - 1) != 0:
            raise ValueError("patch_size must be a power of two and at least 2.")
        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        stages = int(math.log2(patch_size))
        blocks: list[nn.Module] = []
        in_ch = embed_dim
        for stage in range(stages):
            out_ch = decoder_channels if stage < stages - 1 else decoder_channels // 2
            out_ch = max(out_ch, 8)
            blocks.append(ConvBlock(in_ch, out_ch))
            in_ch = out_ch
        self.decoder = nn.ModuleList(blocks)
        self.flow = nn.Conv2d(in_ch, 2, kernel_size=3, padding=1)
        nn.init.normal_(self.flow.weight, mean=0.0, std=1e-5)
        nn.init.zeros_(self.flow.bias)

    def forward(self, fixed: torch.Tensor, moving: torch.Tensor) -> torch.Tensor:
        image_height, image_width = fixed.shape[-2:]
        x = torch.cat([fixed, moving], dim=1)
        tokens = self.patch_embed(x)
        batch, channels, grid_h, grid_w = tokens.shape
        sequence = tokens.flatten(2).transpose(1, 2)
        sequence = sequence + _positional_encoding_2d(
            grid_h, grid_w, channels, sequence.device, sequence.dtype
        )
        encoded = self.encoder(sequence)
        features = encoded.transpose(1, 2).reshape(batch, channels, grid_h, grid_w)
        decoded = features
        for block in self.decoder:
            decoded = F.interpolate(
                decoded, scale_factor=2, mode="bilinear", align_corners=False
            )
            decoded = block(decoded)
        if decoded.shape[-2:] != (image_height, image_width):
            decoded = F.interpolate(
                decoded,
                size=(image_height, image_width),
                mode="bilinear",
                align_corners=False,
            )
        return self.flow(decoded)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "TransMorph2D":
        model_cfg = config.get("model", {})
        return cls(
            in_channels=int(model_cfg.get("in_channels", 2)),
            embed_dim=int(model_cfg.get("embed_dim", 48)),
            patch_size=int(model_cfg.get("patch_size", 8)),
            depth=int(model_cfg.get("depth", 2)),
            num_heads=int(model_cfg.get("num_heads", 4)),
            decoder_channels=int(model_cfg.get("decoder_channels", 32)),
        )


class TransMorph3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        embed_dim: int = 24,
        patch_size: int = 8,
        depth: int = 1,
        num_heads: int = 4,
        decoder_channels: int = 12,
    ):
        super().__init__()
        if patch_size < 2 or patch_size & (patch_size - 1) != 0:
            raise ValueError("patch_size must be a power of two and at least 2.")
        self.patch_size = patch_size
        self.patch_embed = nn.Conv3d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        stages = int(math.log2(patch_size))
        blocks: list[nn.Module] = []
        in_ch = embed_dim
        for stage in range(stages):
            out_ch = decoder_channels if stage < stages - 1 else decoder_channels // 2
            out_ch = max(out_ch, 6)
            blocks.append(ConvBlock3D(in_ch, out_ch))
            in_ch = out_ch
        self.decoder = nn.ModuleList(blocks)
        self.flow = nn.Conv3d(in_ch, 3, kernel_size=3, padding=1)
        nn.init.normal_(self.flow.weight, mean=0.0, std=1e-5)
        nn.init.zeros_(self.flow.bias)

    def forward(self, fixed: torch.Tensor, moving: torch.Tensor) -> torch.Tensor:
        image_depth, image_height, image_width = fixed.shape[-3:]
        x = torch.cat([fixed, moving], dim=1)
        tokens = self.patch_embed(x)
        batch, channels, grid_d, grid_h, grid_w = tokens.shape
        sequence = tokens.flatten(2).transpose(1, 2)
        sequence = sequence + _positional_encoding_3d(
            grid_d, grid_h, grid_w, channels, sequence.device, sequence.dtype
        )
        encoded = self.encoder(sequence)
        features = encoded.transpose(1, 2).reshape(
            batch, channels, grid_d, grid_h, grid_w
        )
        decoded = features
        for block in self.decoder:
            decoded = F.interpolate(
                decoded, scale_factor=2, mode="trilinear", align_corners=False
            )
            decoded = block(decoded)
        if decoded.shape[-3:] != (image_depth, image_height, image_width):
            decoded = F.interpolate(
                decoded,
                size=(image_depth, image_height, image_width),
                mode="trilinear",
                align_corners=False,
            )
        return self.flow(decoded)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "TransMorph3D":
        model_cfg = config.get("model", {})
        return cls(
            in_channels=int(model_cfg.get("in_channels", 2)),
            embed_dim=int(model_cfg.get("embed_dim", 24)),
            patch_size=int(model_cfg.get("patch_size", 8)),
            depth=int(model_cfg.get("depth", 1)),
            num_heads=int(model_cfg.get("num_heads", 4)),
            decoder_channels=int(model_cfg.get("decoder_channels", 12)),
        )


def build_transmorph_model(config: dict[str, Any]) -> nn.Module:
    model_cfg = config.get("model", {})
    dim = int(model_cfg.get("dim", 2))
    if dim == 2:
        return TransMorph2D.from_config(config)
    if dim == 3:
        return TransMorph3D.from_config(config)
    raise ValueError("TransMorph model.dim must be 2 or 3.")

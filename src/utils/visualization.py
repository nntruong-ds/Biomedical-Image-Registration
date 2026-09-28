from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from src.utils.io import ensure_dir
from src.utils.metrics import normalize_image


def to_uint8(image: np.ndarray) -> np.ndarray:
    view = central_slice(np.asarray(image))
    return np.clip(normalize_image(view) * 255.0, 0, 255).astype(np.uint8)


def central_slice(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3:
        return image[image.shape[0] // 2]
    raise ValueError(f"Expected 2D or 3D image for visualization, got {image.shape}")


def overlay_rgb(fixed: np.ndarray, moving_or_registered: np.ndarray) -> np.ndarray:
    fixed_u8 = to_uint8(fixed)
    moving_u8 = to_uint8(moving_or_registered)
    blue = ((fixed_u8.astype(np.float32) + moving_u8.astype(np.float32)) * 0.5).astype(
        np.uint8
    )
    return np.stack([fixed_u8, moving_u8, blue], axis=-1)


def save_overlay(
    fixed: np.ndarray, moving_or_registered: np.ndarray, path: str | Path
) -> Path:
    out_path = Path(path)
    ensure_dir(out_path.parent)
    Image.fromarray(overlay_rgb(fixed, moving_or_registered)).save(out_path)
    return out_path


def _panel(image: np.ndarray, title: str) -> Image.Image:
    rgb = Image.fromarray(np.repeat(to_uint8(image)[..., None], 3, axis=-1))
    header = Image.new("RGB", (rgb.width, 18), (245, 245, 245))
    draw = ImageDraw.Draw(header)
    draw.text((4, 3), title, fill=(20, 20, 20))
    out = Image.new("RGB", (rgb.width, rgb.height + header.height), (255, 255, 255))
    out.paste(header, (0, 0))
    out.paste(rgb, (0, header.height))
    return out


def save_preview(
    fixed: np.ndarray,
    moving: np.ndarray,
    path: str | Path,
    registered: np.ndarray | None = None,
) -> Path:
    panels = [_panel(fixed, "fixed"), _panel(moving, "moving")]
    if registered is not None:
        panels.append(_panel(registered, "registered"))
    panels.append(Image.fromarray(overlay_rgb(fixed, registered if registered is not None else moving)))

    gap = 6
    width = sum(panel.width for panel in panels) + gap * (len(panels) - 1)
    height = max(panel.height for panel in panels)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    x_offset = 0
    for panel in panels:
        canvas.paste(panel, (x_offset, 0))
        x_offset += panel.width + gap

    out_path = Path(path)
    ensure_dir(out_path.parent)
    canvas.save(out_path)
    return out_path

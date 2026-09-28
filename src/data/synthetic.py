from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter

from src.utils.io import ensure_dir, save_array, save_json, write_image
from src.utils.metrics import normalize_image
from src.utils.visualization import save_preview
from src.utils.warp import AffineParams, warp_affine, warp_image_with_displacement


@dataclass(frozen=True)
class SyntheticPairMetadata:
    index: int
    fixed: str
    moving: str
    gt_affine_params: dict[str, float]
    gt_displacement: str
    preview: str


def _ellipse(
    yy: np.ndarray,
    xx: np.ndarray,
    center_y: float,
    center_x: float,
    radius_y: float,
    radius_x: float,
    angle: float,
) -> np.ndarray:
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    y = yy - center_y
    x = xx - center_x
    x_rot = cos_a * x + sin_a * y
    y_rot = -sin_a * x + cos_a * y
    return (x_rot / radius_x) ** 2 + (y_rot / radius_y) ** 2


def generate_medical_like_image(size: int, rng: np.random.Generator) -> np.ndarray:
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, size, dtype=np.float32),
        np.linspace(-1.0, 1.0, size, dtype=np.float32),
        indexing="ij",
    )
    image = np.zeros((size, size), dtype=np.float32)

    body = _ellipse(yy, xx, 0.0, 0.0, 0.78, 0.68, rng.uniform(-0.25, 0.25))
    image += 0.18 * np.exp(-2.0 * body)
    image[body <= 1.0] += 0.35

    for _ in range(rng.integers(5, 9)):
        center_y = float(rng.uniform(-0.42, 0.42))
        center_x = float(rng.uniform(-0.42, 0.42))
        radius_y = float(rng.uniform(0.08, 0.24))
        radius_x = float(rng.uniform(0.07, 0.22))
        angle = float(rng.uniform(0.0, math.pi))
        intensity = float(rng.uniform(0.18, 0.85))
        dist = _ellipse(yy, xx, center_y, center_x, radius_y, radius_x, angle)
        blob = np.exp(-2.2 * dist)
        image += intensity * blob

    for _ in range(rng.integers(2, 5)):
        center_y = float(rng.uniform(-0.35, 0.35))
        center_x = float(rng.uniform(-0.35, 0.35))
        radius_y = float(rng.uniform(0.13, 0.28))
        radius_x = float(rng.uniform(0.13, 0.28))
        angle = float(rng.uniform(0.0, math.pi))
        outer = _ellipse(yy, xx, center_y, center_x, radius_y, radius_x, angle)
        inner = _ellipse(yy, xx, center_y, center_x, radius_y * 0.62, radius_x * 0.62, angle)
        ring = np.logical_and(outer <= 1.0, inner >= 1.0)
        image[ring] += float(rng.uniform(0.08, 0.28))

    low_frequency = gaussian_filter(rng.normal(0.0, 1.0, (size, size)), sigma=size / 14.0)
    low_frequency = normalize_image(low_frequency) - 0.5
    fine_texture = gaussian_filter(rng.normal(0.0, 1.0, (size, size)), sigma=1.2)
    fine_texture = normalize_image(fine_texture) - 0.5
    image += 0.12 * low_frequency + 0.04 * fine_texture
    image *= body <= 1.18
    image = gaussian_filter(image, sigma=0.75)
    return normalize_image(image)


def random_smooth_displacement(
    size: int, rng: np.random.Generator, max_magnitude: float
) -> np.ndarray:
    field = rng.normal(0.0, 1.0, (size, size, 2)).astype(np.float32)
    field[..., 0] = gaussian_filter(field[..., 0], sigma=size / 7.0)
    field[..., 1] = gaussian_filter(field[..., 1], sigma=size / 7.0)
    magnitude = np.sqrt(np.sum(field**2, axis=-1))
    max_observed = float(magnitude.max())
    if max_observed > 1e-8:
        field *= float(rng.uniform(0.45, 1.0) * max_magnitude / max_observed)
    return field.astype(np.float32)


def random_affine_params(size: int, rng: np.random.Generator) -> AffineParams:
    shift = max(3.0, size * 0.07)
    return AffineParams(
        angle_deg=float(rng.uniform(-11.0, 11.0)),
        tx=float(rng.uniform(-shift, shift)),
        ty=float(rng.uniform(-shift, shift)),
        scale_x=float(rng.uniform(0.95, 1.06)),
        scale_y=float(rng.uniform(0.95, 1.06)),
        shear=float(rng.uniform(-0.055, 0.055)),
    )


def generate_pair(index: int, size: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, AffineParams]:
    fixed = generate_medical_like_image(size=size, rng=rng)
    affine = random_affine_params(size=size, rng=rng)
    affine_moved = warp_affine(fixed, affine, output_shape=fixed.shape)
    displacement = random_smooth_displacement(
        size=size, rng=rng, max_magnitude=max(2.0, size * 0.045)
    )
    moving = warp_image_with_displacement(affine_moved, displacement)
    bias = gaussian_filter(rng.normal(0.0, 1.0, fixed.shape), sigma=size / 3.0)
    bias = 1.0 + 0.12 * (normalize_image(bias) - 0.5)
    moving = moving * bias + rng.normal(0.0, 0.015, fixed.shape).astype(np.float32)
    moving = normalize_image(np.clip(moving, 0.0, None))
    return fixed, moving, displacement, affine


def generate_dataset(
    out_dir: str | Path, num_pairs: int, size: int, seed: int = 7
) -> list[SyntheticPairMetadata]:
    if num_pairs < 1:
        raise ValueError("--num_pairs must be at least 1")
    if size < 32:
        raise ValueError("--size must be at least 32 pixels")

    out_path = ensure_dir(out_dir)
    rng = np.random.default_rng(seed)
    metadata: list[SyntheticPairMetadata] = []

    for index in range(num_pairs):
        fixed, moving, displacement, affine = generate_pair(index=index, size=size, rng=rng)
        stem = f"{index:03d}"
        fixed_path = out_path / f"fixed_{stem}.nii.gz"
        moving_path = out_path / f"moving_{stem}.nii.gz"
        field_path = out_path / f"gt_displacement_{stem}.npy"
        params_path = out_path / f"gt_affine_{stem}.json"
        preview_path = out_path / f"preview_{stem}.png"

        write_image(fixed, fixed_path)
        write_image(moving, moving_path)
        save_array(displacement, field_path)
        affine_dict = asdict(affine)
        save_json(
            {
                "description": (
                    "Affine parameters used while synthesizing the moving image. "
                    "The displacement field is stored as (dy, dx) pixel offsets."
                ),
                "params": affine_dict,
            },
            params_path,
        )
        save_preview(fixed, moving, preview_path)
        metadata.append(
            SyntheticPairMetadata(
                index=index,
                fixed=str(fixed_path),
                moving=str(moving_path),
                gt_affine_params=affine_dict,
                gt_displacement=str(field_path),
                preview=str(preview_path),
            )
        )

    save_json(
        {
            "num_pairs": num_pairs,
            "size": size,
            "seed": seed,
            "pairs": [asdict(item) for item in metadata],
        },
        out_path / "dataset.json",
    )
    return metadata


from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import map_coordinates


@dataclass(frozen=True)
class AffineParams:
    angle_deg: float
    tx: float
    ty: float
    scale_x: float = 1.0
    scale_y: float = 1.0
    shear: float = 0.0

    def as_vector(self) -> np.ndarray:
        return np.array(
            [self.angle_deg, self.tx, self.ty, self.scale_x, self.scale_y, self.shear],
            dtype=np.float32,
        )


def coordinate_grid(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape
    yy, xx = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )
    return yy, xx


def warp_image_with_displacement(
    image: np.ndarray, displacement: np.ndarray, cval: float = 0.0
) -> np.ndarray:
    if displacement.shape != image.shape + (2,):
        raise ValueError(
            "Displacement field must have shape (H, W, 2) matching the image; "
            f"got image {image.shape}, field {displacement.shape}"
        )
    yy, xx = coordinate_grid(image.shape)
    src_y = yy + displacement[..., 0]
    src_x = xx + displacement[..., 1]
    warped = map_coordinates(
        image,
        [src_y, src_x],
        order=1,
        mode="constant",
        cval=cval,
        prefilter=False,
    )
    return warped.astype(np.float32)


def affine_matrix(params: AffineParams) -> np.ndarray:
    angle = math.radians(float(params.angle_deg))
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    scale_shear = np.array(
        [[params.scale_x, params.shear], [0.0, params.scale_y]], dtype=np.float32
    )
    return rotation @ scale_shear


def warp_affine(
    image: np.ndarray,
    params: AffineParams,
    output_shape: tuple[int, int] | None = None,
    cval: float = 0.0,
) -> np.ndarray:
    if output_shape is None:
        output_shape = image.shape
    out_h, out_w = output_shape
    in_h, in_w = image.shape
    yy, xx = coordinate_grid(output_shape)
    out_center = np.array([(out_w - 1) * 0.5, (out_h - 1) * 0.5], dtype=np.float32)
    in_center = np.array([(in_w - 1) * 0.5, (in_h - 1) * 0.5], dtype=np.float32)
    coords_xy = np.stack([xx - out_center[0], yy - out_center[1]], axis=0).reshape(2, -1)
    matrix = affine_matrix(params)
    src_xy = matrix @ coords_xy
    src_xy[0] += in_center[0] + float(params.tx)
    src_xy[1] += in_center[1] + float(params.ty)
    src_x = src_xy[0].reshape(output_shape)
    src_y = src_xy[1].reshape(output_shape)
    warped = map_coordinates(
        image,
        [src_y, src_x],
        order=1,
        mode="constant",
        cval=cval,
        prefilter=False,
    )
    return warped.astype(np.float32)


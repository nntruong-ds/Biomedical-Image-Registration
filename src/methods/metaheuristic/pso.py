from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import map_coordinates

from src.utils.io import ensure_dir, read_image, read_sitk_image, save_json, write_image
from src.utils.metrics import before_after_metrics, mse, ncc, normalize_image
from src.utils.visualization import save_overlay
from src.utils.warp import AffineParams, warp_affine


@dataclass(frozen=True)
class PSOConfig:
    particles: int = 24
    iterations: int = 40
    inertia: float = 0.72
    cognitive: float = 1.45
    social: float = 1.45
    metric: str = "ncc"
    transform: str = "affine"
    seed: int = 7


@dataclass(frozen=True)
class Rigid3DParams:
    rx_deg: float
    ry_deg: float
    rz_deg: float
    tx: float
    ty: float
    tz: float

    def as_vector(self) -> np.ndarray:
        return np.array(
            [self.rx_deg, self.ry_deg, self.rz_deg, self.tx, self.ty, self.tz],
            dtype=np.float32,
        )


@dataclass(frozen=True)
class Affine3DParams:
    rx_deg: float
    ry_deg: float
    rz_deg: float
    tx: float
    ty: float
    tz: float
    scale_x: float
    scale_y: float
    scale_z: float
    shear_xy: float
    shear_xz: float
    shear_yz: float

    def as_vector(self) -> np.ndarray:
        return np.array(
            [
                self.rx_deg,
                self.ry_deg,
                self.rz_deg,
                self.tx,
                self.ty,
                self.tz,
                self.scale_x,
                self.scale_y,
                self.scale_z,
                self.shear_xy,
                self.shear_xz,
                self.shear_yz,
            ],
            dtype=np.float32,
        )


def _bounds(shape: tuple[int, ...], transform: str) -> tuple[np.ndarray, np.ndarray]:
    if len(shape) == 2:
        height, width = shape
        shift = max(4.0, 0.14 * max(height, width))
        if transform == "rigid":
            lo = np.array([-18.0, -shift, -shift], dtype=np.float32)
            hi = np.array([18.0, shift, shift], dtype=np.float32)
        elif transform == "affine":
            lo = np.array([-18.0, -shift, -shift, 0.88, 0.88, -0.12], dtype=np.float32)
            hi = np.array([18.0, shift, shift, 1.12, 1.12, 0.12], dtype=np.float32)
        else:
            raise ValueError("PSO transform must be 'rigid' or 'affine'.")
        return lo, hi

    if len(shape) == 3:
        depth, height, width = shape
        shift = max(3.0, 0.08 * max(depth, height, width))
        if transform == "rigid":
            lo = np.array([-8.0, -8.0, -8.0, -shift, -shift, -shift], dtype=np.float32)
            hi = np.array([8.0, 8.0, 8.0, shift, shift, shift], dtype=np.float32)
            return lo, hi
        if transform == "affine":
            lo = np.array(
                [
                    -8.0,
                    -8.0,
                    -8.0,
                    -shift,
                    -shift,
                    -shift,
                    0.90,
                    0.90,
                    0.90,
                    -0.08,
                    -0.08,
                    -0.08,
                ],
                dtype=np.float32,
            )
            hi = np.array(
                [
                    8.0,
                    8.0,
                    8.0,
                    shift,
                    shift,
                    shift,
                    1.10,
                    1.10,
                    1.10,
                    0.08,
                    0.08,
                    0.08,
                ],
                dtype=np.float32,
            )
            return lo, hi
        raise ValueError("3D PSO transform must be 'rigid' or 'affine'.")

    raise ValueError(f"PSO expects a 2D or 3D image, got shape {shape}.")


def _vector_to_params(
    vector: np.ndarray, transform: str, ndim: int
) -> AffineParams | Rigid3DParams | Affine3DParams:
    if ndim == 3:
        if transform == "rigid":
            return Rigid3DParams(
                rx_deg=float(vector[0]),
                ry_deg=float(vector[1]),
                rz_deg=float(vector[2]),
                tx=float(vector[3]),
                ty=float(vector[4]),
                tz=float(vector[5]),
            )
        if transform == "affine":
            return Affine3DParams(
                rx_deg=float(vector[0]),
                ry_deg=float(vector[1]),
                rz_deg=float(vector[2]),
                tx=float(vector[3]),
                ty=float(vector[4]),
                tz=float(vector[5]),
                scale_x=float(vector[6]),
                scale_y=float(vector[7]),
                scale_z=float(vector[8]),
                shear_xy=float(vector[9]),
                shear_xz=float(vector[10]),
                shear_yz=float(vector[11]),
            )
        raise ValueError("3D PSO transform must be 'rigid' or 'affine'.")
    if transform == "rigid":
        return AffineParams(
            angle_deg=float(vector[0]),
            tx=float(vector[1]),
            ty=float(vector[2]),
        )
    return AffineParams(
        angle_deg=float(vector[0]),
        tx=float(vector[1]),
        ty=float(vector[2]),
        scale_x=float(vector[3]),
        scale_y=float(vector[4]),
        shear=float(vector[5]),
    )


def _rotation_matrix_3d(params: Rigid3DParams) -> np.ndarray:
    rx = np.deg2rad(float(params.rx_deg))
    ry = np.deg2rad(float(params.ry_deg))
    rz = np.deg2rad(float(params.rz_deg))
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float32)
    rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
    rot_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return rot_z @ rot_y @ rot_x


def _rotation_matrix_from_angles(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    return _rotation_matrix_3d(
        Rigid3DParams(
            rx_deg=float(rx_deg),
            ry_deg=float(ry_deg),
            rz_deg=float(rz_deg),
            tx=0.0,
            ty=0.0,
            tz=0.0,
        )
    )


def _affine_matrix_3d(params: Affine3DParams) -> np.ndarray:
    rotation = _rotation_matrix_from_angles(params.rx_deg, params.ry_deg, params.rz_deg)
    scale_shear = np.array(
        [
            [params.scale_x, params.shear_xy, params.shear_xz],
            [0.0, params.scale_y, params.shear_yz],
            [0.0, 0.0, params.scale_z],
        ],
        dtype=np.float32,
    )
    return rotation @ scale_shear


def _matrix_3d(params: Rigid3DParams | Affine3DParams) -> np.ndarray:
    if isinstance(params, Rigid3DParams):
        return _rotation_matrix_3d(params)
    return _affine_matrix_3d(params)


def _coordinate_grid_3d(shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    depth, height, width = shape
    zz, yy, xx = np.meshgrid(
        np.arange(depth, dtype=np.float32),
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )
    return zz, yy, xx


def _warp_rigid_3d(
    image: np.ndarray,
    params: Rigid3DParams,
    output_shape: tuple[int, int, int] | None = None,
    cval: float = 0.0,
) -> np.ndarray:
    return _warp_transform_3d(image, params, output_shape=output_shape, cval=cval)


def _warp_affine_3d(
    image: np.ndarray,
    params: Affine3DParams,
    output_shape: tuple[int, int, int] | None = None,
    cval: float = 0.0,
) -> np.ndarray:
    return _warp_transform_3d(image, params, output_shape=output_shape, cval=cval)


def _warp_transform_3d(
    image: np.ndarray,
    params: Rigid3DParams | Affine3DParams,
    output_shape: tuple[int, int, int] | None = None,
    cval: float = 0.0,
) -> np.ndarray:
    if image.ndim != 3:
        raise ValueError(f"3D warp expects a 3D image, got shape {image.shape}.")
    if output_shape is None:
        output_shape = image.shape
    out_d, out_h, out_w = output_shape
    in_d, in_h, in_w = image.shape
    zz, yy, xx = _coordinate_grid_3d(output_shape)
    out_center = np.array(
        [(out_w - 1) * 0.5, (out_h - 1) * 0.5, (out_d - 1) * 0.5],
        dtype=np.float32,
    )
    in_center = np.array(
        [(in_w - 1) * 0.5, (in_h - 1) * 0.5, (in_d - 1) * 0.5],
        dtype=np.float32,
    )
    coords_xyz = np.stack(
        [xx - out_center[0], yy - out_center[1], zz - out_center[2]], axis=0
    ).reshape(3, -1)
    src_xyz = _matrix_3d(params) @ coords_xyz
    src_xyz[0] += in_center[0] + float(params.tx)
    src_xyz[1] += in_center[1] + float(params.ty)
    src_xyz[2] += in_center[2] + float(params.tz)
    src_x = src_xyz[0].reshape(output_shape)
    src_y = src_xyz[1].reshape(output_shape)
    src_z = src_xyz[2].reshape(output_shape)
    warped = map_coordinates(
        image,
        [src_z, src_y, src_x],
        order=1,
        mode="constant",
        cval=cval,
        prefilter=False,
    )
    return warped.astype(np.float32)


def _params_to_json(params: AffineParams | Rigid3DParams | Affine3DParams) -> dict[str, Any]:
    if isinstance(params, Rigid3DParams):
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, :3] = _rotation_matrix_3d(params)
        matrix[:3, 3] = [params.tx, params.ty, params.tz]
        return {
            "description": (
                "Transform maps fixed output voxel coordinates to moving image sample "
                "coordinates. Translation is in voxels along x/y/z."
            ),
            "params": asdict(params),
            "matrix_4x4": matrix.tolist(),
        }
    if isinstance(params, Affine3DParams):
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, :3] = _affine_matrix_3d(params)
        matrix[:3, 3] = [params.tx, params.ty, params.tz]
        return {
            "description": (
                "Affine transform maps fixed centered output voxel coordinates to "
                "moving image sample coordinates. Translation is in voxels along x/y/z."
            ),
            "params": asdict(params),
            "matrix_4x4": matrix.tolist(),
        }

    matrix = np.eye(3, dtype=np.float32)
    from src.utils.warp import affine_matrix

    matrix[:2, :2] = affine_matrix(params)
    matrix[0, 2] = params.tx
    matrix[1, 2] = params.ty
    return {
        "description": (
            "Transform maps fixed output pixel coordinates to moving image sample "
            "coordinates. Translation is in pixels."
        ),
        "params": asdict(params),
        "matrix_3x3": matrix.tolist(),
    }


def _score(
    fixed: np.ndarray,
    moving: np.ndarray,
    vector: np.ndarray,
    transform: str,
    metric: str,
) -> float:
    params = _vector_to_params(vector, transform, fixed.ndim)
    if fixed.ndim == 3:
        if not isinstance(params, (Rigid3DParams, Affine3DParams)):
            raise TypeError("3D PSO expected 3D transform parameters.")
        warped = _warp_transform_3d(moving, params, output_shape=fixed.shape)
    else:
        warped = warp_affine(moving, params, output_shape=fixed.shape)
    if metric == "mse":
        return mse(fixed, warped)
    if metric == "ncc":
        return 1.0 - ncc(fixed, warped)
    raise ValueError("PSO metric must be 'ncc' or 'mse'.")


def _identity_vector(ndim: int, transform: str) -> np.ndarray:
    if ndim == 3:
        if transform == "rigid":
            return np.zeros(6, dtype=np.float32)
        if transform == "affine":
            return np.array(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                dtype=np.float32,
            )
        raise ValueError("3D PSO transform must be 'rigid' or 'affine'.")
    if transform == "affine":
        return np.array([0.0, 0.0, 0.0, 1.0, 1.0, 0.0], dtype=np.float32)
    return np.array([0.0, 0.0, 0.0], dtype=np.float32)


def run_pso(
    fixed_path: str | Path,
    moving_path: str | Path,
    out_dir: str | Path,
    config: PSOConfig | None = None,
) -> dict[str, Any]:
    cfg = config or PSOConfig()
    if cfg.particles < 2:
        raise ValueError("PSO requires at least 2 particles.")
    if cfg.iterations < 1:
        raise ValueError("PSO requires at least 1 iteration.")

    start = time.time()
    out = ensure_dir(out_dir)
    fixed = normalize_image(read_image(fixed_path))
    moving = normalize_image(read_image(moving_path))
    if fixed.shape != moving.shape:
        raise ValueError(
            f"PSO expects fixed and moving images with equal shape, got {fixed.shape} and {moving.shape}."
        )

    lo, hi = _bounds(fixed.shape, cfg.transform)
    span = hi - lo
    rng = np.random.default_rng(cfg.seed)
    positions = rng.uniform(lo, hi, size=(cfg.particles, lo.size)).astype(np.float32)
    velocities = rng.uniform(-0.05 * span, 0.05 * span, size=positions.shape).astype(np.float32)

    identity = _identity_vector(fixed.ndim, cfg.transform)
    positions[0] = identity
    velocities[0] = 0.0

    personal_best = positions.copy()
    personal_scores = np.full(cfg.particles, np.inf, dtype=np.float32)
    global_best = positions[0].copy()
    global_score = float("inf")
    history: list[dict[str, float | int]] = []

    for iteration in range(cfg.iterations):
        for idx in range(cfg.particles):
            score = _score(fixed, moving, positions[idx], cfg.transform, cfg.metric)
            if score < personal_scores[idx]:
                personal_scores[idx] = score
                personal_best[idx] = positions[idx].copy()
            if score < global_score:
                global_score = float(score)
                global_best = positions[idx].copy()

        history.append({"iteration": iteration, "best_score": float(global_score)})

        r1 = rng.random(size=positions.shape, dtype=np.float32)
        r2 = rng.random(size=positions.shape, dtype=np.float32)
        velocities = (
            cfg.inertia * velocities
            + cfg.cognitive * r1 * (personal_best - positions)
            + cfg.social * r2 * (global_best[None, :] - positions)
        )
        positions = np.clip(positions + velocities, lo, hi)

    best_params = _vector_to_params(global_best, cfg.transform, fixed.ndim)
    if fixed.ndim == 3:
        if not isinstance(best_params, (Rigid3DParams, Affine3DParams)):
            raise TypeError("3D PSO expected 3D transform parameters.")
        registered = normalize_image(
            _warp_transform_3d(moving, best_params, output_shape=fixed.shape)
        )
    else:
        registered = normalize_image(
            warp_affine(moving, best_params, output_shape=fixed.shape)
        )
    metrics = before_after_metrics(fixed, moving, registered)
    reference = read_sitk_image(fixed_path)
    write_image(registered, out / "registered.nii.gz", reference=reference)
    save_json(_params_to_json(best_params), out / "transform_params.json")
    save_overlay(fixed, registered, out / "overlay.png")
    save_json(metrics, out / "metrics.json")

    log = {
        "method": "pso",
        "config": asdict(cfg),
        "elapsed_seconds": time.time() - start,
        "best_score": float(global_score),
        "best_params": asdict(best_params),
        "history": history,
        "fixed": str(fixed_path),
        "moving": str(moving_path),
        "outputs": {
            "registered": str(out / "registered.nii.gz"),
            "transform_params": str(out / "transform_params.json"),
            "overlay": str(out / "overlay.png"),
            "metrics": str(out / "metrics.json"),
        },
        "metrics": metrics,
    }
    save_json(log, out / "log.json")
    return log

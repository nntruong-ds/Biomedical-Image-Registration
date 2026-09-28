from __future__ import annotations

from typing import Any

import numpy as np


def normalize_image(image: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    values = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("Image contains no finite values.")
    min_value = float(values[finite].min())
    max_value = float(values[finite].max())
    if max_value - min_value < eps:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - min_value) / (max_value - min_value)).astype(np.float32)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    lhs = np.asarray(a, dtype=np.float32)
    rhs = np.asarray(b, dtype=np.float32)
    if lhs.shape != rhs.shape:
        raise ValueError(f"MSE requires equal shapes, got {lhs.shape} and {rhs.shape}")
    return float(np.mean((lhs - rhs) ** 2))


def ncc(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    lhs = np.asarray(a, dtype=np.float32)
    rhs = np.asarray(b, dtype=np.float32)
    if lhs.shape != rhs.shape:
        raise ValueError(f"NCC requires equal shapes, got {lhs.shape} and {rhs.shape}")
    lhs = lhs - float(lhs.mean())
    rhs = rhs - float(rhs.mean())
    denom = float(np.sqrt(np.sum(lhs * lhs) * np.sum(rhs * rhs)))
    if denom < eps:
        return 0.0
    return float(np.sum(lhs * rhs) / denom)


def before_after_metrics(
    fixed: np.ndarray, moving: np.ndarray, registered: np.ndarray
) -> dict[str, Any]:
    return {
        "before": {
            "mse": mse(fixed, moving),
            "ncc": ncc(fixed, moving),
        },
        "after": {
            "mse": mse(fixed, registered),
            "ncc": ncc(fixed, registered),
        },
    }


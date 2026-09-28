from __future__ import annotations

import numpy as np

from src.utils.metrics import mse, ncc, normalize_image


def test_metrics_identity() -> None:
    image = np.arange(16, dtype=np.float32).reshape(4, 4)
    assert mse(image, image) == 0.0
    assert ncc(image, image) > 0.999


def test_normalize_constant_image() -> None:
    image = np.ones((8, 8), dtype=np.float32)
    normalized = normalize_image(image)
    assert normalized.shape == image.shape
    assert float(normalized.max()) == 0.0


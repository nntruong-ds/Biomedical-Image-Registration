from __future__ import annotations

import numpy as np

from src.methods.metaheuristic.pso import (
    Affine3DParams,
    _identity_vector,
    _vector_to_params,
    _warp_affine_3d,
)


def test_3d_affine_identity_vector() -> None:
    vector = _identity_vector(3, "affine")
    params = _vector_to_params(vector, "affine", 3)

    assert isinstance(params, Affine3DParams)
    assert params.scale_x == 1.0
    assert params.scale_y == 1.0
    assert params.scale_z == 1.0
    assert params.shear_xy == 0.0
    assert params.shear_xz == 0.0
    assert params.shear_yz == 0.0


def test_3d_affine_identity_warp() -> None:
    image = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
    params = Affine3DParams(
        rx_deg=0.0,
        ry_deg=0.0,
        rz_deg=0.0,
        tx=0.0,
        ty=0.0,
        tz=0.0,
        scale_x=1.0,
        scale_y=1.0,
        scale_z=1.0,
        shear_xy=0.0,
        shear_xz=0.0,
        shear_yz=0.0,
    )

    warped = _warp_affine_3d(image, params)

    assert np.allclose(warped, image)

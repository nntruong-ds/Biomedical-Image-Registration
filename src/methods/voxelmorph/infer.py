from __future__ import annotations

from pathlib import Path
from typing import Any

from src.methods.deep_common import run_deep_inference
from src.methods.voxelmorph.model import build_voxelmorph_model


def _builder(config: dict[str, Any]):
    return build_voxelmorph_model(config)


def run_voxelmorph(
    fixed_path: str | Path,
    moving_path: str | Path,
    checkpoint_path: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    return run_deep_inference(
        model_builder=_builder,
        model_name="voxelmorph",
        fixed_path=fixed_path,
        moving_path=moving_path,
        checkpoint_path=checkpoint_path,
        out_dir=out_dir,
    )

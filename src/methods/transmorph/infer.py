from __future__ import annotations

from pathlib import Path
from typing import Any

from src.methods.deep_common import run_deep_inference
from src.methods.transmorph.model import build_transmorph_model


def _builder(config: dict[str, Any]):
    return build_transmorph_model(config)


def run_transmorph(
    fixed_path: str | Path,
    moving_path: str | Path,
    checkpoint_path: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    return run_deep_inference(
        model_builder=_builder,
        model_name="transmorph",
        fixed_path=fixed_path,
        moving_path=moving_path,
        checkpoint_path=checkpoint_path,
        out_dir=out_dir,
    )

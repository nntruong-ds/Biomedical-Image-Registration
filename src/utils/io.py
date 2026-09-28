from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _sitk():
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise RuntimeError(
            "SimpleITK is required for NIfTI I/O and classical registration. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc
    return sitk


def read_sitk_image(path: str | Path):
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Image not found: {file_path}")
    sitk = _sitk()
    return sitk.ReadImage(str(file_path))


def read_image(path: str | Path) -> np.ndarray:
    sitk = _sitk()
    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image).astype(np.float32)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim not in (2, 3):
        raise ValueError(
            f"Expected a 2D or 3D image at {path}, got array shape {tuple(array.shape)}"
        )
    return array


def write_image(
    array: np.ndarray, path: str | Path, reference: Any | None = None
) -> Path:
    sitk = _sitk()
    out_path = Path(path)
    ensure_dir(out_path.parent)
    image = sitk.GetImageFromArray(np.asarray(array, dtype=np.float32))
    if reference is not None and list(reference.GetSize()) == list(image.GetSize()):
        image.CopyInformation(reference)
    sitk.WriteImage(image, str(out_path))
    return out_path


def write_vector_field(
    field: np.ndarray, path: str | Path, reference: Any | None = None
) -> Path:
    if field.ndim not in (3, 4) or field.shape[-1] not in (2, 3):
        raise ValueError(f"Expected field shape (H, W, 2) or (D, H, W, 3), got {field.shape}")
    sitk = _sitk()
    out_path = Path(path)
    ensure_dir(out_path.parent)
    image = sitk.GetImageFromArray(np.asarray(field, dtype=np.float32), isVector=True)
    if reference is not None and list(reference.GetSize()) == list(image.GetSize()):
        image.CopyInformation(reference)
    sitk.WriteImage(image, str(out_path))
    return out_path


def save_json(data: dict[str, Any], path: str | Path) -> Path:
    out_path = Path(path)
    ensure_dir(out_path.parent)
    with out_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)
        file.write("\n")
    return out_path


def save_array(array: np.ndarray, path: str | Path) -> Path:
    out_path = Path(path)
    ensure_dir(out_path.parent)
    np.save(out_path, array)
    return out_path

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.io import (
    ensure_dir,
    read_image,
    read_sitk_image,
    save_array,
    save_json,
    write_image,
    write_vector_field,
)
from src.utils.metrics import before_after_metrics, normalize_image
from src.utils.visualization import save_overlay


def _sitk():
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise RuntimeError(
            "SimpleITK is required for classical Demons registration. "
            "Install it with: pip install -r requirements.txt"
        ) from exc
    return sitk


def _normalized_sitk(path: str | Path):
    sitk = _sitk()
    image = sitk.Cast(sitk.ReadImage(str(path)), sitk.sitkFloat32)
    return sitk.RescaleIntensity(image, 0.0, 1.0)


def register_demons(
    fixed_path: str | Path,
    moving_path: str | Path,
    out_dir: str | Path,
    iterations: int = 50,
    smoothing_sigma: float = 1.3,
    diffeomorphic: bool = True,
) -> dict[str, Any]:
    if iterations < 1:
        raise ValueError("Classical registration iterations must be >= 1")

    sitk = _sitk()
    start = time.time()
    out = ensure_dir(out_dir)
    fixed_sitk = _normalized_sitk(fixed_path)
    moving_sitk = _normalized_sitk(moving_path)
    reference = read_sitk_image(fixed_path)

    if fixed_sitk.GetSize() != moving_sitk.GetSize():
        raise ValueError(
            "Classical Demons baseline expects fixed and moving images with the "
            f"same size, got {fixed_sitk.GetSize()} and {moving_sitk.GetSize()}."
        )

    if diffeomorphic:
        registration = sitk.DiffeomorphicDemonsRegistrationFilter()
        backend = "SimpleITK DiffeomorphicDemons"
    else:
        registration = sitk.DemonsRegistrationFilter()
        backend = "SimpleITK Demons"
    registration.SetNumberOfIterations(int(iterations))
    registration.SetStandardDeviations(float(smoothing_sigma))

    displacement = registration.Execute(fixed_sitk, moving_sitk)
    field = sitk.GetArrayFromImage(displacement).astype(np.float32)
    transform = sitk.DisplacementFieldTransform(sitk.Image(displacement))
    registered_sitk = sitk.Resample(
        moving_sitk,
        fixed_sitk,
        transform,
        sitk.sitkLinear,
        0.0,
        sitk.sitkFloat32,
    )

    registered = sitk.GetArrayFromImage(registered_sitk).astype(np.float32)
    if field.ndim in (3, 4) and field.shape[-1] in (2, 3):
        field_to_save = field
    else:
        field_to_save = np.asarray(field, dtype=np.float32)

    fixed = normalize_image(read_image(fixed_path))
    moving = normalize_image(read_image(moving_path))
    registered = normalize_image(registered)
    metrics = before_after_metrics(fixed, moving, registered)

    write_image(registered, out / "registered.nii.gz", reference=reference)
    save_array(field_to_save, out / "deformation_field.npy")
    write_vector_field(field_to_save, out / "deformation_field.nii.gz", reference=reference)
    save_overlay(fixed, registered, out / "overlay.png")
    save_json(metrics, out / "metrics.json")

    log = {
        "method": "classical",
        "backend": backend,
        "iterations": int(iterations),
        "smoothing_sigma": float(smoothing_sigma),
        "elapsed_seconds": time.time() - start,
        "final_metric": float(registration.GetMetric()),
        "fixed": str(fixed_path),
        "moving": str(moving_path),
        "outputs": {
            "registered": str(out / "registered.nii.gz"),
            "deformation_field_npy": str(out / "deformation_field.npy"),
            "deformation_field_nifti": str(out / "deformation_field.nii.gz"),
            "overlay": str(out / "overlay.png"),
            "metrics": str(out / "metrics.json"),
        },
        "metrics": metrics,
    }
    save_json(log, out / "log.json")
    return log


def register_antspyx_syn(
    fixed_path: str | Path,
    moving_path: str | Path,
    out_dir: str | Path,
    iterations: int = 50,
) -> dict[str, Any]:
    try:
        import ants
    except ImportError as exc:
        raise RuntimeError(
            "ANTsPyX is not installed. Install antspyx or use the default "
            "SimpleITK Demons backend."
        ) from exc

    start = time.time()
    out = ensure_dir(out_dir)
    fixed = ants.image_read(str(fixed_path))
    moving = ants.image_read(str(moving_path))
    registration = ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform="SyN",
        reg_iterations=(iterations, max(iterations // 2, 1), max(iterations // 4, 1)),
        outprefix=str(out / "ants_"),
    )
    registered_ants = registration["warpedmovout"]
    ants.image_write(registered_ants, str(out / "registered.nii.gz"))

    copied_transforms: list[str] = []
    for transform_path in registration.get("fwdtransforms", []):
        src = Path(transform_path)
        if src.exists():
            dst = out / src.name
            if src.resolve() != dst.resolve():
                shutil.copy2(src, dst)
            copied_transforms.append(str(dst))

    fixed_np = normalize_image(read_image(fixed_path))
    moving_np = normalize_image(read_image(moving_path))
    registered_np = normalize_image(read_image(out / "registered.nii.gz"))
    metrics = before_after_metrics(fixed_np, moving_np, registered_np)
    save_overlay(fixed_np, registered_np, out / "overlay.png")
    save_json(metrics, out / "metrics.json")
    save_json({"fwdtransforms": copied_transforms}, out / "transform_params.json")

    log = {
        "method": "classical",
        "backend": "ANTsPyX SyN",
        "iterations": int(iterations),
        "elapsed_seconds": time.time() - start,
        "fixed": str(fixed_path),
        "moving": str(moving_path),
        "outputs": {
            "registered": str(out / "registered.nii.gz"),
            "transforms": copied_transforms,
            "overlay": str(out / "overlay.png"),
            "metrics": str(out / "metrics.json"),
        },
        "metrics": metrics,
    }
    save_json(log, out / "log.json")
    return log


def run_classical(
    fixed_path: str | Path,
    moving_path: str | Path,
    out_dir: str | Path,
    backend: str = "diffeomorphic_demons",
    iterations: int = 50,
    smoothing_sigma: float = 1.3,
) -> dict[str, Any]:
    if backend == "diffeomorphic_demons":
        return register_demons(
            fixed_path=fixed_path,
            moving_path=moving_path,
            out_dir=out_dir,
            iterations=iterations,
            smoothing_sigma=smoothing_sigma,
            diffeomorphic=True,
        )
    if backend == "demons":
        return register_demons(
            fixed_path=fixed_path,
            moving_path=moving_path,
            out_dir=out_dir,
            iterations=iterations,
            smoothing_sigma=smoothing_sigma,
            diffeomorphic=False,
        )
    if backend == "ants_syn":
        return register_antspyx_syn(
            fixed_path=fixed_path,
            moving_path=moving_path,
            out_dir=out_dir,
            iterations=iterations,
        )
    raise ValueError(
        "Unknown classical backend. Choose one of: diffeomorphic_demons, demons, ants_syn"
    )

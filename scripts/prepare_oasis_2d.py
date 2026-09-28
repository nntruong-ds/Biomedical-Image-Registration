from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import zoom

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import ensure_dir, save_json, write_image
from src.utils.metrics import normalize_image
from src.utils.visualization import save_preview


def _sitk():
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise RuntimeError(
            "SimpleITK is required to read OASIS Analyze .hdr/.img files. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc
    return sitk


def _find_oasis_volumes(root: Path) -> list[Path]:
    patterns = [
        "*/PROCESSED/MPRAGE/T88_111/*masked_gfc.hdr",
        "disc*/**/PROCESSED/MPRAGE/T88_111/*masked_gfc.hdr",
    ]
    found: list[Path] = []
    for pattern in patterns:
        found.extend(root.glob(pattern))
    unique = sorted(set(found))
    if not unique:
        raise FileNotFoundError(
            f"No OASIS processed MPRAGE masked Analyze headers found under {root}. "
            "Expected paths like OAS1_0001_MR1/PROCESSED/MPRAGE/T88_111/*masked_gfc.hdr."
        )
    return unique


def _best_slice(volume: np.ndarray, axis: int) -> np.ndarray:
    if axis < 0 or axis > 2:
        raise ValueError("axis must be 0, 1, or 2")
    moved = np.moveaxis(volume, axis, 0)
    scores = []
    for idx in range(moved.shape[0]):
        image = moved[idx]
        foreground = image > np.percentile(image, 70)
        scores.append(float(foreground.sum()) * float(image.mean()))
    slice_index = int(np.argmax(scores))
    return moved[slice_index].astype(np.float32)


def _center_square(image: np.ndarray) -> np.ndarray:
    height, width = image.shape
    side = min(height, width)
    y0 = max((height - side) // 2, 0)
    x0 = max((width - side) // 2, 0)
    return image[y0 : y0 + side, x0 : x0 + side]


def _resize(image: np.ndarray, size: int) -> np.ndarray:
    image = _center_square(image)
    factor_y = size / image.shape[0]
    factor_x = size / image.shape[1]
    resized = zoom(image, (factor_y, factor_x), order=1)
    if resized.shape != (size, size):
        resized = resized[:size, :size]
        pad_y = max(size - resized.shape[0], 0)
        pad_x = max(size - resized.shape[1], 0)
        resized = np.pad(resized, ((0, pad_y), (0, pad_x)))
    return resized.astype(np.float32)


def _load_slice(path: Path, axis: int, size: int) -> np.ndarray:
    sitk = _sitk()
    image = sitk.ReadImage(str(path))
    volume = sitk.GetArrayFromImage(image).astype(np.float32)
    slice_2d = _best_slice(volume, axis=axis)
    positive = slice_2d[slice_2d > 0]
    if positive.size:
        high = float(np.percentile(positive, 99.5))
        slice_2d = np.clip(slice_2d, 0.0, high)
    return normalize_image(_resize(slice_2d, size=size))


def prepare_oasis_2d(
    root: str | Path,
    out_dir: str | Path,
    num_pairs: int,
    size: int,
    axis: int,
) -> list[dict[str, str | int]]:
    if num_pairs < 0:
        raise ValueError("--num_pairs must be non-negative; use 0 for all pairs")
    if size < 32:
        raise ValueError("--size must be at least 32")

    root_path = Path(root)
    out_path = ensure_dir(out_dir)
    volumes = _find_oasis_volumes(root_path)
    if num_pairs == 0:
        num_pairs = len(volumes) - 1
    if len(volumes) < num_pairs + 1:
        raise ValueError(
            f"Need at least {num_pairs + 1} volumes to create {num_pairs} pairs; "
            f"found {len(volumes)} under {root_path}."
        )

    images = [_load_slice(path, axis=axis, size=size) for path in volumes[: num_pairs + 1]]
    pairs: list[dict[str, str | int]] = []
    for idx in range(num_pairs):
        fixed = images[idx]
        moving = images[idx + 1]
        stem = f"{idx:03d}"
        fixed_path = out_path / f"fixed_{stem}.nii.gz"
        moving_path = out_path / f"moving_{stem}.nii.gz"
        preview_path = out_path / f"preview_{stem}.png"
        source_path = out_path / f"source_{stem}.json"
        write_image(fixed, fixed_path)
        write_image(moving, moving_path)
        save_preview(fixed, moving, preview_path)
        source = {
            "fixed_source": str(volumes[idx]),
            "moving_source": str(volumes[idx + 1]),
            "slice_axis": axis,
            "size": size,
            "note": "Real OASIS data do not include ground-truth deformation fields.",
        }
        save_json(source, source_path)
        pairs.append(
            {
                "index": idx,
                "fixed": str(fixed_path),
                "moving": str(moving_path),
                "preview": str(preview_path),
                "source": str(source_path),
            }
        )

    dataset = {
        "dataset": "OASIS-1 disc-derived 2D slices",
        "root": str(root_path),
        "num_pairs": num_pairs,
        "size": size,
        "slice_axis": axis,
        "pairs": pairs,
    }
    with (out_path / "dataset.json").open("w", encoding="utf-8") as file:
        json.dump(dataset, file, indent=2, sort_keys=True)
        file.write("\n")
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare 2D NIfTI pairs from extracted OASIS-1 volumes."
    )
    parser.add_argument("--root", required=True, help="Extracted OASIS root, e.g. data/oasis/disc1")
    parser.add_argument("--out", required=True, help="Output directory for 2D NIfTI pairs")
    parser.add_argument(
        "--num_pairs",
        type=int,
        default=8,
        help="Number of consecutive pairs to create. Use 0 for all available pairs.",
    )
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument(
        "--axis",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Volume axis to slice after SimpleITK array loading. 0 is axial.",
    )
    args = parser.parse_args()

    pairs = prepare_oasis_2d(
        root=args.root,
        out_dir=args.out,
        num_pairs=args.num_pairs,
        size=args.size,
        axis=args.axis,
    )
    print(f"Prepared {len(pairs)} OASIS 2D pairs in {args.out}")


if __name__ == "__main__":
    main()

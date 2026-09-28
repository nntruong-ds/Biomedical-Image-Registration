from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import ensure_dir, save_json, write_image
from src.utils.metrics import normalize_image
from src.utils.seed import set_seed
from src.utils.visualization import save_preview


def _subject_mri_dirs(root: Path, image_name: str) -> list[Path]:
    return sorted(path.parent for path in root.glob(f"disc*/OAS*_MR*/mri/{image_name}"))


def _resize_volume(volume: np.ndarray, size: int, order: int) -> np.ndarray:
    factors = (size / volume.shape[0], size / volume.shape[1], size / volume.shape[2])
    return zoom(volume, factors, order=order).astype(np.float32)


def _load_volume(path: Path, size: int, label: bool = False) -> np.ndarray:
    image = nib.load(str(path))
    data = np.asarray(image.get_fdata(dtype=np.float32), dtype=np.float32)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D FreeSurfer volume at {path}, got {data.shape}.")
    if label:
        return np.rint(_resize_volume(data, size, order=0)).astype(np.float32)
    mask = data > 0
    if np.any(mask):
        lo, hi = np.percentile(data[mask], [1.0, 99.0])
        if hi > lo:
            data = np.clip((data - lo) / (hi - lo), 0.0, 1.0)
        else:
            data = normalize_image(data)
    else:
        data = normalize_image(data)
    return _resize_volume(data, size, order=1)


def prepare_freesurfer_pairs(
    subjects_root: str | Path,
    out_dir: str | Path,
    num_pairs: int,
    size: int,
    image_name: str,
    label_name: str,
    seed: int,
) -> dict[str, object]:
    if num_pairs < 1:
        raise ValueError("--num_pairs must be >= 1.")
    if size < 16:
        raise ValueError("--size must be >= 16.")

    set_seed(seed)
    subjects_root = Path(subjects_root)
    if not subjects_root.exists():
        raise FileNotFoundError(f"FreeSurfer subjects directory not found: {subjects_root}")
    subject_dirs = _subject_mri_dirs(subjects_root, image_name)
    if len(subject_dirs) < num_pairs + 1:
        raise RuntimeError(
            f"Need at least {num_pairs + 1} extracted subjects with {image_name}; "
            f"found {len(subject_dirs)} under {subjects_root}."
        )

    out = ensure_dir(out_dir)
    pairs: list[dict[str, object]] = []
    for idx in range(num_pairs):
        fixed_dir = subject_dirs[idx]
        moving_dir = subject_dirs[idx + 1]
        fixed = _load_volume(fixed_dir / image_name, size=size)
        moving = _load_volume(moving_dir / image_name, size=size)
        fixed_label_path = fixed_dir / label_name
        moving_label_path = moving_dir / label_name
        fixed_label = (
            _load_volume(fixed_label_path, size=size, label=True)
            if fixed_label_path.exists()
            else None
        )
        moving_label = (
            _load_volume(moving_label_path, size=size, label=True)
            if moving_label_path.exists()
            else None
        )

        suffix = f"{idx:03d}.nii.gz"
        write_image(fixed, out / f"fixed_{suffix}")
        write_image(moving, out / f"moving_{suffix}")
        if fixed_label is not None:
            write_image(fixed_label, out / f"fixed_label_{suffix}")
        if moving_label is not None:
            write_image(moving_label, out / f"moving_label_{suffix}")
        save_preview(fixed, moving, out / f"preview_{idx:03d}.png")

        pair = {
            "index": idx,
            "fixed_subject": str(fixed_dir.parent),
            "moving_subject": str(moving_dir.parent),
            "fixed": str(out / f"fixed_{suffix}"),
            "moving": str(out / f"moving_{suffix}"),
            "fixed_label": str(out / f"fixed_label_{suffix}") if fixed_label is not None else None,
            "moving_label": str(out / f"moving_label_{suffix}") if moving_label is not None else None,
            "image_name": image_name,
            "label_name": label_name,
            "shape": [size, size, size],
        }
        save_json(pair, out / f"pair_{idx:03d}.json")
        pairs.append(pair)

    summary = {
        "subjects_root": str(subjects_root),
        "out_dir": str(out),
        "image_name": image_name,
        "label_name": label_name,
        "num_pairs": num_pairs,
        "size": size,
        "seed": seed,
        "pairs": pairs,
    }
    with (out / "dataset.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, sort_keys=True)
        file.write("\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build small 3D NIfTI registration pairs from extracted OASIS FreeSurfer data."
    )
    parser.add_argument(
        "--subjects-root",
        default="data/oasis/freesurfer/subjects",
        help="Extracted FreeSurfer subject root.",
    )
    parser.add_argument("--out", default="data/oasis/freesurfer_3d_smoke")
    parser.add_argument("--num_pairs", type=int, default=3)
    parser.add_argument("--size", type=int, default=48)
    parser.add_argument("--image", default="T1.mgz")
    parser.add_argument("--label", default="aparc+aseg.mgz")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    summary = prepare_freesurfer_pairs(
        subjects_root=args.subjects_root,
        out_dir=args.out,
        num_pairs=args.num_pairs,
        size=args.size,
        image_name=args.image,
        label_name=args.label,
        seed=args.seed,
    )
    print(
        f"Prepared {summary['num_pairs']} FreeSurfer 3D pairs at {summary['out_dir']} "
        f"with shape {summary['size']}^3"
    )


if __name__ == "__main__":
    main()

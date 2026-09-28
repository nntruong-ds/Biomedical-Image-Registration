from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.utils.io import read_image
from src.utils.metrics import normalize_image


def list_fixed_image_paths(root: str | Path) -> list[Path]:
    data_root = Path(root)
    return sorted(
        path
        for path in data_root.glob("fixed_*.nii.gz")
        if not path.name.startswith("fixed_label_")
    )


class SyntheticPairDataset(Dataset[dict[str, torch.Tensor | str]]):
    def __init__(self, root: str | Path):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(
                f"Synthetic data directory not found: {self.root}. "
                "Generate it with scripts/make_synthetic_data.py."
            )
        self.fixed_paths = list_fixed_image_paths(self.root)
        if not self.fixed_paths:
            raise FileNotFoundError(
                f"No fixed_*.nii.gz files found in {self.root}. "
                "Generate data with: python scripts/make_synthetic_data.py --out "
                f"{self.root} --num_pairs 20 --size 128"
            )

    def __len__(self) -> int:
        return len(self.fixed_paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        fixed_path = self.fixed_paths[index]
        suffix = fixed_path.name.removeprefix("fixed_")
        moving_path = self.root / f"moving_{suffix}"
        if not moving_path.exists():
            raise FileNotFoundError(f"Missing paired moving image: {moving_path}")
        fixed = normalize_image(read_image(fixed_path))
        moving = normalize_image(read_image(moving_path))
        return {
            "fixed": torch.from_numpy(fixed[None, ...]),
            "moving": torch.from_numpy(moving[None, ...]),
            "fixed_path": str(fixed_path),
            "moving_path": str(moving_path),
        }

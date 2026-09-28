from __future__ import annotations

from pathlib import Path

from src.data.dataset import SyntheticPairDataset, list_fixed_image_paths
from src.data.synthetic import generate_dataset


def test_synthetic_generation(tmp_path: Path) -> None:
    metadata = generate_dataset(tmp_path / "synthetic", num_pairs=1, size=32, seed=5)
    assert len(metadata) == 1
    pair = metadata[0]
    assert Path(pair.fixed).exists()
    assert Path(pair.moving).exists()
    assert Path(pair.gt_displacement).exists()
    assert Path(pair.preview).exists()


def test_pair_dataset_ignores_label_volumes(tmp_path: Path) -> None:
    root = tmp_path / "pairs"
    root.mkdir()
    for name in (
        "fixed_000.nii.gz",
        "moving_000.nii.gz",
        "fixed_label_000.nii.gz",
        "moving_label_000.nii.gz",
    ):
        (root / name).touch()

    fixed_paths = list_fixed_image_paths(root)
    dataset = SyntheticPairDataset(root)

    assert [path.name for path in fixed_paths] == ["fixed_000.nii.gz"]
    assert len(dataset) == 1
    assert dataset.fixed_paths == fixed_paths

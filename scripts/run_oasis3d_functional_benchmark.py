from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import map_coordinates

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.methods.classical.register import run_classical
from src.methods.deep_common import train_unsupervised
from src.methods.metaheuristic.pso import PSOConfig, run_pso
from src.methods.transmorph.infer import run_transmorph
from src.methods.transmorph.model import build_transmorph_model
from src.methods.voxelmorph.infer import run_voxelmorph
from src.methods.voxelmorph.model import build_voxelmorph_model
from src.utils.io import ensure_dir, read_image, read_sitk_image, save_json, write_image
from src.utils.metrics import before_after_metrics
from src.utils.seed import set_seed


METHODS = ("classical", "pso", "voxelmorph", "transmorph")
LEARNED_METHODS = ("voxelmorph", "transmorph")
SELECTED_LABELS = {
    2: "Left-Cerebral-White-Matter",
    3: "Left-Cerebral-Cortex",
    4: "Left-Lateral-Ventricle",
    10: "Left-Thalamus",
    11: "Left-Caudate",
    12: "Left-Putamen",
    13: "Left-Pallidum",
    17: "Left-Hippocampus",
    18: "Left-Amygdala",
    26: "Left-Accumbens-area",
    41: "Right-Cerebral-White-Matter",
    42: "Right-Cerebral-Cortex",
    43: "Right-Lateral-Ventricle",
    49: "Right-Thalamus",
    50: "Right-Caudate",
    51: "Right-Putamen",
    52: "Right-Pallidum",
    53: "Right-Hippocampus",
    54: "Right-Amygdala",
    58: "Right-Accumbens-area",
}


@dataclass(frozen=True)
class Pair:
    index: int
    fixed: Path
    moving: Path
    fixed_label: Path | None
    moving_label: Path | None
    fixed_subject: str | None = None
    moving_subject: str | None = None


def _load_pairs(dataset_root: Path) -> list[Pair]:
    fixed_paths = sorted(
        path
        for path in dataset_root.glob("fixed_*.nii.gz")
        if not path.name.startswith("fixed_label_")
    )
    pairs: list[Pair] = []
    for fixed in fixed_paths:
        suffix = fixed.name.removeprefix("fixed_")
        index = int(suffix.split(".")[0])
        moving = dataset_root / f"moving_{suffix}"
        if not moving.exists():
            raise FileNotFoundError(f"Missing paired moving image: {moving}")
        pair_json = dataset_root / f"pair_{index:03d}.json"
        fixed_subject = None
        moving_subject = None
        if pair_json.exists():
            payload = json.loads(pair_json.read_text(encoding="utf-8"))
            fixed_subject = payload.get("fixed_subject")
            moving_subject = payload.get("moving_subject")
        fixed_label = dataset_root / f"fixed_label_{suffix}"
        moving_label = dataset_root / f"moving_label_{suffix}"
        pairs.append(
            Pair(
                index=index,
                fixed=fixed,
                moving=moving,
                fixed_label=fixed_label if fixed_label.exists() else None,
                moving_label=moving_label if moving_label.exists() else None,
                fixed_subject=fixed_subject,
                moving_subject=moving_subject,
            )
        )
    if not pairs:
        raise FileNotFoundError(f"No fixed_*.nii.gz pairs found in {dataset_root}")
    return pairs


def _split_pairs(
    pairs: list[Pair], seed: int, train_fraction: float, val_fraction: float
) -> dict[str, list[Pair]]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pairs))
    n_total = len(pairs)
    n_train = int(round(train_fraction * n_total))
    n_val = int(round(val_fraction * n_total))
    n_train = min(max(n_train, 1), n_total - 2)
    n_val = min(max(n_val, 1), n_total - n_train - 1)
    train_idx = order[:n_train]
    val_idx = order[n_train : n_train + n_val]
    test_idx = order[n_train + n_val :]
    return {
        "train": [pairs[int(idx)] for idx in train_idx],
        "val": [pairs[int(idx)] for idx in val_idx],
        "test": [pairs[int(idx)] for idx in test_idx],
    }


def _link_or_replace(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def _make_subset_root(dataset_root: Path, pairs: list[Pair], out_root: Path) -> Path:
    ensure_dir(out_root)
    for pair in pairs:
        suffix = f"{pair.index:03d}.nii.gz"
        _link_or_replace(pair.fixed, out_root / f"fixed_{suffix}")
        _link_or_replace(pair.moving, out_root / f"moving_{suffix}")
        if pair.fixed_label is not None:
            _link_or_replace(pair.fixed_label, out_root / f"fixed_label_{suffix}")
        if pair.moving_label is not None:
            _link_or_replace(pair.moving_label, out_root / f"moving_label_{suffix}")
        pair_json = dataset_root / f"pair_{pair.index:03d}.json"
        if pair_json.exists():
            _link_or_replace(pair_json, out_root / f"pair_{pair.index:03d}.json")
    return out_root


def _deep_config(
    dataset_root: Path,
    output_dir: Path,
    model_name: str,
    epochs: int,
    batch_size: int,
    seed: int,
    device: str,
    num_workers: int,
) -> dict[str, Any]:
    model: dict[str, Any]
    if model_name == "voxelmorph":
        model = {"dim": 3, "in_channels": 2, "base_channels": 4}
    elif model_name == "transmorph":
        model = {
            "dim": 3,
            "in_channels": 2,
            "embed_dim": 24,
            "patch_size": 8,
            "depth": 1,
            "num_heads": 4,
            "decoder_channels": 12,
        }
    else:
        raise ValueError(f"Unknown learned method: {model_name}")
    return {
        "seed": seed,
        "device": device,
        "data": {"root": str(dataset_root)},
        "output_dir": str(output_dir),
        "model": model,
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": 1e-3,
            "image_loss": "mse",
            "smooth_weight": 0.05,
            "num_workers": num_workers,
        },
    }


def _train_one_learned(task: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    model_name = str(task["model_name"])
    train_root = Path(task["train_root"])
    method_out = Path(task["method_out"])
    cfg = _deep_config(
        train_root,
        method_out,
        model_name,
        int(task["epochs"]),
        int(task["batch_size"]),
        int(task["seed"]),
        str(task["device"]),
        int(task["num_workers"]),
    )
    if model_name == "voxelmorph":
        builder = build_voxelmorph_model
    elif model_name == "transmorph":
        builder = build_transmorph_model
    else:
        raise ValueError(f"Unknown learned method: {model_name}")

    print(
        f"Training {model_name} on train split for {task['epochs']} epochs "
        f"on {task['device']}...",
        flush=True,
    )
    log = train_unsupervised(builder(cfg), cfg, model_name, method_out)
    return model_name, {
        "elapsed_seconds": log["elapsed_seconds"],
        "best_loss": log["best_loss"],
        "checkpoint": str(method_out / "best.pt"),
        "trained_on": str(train_root),
        "device": str(task["device"]),
        "num_pairs": log.get("num_pairs"),
    }


def _train_learned(
    train_root: Path,
    out: Path,
    epochs: int,
    batch_size: int,
    seed: int,
    device: str,
    voxelmorph_device: str | None,
    transmorph_device: str | None,
    parallel: bool,
    num_workers: int,
) -> dict[str, dict[str, Any]]:
    if parallel and num_workers > 0:
        print(
            "Parallel learned training uses separate model processes; forcing "
            "training_num_workers=0 to avoid nested DataLoader workers.",
            flush=True,
        )
        num_workers = 0
    tasks = [
        {
            "model_name": "voxelmorph",
            "train_root": str(train_root),
            "method_out": str(out / "training" / "voxelmorph"),
            "epochs": epochs,
            "batch_size": batch_size,
            "seed": seed,
            "device": voxelmorph_device or device,
            "num_workers": num_workers,
        },
        {
            "model_name": "transmorph",
            "train_root": str(train_root),
            "method_out": str(out / "training" / "transmorph"),
            "epochs": epochs,
            "batch_size": batch_size,
            "seed": seed,
            "device": transmorph_device or device,
            "num_workers": num_workers,
        },
    ]
    if parallel:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=2) as pool:
            results = pool.map(_train_one_learned, tasks)
    else:
        results = [_train_one_learned(task) for task in tasks]
    return {name: payload for name, payload in results}


def _coords_3d(shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    depth, height, width = shape
    return np.meshgrid(
        np.arange(depth, dtype=np.float32),
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )


def _warp_label_dense(
    label: np.ndarray, field: np.ndarray, component_order: str
) -> np.ndarray:
    if label.ndim != 3 or field.shape != label.shape + (3,):
        raise ValueError(f"Expected label (D,H,W) and field (D,H,W,3), got {label.shape}, {field.shape}")
    zz, yy, xx = _coords_3d(label.shape)
    if component_order == "zyx":
        src_z = zz + field[..., 0]
        src_y = yy + field[..., 1]
        src_x = xx + field[..., 2]
    elif component_order == "xyz":
        src_z = zz + field[..., 2]
        src_y = yy + field[..., 1]
        src_x = xx + field[..., 0]
    else:
        raise ValueError("component_order must be 'zyx' or 'xyz'.")
    warped = map_coordinates(
        np.rint(label).astype(np.float32),
        [src_z, src_y, src_x],
        order=0,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    return np.rint(warped).astype(np.int32)


def _rotation_matrix_3d(params: dict[str, float]) -> np.ndarray:
    rx = np.deg2rad(float(params["rx_deg"]))
    ry = np.deg2rad(float(params["ry_deg"]))
    rz = np.deg2rad(float(params["rz_deg"]))
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float32)
    rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
    rot_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return rot_z @ rot_y @ rot_x


def _warp_label_pso_3d(label: np.ndarray, matrix_4x4: list[list[float]]) -> np.ndarray:
    label = np.rint(label).astype(np.float32)
    out_d, out_h, out_w = label.shape
    zz, yy, xx = _coords_3d(label.shape)
    center = np.array([(out_w - 1) * 0.5, (out_h - 1) * 0.5, (out_d - 1) * 0.5], dtype=np.float32)
    coords_xyz = np.stack([xx - center[0], yy - center[1], zz - center[2]], axis=0).reshape(3, -1)
    matrix = np.asarray(matrix_4x4, dtype=np.float32)
    src_xyz = matrix[:3, :3] @ coords_xyz
    src_xyz[0] += center[0] + float(matrix[0, 3])
    src_xyz[1] += center[1] + float(matrix[1, 3])
    src_xyz[2] += center[2] + float(matrix[2, 3])
    warped = map_coordinates(
        label,
        [src_xyz[2].reshape(label.shape), src_xyz[1].reshape(label.shape), src_xyz[0].reshape(label.shape)],
        order=0,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    return np.rint(warped).astype(np.int32)


def _dice_per_label(
    fixed_label: np.ndarray, moving_label: np.ndarray, labels: dict[int, str]
) -> dict[str, float]:
    fixed_i = np.rint(fixed_label).astype(np.int32)
    moving_i = np.rint(moving_label).astype(np.int32)
    out: dict[str, float] = {}
    for value, name in labels.items():
        fixed_mask = fixed_i == value
        moving_mask = moving_i == value
        denom = int(fixed_mask.sum() + moving_mask.sum())
        if denom == 0:
            continue
        out[name] = float(2.0 * np.logical_and(fixed_mask, moving_mask).sum() / denom)
    return out


def _mean(values: dict[str, float]) -> float | None:
    return statistics.fmean(values.values()) if values else None


def _jacobian_folding_percent(field: np.ndarray, component_order: str) -> float:
    if component_order == "zyx":
        disp = field
    elif component_order == "xyz":
        disp = field[..., [2, 1, 0]]
    else:
        raise ValueError("component_order must be 'zyx' or 'xyz'.")
    gz0, gy0, gx0 = np.gradient(disp[..., 0], edge_order=1)
    gz1, gy1, gx1 = np.gradient(disp[..., 1], edge_order=1)
    gz2, gy2, gx2 = np.gradient(disp[..., 2], edge_order=1)
    j00 = 1.0 + gz0
    j01 = gy0
    j02 = gx0
    j10 = gz1
    j11 = 1.0 + gy1
    j12 = gx1
    j20 = gz2
    j21 = gy2
    j22 = 1.0 + gx2
    determinant = (
        j00 * (j11 * j22 - j12 * j21)
        - j01 * (j10 * j22 - j12 * j20)
        + j02 * (j10 * j21 - j11 * j20)
    )
    return float(np.mean(determinant <= 0.0) * 100.0)


def _label_metrics(
    method: str,
    pair: Pair,
    method_out: Path,
    reference_path: Path,
) -> dict[str, Any]:
    if pair.fixed_label is None or pair.moving_label is None:
        return {"labels_available": False}
    fixed_label = np.rint(read_image(pair.fixed_label)).astype(np.int32)
    moving_label = np.rint(read_image(pair.moving_label)).astype(np.int32)
    before = _dice_per_label(fixed_label, moving_label, SELECTED_LABELS)
    warped: np.ndarray | None = None
    jacobian: float | None = None

    if method == "pso":
        transform_payload = json.loads((method_out / "transform_params.json").read_text(encoding="utf-8"))
        warped = _warp_label_pso_3d(moving_label, transform_payload["matrix_4x4"])
    elif method == "classical":
        field = np.load(method_out / "deformation_field.npy")
        warped = _warp_label_dense(moving_label, field, component_order="xyz")
        jacobian = _jacobian_folding_percent(field, component_order="xyz")
    elif method in LEARNED_METHODS:
        field = np.load(method_out / "deformation_field.npy")
        warped = _warp_label_dense(moving_label, field, component_order="zyx")
        jacobian = _jacobian_folding_percent(field, component_order="zyx")
    else:
        raise ValueError(f"Unsupported method for label metrics: {method}")

    after = _dice_per_label(fixed_label, warped, SELECTED_LABELS)
    write_image(
        warped.astype(np.float32),
        method_out / "registered_label.nii.gz",
        reference=read_sitk_image(reference_path),
    )
    before_mean = _mean(before)
    after_mean = _mean(after)
    return {
        "labels_available": True,
        "label_source": "aparc+aseg/aseg volume, nearest-neighbor warped",
        "selected_labels": SELECTED_LABELS,
        "dice_before": before,
        "dice_after": after,
        "dice_before_mean": before_mean,
        "dice_after_mean": after_mean,
        "dice_delta_mean": (after_mean - before_mean) if before_mean is not None and after_mean is not None else None,
        "registered_label": str(method_out / "registered_label.nii.gz"),
        "jacobian_folding_percent": jacobian,
    }


def _label_metrics_safe(method: str, pair: Pair, method_out: Path, reference_path: Path) -> dict[str, Any]:
    try:
        return _label_metrics(method, pair, method_out, reference_path)
    except Exception as exc:
        return {"labels_available": pair.fixed_label is not None and pair.moving_label is not None, "label_error": str(exc)}


def _run_method(
    method: str,
    pair: Pair,
    method_out: Path,
    args: argparse.Namespace,
    training: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    ensure_dir(method_out)
    started = time.perf_counter()
    if method == "classical":
        log = run_classical(
            pair.fixed,
            pair.moving,
            method_out,
            iterations=args.classical_iterations,
            smoothing_sigma=args.classical_smoothing_sigma,
        )
    elif method == "pso":
        log = run_pso(
            pair.fixed,
            pair.moving,
            method_out,
            PSOConfig(
                particles=args.pso_particles,
                iterations=args.pso_iterations,
                metric=args.pso_metric,
                transform=args.pso_transform,
                seed=args.seed + pair.index,
            ),
        )
    elif method == "voxelmorph":
        log = run_voxelmorph(pair.fixed, pair.moving, training["voxelmorph"]["checkpoint"], method_out)
    elif method == "transmorph":
        log = run_transmorph(pair.fixed, pair.moving, training["transmorph"]["checkpoint"], method_out)
    else:
        raise ValueError(f"Unknown method: {method}")

    run_seconds = time.perf_counter() - started
    metrics = log["metrics"]
    labels = _label_metrics_safe(method, pair, method_out, pair.fixed)
    row: dict[str, Any] = {
        "method": method,
        "pair_index": pair.index,
        "fixed": str(pair.fixed),
        "moving": str(pair.moving),
        "fixed_subject": pair.fixed_subject,
        "moving_subject": pair.moving_subject,
        "success": True,
        "error": "",
        "run_seconds": run_seconds,
        "before_mse": metrics["before"]["mse"],
        "after_mse": metrics["after"]["mse"],
        "delta_mse": metrics["before"]["mse"] - metrics["after"]["mse"],
        "before_ncc": metrics["before"]["ncc"],
        "after_ncc": metrics["after"]["ncc"],
        "delta_ncc": metrics["after"]["ncc"] - metrics["before"]["ncc"],
        "registered": log["outputs"]["registered"],
        "overlay": log["outputs"]["overlay"],
        "log": str(method_out / "log.json"),
        "labels_available": labels.get("labels_available", False),
        "label_error": labels.get("label_error", ""),
        "dice_before_mean": labels.get("dice_before_mean"),
        "dice_after_mean": labels.get("dice_after_mean"),
        "dice_delta_mean": labels.get("dice_delta_mean"),
        "jacobian_folding_percent": labels.get("jacobian_folding_percent"),
    }
    save_json({"row": row, "label_metrics": labels}, method_out / "benchmark_metrics.json")
    return row


def _failure_row(method: str, pair: Pair, error: Exception) -> dict[str, Any]:
    return {
        "method": method,
        "pair_index": pair.index,
        "fixed": str(pair.fixed),
        "moving": str(pair.moving),
        "fixed_subject": pair.fixed_subject,
        "moving_subject": pair.moving_subject,
        "success": False,
        "error": str(error),
        "run_seconds": None,
        "before_mse": None,
        "after_mse": None,
        "delta_mse": None,
        "before_ncc": None,
        "after_ncc": None,
        "delta_ncc": None,
        "registered": "",
        "overlay": "",
        "log": "",
        "labels_available": pair.fixed_label is not None and pair.moving_label is not None,
        "label_error": "",
        "dice_before_mean": None,
        "dice_after_mean": None,
        "dice_delta_mean": None,
        "jacobian_folding_percent": None,
    }


def _write_rows(rows: list[dict[str, Any]], csv_path: Path, json_path: Path) -> None:
    ensure_dir(csv_path.parent)
    fields = [
        "split",
        "method",
        "pair_index",
        "fixed",
        "moving",
        "fixed_subject",
        "moving_subject",
        "success",
        "error",
        "run_seconds",
        "before_mse",
        "after_mse",
        "delta_mse",
        "before_ncc",
        "after_ncc",
        "delta_ncc",
        "labels_available",
        "label_error",
        "dice_before_mean",
        "dice_after_mean",
        "dice_delta_mean",
        "jacobian_folding_percent",
        "registered",
        "overlay",
        "log",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
    save_json({"rows": rows}, json_path)


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for method in METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        successful = [row for row in method_rows if row.get("success")]
        entry: dict[str, Any] = {
            "num_rows": len(method_rows),
            "success_count": len(successful),
            "failure_count": len(method_rows) - len(successful),
        }
        for key in (
            "run_seconds",
            "before_mse",
            "after_mse",
            "delta_mse",
            "before_ncc",
            "after_ncc",
            "delta_ncc",
            "dice_before_mean",
            "dice_after_mean",
            "dice_delta_mean",
            "jacobian_folding_percent",
        ):
            values = [float(row[key]) for row in successful if row.get(key) is not None]
            if values:
                entry[f"mean_{key}"] = statistics.fmean(values)
                entry[f"std_{key}"] = statistics.pstdev(values) if len(values) > 1 else 0.0
            else:
                entry[f"mean_{key}"] = None
                entry[f"std_{key}"] = None
        summary[method] = entry
    return summary


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    metadata = summary["metadata"]
    lines = [
        "# OASIS-1 3D Functional Benchmark",
        "",
        "This is a 64^3 OASIS-1 FreeSurfer-derived 3D functional benchmark. It is not an exact VoxelMorph or TransMorph paper reproduction.",
        "",
        "## Dataset",
        "",
        f"- Subjects: {metadata['num_subjects']}",
        f"- Total pairs: {metadata['num_pairs']}",
        f"- Train pairs: {metadata['splits']['train']}",
        f"- Validation pairs: {metadata['splits']['val']}",
        f"- Test pairs: {metadata['splits']['test']}",
        f"- Resolution: {metadata['resolution']}",
        f"- Label source: {metadata['label_source']}",
        "",
        "Learning methods are trained only on the train split. The final benchmark table below uses test pairs only.",
        "",
        "## Test Results",
        "",
        "| Method | Success | Fail | Mean sec | After MSE | Delta MSE | After NCC | Delta NCC | Dice after | Dice delta | Jacobian folding % |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        entry = summary["test_methods"][method]
        def fmt(key: str) -> str:
            value = entry.get(key)
            return "" if value is None else f"{value:.6f}"
        lines.append(
            "| {method} | {ok} | {fail} | {sec} | {mse} | {dmse} | {ncc} | {dncc} | {dice} | {ddice} | {fold} |".format(
                method=method,
                ok=entry["success_count"],
                fail=entry["failure_count"],
                sec=fmt("mean_run_seconds"),
                mse=fmt("mean_after_mse"),
                dmse=fmt("mean_delta_mse"),
                ncc=fmt("mean_after_ncc"),
                dncc=fmt("mean_delta_ncc"),
                dice=fmt("mean_dice_after_mean"),
                ddice=fmt("mean_dice_delta_mean"),
                fold=fmt("mean_jacobian_folding_percent"),
            )
        )
    lines.extend(
        [
            "",
            "## Training",
            "",
        ]
    )
    for method, item in summary["training"].items():
        best_loss = item.get("best_loss")
        best_loss_text = "reused" if best_loss is None else f"{float(best_loss):.6f}"
        lines.append(
            f"- {method}: seconds={item['elapsed_seconds']:.4f}, best_loss={best_loss_text}, checkpoint={item['checkpoint']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _subjects_from_pairs(pairs: list[Pair]) -> set[str]:
    subjects: set[str] = set()
    for pair in pairs:
        if pair.fixed_subject:
            subjects.add(pair.fixed_subject)
        if pair.moving_subject:
            subjects.add(pair.moving_subject)
    return subjects


def run(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    dataset_root = Path(args.dataset_root)
    out = ensure_dir(args.out)
    pairs = _load_pairs(dataset_root)
    if args.num_pairs is not None:
        pairs = pairs[: args.num_pairs]
    splits = _split_pairs(pairs, args.seed, args.train_fraction, args.val_fraction)
    split_payload = {
        "seed": args.seed,
        "train_fraction": args.train_fraction,
        "val_fraction": args.val_fraction,
        "test_fraction": 1.0 - args.train_fraction - args.val_fraction,
        "train": [pair.index for pair in splits["train"]],
        "val": [pair.index for pair in splits["val"]],
        "test": [pair.index for pair in splits["test"]],
    }
    save_json(split_payload, out / "splits.json")

    train_root = _make_subset_root(dataset_root, splits["train"], out / "splits" / "train")
    if args.skip_training:
        if args.voxelmorph_checkpoint is None or args.transmorph_checkpoint is None:
            raise ValueError(
                "--skip-training requires --voxelmorph-checkpoint and --transmorph-checkpoint"
            )
        training = {
            "voxelmorph": {
                "elapsed_seconds": 0.0,
                "best_loss": None,
                "checkpoint": args.voxelmorph_checkpoint,
                "trained_on": str(train_root),
                "device": "checkpoint_config",
                "num_pairs": len(splits["train"]),
                "reused_checkpoint": True,
            },
            "transmorph": {
                "elapsed_seconds": 0.0,
                "best_loss": None,
                "checkpoint": args.transmorph_checkpoint,
                "trained_on": str(train_root),
                "device": "checkpoint_config",
                "num_pairs": len(splits["train"]),
                "reused_checkpoint": True,
            },
        }
    else:
        training = _train_learned(
            train_root,
            out,
            args.deep_epochs,
            args.batch_size,
            args.seed,
            args.device,
            args.voxelmorph_device,
            args.transmorph_device,
            args.parallel_learned_training,
            args.training_num_workers,
        )

    validation_rows: list[dict[str, Any]] = []
    if args.run_validation:
        print(f"Running learned-method validation checks on {len(splits['val'])} pairs...", flush=True)
        for count, pair in enumerate(splits["val"], start=1):
            for method in LEARNED_METHODS:
                method_out = out / "validation_runs" / method / f"pair_{pair.index:03d}"
                try:
                    row = _run_method(method, pair, method_out, args, training)
                except Exception as exc:
                    row = _failure_row(method, pair, exc)
                row["split"] = "val"
                validation_rows.append(row)
            print(f"  validation {count}/{len(splits['val'])} done", flush=True)
        _write_rows(validation_rows, out / "validation_results.csv", out / "validation_results.json")

    test_rows: list[dict[str, Any]] = []
    print(f"Running final test benchmark on {len(splits['test'])} pairs...", flush=True)
    for count, pair in enumerate(splits["test"], start=1):
        for method in METHODS:
            method_out = out / "test_runs" / method / f"pair_{pair.index:03d}"
            try:
                row = _run_method(method, pair, method_out, args, training)
            except Exception as exc:
                row = _failure_row(method, pair, exc)
            row["split"] = "test"
            test_rows.append(row)
            status = "done" if row["success"] else "failed"
            print(f"  test {count}/{len(splits['test'])} {method} {status}", flush=True)

    _write_rows(test_rows, out / "benchmark_results.csv", out / "benchmark_results.json")
    metadata = {
        "benchmark": "OASIS-1 3D functional benchmark",
        "not_exact_paper_reproduction": True,
        "dataset_root": str(dataset_root),
        "num_subjects": len(_subjects_from_pairs(pairs)),
        "num_pairs": len(pairs),
        "splits": {name: len(items) for name, items in splits.items()},
        "resolution": "64^3",
        "label_source": "FreeSurfer aseg/aparc+aseg volume; label/*.label files are not used for Dice",
        "selected_labels": SELECTED_LABELS,
        "device": args.device,
        "voxelmorph_device": args.voxelmorph_device or args.device,
        "transmorph_device": args.transmorph_device or args.device,
        "parallel_learned_training": args.parallel_learned_training,
        "skip_training": args.skip_training,
        "training_num_workers": args.training_num_workers,
        "pso_transform": args.pso_transform,
        "pso_particles": args.pso_particles,
        "pso_iterations": args.pso_iterations,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    summary = {
        "metadata": metadata,
        "training": training,
        "validation_methods": _summarize_rows(validation_rows) if validation_rows else {},
        "test_methods": _summarize_rows(test_rows),
    }
    save_json(summary, out / "benchmark_summary.json")
    _write_markdown(summary, out / "benchmark_summary.md")
    print(f"Benchmark complete. Summary: {out / 'benchmark_summary.md'}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the OASIS-1 FreeSurfer-derived 3D functional benchmark.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-pairs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--deep-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--voxelmorph-checkpoint", default=None)
    parser.add_argument("--transmorph-checkpoint", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--voxelmorph-device", default=None)
    parser.add_argument("--transmorph-device", default=None)
    parser.add_argument("--parallel-learned-training", action="store_true")
    parser.add_argument("--training-num-workers", type=int, default=0)
    parser.add_argument("--classical-iterations", type=int, default=10)
    parser.add_argument("--classical-smoothing-sigma", type=float, default=1.3)
    parser.add_argument("--pso-particles", type=int, default=8)
    parser.add_argument("--pso-iterations", type=int, default=10)
    parser.add_argument("--pso-metric", choices=["ncc", "mse"], default="ncc")
    parser.add_argument("--pso-transform", choices=["rigid", "affine"], default="rigid")
    parser.add_argument("--no-validation", dest="run_validation", action="store_false")
    parser.set_defaults(run_validation=True)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

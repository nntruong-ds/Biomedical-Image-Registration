from __future__ import annotations

import argparse
import gzip
import json
import struct
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.metrics import normalize_image


METHODS = ("classical", "pso", "voxelmorph", "transmorph")
LEARNED_METHODS = ("voxelmorph", "transmorph")
DENSE_METHODS = ("classical", "voxelmorph", "transmorph")
METHOD_LABELS = {
    "classical": "Classical",
    "pso": "PSO",
    "voxelmorph": "VoxelMorph",
    "transmorph": "TransMorph",
}
METHOD_COLORS = {
    "classical": "#4C78A8",
    "pso": "#F58518",
    "voxelmorph": "#54A24B",
    "transmorph": "#B279A2",
}
NUMERIC_COLUMNS = (
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
)


def _read_nifti_minimal(path: str | Path) -> np.ndarray:
    file_path = Path(path)
    if file_path.suffix == ".gz":
        with gzip.open(file_path, "rb") as file:
            raw = file.read()
    else:
        raw = file_path.read_bytes()

    if len(raw) < 352:
        raise ValueError(f"File is too small to be a NIfTI image: {file_path}")
    if struct.unpack("<i", raw[:4])[0] == 348:
        endian = "<"
    elif struct.unpack(">i", raw[:4])[0] == 348:
        endian = ">"
    else:
        raise ValueError(f"Unsupported NIfTI header: {file_path}")

    dims = struct.unpack(endian + "8h", raw[40:56])
    ndim = int(dims[0])
    shape = tuple(int(value) for value in dims[1 : ndim + 1])
    datatype = struct.unpack(endian + "h", raw[70:72])[0]
    vox_offset = int(struct.unpack(endian + "f", raw[108:112])[0])
    slope = float(struct.unpack(endian + "f", raw[112:116])[0])
    intercept = float(struct.unpack(endian + "f", raw[116:120])[0])
    dtype_map = {
        2: "u1",
        4: "i2",
        8: "i4",
        16: "f4",
        64: "f8",
        512: "u2",
        768: "u4",
    }
    if datatype not in dtype_map:
        raise ValueError(f"Unsupported NIfTI datatype {datatype}: {file_path}")

    dtype = np.dtype(dtype_map[datatype]).newbyteorder(endian)
    data = np.frombuffer(raw, dtype=dtype, count=int(np.prod(shape)), offset=vox_offset)
    image = np.array(data.reshape(shape, order="F"), dtype=np.float32)
    if slope not in (0.0, 1.0):
        image = image * slope
    if intercept != 0.0:
        image = image + intercept
    if image.ndim == 2:
        return image.T
    if image.ndim == 3:
        return np.transpose(image, (2, 1, 0))
    if image.ndim == 4 and image.shape[-1] == 1:
        return np.transpose(image[..., 0], (2, 1, 0))
    raise ValueError(f"Expected a 2D or 3D NIfTI image, got shape {image.shape}: {file_path}")


def read_volume(path: str | Path) -> np.ndarray:
    try:
        from src.utils.io import read_image

        return read_image(path)
    except Exception:
        return _read_nifti_minimal(path)


def _success_mask(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(("true", "1", "yes"))


def _load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "success" in df.columns:
        df = df[_success_mask(df["success"])].copy()
    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df["pair_index"] = pd.to_numeric(df["pair_index"]).astype(int)
    return df


def _methods_in(df: pd.DataFrame) -> list[str]:
    present = set(df["method"])
    return [method for method in METHODS if method in present]


def _save(fig: plt.Figure, out_dir: Path, name: str, dpi: int = 170) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def _mid_slice(volume: np.ndarray, axis: int = 0, index: int | None = None) -> np.ndarray:
    if index is None:
        index = volume.shape[axis] // 2
    if axis == 0:
        return volume[index, :, :]
    if axis == 1:
        return volume[:, index, :]
    return volume[:, :, index]


def _display(image: np.ndarray) -> np.ndarray:
    return normalize_image(np.nan_to_num(image.astype(np.float32), nan=0.0))


def _overlay_rgb(fixed: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    fixed_s = _display(fixed)
    candidate_s = _display(candidate)
    rgb = np.zeros(fixed_s.shape + (3,), dtype=np.float32)
    rgb[..., 0] = fixed_s
    rgb[..., 1] = candidate_s
    rgb[..., 2] = 0.25 * fixed_s
    return np.clip(rgb, 0.0, 1.0)


def _field_to_zyx(field: np.ndarray, method: str) -> np.ndarray:
    return field[..., [2, 1, 0]] if method == "classical" else field


def _jacobian_det(field_zyx: np.ndarray) -> np.ndarray:
    gz0, gy0, gx0 = np.gradient(field_zyx[..., 0], edge_order=1)
    gz1, gy1, gx1 = np.gradient(field_zyx[..., 1], edge_order=1)
    gz2, gy2, gx2 = np.gradient(field_zyx[..., 2], edge_order=1)
    j00, j01, j02 = 1.0 + gz0, gy0, gx0
    j10, j11, j12 = gz1, 1.0 + gy1, gx1
    j20, j21, j22 = gz2, gy2, 1.0 + gx2
    return (
        j00 * (j11 * j22 - j12 * j21)
        - j01 * (j10 * j22 - j12 * j20)
        + j02 * (j10 * j21 - j11 * j20)
    )


def plot_summary_metrics(df: pd.DataFrame, out_dir: Path) -> None:
    methods = _methods_in(df)
    grouped = df.groupby("method").mean(numeric_only=True).reindex(methods)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.3))
    plots = [
        ("after_mse", "MSE"),
        ("after_ncc", "NCC"),
        ("dice_after_mean", "Dice"),
        ("run_seconds", "Runtime / pair (s)"),
    ]
    for ax, (column, title) in zip(axes.flat, plots):
        values = grouped[column].to_numpy(dtype=float)
        bars = ax.bar(
            [METHOD_LABELS[m] for m in methods],
            values,
            color=[METHOD_COLORS[m] for m in methods],
        )
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=20)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    _save(fig, out_dir, "summary_metrics.png")


def plot_metric_distributions(df: pd.DataFrame, out_dir: Path) -> None:
    methods = _methods_in(df)
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.4))
    plots = [
        ("after_mse", "MSE"),
        ("after_ncc", "NCC"),
        ("dice_after_mean", "Dice"),
        ("run_seconds", "Runtime / pair (s)"),
        ("delta_ncc", "NCC improvement"),
        ("jacobian_folding_percent", "Jacobian folding %"),
    ]
    for ax, (column, title) in zip(axes.flat, plots):
        data = [df.loc[df["method"] == method, column].dropna().to_numpy() for method in methods]
        box = ax.boxplot(data, patch_artist=True, widths=0.62, showmeans=True)
        for patch, method in zip(box["boxes"], methods):
            patch.set_facecolor(METHOD_COLORS[method])
            patch.set_alpha(0.72)
        ax.set_title(title)
        ax.set_xticklabels([METHOD_LABELS[m] for m in methods], rotation=25)
        ax.grid(axis="y", alpha=0.25)
        if column == "run_seconds":
            ax.set_yscale("log")
    _save(fig, out_dir, "metric_distributions.png")


def plot_runtime_quality(df: pd.DataFrame, out_dir: Path) -> None:
    methods = _methods_in(df)
    grouped = df.groupby("method").mean(numeric_only=True).reindex(methods)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    panels = [("after_ncc", "Mean NCC"), ("dice_after_mean", "Mean Dice")]
    for ax, (y_col, y_label) in zip(axes, panels):
        for method in methods:
            x = grouped.loc[method, "run_seconds"]
            y = grouped.loc[method, y_col]
            ax.scatter(
                x,
                y,
                s=190,
                color=METHOD_COLORS[method],
                edgecolor="black",
                linewidth=0.7,
            )
            ax.annotate(METHOD_LABELS[method], xy=(x, y), xytext=(7, 4), textcoords="offset points")
        ax.set_xlabel("Mean runtime per pair (s)")
        ax.set_ylabel(y_label)
        ax.grid(alpha=0.25)
    _save(fig, out_dir, "runtime_quality_tradeoff.png")


def plot_label_heatmap(df: pd.DataFrame, benchmark_dir: Path, out_dir: Path) -> None:
    records: list[dict[str, Any]] = []
    for row in df.itertuples(index=False):
        path = (
            benchmark_dir
            / "test_runs"
            / row.method
            / f"pair_{int(row.pair_index):03d}"
            / "benchmark_metrics.json"
        )
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        dice_after = payload.get("label_metrics", {}).get("dice_after", {}) or {}
        for label, value in dice_after.items():
            records.append({"method": row.method, "label": label, "dice_after": float(value)})
    label_df = pd.DataFrame.from_records(records)
    if label_df.empty:
        return
    methods = _methods_in(df)
    pivot = (
        label_df.pivot_table(index="method", columns="label", values="dice_after", aggfunc="mean")
        .reindex(methods)
    )
    pivot = pivot[pivot.mean(axis=0).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(16.5, 4.8))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_title("Mean Dice by Anatomical Label")
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels([METHOD_LABELS[m] for m in methods])
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=65, ha="right", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02, label="Dice")
    _save(fig, out_dir, "label_dice_heatmap.png")


def _select_pair(df: pd.DataFrame, requested: int | None) -> int:
    if requested is not None:
        return int(requested)
    means = df.groupby("pair_index")["after_ncc"].mean().sort_values()
    return int(means.index[len(means) // 2])


def plot_qualitative_gallery(df: pd.DataFrame, out_dir: Path, pair_index: int) -> None:
    methods = _methods_in(df)
    pair_df = df[df["pair_index"] == pair_index]
    if pair_df.empty:
        raise ValueError(f"No rows for pair {pair_index:03d}")
    base = pair_df.iloc[0]
    fixed = read_volume(base["fixed"])
    moving = read_volume(base["moving"])
    fixed_slice = _mid_slice(fixed, axis=0)
    moving_slice = _mid_slice(moving, axis=0)

    rows: list[tuple[str, str, np.ndarray]] = [("Before", "", moving_slice)]
    for method in methods:
        row = pair_df[pair_df["method"] == method].iloc[0]
        registered = read_volume(row["registered"])
        rows.append((METHOD_LABELS[method], "", _mid_slice(registered, axis=0)))

    fig, axes = plt.subplots(len(rows), 4, figsize=(13.6, 3.05 * len(rows)))
    for row_index, (row_label, metric_text, candidate) in enumerate(rows):
        diff = np.abs(_display(fixed_slice) - _display(candidate))
        panels = [
            (fixed_slice, "Fixed", "gray"),
            (candidate, "Moving" if row_label == "Before" else "Registered", "gray"),
            (_overlay_rgb(fixed_slice, candidate), "Overlay", None),
            (diff, "|fixed - candidate|", "magma"),
        ]
        axes[row_index, 0].set_ylabel(
            f"{row_label}\n{metric_text}".strip(),
            fontsize=10,
        )
        for col_index, (image, title, cmap) in enumerate(panels):
            ax = axes[row_index, col_index]
            if cmap is None:
                ax.imshow(image)
            else:
                ax.imshow(image, cmap=cmap)
            ax.set_title(title, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
    _save(fig, out_dir, f"qualitative_pair_{pair_index:03d}.png", dpi=150)


def plot_deformation_diagnostics(
    df: pd.DataFrame, benchmark_dir: Path, out_dir: Path, pair_index: int
) -> None:
    pair_df = df[df["pair_index"] == pair_index]
    fig, axes = plt.subplots(len(DENSE_METHODS), 3, figsize=(12.8, 3.6 * len(DENSE_METHODS)))
    for row_index, method in enumerate(DENSE_METHODS):
        field_path = benchmark_dir / "test_runs" / method / f"pair_{pair_index:03d}" / "deformation_field.npy"
        field = _field_to_zyx(np.load(field_path), method)
        magnitude = np.linalg.norm(field, axis=-1)
        jacobian = _jacobian_det(field)
        fold = jacobian <= 0.0
        panels = [
            (_mid_slice(magnitude, axis=0), "Displacement magnitude", "viridis", None),
            (_mid_slice(jacobian, axis=0), "Jacobian determinant", "coolwarm", None),
            (_mid_slice(fold.astype(np.float32), axis=0), "Folding mask", "Reds", (0.0, 1.0)),
        ]
        axes[row_index, 0].set_ylabel(METHOD_LABELS[method], fontsize=10)
        for col_index, (image, title, cmap, limits) in enumerate(panels):
            ax = axes[row_index, col_index]
            if limits is None:
                shown = ax.imshow(image, cmap=cmap)
            else:
                shown = ax.imshow(image, cmap=cmap, vmin=limits[0], vmax=limits[1])
            ax.set_title(title, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(shown, ax=ax, fraction=0.035, pad=0.02)
    _save(fig, out_dir, f"deformation_pair_{pair_index:03d}.png", dpi=150)


def _load_training_logs(benchmark_dir: Path) -> dict[str, dict[str, Any]]:
    logs: dict[str, dict[str, Any]] = {}
    for method in LEARNED_METHODS:
        path = benchmark_dir / "training" / method / "training_log.json"
        if path.exists():
            logs[method] = json.loads(path.read_text(encoding="utf-8"))
    return logs


def plot_training_curves(benchmark_dir: Path, out_dir: Path) -> None:
    logs = _load_training_logs(benchmark_dir)
    if not logs:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    plots = [
        ("loss", "Total loss"),
        ("image_loss", "Image similarity loss"),
        ("smoothness_loss", "Smoothness loss"),
    ]
    for ax, (key, title) in zip(axes, plots):
        for method, log in logs.items():
            history = pd.DataFrame(log.get("history", []))
            if history.empty or key not in history:
                continue
            ax.plot(
                history["epoch"],
                history[key],
                marker="o",
                linewidth=2.0,
                markersize=4.0,
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
            )
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    _save(fig, out_dir, "training_loss_curves.png")


def plot_training_summary(benchmark_dir: Path, out_dir: Path) -> None:
    logs = _load_training_logs(benchmark_dir)
    if not logs:
        return
    summary = json.loads((benchmark_dir / "benchmark_summary.json").read_text(encoding="utf-8"))
    train_pairs = int(summary.get("metadata", {}).get("splits", {}).get("train", 0))
    records = []
    for method, log in logs.items():
        records.append(
            {
                "method": method,
                "elapsed_seconds": float(log.get("elapsed_seconds", 0.0)),
                "best_loss": float(log.get("best_loss", np.nan)),
                "num_pairs": int(log.get("num_pairs", 0)),
                "epochs": len(log.get("history", [])),
            }
        )
    df = pd.DataFrame.from_records(records).set_index("method").reindex(LEARNED_METHODS)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))
    panels = [
        ("elapsed_seconds", "Training time (s)", "{:.0f}"),
        ("best_loss", "Best training loss", "{:.4f}"),
        ("num_pairs", "Samples seen by loader", "{:.0f}"),
    ]
    for ax, (column, title, fmt) in zip(axes, panels):
        values = df[column].to_numpy(dtype=float)
        labels = [METHOD_LABELS[m] for m in df.index]
        bars = ax.bar(labels, values, color=[METHOD_COLORS[m] for m in df.index])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                fmt.format(value),
                ha="center",
                va="bottom",
                fontsize=9,
            )
        if column == "num_pairs" and train_pairs:
            ax.axhline(train_pairs, linestyle="--", color="black", linewidth=1.2)
    _save(fig, out_dir, "training_summary.png")


def plot_validation_vs_test(benchmark_dir: Path, test_df: pd.DataFrame, out_dir: Path) -> None:
    val_path = benchmark_dir / "validation_results.csv"
    if not val_path.exists():
        return
    val_df = _load_results(val_path)
    val_df = val_df[val_df["method"].isin(LEARNED_METHODS)]
    test_df = test_df[test_df["method"].isin(LEARNED_METHODS)]
    if val_df.empty or test_df.empty:
        return
    metrics = [
        ("after_mse", "MSE"),
        ("after_ncc", "NCC"),
        ("dice_after_mean", "Dice"),
        ("jacobian_folding_percent", "Folding %"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8))
    for ax, (column, title) in zip(axes.flat, metrics):
        val_means = val_df.groupby("method")[column].mean().reindex(LEARNED_METHODS)
        test_means = test_df.groupby("method")[column].mean().reindex(LEARNED_METHODS)
        x = np.arange(len(LEARNED_METHODS))
        width = 0.36
        ax.bar(x - width / 2, val_means, width, label="Validation", color="#9ecae9")
        ax.bar(x + width / 2, test_means, width, label="Test", color="#fdae6b")
        ax.set_title(title, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS[m] for m in LEARNED_METHODS])
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
    _save(fig, out_dir, "validation_vs_test_learned.png")


def plot_method_win_counts(df: pd.DataFrame, out_dir: Path) -> None:
    methods = _methods_in(df)
    tasks = [
        ("after_ncc", "NCC winner", False, methods),
        ("dice_after_mean", "Dice winner", False, methods),
        ("run_seconds", "Runtime winner", True, methods),
        ("jacobian_folding_percent", "Folding winner", True, [m for m in methods if m in DENSE_METHODS]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8))
    for ax, (column, title, ascending, allowed_methods) in zip(axes.flat, tasks):
        winners: list[str] = []
        for _, group in df[df["method"].isin(allowed_methods)].groupby("pair_index"):
            group = group.dropna(subset=[column])
            if group.empty:
                continue
            winners.append(group.sort_values(column, ascending=ascending).iloc[0]["method"])
        counts = pd.Series(winners).value_counts().reindex(allowed_methods, fill_value=0)
        ax.bar(
            [METHOD_LABELS[m] for m in counts.index],
            counts.to_numpy(dtype=float),
            color=[METHOD_COLORS[m] for m in counts.index],
        )
        ax.set_title(title)
        ax.set_ylabel("Number of test pairs")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)
    _save(fig, out_dir, "method_win_counts.png")


def plot_ncc_dice_scatter(df: pd.DataFrame, out_dir: Path) -> None:
    methods = _methods_in(df)
    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    for method in methods:
        method_df = df[df["method"] == method]
        ax.scatter(
            method_df["after_ncc"],
            method_df["dice_after_mean"],
            s=28,
            alpha=0.38,
            color=METHOD_COLORS[method],
            label=f"{METHOD_LABELS[method]} pairs",
        )
        mean_x = method_df["after_ncc"].mean()
        mean_y = method_df["dice_after_mean"].mean()
        ax.scatter(
            [mean_x],
            [mean_y],
            s=210,
            color=METHOD_COLORS[method],
            edgecolor="black",
            linewidth=0.9,
        )
        ax.annotate(METHOD_LABELS[method], (mean_x, mean_y), xytext=(7, 4), textcoords="offset points")
    ax.set_title("NCC vs Dice")
    ax.set_xlabel("NCC")
    ax.set_ylabel("Dice")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")
    _save(fig, out_dir, "ncc_vs_dice_scatter.png")


def generate(args: argparse.Namespace) -> None:
    benchmark_dir = Path(args.benchmark_dir)
    out_dir = Path(args.out_dir) if args.out_dir else Path("visualize") / benchmark_dir.name
    df = _load_results(benchmark_dir / "benchmark_results.csv")
    pair_index = _select_pair(df, args.pair_index)

    plot_summary_metrics(df, out_dir)
    plot_metric_distributions(df, out_dir)
    plot_runtime_quality(df, out_dir)
    plot_label_heatmap(df, benchmark_dir, out_dir)
    plot_qualitative_gallery(df, out_dir, pair_index)
    plot_deformation_diagnostics(df, benchmark_dir, out_dir, pair_index)
    plot_training_curves(benchmark_dir, out_dir)
    plot_training_summary(benchmark_dir, out_dir)
    plot_validation_vs_test(benchmark_dir, df, out_dir)
    plot_method_win_counts(df, out_dir)
    plot_ncc_dice_scatter(df, out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate OASIS-1 3D benchmark figures.")
    parser.add_argument(
        "--benchmark-dir",
        default="outputs/benchmark/oasis1_3d_functional_20260527_025659",
    )
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--pair-index", type=int, default=None)
    args = parser.parse_args()
    generate(args)


if __name__ == "__main__":
    main()

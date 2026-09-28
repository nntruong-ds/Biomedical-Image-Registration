from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import map_coordinates

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_oasis3d_visualizations import read_volume
from src.methods.metaheuristic.pso import (
    _bounds,
    _identity_vector,
    _score,
    _vector_to_params,
    _warp_transform_3d,
)
from src.utils.metrics import normalize_image


METHODS = ("classical", "pso", "voxelmorph", "transmorph")
DENSE_METHODS = ("classical", "voxelmorph", "transmorph")
METHOD_LABELS = {
    "classical": "Classical",
    "pso": "PSO",
    "voxelmorph": "VoxelMorph",
    "transmorph": "TransMorph",
}


def _success_mask(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(("true", "1", "yes"))


def load_results(benchmark_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(benchmark_dir / "benchmark_results.csv")
    if "success" in df.columns:
        df = df[_success_mask(df["success"])].copy()
    for column in (
        "run_seconds",
        "after_mse",
        "delta_mse",
        "after_ncc",
        "delta_ncc",
        "dice_after_mean",
        "dice_delta_mean",
    ):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df["pair_index"] = pd.to_numeric(df["pair_index"]).astype(int)
    return df


def row_for(df: pd.DataFrame, pair_index: int, method: str) -> pd.Series:
    rows = df[(df["pair_index"] == pair_index) & (df["method"] == method)]
    if rows.empty:
        raise ValueError(f"No benchmark row for pair={pair_index}, method={method}")
    return rows.iloc[0]


def mid_slice(volume: np.ndarray, axis: int = 0, index: int | None = None) -> np.ndarray:
    if index is None:
        index = volume.shape[axis] // 2
    if axis == 0:
        return volume[index, :, :]
    if axis == 1:
        return volume[:, index, :]
    return volume[:, :, index]


def as_display(image: np.ndarray) -> np.ndarray:
    return normalize_image(np.nan_to_num(image.astype(np.float32), nan=0.0))


def overlay_rgb(fixed: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    fixed_s = as_display(fixed)
    candidate_s = as_display(candidate)
    rgb = np.zeros(fixed_s.shape + (3,), dtype=np.float32)
    rgb[..., 0] = fixed_s
    rgb[..., 1] = candidate_s
    rgb[..., 2] = 0.25 * fixed_s
    return np.clip(rgb, 0.0, 1.0)


def dense_field_to_zyx(field: np.ndarray, method: str) -> np.ndarray:
    return field[..., [2, 1, 0]] if method == "classical" else field


def coords_3d(shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return np.meshgrid(
        np.arange(shape[0], dtype=np.float32),
        np.arange(shape[1], dtype=np.float32),
        np.arange(shape[2], dtype=np.float32),
        indexing="ij",
    )


def warp_dense_scaled(
    moving: np.ndarray, field: np.ndarray, method: str, scale: float
) -> np.ndarray:
    field_zyx = dense_field_to_zyx(field, method)
    zz, yy, xx = coords_3d(moving.shape)
    warped = map_coordinates(
        moving.astype(np.float32),
        [
            zz + scale * field_zyx[..., 0],
            yy + scale * field_zyx[..., 1],
            xx + scale * field_zyx[..., 2],
        ],
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    return warped.astype(np.float32)


def jacobian_det_3d(field_zyx: np.ndarray) -> np.ndarray:
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


def fig_to_frame(fig: plt.Figure, dpi: int = 105) -> Image.Image:
    fig.tight_layout(pad=0.8)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def pad_frames(frames: list[Image.Image]) -> list[Image.Image]:
    max_width = max(frame.width for frame in frames)
    max_height = max(frame.height for frame in frames)
    padded = []
    for frame in frames:
        if frame.size == (max_width, max_height):
            padded.append(frame)
            continue
        canvas = Image.new("RGB", (max_width, max_height), "white")
        offset = ((max_width - frame.width) // 2, (max_height - frame.height) // 2)
        canvas.paste(frame, offset)
        padded.append(canvas)
    return padded


def save_gif(frames: list[Image.Image], path: Path, duration: int = 120) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = pad_frames(frames)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        optimize=True,
    )
    print(f"Saved {len(frames)} frames: {path}")
    return path


def registration_frame(
    fixed_slice: np.ndarray,
    candidate_slice: np.ndarray,
    candidate_title: str,
    fixed_title: str = "Fixed",
    dpi: int = 105,
) -> Image.Image:
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.0))
    panels = [
        (fixed_slice, fixed_title, "gray"),
        (candidate_slice, candidate_title, "gray"),
        (overlay_rgb(fixed_slice, candidate_slice), "Overlay", None),
    ]
    for ax, (image, title, cmap) in zip(axes, panels):
        if cmap is None:
            ax.imshow(image)
        else:
            ax.imshow(image, cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    return fig_to_frame(fig, dpi=dpi)


def selected_cases(df: pd.DataFrame) -> list[dict[str, int | str]]:
    pair_mean = df.groupby("pair_index").mean(numeric_only=True)
    return [
        {"name": "best_delta_ncc", "pair": int(pair_mean["delta_ncc"].idxmax())},
        {
            "name": "median_after_ncc",
            "pair": int((pair_mean["after_ncc"] - pair_mean["after_ncc"].median()).abs().idxmin()),
        },
        {"name": "worst_delta_ncc", "pair": int(pair_mean["delta_ncc"].idxmin())},
    ]


def generate_moving_registered_fade(
    df: pd.DataFrame, gif_dir: Path, methods: list[str], frames_per_gif: int
) -> None:
    out = gif_dir / "01_moving_to_registered_fade"
    for case in selected_cases(df):
        pair_index = int(case["pair"])
        base = row_for(df, pair_index, methods[0])
        fixed = read_volume(base["fixed"])
        moving = read_volume(base["moving"])
        fixed_slice = mid_slice(fixed, axis=0)
        moving_slice = mid_slice(moving, axis=0)
        for method in methods:
            row = row_for(df, pair_index, method)
            registered_slice = mid_slice(read_volume(row["registered"]), axis=0)
            frames = []
            for frame_idx in range(frames_per_gif):
                before = frame_idx % 2 == 0
                candidate = moving_slice if before else registered_slice
                state = "Before" if before else "After"
                frames.append(
                    registration_frame(
                        fixed_slice,
                        candidate,
                        f"{METHOD_LABELS[method]} {state}",
                    )
                )
            save_gif(
                frames,
                out / f"{case['name']}_pair_{pair_index:03d}_{method}_fade.gif",
                duration=520,
            )


def generate_axial_sweep(df: pd.DataFrame, gif_dir: Path, methods: list[str], pair_index: int) -> None:
    out = gif_dir / "02_axial_slice_sweep"
    base = row_for(df, pair_index, methods[0])
    fixed = read_volume(base["fixed"])
    slice_indices = np.linspace(8, fixed.shape[0] - 9, 22).astype(int)
    for method in methods:
        row = row_for(df, pair_index, method)
        registered = read_volume(row["registered"])
        frames = []
        for z in slice_indices:
            fixed_slice = mid_slice(fixed, axis=0, index=int(z))
            registered_slice = mid_slice(registered, axis=0, index=int(z))
            frames.append(
                registration_frame(
                    fixed_slice,
                    registered_slice,
                    f"{METHOD_LABELS[method]} z={int(z)}",
                    fixed_title=f"Fixed z={int(z)}",
                )
            )
        save_gif(frames, out / f"pair_{pair_index:03d}_{method}_axial_sweep.gif", duration=240)


def generate_deformation_sweep(benchmark_dir: Path, gif_dir: Path, pair_index: int) -> None:
    out = gif_dir / "03_final_field_scale"
    for method in DENSE_METHODS:
        field = np.load(
            benchmark_dir / "test_runs" / method / f"pair_{pair_index:03d}" / "deformation_field.npy"
        )
        field_zyx = dense_field_to_zyx(field, method)
        magnitude = np.linalg.norm(field_zyx, axis=-1)
        jacobian = jacobian_det_3d(field_zyx)
        folding = jacobian <= 0.0
        slice_indices = np.linspace(8, magnitude.shape[0] - 9, 24).astype(int)
        frames = []
        for z in slice_indices:
            fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.2))
            panels = [
                (magnitude[int(z)], "Disp.", "viridis", None),
                (jacobian[int(z)], "Jacobian", "coolwarm", None),
                (folding[int(z)].astype(np.float32), "Folding", "Reds", (0.0, 1.0)),
            ]
            axes[0].set_ylabel(f"{METHOD_LABELS[method]}\nz={int(z)}", fontsize=9)
            for ax, (image, title, cmap, limits) in zip(axes, panels):
                if limits is None:
                    shown = ax.imshow(image, cmap=cmap)
                else:
                    shown = ax.imshow(image, cmap=cmap, vmin=limits[0], vmax=limits[1])
                ax.set_title(title, fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
                fig.colorbar(shown, ax=ax, fraction=0.035, pad=0.02)
            frames.append(fig_to_frame(fig, dpi=105))
        save_gif(frames, out / f"pair_{pair_index:03d}_{method}_field_scale.gif", duration=210)


def label_edges(label_slice: np.ndarray) -> np.ndarray:
    labels = label_slice.astype(np.int32)
    valid = labels > 0
    edges = np.zeros(labels.shape, dtype=bool)
    edges[:-1, :] |= (labels[:-1, :] != labels[1:, :]) & (valid[:-1, :] | valid[1:, :])
    edges[:, :-1] |= (labels[:, :-1] != labels[:, 1:]) & (valid[:, :-1] | valid[:, 1:])
    return edges


def edge_rgba(edges: np.ndarray, color: tuple[float, float, float], alpha: float) -> np.ndarray:
    rgba = np.zeros(edges.shape + (4,), dtype=np.float32)
    rgba[..., 0] = color[0]
    rgba[..., 1] = color[1]
    rgba[..., 2] = color[2]
    rgba[..., 3] = edges.astype(np.float32) * alpha
    return rgba


def label_contour_frame(
    fixed_slice: np.ndarray,
    fixed_edges: np.ndarray,
    moving_edges: np.ndarray,
    registered_edges: np.ndarray,
    method: str,
    z: int,
) -> Image.Image:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.55))
    background = as_display(fixed_slice) * 0.82
    panels = [
        ("Fixed + Moving", moving_edges, (1.0, 0.92, 0.0)),
        ("Fixed + Registered", registered_edges, (0.0, 0.95, 1.0)),
    ]
    axes[0].set_ylabel(f"{METHOD_LABELS[method]}\nz={z}", fontsize=9)
    for ax, (title, candidate_edges, color) in zip(axes, panels):
        ax.imshow(background, cmap="gray", vmin=0.0, vmax=1.0)
        ax.imshow(edge_rgba(fixed_edges, (1.0, 0.05, 0.05), 0.95))
        ax.imshow(edge_rgba(candidate_edges, color, 0.95))
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    return fig_to_frame(fig, dpi=115)


def _label_path(image_path: str, prefix: str) -> str:
    return image_path.replace(f"/{prefix}_", f"/{prefix}_label_")


def generate_label_contours(
    df: pd.DataFrame, benchmark_dir: Path, gif_dir: Path, methods: list[str], pair_index: int
) -> None:
    out = gif_dir / "04_label_contours"
    base = row_for(df, pair_index, methods[0])
    fixed = read_volume(base["fixed"])
    fixed_label = read_volume(_label_path(str(base["fixed"]), "fixed"))
    moving_label = read_volume(_label_path(str(base["moving"]), "moving"))
    slice_indices = np.linspace(8, fixed.shape[0] - 9, 18).astype(int)
    fixed_edge_slices = {
        int(z): label_edges(mid_slice(fixed_label, axis=0, index=int(z))) for z in slice_indices
    }
    moving_edge_slices = {
        int(z): label_edges(mid_slice(moving_label, axis=0, index=int(z))) for z in slice_indices
    }
    for method in methods:
        registered_label = read_volume(
            benchmark_dir
            / "test_runs"
            / method
            / f"pair_{pair_index:03d}"
            / "registered_label.nii.gz"
        )
        frames = []
        for z in slice_indices:
            z_int = int(z)
            frames.append(
                label_contour_frame(
                    mid_slice(fixed, axis=0, index=z_int),
                    fixed_edge_slices[z_int],
                    moving_edge_slices[z_int],
                    label_edges(mid_slice(registered_label, axis=0, index=z_int)),
                    method,
                    z_int,
                )
            )
        save_gif(frames, out / f"pair_{pair_index:03d}_{method}_label_contours.gif", duration=170)


def generate_pso_progress(
    df: pd.DataFrame,
    gif_dir: Path,
    pair_index: int,
    progress_frames: int,
    transform: str = "affine",
    particles: int = 12,
) -> list[np.ndarray]:
    out = gif_dir / "05_true_iterations"
    base = row_for(df, pair_index, "pso")
    fixed = normalize_image(read_volume(base["fixed"]))
    moving = normalize_image(read_volume(base["moving"]))
    fixed_slice = mid_slice(fixed, axis=0)
    lo, hi = _bounds(fixed.shape, transform)
    span = hi - lo
    rng = np.random.default_rng(123 + pair_index)
    positions = rng.uniform(lo, hi, size=(particles, lo.size)).astype(np.float32)
    velocities = rng.uniform(-0.05 * span, 0.05 * span, size=positions.shape).astype(np.float32)
    positions[0] = _identity_vector(fixed.ndim, transform)
    velocities[0] = 0.0
    personal_best = positions.copy()
    personal_scores = np.full(particles, np.inf, dtype=np.float32)
    global_best = positions[0].copy()
    global_score = float("inf")
    frames = []
    progress_volumes = []
    for iteration in range(progress_frames):
        for idx in range(particles):
            score = _score(fixed, moving, positions[idx], transform, "ncc")
            if score < personal_scores[idx]:
                personal_scores[idx] = score
                personal_best[idx] = positions[idx].copy()
            if score < global_score:
                global_score = float(score)
                global_best = positions[idx].copy()
        params = _vector_to_params(global_best, transform, fixed.ndim)
        registered = normalize_image(_warp_transform_3d(moving, params, output_shape=fixed.shape))
        progress_volumes.append(registered)
        frames.append(
            registration_frame(
                fixed_slice,
                mid_slice(registered, axis=0),
                f"PSO {transform} iter {iteration + 1:02d}/{progress_frames}",
            )
        )
        r1 = rng.random(size=positions.shape, dtype=np.float32)
        r2 = rng.random(size=positions.shape, dtype=np.float32)
        velocities = (
            0.72 * velocities
            + 1.45 * r1 * (personal_best - positions)
            + 1.45 * r2 * (global_best[None, :] - positions)
        )
        positions = np.clip(positions + velocities, lo, hi)
    save_gif(frames, out / f"pair_{pair_index:03d}_pso_{transform}_true_iterations.gif", duration=180)
    return progress_volumes


def generate_progress_gifs(
    df: pd.DataFrame,
    benchmark_dir: Path,
    gif_dir: Path,
    methods: list[str],
    pair_index: int,
    progress_frames: int,
    pso_transform: str,
    pso_particles: int,
) -> None:
    _ = benchmark_dir, methods
    out = gif_dir / "05_true_iterations"
    stale_patterns = (
        "*_classical_increasing_iterations.gif",
        "*_voxelmorph_progress_proxy.gif",
        "*_transmorph_progress_proxy.gif",
        "*_all_methods_aligned_progress.gif",
        "*_pso_true_iterations.gif",
        "*_pso_*_true_iterations.gif",
    )
    for pattern in stale_patterns:
        for path in out.glob(pattern):
            path.unlink()

    pso_df = df[df["method"] == "pso"].dropna(subset=["dice_delta_mean"])
    extra_pair = int(
        pso_df[pso_df["pair_index"] != pair_index]
        .sort_values("dice_delta_mean", ascending=False)
        .iloc[0]["pair_index"]
    )
    for pso_pair in (pair_index, extra_pair):
        generate_pso_progress(
            df,
            gif_dir,
            pso_pair,
            progress_frames,
            transform=pso_transform,
            particles=pso_particles,
        )


def generate(args: argparse.Namespace) -> None:
    benchmark_dir = Path(args.benchmark_dir)
    gif_dir = Path(args.gif_dir)
    df = load_results(benchmark_dir)
    methods = [method for method in METHODS if method in set(df["method"])]
    pair_mean = df.groupby("pair_index").mean(numeric_only=True)
    demo_pair = int((pair_mean["after_ncc"] - pair_mean["after_ncc"].median()).abs().idxmin())

    generate_moving_registered_fade(df, gif_dir, methods, frames_per_gif=16)
    generate_axial_sweep(df, gif_dir, methods, demo_pair)
    generate_deformation_sweep(benchmark_dir, gif_dir, demo_pair)
    generate_label_contours(df, benchmark_dir, gif_dir, methods, demo_pair)
    generate_progress_gifs(
        df,
        benchmark_dir,
        gif_dir,
        methods,
        demo_pair,
        progress_frames=args.pso_visual_frames,
        pso_transform=args.pso_visual_transform,
        pso_particles=args.pso_visual_particles,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate OASIS-1 3D benchmark GIFs.")
    parser.add_argument(
        "--benchmark-dir",
        default="outputs/benchmark/oasis1_3d_functional_20260527_025659",
    )
    parser.add_argument("--gif-dir", default="visualize/oasis3d_benchmark_gifs")
    parser.add_argument("--pso-visual-transform", choices=["rigid", "affine"], default="affine")
    parser.add_argument("--pso-visual-frames", type=int, default=40)
    parser.add_argument("--pso-visual-particles", type=int, default=12)
    args = parser.parse_args()
    generate(args)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import list_fixed_image_paths
from src.methods.classical.register import run_classical
from src.methods.deep_common import train_unsupervised
from src.methods.metaheuristic.pso import PSOConfig, run_pso
from src.methods.transmorph.infer import run_transmorph
from src.methods.transmorph.model import build_transmorph_model
from src.methods.voxelmorph.infer import run_voxelmorph
from src.methods.voxelmorph.model import build_voxelmorph_model
from src.utils.io import ensure_dir, read_image, save_json
from src.utils.seed import set_seed


METHODS = ("classical", "pso", "voxelmorph", "transmorph")


def _pairs(dataset_root: Path, num_pairs: int | None) -> list[tuple[int, Path, Path]]:
    fixed_paths = list_fixed_image_paths(dataset_root)
    if num_pairs is not None:
        fixed_paths = fixed_paths[:num_pairs]
    pairs: list[tuple[int, Path, Path]] = []
    for fixed in fixed_paths:
        suffix = fixed.name.removeprefix("fixed_")
        moving = dataset_root / f"moving_{suffix}"
        if not moving.exists():
            raise FileNotFoundError(f"Missing moving image for {fixed}: {moving}")
        index = int(suffix.split(".")[0])
        pairs.append((index, fixed, moving))
    if not pairs:
        raise FileNotFoundError(
            f"No fixed_*.nii.gz / moving_*.nii.gz pairs found under {dataset_root}"
        )
    return pairs


def _deep_config(
    dataset_root: Path,
    output_dir: Path,
    model_name: str,
    epochs: int,
    batch_size: int,
    seed: int,
    device: str,
    dim: int,
) -> dict[str, Any]:
    if model_name == "voxelmorph":
        model = {"dim": dim, "in_channels": 2, "base_channels": 8 if dim == 2 else 4}
    elif model_name == "transmorph":
        model = {
            "dim": dim,
            "in_channels": 2,
            "embed_dim": 32 if dim == 2 else 24,
            "patch_size": 8,
            "depth": 1,
            "num_heads": 4,
            "decoder_channels": 16 if dim == 2 else 12,
        }
    else:
        raise ValueError(f"Unknown deep model: {model_name}")
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
            "num_workers": 0,
        },
    }


def _row(
    method: str,
    pair_index: int,
    fixed: Path,
    moving: Path,
    log: dict[str, Any],
    elapsed: float,
) -> dict[str, Any]:
    metrics = log["metrics"]
    before = metrics["before"]
    after = metrics["after"]
    return {
        "method": method,
        "pair_index": pair_index,
        "fixed": str(fixed),
        "moving": str(moving),
        "run_seconds": elapsed,
        "before_mse": before["mse"],
        "after_mse": after["mse"],
        "delta_mse": before["mse"] - after["mse"],
        "before_ncc": before["ncc"],
        "after_ncc": after["ncc"],
        "delta_ncc": after["ncc"] - before["ncc"],
        "registered": log["outputs"]["registered"],
        "overlay": log["outputs"]["overlay"],
        "log": str(Path(log["outputs"]["registered"]).parent / "log.json"),
    }


def _summarize(rows: list[dict[str, Any]], training: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {"methods": {}, "training": training}
    for method in METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        if not method_rows:
            continue
        entry: dict[str, Any] = {"num_pairs": len(method_rows)}
        for key in ("run_seconds", "after_mse", "delta_mse", "after_ncc", "delta_ncc"):
            values = [float(row[key]) for row in method_rows]
            entry[f"mean_{key}"] = statistics.fmean(values)
            entry[f"std_{key}"] = statistics.pstdev(values) if len(values) > 1 else 0.0
        summary["methods"][method] = entry
    return summary


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    fields = [
        "method",
        "pair_index",
        "fixed",
        "moving",
        "run_seconds",
        "before_mse",
        "after_mse",
        "delta_mse",
        "before_ncc",
        "after_ncc",
        "delta_ncc",
        "registered",
        "overlay",
        "log",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Registration Benchmark",
        "",
        "This is a small functional benchmark over prepared NIfTI pairs. It is not a final scientific evaluation.",
        "",
        "| Method | Pairs | Mean sec/pair | Mean after MSE | Mean delta MSE | Mean after NCC | Mean delta NCC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        entry = summary["methods"].get(method)
        if not entry:
            continue
        lines.append(
            "| {method} | {pairs} | {sec:.4f} | {mse:.6f} | {dmse:.6f} | {ncc:.6f} | {dncc:.6f} |".format(
                method=method,
                pairs=entry["num_pairs"],
                sec=entry["mean_run_seconds"],
                mse=entry["mean_after_mse"],
                dmse=entry["mean_delta_mse"],
                ncc=entry["mean_after_ncc"],
                dncc=entry["mean_delta_ncc"],
            )
        )
    lines.append("")
    lines.append("Training outputs:")
    for name, item in summary.get("training", {}).items():
        lines.append(
            f"- {name}: seconds={item.get('elapsed_seconds', 0.0):.4f}, "
            f"best_loss={item.get('best_loss', 0.0):.6f}, checkpoint={item.get('checkpoint', '')}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    dataset_root = Path(args.dataset_root)
    out = ensure_dir(args.out)
    pairs = _pairs(dataset_root, args.num_pairs)
    dim = int(read_image(pairs[0][1]).ndim)
    if dim not in (2, 3):
        raise ValueError(f"Benchmark expects 2D or 3D images, got {dim}D.")
    pso_transform = args.pso_transform
    rows: list[dict[str, Any]] = []

    training: dict[str, Any] = {}
    if not args.skip_training:
        vox_out = out / "training" / "voxelmorph"
        vox_cfg = _deep_config(
            dataset_root,
            vox_out,
            "voxelmorph",
            args.deep_epochs,
            args.batch_size,
            args.seed,
            args.device,
            dim,
        )
        print(
            f"Training VoxelMorph on {len(pairs)} pairs for {args.deep_epochs} epochs...",
            flush=True,
        )
        vox_log = train_unsupervised(
            build_voxelmorph_model(vox_cfg), vox_cfg, "voxelmorph", vox_out
        )
        training["voxelmorph"] = {
            "elapsed_seconds": vox_log["elapsed_seconds"],
            "best_loss": vox_log["best_loss"],
            "checkpoint": str(vox_out / "best.pt"),
        }
        print(
            f"VoxelMorph training complete: best_loss={vox_log['best_loss']:.6f}",
            flush=True,
        )

        trans_out = out / "training" / "transmorph"
        trans_cfg = _deep_config(
            dataset_root,
            trans_out,
            "transmorph",
            args.deep_epochs,
            args.batch_size,
            args.seed,
            args.device,
            dim,
        )
        print(
            f"Training TransMorph on {len(pairs)} pairs for {args.deep_epochs} epochs...",
            flush=True,
        )
        trans_log = train_unsupervised(
            build_transmorph_model(trans_cfg), trans_cfg, "transmorph", trans_out
        )
        training["transmorph"] = {
            "elapsed_seconds": trans_log["elapsed_seconds"],
            "best_loss": trans_log["best_loss"],
            "checkpoint": str(trans_out / "best.pt"),
        }
        print(
            f"TransMorph training complete: best_loss={trans_log['best_loss']:.6f}",
            flush=True,
        )
    else:
        if args.voxelmorph_checkpoint is None or args.transmorph_checkpoint is None:
            raise ValueError(
                "--skip-training requires --voxelmorph-checkpoint and --transmorph-checkpoint"
            )
        training["voxelmorph"] = {"checkpoint": args.voxelmorph_checkpoint}
        training["transmorph"] = {"checkpoint": args.transmorph_checkpoint}

    for pair_number, (pair_index, fixed, moving) in enumerate(pairs, start=1):
        pair_name = f"pair_{pair_index:03d}"
        print(f"[{pair_number}/{len(pairs)}] Benchmarking {pair_name}", flush=True)
        started = time.perf_counter()
        log = run_classical(
            fixed,
            moving,
            out / "runs" / "classical" / pair_name,
            iterations=args.classical_iterations,
            smoothing_sigma=args.classical_smoothing_sigma,
        )
        rows.append(_row("classical", pair_index, fixed, moving, log, time.perf_counter() - started))
        print("  classical done", flush=True)

        started = time.perf_counter()
        log = run_pso(
            fixed,
            moving,
            out / "runs" / "pso" / pair_name,
            PSOConfig(
                particles=args.pso_particles,
                iterations=args.pso_iterations,
                metric=args.pso_metric,
                transform=pso_transform,
                seed=args.seed + pair_index,
            ),
        )
        rows.append(_row("pso", pair_index, fixed, moving, log, time.perf_counter() - started))
        print("  pso done", flush=True)

        started = time.perf_counter()
        log = run_voxelmorph(
            fixed,
            moving,
            training["voxelmorph"]["checkpoint"],
            out / "runs" / "voxelmorph" / pair_name,
        )
        rows.append(_row("voxelmorph", pair_index, fixed, moving, log, time.perf_counter() - started))
        print("  voxelmorph done", flush=True)

        started = time.perf_counter()
        log = run_transmorph(
            fixed,
            moving,
            training["transmorph"]["checkpoint"],
            out / "runs" / "transmorph" / pair_name,
        )
        rows.append(_row("transmorph", pair_index, fixed, moving, log, time.perf_counter() - started))
        print("  transmorph done", flush=True)

    _write_csv(rows, out / "benchmark_results.csv")
    save_json({"rows": rows}, out / "benchmark_results.json")
    summary = _summarize(rows, training)
    save_json(summary, out / "benchmark_summary.json")
    _write_markdown(summary, out / "benchmark_summary.md")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and benchmark all 4 methods.")
    parser.add_argument("--dataset-root", default="data/oasis_2d")
    parser.add_argument("--out", default="outputs/benchmark/oasis")
    parser.add_argument("--num-pairs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--voxelmorph-checkpoint", default=None)
    parser.add_argument("--transmorph-checkpoint", default=None)
    parser.add_argument("--deep-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--device",
        default="auto",
        help="Deep learning device: auto, cpu, cuda, or cuda:<index>.",
    )
    parser.add_argument("--classical-iterations", type=int, default=30)
    parser.add_argument("--classical-smoothing-sigma", type=float, default=1.3)
    parser.add_argument("--pso-particles", type=int, default=16)
    parser.add_argument("--pso-iterations", type=int, default=25)
    parser.add_argument("--pso-metric", choices=["ncc", "mse"], default="ncc")
    parser.add_argument("--pso-transform", choices=["affine", "rigid"], default="affine")
    args = parser.parse_args()

    summary = run_benchmark(args)
    print(f"Benchmark complete. Summary: {Path(args.out) / 'benchmark_summary.md'}")
    for method in METHODS:
        item = summary["methods"].get(method)
        if item:
            print(
                f"{method}: mean_after_mse={item['mean_after_mse']:.6f} "
                f"mean_after_ncc={item['mean_after_ncc']:.6f} "
                f"mean_sec={item['mean_run_seconds']:.4f}"
            )


if __name__ == "__main__":
    main()

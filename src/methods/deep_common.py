from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import SyntheticPairDataset
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
from src.utils.seed import set_seed
from src.utils.visualization import save_overlay


class SpatialTransformer(nn.Module):
    def forward(self, image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        if image.ndim not in (4, 5) or flow.ndim != image.ndim:
            raise ValueError("SpatialTransformer expects BCHW or BCDHW tensors.")
        spatial_dims = image.ndim - 2
        if image.shape[0] != flow.shape[0] or flow.shape[1] != spatial_dims:
            raise ValueError(
                f"Expected flow shape (B, {spatial_dims}, ...), got {tuple(flow.shape)}"
            )
        device = image.device
        dtype = image.dtype
        batch = image.shape[0]
        spatial_shape = image.shape[2:]
        axes = [
            torch.arange(size, device=device, dtype=dtype)
            for size in spatial_shape
        ]
        meshes = torch.meshgrid(*axes, indexing="ij")
        samples = [
            mesh.unsqueeze(0).expand(batch, *spatial_shape) + flow[:, dim]
            for dim, mesh in enumerate(meshes)
        ]
        normalized = [
            2.0 * samples[dim] / max(spatial_shape[dim] - 1, 1) - 1.0
            for dim in range(spatial_dims)
        ]
        if spatial_dims == 2:
            grid = torch.stack([normalized[1], normalized[0]], dim=-1)
        else:
            grid = torch.stack([normalized[2], normalized[1], normalized[0]], dim=-1)
        return F.grid_sample(
            image,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )


def smoothness_loss(flow: torch.Tensor) -> torch.Tensor:
    losses = []
    for dim in range(2, flow.ndim):
        left = [slice(None)] * flow.ndim
        right = [slice(None)] * flow.ndim
        left[dim] = slice(1, None)
        right[dim] = slice(None, -1)
        losses.append(torch.mean((flow[tuple(left)] - flow[tuple(right)]) ** 2))
    return sum(losses)


def ncc_loss(fixed: torch.Tensor, moving: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    dims = tuple(range(1, fixed.ndim))
    fixed_centered = fixed - fixed.mean(dim=dims, keepdim=True)
    moving_centered = moving - moving.mean(dim=dims, keepdim=True)
    numerator = torch.sum(fixed_centered * moving_centered, dim=dims)
    denom = torch.sqrt(
        torch.sum(fixed_centered**2, dim=dims)
        * torch.sum(moving_centered**2, dim=dims)
        + eps
    )
    return torch.mean(1.0 - numerator / denom)


def image_loss(fixed: torch.Tensor, registered: torch.Tensor, name: str) -> torch.Tensor:
    if name == "mse":
        return F.mse_loss(registered, fixed)
    if name == "ncc":
        return ncc_loss(fixed, registered)
    raise ValueError("image_loss must be 'mse' or 'ncc'.")


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return data


def torch_load(path: str | Path, map_location: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def resolve_device(requested: str | None = None) -> torch.device:
    choice = (requested or "auto").lower()
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cpu":
        return torch.device("cpu")
    if choice.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested device '{requested}', but PyTorch cannot access CUDA."
            )
        return torch.device(choice)
    raise ValueError("Device must be 'auto', 'cpu', 'cuda', or 'cuda:<index>'.")


def train_unsupervised(
    model: nn.Module,
    config: dict[str, Any],
    model_name: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    seed = int(config.get("seed", 7))
    set_seed(seed)
    device = resolve_device(str(config.get("device", "auto")))
    data_root = Path(config.get("data", {}).get("root", "data/synthetic"))
    training_cfg = config.get("training", {})
    epochs = int(training_cfg.get("epochs", 2))
    batch_size = int(training_cfg.get("batch_size", 4))
    learning_rate = float(training_cfg.get("learning_rate", 1e-3))
    smooth_weight = float(training_cfg.get("smooth_weight", 0.05))
    loss_name = str(training_cfg.get("image_loss", "mse"))
    num_workers = int(training_cfg.get("num_workers", 0))

    if epochs < 1:
        raise ValueError("training.epochs must be >= 1")

    dataset = SyntheticPairDataset(data_root)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
    )

    out = ensure_dir(output_dir)
    model = model.to(device)
    transformer = SpatialTransformer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_loss = float("inf")
    history: list[dict[str, float | int]] = []
    start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_total = 0.0
        epoch_sim = 0.0
        epoch_smooth = 0.0
        count = 0
        for batch in loader:
            fixed = batch["fixed"].to(device=device, dtype=torch.float32)
            moving = batch["moving"].to(device=device, dtype=torch.float32)
            flow = model(fixed, moving)
            registered = transformer(moving, flow)
            sim = image_loss(fixed, registered, loss_name)
            smooth = smoothness_loss(flow)
            loss = sim + smooth_weight * smooth

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_count = fixed.shape[0]
            count += batch_count
            epoch_total += float(loss.detach().cpu()) * batch_count
            epoch_sim += float(sim.detach().cpu()) * batch_count
            epoch_smooth += float(smooth.detach().cpu()) * batch_count

        row = {
            "epoch": epoch,
            "loss": epoch_total / count,
            "image_loss": epoch_sim / count,
            "smoothness_loss": epoch_smooth / count,
        }
        history.append(row)
        if row["loss"] < best_loss:
            best_loss = float(row["loss"])
            torch.save(
                {
                    "model_name": model_name,
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "best_loss": best_loss,
                },
                out / "best.pt",
            )

    torch.save(
        {
            "model_name": model_name,
            "model_state_dict": model.state_dict(),
            "config": config,
            "epoch": epochs,
            "best_loss": best_loss,
        },
        out / "last.pt",
    )
    log = {
        "model": model_name,
        "device": str(device),
        "data_root": str(data_root),
        "num_pairs": len(dataset),
        "elapsed_seconds": time.time() - start,
        "best_loss": best_loss,
        "history": history,
        "outputs": {
            "best_checkpoint": str(out / "best.pt"),
            "last_checkpoint": str(out / "last.pt"),
        },
    }
    save_json(log, out / "training_log.json")
    with (out / "resolved_config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, sort_keys=True)
        file.write("\n")
    return log


def run_deep_inference(
    model_builder: Callable[[dict[str, Any]], nn.Module],
    model_name: str,
    fixed_path: str | Path,
    moving_path: str | Path,
    checkpoint_path: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}. Train first with the matching train module."
        )
    start = time.time()
    out = ensure_dir(out_dir)
    payload = torch_load(checkpoint, torch.device("cpu"))
    config = payload.get("config", {})
    device = resolve_device(str(config.get("device", "auto")))
    payload = torch_load(checkpoint, device)
    model = model_builder(config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    fixed_np = normalize_image(read_image(fixed_path))
    moving_np = normalize_image(read_image(moving_path))
    if fixed_np.shape != moving_np.shape:
        raise ValueError(
            f"Deep models expect equal image shapes, got {fixed_np.shape} and {moving_np.shape}."
        )
    fixed = torch.from_numpy(fixed_np[None, None]).to(device=device, dtype=torch.float32)
    moving = torch.from_numpy(moving_np[None, None]).to(device=device, dtype=torch.float32)
    transformer = SpatialTransformer().to(device)

    with torch.no_grad():
        flow = model(fixed, moving)
        registered = transformer(moving, flow)

    registered_np = normalize_image(
        registered.squeeze(0).squeeze(0).detach().cpu().numpy()
    )
    field = np.moveaxis(
        flow.squeeze(0).detach().cpu().numpy(), 0, -1
    ).astype("float32")
    reference = read_sitk_image(fixed_path)
    metrics = before_after_metrics(fixed_np, moving_np, registered_np)

    write_image(registered_np, out / "registered.nii.gz", reference=reference)
    save_array(field, out / "deformation_field.npy")
    write_vector_field(field, out / "deformation_field.nii.gz", reference=reference)
    save_overlay(fixed_np, registered_np, out / "overlay.png")
    save_json(metrics, out / "metrics.json")

    log = {
        "method": model_name,
        "checkpoint": str(checkpoint),
        "device": str(device),
        "elapsed_seconds": time.time() - start,
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

from __future__ import annotations

import argparse

from src.methods.deep_common import load_yaml, train_unsupervised
from src.methods.voxelmorph.model import build_voxelmorph_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a minimal VoxelMorph-style model.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()

    config = load_yaml(args.config)
    output_dir = config.get("output_dir", "outputs/voxelmorph")
    model = build_voxelmorph_model(config)
    log = train_unsupervised(model, config, model_name="voxelmorph", output_dir=output_dir)
    print(f"Saved best checkpoint: {log['outputs']['best_checkpoint']}")


if __name__ == "__main__":
    main()

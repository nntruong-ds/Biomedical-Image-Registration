from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.synthetic import generate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic 2D registration pairs.")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--num_pairs", type=int, required=True)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    metadata = generate_dataset(
        out_dir=args.out,
        num_pairs=args.num_pairs,
        size=args.size,
        seed=args.seed,
    )
    print(f"Generated {len(metadata)} synthetic pairs in {args.out}")


if __name__ == "__main__":
    main()

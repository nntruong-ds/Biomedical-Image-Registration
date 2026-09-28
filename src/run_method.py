from __future__ import annotations

import argparse

from src.methods.classical.register import run_classical
from src.methods.metaheuristic.pso import PSOConfig, run_pso
from src.methods.transmorph.infer import run_transmorph
from src.methods.voxelmorph.infer import run_voxelmorph


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one biomedical registration method.")
    parser.add_argument(
        "--method",
        choices=["classical", "pso", "voxelmorph", "transmorph"],
        required=True,
    )
    parser.add_argument("--fixed", required=True, help="Path to fixed 2D NIfTI image.")
    parser.add_argument("--moving", required=True, help="Path to moving 2D NIfTI image.")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint for learned methods.")

    parser.add_argument(
        "--backend",
        default="diffeomorphic_demons",
        choices=["diffeomorphic_demons", "demons", "ants_syn"],
        help="Classical backend.",
    )
    parser.add_argument("--iterations", type=int, default=None, help="Iteration count.")
    parser.add_argument("--smoothing-sigma", type=float, default=1.3)

    parser.add_argument("--particles", type=int, default=24, help="PSO particles.")
    parser.add_argument(
        "--metric", choices=["ncc", "mse"], default="ncc", help="PSO objective metric."
    )
    parser.add_argument(
        "--transform", choices=["affine", "rigid"], default="affine", help="PSO transform."
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if args.method == "classical":
        log = run_classical(
            fixed_path=args.fixed,
            moving_path=args.moving,
            out_dir=args.out,
            backend=args.backend,
            iterations=args.iterations or 50,
            smoothing_sigma=args.smoothing_sigma,
        )
    elif args.method == "pso":
        log = run_pso(
            fixed_path=args.fixed,
            moving_path=args.moving,
            out_dir=args.out,
            config=PSOConfig(
                particles=args.particles,
                iterations=args.iterations or 40,
                metric=args.metric,
                transform=args.transform,
                seed=args.seed,
            ),
        )
    elif args.method == "voxelmorph":
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required for --method voxelmorph")
        log = run_voxelmorph(args.fixed, args.moving, args.checkpoint, args.out)
    elif args.method == "transmorph":
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required for --method transmorph")
        log = run_transmorph(args.fixed, args.moving, args.checkpoint, args.out)
    else:
        raise SystemExit(f"Unsupported method: {args.method}")

    after = log["metrics"]["after"]
    print(
        f"{args.method} complete: registered={log['outputs']['registered']} "
        f"after_mse={after['mse']:.6f} after_ncc={after['ncc']:.6f}"
    )


if __name__ == "__main__":
    main()


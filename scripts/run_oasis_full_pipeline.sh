#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON:-python}"
if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT/.venv/bin/python"
fi

DEVICE="${DEVICE:-auto}"
SIZE="${SIZE:-128}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-4}"
CLASSICAL_ITERATIONS="${CLASSICAL_ITERATIONS:-50}"
PSO_PARTICLES="${PSO_PARTICLES:-24}"
PSO_ITERATIONS="${PSO_ITERATIONS:-40}"

"$PYTHON_BIN" scripts/extract_oasis_raw.py --raw data/oasis/raw --out data/oasis
"$PYTHON_BIN" scripts/prepare_oasis_2d.py \
  --root data/oasis \
  --out data/oasis_2d_all \
  --num_pairs 0 \
  --size "$SIZE"

"$PYTHON_BIN" scripts/run_benchmark.py \
  --dataset-root data/oasis_2d_all \
  --out outputs/benchmark/oasis_all \
  --device "$DEVICE" \
  --deep-epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --classical-iterations "$CLASSICAL_ITERATIONS" \
  --pso-particles "$PSO_PARTICLES" \
  --pso-iterations "$PSO_ITERATIONS"

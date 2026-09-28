#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON:-python}"
if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT/.venv/bin/python"
fi

SMOKE_ROOT="$ROOT/outputs/smoke"
DATA_DIR="$SMOKE_ROOT/data"
rm -rf "$SMOKE_ROOT"
mkdir -p "$SMOKE_ROOT"

"$PYTHON_BIN" scripts/make_synthetic_data.py --out "$DATA_DIR" --num_pairs 4 --size 64 --seed 123

"$PYTHON_BIN" -m src.run_method \
  --method classical \
  --fixed "$DATA_DIR/fixed_000.nii.gz" \
  --moving "$DATA_DIR/moving_000.nii.gz" \
  --out "$SMOKE_ROOT/classical" \
  --iterations 8

"$PYTHON_BIN" -m src.run_method \
  --method pso \
  --fixed "$DATA_DIR/fixed_000.nii.gz" \
  --moving "$DATA_DIR/moving_000.nii.gz" \
  --out "$SMOKE_ROOT/pso" \
  --particles 8 \
  --iterations 8

"$PYTHON_BIN" -m src.methods.voxelmorph.train --config configs/voxelmorph_smoke.yaml
"$PYTHON_BIN" -m src.run_method \
  --method voxelmorph \
  --fixed "$DATA_DIR/fixed_000.nii.gz" \
  --moving "$DATA_DIR/moving_000.nii.gz" \
  --checkpoint "$SMOKE_ROOT/voxelmorph/best.pt" \
  --out "$SMOKE_ROOT/voxelmorph_infer"

"$PYTHON_BIN" -m src.methods.transmorph.train --config configs/transmorph_smoke.yaml
"$PYTHON_BIN" -m src.run_method \
  --method transmorph \
  --fixed "$DATA_DIR/fixed_000.nii.gz" \
  --moving "$DATA_DIR/moving_000.nii.gz" \
  --checkpoint "$SMOKE_ROOT/transmorph/best.pt" \
  --out "$SMOKE_ROOT/transmorph_infer"

for method_dir in classical pso voxelmorph_infer transmorph_infer; do
  test -f "$SMOKE_ROOT/$method_dir/registered.nii.gz"
  test -f "$SMOKE_ROOT/$method_dir/overlay.png"
  test -f "$SMOKE_ROOT/$method_dir/log.json"
  test -f "$SMOKE_ROOT/$method_dir/metrics.json"
done

test -f "$SMOKE_ROOT/classical/deformation_field.npy"
test -f "$SMOKE_ROOT/pso/transform_params.json"
test -f "$SMOKE_ROOT/voxelmorph_infer/deformation_field.npy"
test -f "$SMOKE_ROOT/transmorph_infer/deformation_field.npy"

echo "ALL 4 METHODS SMOKE TEST PASSED"


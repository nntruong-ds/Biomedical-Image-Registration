#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
ARCHIVE="${ARCHIVE:-data/oasis/freesurfer/raw/oasis_cs_freesurfer_disc1.tar.gz}"
SUBJECTS_ROOT="${SUBJECTS_ROOT:-data/oasis/freesurfer/subjects}"
DATA_ROOT="${DATA_ROOT:-data/oasis/freesurfer_3d_smoke}"
OUT_ROOT="${OUT_ROOT:-outputs/freesurfer_smoke}"
SIZE="${SIZE:-32}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -d "$SUBJECTS_ROOT/disc1" ]]; then
  if [[ ! -f "$ARCHIVE" ]]; then
    echo "Need either extracted disc1 at $SUBJECTS_ROOT/disc1 or archive $ARCHIVE" >&2
    exit 1
  fi
  "$PYTHON_BIN" scripts/extract_freesurfer_archive.py \
    --archive "$ARCHIVE" \
    --out "$SUBJECTS_ROOT" \
    --delete-archive
fi

"$PYTHON_BIN" scripts/prepare_freesurfer_3d.py \
  --subjects-root "$SUBJECTS_ROOT" \
  --out "$DATA_ROOT" \
  --num_pairs 3 \
  --size "$SIZE"

rm -rf "$OUT_ROOT/classical" "$OUT_ROOT/pso" "$OUT_ROOT/voxelmorph" "$OUT_ROOT/transmorph"

"$PYTHON_BIN" -m src.run_method \
  --method classical \
  --fixed "$DATA_ROOT/fixed_000.nii.gz" \
  --moving "$DATA_ROOT/moving_000.nii.gz" \
  --out "$OUT_ROOT/classical" \
  --iterations "${CLASSICAL_ITERS:-2}"

"$PYTHON_BIN" -m src.run_method \
  --method pso \
  --fixed "$DATA_ROOT/fixed_000.nii.gz" \
  --moving "$DATA_ROOT/moving_000.nii.gz" \
  --out "$OUT_ROOT/pso" \
  --transform rigid \
  --particles "${PSO_PARTICLES:-4}" \
  --iterations "${PSO_ITERS:-3}"

"$PYTHON_BIN" -m src.methods.voxelmorph.train \
  --config configs/voxelmorph_freesurfer_smoke.yaml

"$PYTHON_BIN" -m src.run_method \
  --method voxelmorph \
  --fixed "$DATA_ROOT/fixed_000.nii.gz" \
  --moving "$DATA_ROOT/moving_000.nii.gz" \
  --checkpoint "$OUT_ROOT/voxelmorph_train/best.pt" \
  --out "$OUT_ROOT/voxelmorph"

"$PYTHON_BIN" -m src.methods.transmorph.train \
  --config configs/transmorph_freesurfer_smoke.yaml

"$PYTHON_BIN" -m src.run_method \
  --method transmorph \
  --fixed "$DATA_ROOT/fixed_000.nii.gz" \
  --moving "$DATA_ROOT/moving_000.nii.gz" \
  --checkpoint "$OUT_ROOT/transmorph_train/best.pt" \
  --out "$OUT_ROOT/transmorph"

for method in classical pso voxelmorph transmorph; do
  test -f "$OUT_ROOT/$method/registered.nii.gz"
  test -f "$OUT_ROOT/$method/overlay.png"
  test -f "$OUT_ROOT/$method/metrics.json"
  test -f "$OUT_ROOT/$method/log.json"
done
test -f "$OUT_ROOT/classical/deformation_field.nii.gz"
test -f "$OUT_ROOT/pso/transform_params.json"
test -f "$OUT_ROOT/voxelmorph/deformation_field.nii.gz"
test -f "$OUT_ROOT/transmorph/deformation_field.nii.gz"

echo "ALL 4 FREESURFER METHODS SMOKE TEST PASSED"

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
SUBJECTS_ROOT="${SUBJECTS_ROOT:-data/oasis/freesurfer/subjects}"
SIZE="${SIZE:-64}"
SEED="${SEED:-123}"
DEEP_EPOCHS="${DEEP_EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-1}"
PARALLEL_LEARNED_TRAINING="${PARALLEL_LEARNED_TRAINING:-0}"
VOXELMORPH_DEVICE="${VOXELMORPH_DEVICE:-cuda:0}"
TRANSMORPH_DEVICE="${TRANSMORPH_DEVICE:-cuda:1}"
TRAINING_NUM_WORKERS="${TRAINING_NUM_WORKERS:-0}"
CLASSICAL_ITERS="${CLASSICAL_ITERS:-10}"
CLASSICAL_SMOOTHING_SIGMA="${CLASSICAL_SMOOTHING_SIGMA:-1.3}"
PSO_PARTICLES="${PSO_PARTICLES:-8}"
PSO_ITERS="${PSO_ITERS:-10}"
PSO_TRANSFORM="${PSO_TRANSFORM:-rigid}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found or not executable: $PYTHON_BIN" >&2
  echo "Create the local venv first, then install requirements into it." >&2
  exit 1
fi

if [[ ! -d "$SUBJECTS_ROOT" ]]; then
  echo "FreeSurfer subjects root not found: $SUBJECTS_ROOT" >&2
  exit 1
fi

subject_count="$(find "$SUBJECTS_ROOT" -path '*/mri/T1.mgz' | wc -l | tr -d '[:space:]')"
if [[ "$subject_count" -lt 2 ]]; then
  echo "Need at least 2 FreeSurfer subjects with mri/T1.mgz; found $subject_count." >&2
  exit 1
fi

max_pairs=$((subject_count - 1))
NUM_PAIRS="${NUM_PAIRS:-$max_pairs}"
if [[ "$NUM_PAIRS" -gt "$max_pairs" ]]; then
  echo "Requested NUM_PAIRS=$NUM_PAIRS, but only $max_pairs adjacent pairs are available." >&2
  exit 1
fi

DATA_ROOT="${DATA_ROOT:-data/oasis/freesurfer_3d_oasis1_${SIZE}_${NUM_PAIRS}_${RUN_ID}}"
OUT_ROOT="${OUT_ROOT:-outputs/benchmark/oasis1_3d_functional_${RUN_ID}}"

mkdir -p "$(dirname "$DATA_ROOT")" "$OUT_ROOT"

echo "Preparing OASIS-1 FreeSurfer-derived 3D data"
echo "  subjects: $subject_count"
echo "  pairs: $NUM_PAIRS"
echo "  size: ${SIZE}^3"
echo "  data root: $DATA_ROOT"

"$PYTHON_BIN" scripts/prepare_freesurfer_3d.py \
  --subjects-root "$SUBJECTS_ROOT" \
  --out "$DATA_ROOT" \
  --num_pairs "$NUM_PAIRS" \
  --size "$SIZE" \
  --image T1.mgz \
  --label aparc+aseg.mgz \
  --seed "$SEED"

echo "Running split-aware OASIS-1 3D functional benchmark"
echo "  output root: $OUT_ROOT"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}"
echo "  device argument: cuda:0"

extra_args=()
if [[ "$PARALLEL_LEARNED_TRAINING" == "1" ]]; then
  extra_args+=(--parallel-learned-training)
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" \
OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" \
ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="${ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS:-8}" \
"$PYTHON_BIN" scripts/run_oasis3d_functional_benchmark.py \
  --dataset-root "$DATA_ROOT" \
  --out "$OUT_ROOT" \
  --num-pairs "$NUM_PAIRS" \
  --seed "$SEED" \
  --train-fraction 0.70 \
  --val-fraction 0.10 \
  --deep-epochs "$DEEP_EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --device cuda:0 \
  --voxelmorph-device "$VOXELMORPH_DEVICE" \
  --transmorph-device "$TRANSMORPH_DEVICE" \
  --training-num-workers "$TRAINING_NUM_WORKERS" \
  --classical-iterations "$CLASSICAL_ITERS" \
  --classical-smoothing-sigma "$CLASSICAL_SMOOTHING_SIGMA" \
  --pso-particles "$PSO_PARTICLES" \
  --pso-iterations "$PSO_ITERS" \
  --pso-transform "$PSO_TRANSFORM" \
  "${extra_args[@]}"

echo "OASIS-1 3D functional benchmark complete"
echo "  benchmark_results.csv: $OUT_ROOT/benchmark_results.csv"
echo "  benchmark_results.json: $OUT_ROOT/benchmark_results.json"
echo "  benchmark_summary.md: $OUT_ROOT/benchmark_summary.md"

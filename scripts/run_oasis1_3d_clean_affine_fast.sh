#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
DATA_ROOT="${DATA_ROOT:-data/oasis/freesurfer_3d_oasis1_64_424_20260527_025659}"
RUN_ID="${RUN_ID:-clean_affine_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-outputs/benchmark/oasis1_3d_functional_${RUN_ID}}"

DEEP_EPOCHS="${DEEP_EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-2}"
TRAINING_NUM_WORKERS="${TRAINING_NUM_WORKERS:-0}"
DEVICE="${DEVICE:-cuda:0}"
VOXELMORPH_DEVICE="${VOXELMORPH_DEVICE:-$DEVICE}"
TRANSMORPH_DEVICE="${TRANSMORPH_DEVICE:-$DEVICE}"
CLASSICAL_ITERS="${CLASSICAL_ITERS:-10}"
CLASSICAL_SMOOTHING_SIGMA="${CLASSICAL_SMOOTHING_SIGMA:-1.3}"
PSO_PARTICLES="${PSO_PARTICLES:-16}"
PSO_ITERS="${PSO_ITERS:-30}"
PSO_TRANSFORM="${PSO_TRANSFORM:-affine}"
SEED="${SEED:-123}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "Prepared dataset root not found: $DATA_ROOT" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT"

echo "Running clean OASIS-1 3D benchmark with PSO ${PSO_TRANSFORM}"
echo "  data root: $DATA_ROOT"
echo "  output root: $OUT_ROOT"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2}"
echo "  learned training: VoxelMorph ${VOXELMORPH_DEVICE}, TransMorph ${TRANSMORPH_DEVICE}"
echo "  epochs=${DEEP_EPOCHS}, batch_size=${BATCH_SIZE}, workers=${TRAINING_NUM_WORKERS}"
echo "  PSO particles=${PSO_PARTICLES}, iterations=${PSO_ITERS}"

extra_args=()
if [[ "${PARALLEL_LEARNED_TRAINING:-0}" == "1" ]]; then
  extra_args+=(--parallel-learned-training)
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}" \
OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" \
ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="${ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS:-8}" \
"$PYTHON_BIN" scripts/run_oasis3d_functional_benchmark.py \
  --dataset-root "$DATA_ROOT" \
  --out "$OUT_ROOT" \
  --num-pairs 424 \
  --seed "$SEED" \
  --train-fraction 0.70 \
  --val-fraction 0.10 \
  --deep-epochs "$DEEP_EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --voxelmorph-device "$VOXELMORPH_DEVICE" \
  --transmorph-device "$TRANSMORPH_DEVICE" \
  --training-num-workers "$TRAINING_NUM_WORKERS" \
  --classical-iterations "$CLASSICAL_ITERS" \
  --classical-smoothing-sigma "$CLASSICAL_SMOOTHING_SIGMA" \
  --pso-particles "$PSO_PARTICLES" \
  --pso-iterations "$PSO_ITERS" \
  --pso-transform "$PSO_TRANSFORM" \
  "${extra_args[@]}"

echo "Clean affine benchmark complete"
echo "  $OUT_ROOT/benchmark_summary.md"

#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
DOWNLOAD_PID_FILE="${DOWNLOAD_PID_FILE:-outputs/logs/freesurfer_download.pid}"
SUBJECTS_ROOT="${SUBJECTS_ROOT:-data/oasis/freesurfer/subjects}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
DATA_ROOT="${DATA_ROOT:-data/oasis/freesurfer_3d_benchmark_${RUN_ID}}"
OUT_ROOT="${OUT_ROOT:-outputs/benchmark/freesurfer_3d_${RUN_ID}}"
GPU_INDEX="${GPU_INDEX:-auto}"
SIZE="${SIZE:-64}"
NUM_PAIRS="${NUM_PAIRS:-100}"
DEEP_EPOCHS="${DEEP_EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-1}"
CLASSICAL_ITERS="${CLASSICAL_ITERS:-10}"
PSO_PARTICLES="${PSO_PARTICLES:-8}"
PSO_ITERS="${PSO_ITERS:-10}"
SEED="${SEED:-123}"
SLEEP_SECONDS="${SLEEP_SECONDS:-300}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="${ITK_THREADS:-8}"

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

log() {
  echo "[$(timestamp)] $*"
}

count_discs() {
  find "$SUBJECTS_ROOT" -maxdepth 1 -type d -name 'disc*' 2>/dev/null | wc -l
}

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

log "Benchmark pipeline started"
log "Waiting for FreeSurfer download/extract job to finish"
if [[ -f "$DOWNLOAD_PID_FILE" ]]; then
  download_pid="$(cat "$DOWNLOAD_PID_FILE")"
  if [[ -n "$download_pid" ]] && kill -0 "$download_pid" 2>/dev/null; then
    while kill -0 "$download_pid" 2>/dev/null; do
      log "Download PID $download_pid still running; extracted discs=$(count_discs)/11"
      sleep "$SLEEP_SECONDS"
    done
  else
    log "Download PID file exists, but process is not running"
  fi
else
  log "No download PID file found at $DOWNLOAD_PID_FILE"
fi

disc_count="$(count_discs)"
if [[ "$disc_count" -lt 11 ]]; then
  echo "Only $disc_count FreeSurfer discs are extracted under $SUBJECTS_ROOT; expected 11." >&2
  echo "Not starting benchmark." >&2
  exit 2
fi
log "All 11 FreeSurfer discs are extracted"

if [[ "$GPU_INDEX" == "auto" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_INDEX="$(
      nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | sort -t, -k2 -n \
        | head -1 \
        | cut -d, -f1 \
        | tr -d ' '
    )"
  else
    echo "GPU_INDEX=auto requested, but nvidia-smi is not available." >&2
    exit 3
  fi
fi
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
log "Using one physical GPU: $GPU_INDEX as cuda:0"

log "Preparing FreeSurfer benchmark pairs: root=$DATA_ROOT size=${SIZE} num_pairs=${NUM_PAIRS}"
"$PYTHON_BIN" scripts/prepare_freesurfer_3d.py \
  --subjects-root "$SUBJECTS_ROOT" \
  --out "$DATA_ROOT" \
  --num_pairs "$NUM_PAIRS" \
  --size "$SIZE" \
  --seed "$SEED"

log "Running four-method benchmark: out=$OUT_ROOT"
"$PYTHON_BIN" scripts/run_benchmark.py \
  --dataset-root "$DATA_ROOT" \
  --out "$OUT_ROOT" \
  --num-pairs "$NUM_PAIRS" \
  --deep-epochs "$DEEP_EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --device cuda:0 \
  --classical-iterations "$CLASSICAL_ITERS" \
  --pso-transform rigid \
  --pso-particles "$PSO_PARTICLES" \
  --pso-iterations "$PSO_ITERS" \
  --seed "$SEED"

log "Benchmark complete"
log "Summary: $OUT_ROOT/benchmark_summary.md"

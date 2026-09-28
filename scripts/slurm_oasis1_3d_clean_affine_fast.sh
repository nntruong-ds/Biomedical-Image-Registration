#!/usr/bin/env bash
#SBATCH --job-name=biomedreg-oasis-affine
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/oasis_affine_%j.out
#SBATCH --error=logs/slurm/oasis_affine_%j.err

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
mkdir -p logs/slurm

export PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="${ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS:-8}"
export RUN_ID="${RUN_ID:-clean_affine_${SLURM_JOB_ID:-manual}}"

bash scripts/run_oasis1_3d_clean_affine_fast.sh

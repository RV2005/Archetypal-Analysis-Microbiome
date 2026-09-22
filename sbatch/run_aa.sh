#!/usr/bin/env bash
#SBATCH --job-name=aa_pipeline
#SBATCH --partition=compute
#SBATCH --time=48:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=11
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

# Usage:
#   sbatch --export=ALL,CONFIG=configs/experiments/dog_lifestyle.yaml,CONDA_ENV=GHG_work sbatch/run_aa.sh
# Optional:
#   STAGES=fit,evaluate,select     run only those requested stages (+ reusable dependencies)
#   OVERWRITE=1                   overwrite explicitly requested stages
#   CONDA_ENV=...                 activate this conda environment when conda is available

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="${CONFIG:-configs/experiments/dog_lifestyle.yaml}"
STAGES="${STAGES:-all}"
OVERWRITE="${OVERWRITE:-0}"

if [[ -n "${CONDA_ENV:-}" ]] && command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  # shellcheck disable=SC1090
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

# AA K values are parallelized as processes. Keep BLAS/OpenMP at one thread per process
# to prevent nested oversubscription inside a single Slurm allocation.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

args=(python -m archetypal_analysis.cli run --config "$CONFIG" --stages "$STAGES")
if [[ "$OVERWRITE" == "1" || "$OVERWRITE" == "true" ]]; then
  args+=(--overwrite)
fi

printf 'Running: '
printf '%q ' "${args[@]}"
printf '\n'
"${args[@]}"
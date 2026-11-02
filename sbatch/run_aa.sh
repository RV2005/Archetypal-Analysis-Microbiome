#!/usr/bin/env bash
#SBATCH --job-name=aa_pipeline
#SBATCH --partition=compute
#SBATCH --time=48:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=24
#SBATCH --output=/scratch/rv2340/Archetypal_Analysis_Project/slurm_outs/pipeline/pipeline-%x-%j.out

# Usage:
#   sbatch --export=ALL,CONFIG=configs/experiments/dog_lifestyle.yaml,CONDA_ENV=GHG_work sbatch/run_aa.sh
# Optional:
#   STAGES=fit,evaluate,select     run only those requested stages (+ reusable dependencies)
#   OVERWRITE=1                   overwrite explicitly requested stages
#   CONDA_ENV=...                 activate this conda environment when conda is available

ROOT="/scratch/rv2340/Archetypal_Analysis_Project/Archetypal-Analysis"
cd "$ROOT"

CONFIG="${CONFIG:-configs/experiments/dog_lifestyle.yaml}"
STAGES="${STAGES:-all}"
OVERWRITE="${OVERWRITE:-0}"

CONDA_SOURCE="${NCA_CONDA_SOURCE:-/share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate}"
CONDA_ENV="${NCA_CONDA_ENV:-GHG_work}"

source "${CONDA_SOURCE}" "${CONDA_ENV}"

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
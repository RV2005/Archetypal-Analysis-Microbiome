#!/bin/bash
#SBATCH --job-name=validate_taxa
#SBATCH --partition=compute
#SBATCH --time=48:00:00
#SBATCH --mem=50GB
#SBATCH --cpus-per-task=2
#SBATCH --output=/scratch/rv2340/Archetypal_Analysis_Project/slurm_outs/taxa_validation/%A_%a.out


source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate GHG_work


# Example:
#   DATASET=dog OUT_DIR=./results_validate_taxa ./run_validate_taxa.sh

DATASET="${DATASET:-sample_type}"
CONFIG_FILE="${CONFIG_FILE:-./dataset_configs.json}"
OUT_DIR="${OUT_DIR:-./results_validate_taxa}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${DATASET}}"

SPLITTER="${SPLITTER:-sss}"
N_RESAMPLES="${N_RESAMPLES:-20}"
TEST_SIZE="${TEST_SIZE:-0.2}"
CV_FOLDS="${CV_FOLDS:-5}"
N_ESTIMATORS="${N_ESTIMATORS:-500}"
N_PERMUTATION_REPEATS="${N_PERMUTATION_REPEATS:-10}"
N_JOBS="${N_JOBS:--1}"
RANDOM_SEED="${RANDOM_SEED:-42}"
AA_K_VALUES="${AA_K_VALUES:-5,10,15}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmpdata/mpl}"
export MPLCONFIGDIR

echo "Running validation for dataset: ${DATASET}"
echo "Configuration:"
echo "  Dataset: ${DATASET}"
echo "  Config file: ${CONFIG_FILE}"
echo "  Experiment name: ${EXPERIMENT_NAME}"
echo "  Output directory: ${OUT_DIR}"
echo "  Splitter: ${SPLITTER}"
echo "  Number of resamples: ${N_RESAMPLES}"
echo "  Test size: ${TEST_SIZE}"
echo "  CV folds: ${CV_FOLDS}"
echo "  Number of estimators: ${N_ESTIMATORS}"
echo "  Number of permutation repeats: ${N_PERMUTATION_REPEATS}"
echo "  Number of jobs: ${N_JOBS}"
echo "  Random seed: ${RANDOM_SEED}"
echo "  AA k values: ${AA_K_VALUES}"

"${PYTHON_BIN}" ./validate_taxa.py \
  --dataset "${DATASET}" \
  --dataset_config_file "${CONFIG_FILE}" \
  --experiment_name "${EXPERIMENT_NAME}" \
  --output_dir "${OUT_DIR}" \
  --splitter "${SPLITTER}" \
  --n_resamples "${N_RESAMPLES}" \
  --test_size "${TEST_SIZE}" \
  --cv_folds "${CV_FOLDS}" \
  --n_estimators "${N_ESTIMATORS}" \
  --n_permutation_repeats "${N_PERMUTATION_REPEATS}" \
  --n_jobs "${N_JOBS}" \
  --random_seed "${RANDOM_SEED}" \
  --aa_k_values "${AA_K_VALUES}" \
  --run_ablation

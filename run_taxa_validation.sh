#!/bin/bash
#SBATCH --job-name=rvf_gpu_together
#SBATCH --partition=compute
#SBATCH --time=48:00:00
#SBATCH --mem=50GB
#SBATCH --cpus-per-task=2
#SBATCH --output=/scratch/rv2340/randomvoronoi/slurm_logs/taxa_validation/%A_%a.out


source /opt/miniconda3/bin/activate
conda activate GHG_work

DATASET="${DATASET:-sample_type}"
DATASET_CONFIG_FILE="${DATASET_CONFIG_FILE:-dataset_configs.json}"
OUTPUT_DIR="${OUTPUT_DIR:-./results_validate_fuzzy_taxa/${DATASET}}"

python_cmd="python microbiome_validate_fuzzy_taxa.py \
  --dataset ${DATASET} \
  --dataset_config_file ${DATASET_CONFIG_FILE} \
  --output_dir ${OUTPUT_DIR} \
  --random_seed 123
"

# Echo running command for logging
echo "Running fuzzy taxa validation with the following command:"
echo "$python_cmd"

# Run the command
if eval $python_cmd; then
    echo "Fuzzy taxa validation completed successfully."
else
    echo "Error: Fuzzy taxa validation failed." >&2
    exit 1
fi

# Echo job completed
echo ""
echo "=========================================="
echo "Job Summary:"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Dataset: $DATASET"
echo "Dataset Config: $DATASET_CONFIG_FILE"
echo "Output Directory: $OUTPUT_DIR"
echo "Output File: /scratch/rv2340/randomvoronoi/slurm_logs/taxa_validation/${SLURM_JOB_ID}_%a.out"
echo "=========================================="

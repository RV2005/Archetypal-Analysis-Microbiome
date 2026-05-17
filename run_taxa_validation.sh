#!/bin/bash
#SBATCH --job-name=rvf_gpu_together
#SBATCH --partition=compute
#SBATCH --time=48:00:00
#SBATCH --mem=50GB
#SBATCH --cpus-per-task=2
#SBATCH --output=/scratch/rv2340/randomvoronoi/slurm_logs/taxa_validation/%A_%a.out


source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate GHG_work

python_cmd="python microbiome_validate_fuzzy_taxa.py \
  --otu_file your_otu_table.tsv \
  --metadata_file your_metadata.tsv \
  --metadata_sample_col sample_id \
  --clinical_col disease_status \
  --top_taxa_file your_fuzzy_taxa.tsv \
  --output_dir ./results_basic \
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
echo "Output File: /scratch/rv2340/randomvoronoi/slurm_logs/taxa_validation/${SLURM_JOB_ID}_%a.out"
echo "=========================================="
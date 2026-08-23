#!/bin/bash
#SBATCH --job-name=archetypal_analysis
#SBATCH --partition=compute
#SBATCH --time=48:00:00
#SBATCH --mem=40GB
#SBATCH --cpus-per-task=1
#SBATCH -a 3-13
#SBATCH --output=/scratch/rv2340/Archetypal_Analysis_Project/slurm_outs/archetypal_analysis/%x_%A_%a.out


export OMP_NUM_THREADS=11

source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate GHG_work

# OTU_FILE="/scratch/rv2340/Archetypal_Analysis_Project/Extra_Work/datasets/raghavi_otu.tsv"
# OUT_DIR="/scratch/rv2340/Archetypal_Analysis_Project/Extra_Work/outputs/raghavi"

# OTU_FILE="/scratch/rv2340/Archetypal_Analysis_Project/Extra_Work/datasets/virome_anique_otu.tsv"
# OUT_DIR="/scratch/rv2340/Archetypal_Analysis_Project/Extra_Work/outputs/virome_anique"

# OTU_FILE="/scratch/rv2340/Archetypal_Analysis_Project/Extra_Work/datasets/infant2_raw_kusha_otu.tsv"
# OUT_DIR="/scratch/rv2340/Archetypal_Analysis_Project/Extra_Work/outputs/infant2_kusha"

# OTU_FILE="/scratch/rv2340/Archetypal_Analysis_Project/Extra_Work/datasets/infant1_kusha_otu.tsv"
# OUT_DIR="/scratch/rv2340/Archetypal_Analysis_Project/Extra_Work/outputs/infant1_kusha"

OTU_FILE="/scratch/rv2340/Archetypal_Analysis_Project/Extra_Work/datasets/jiaming_otu.tsv"
OUT_DIR="/scratch/rv2340/Archetypal_Analysis_Project/Extra_Work/outputs/jiaming"


FRAC=1.0
N_ARCHS=$SLURM_ARRAY_TASK_ID


echo "Running Archetypal Analysis with $SLURM_ARRAY_TASK_ID archetypes, fraction $FRAC, and OTU file $OTU_FILE"

cd "/scratch/rv2340/Archetypal_Analysis_Project/Archetypal-Analysis/HPC_scripts"
python "AA_GHG_HPC_190526.py" \
    --n_archetypes $N_ARCHS \
    --frac $FRAC \
    --otu_file $OTU_FILE \
    --out_dir $OUT_DIR \
    --log_transform true


echo "Job completed for $N_ARCHS archetypes."


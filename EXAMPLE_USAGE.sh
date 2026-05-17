#!/usr/bin/env bash
# Example usage script for microbiome_validate_fuzzy_taxa.py

# This script demonstrates how to run the fuzzy taxa validation pipeline
# with different configurations.

# ============================================================================
# BASIC USAGE - Minimal arguments
# ============================================================================

# Run with minimal required arguments:
# python microbiome_validate_fuzzy_taxa.py \
#   --otu_file your_otu_table.tsv \
#   --metadata_file your_metadata.tsv \
#   --metadata_sample_col sample_id \
#   --clinical_col disease_status \
#   --top_taxa_file your_fuzzy_taxa.tsv \
#   --output_dir ./results_basic \
#   --random_seed 123


# ============================================================================
# STANDARD USAGE - With quality control parameters
# ============================================================================

# Run with more stringent QC and balanced parameters:
# python microbiome_validate_fuzzy_taxa.py \
#   --otu_file your_otu_table.tsv \
#   --metadata_file your_metadata.tsv \
#   --metadata_sample_col sample_id \
#   --clinical_col disease_status \
#   --top_taxa_file your_fuzzy_taxa.tsv \
#   --output_dir ./results_standard \
#   --min_prevalence 0.05 \
#   --min_total_count 10 \
#   --n_resamples 50 \
#   --test_size 0.2 \
#   --random_seed 123


# ============================================================================
# COMPREHENSIVE USAGE - All methods enabled, with covariates
# ============================================================================

# Run with all methods (DA, RF, Boruta) and covariates:
# python microbiome_validate_fuzzy_taxa.py \
#   --otu_file your_otu_table.tsv \
#   --metadata_file your_metadata.tsv \
#   --metadata_sample_col sample_id \
#   --clinical_col disease_status \
#   --top_taxa_file your_fuzzy_taxa.tsv \
#   --output_dir ./results_comprehensive \
#   --covariates age,sex,batch \
#   --min_prevalence 0.05 \
#   --min_total_count 10 \
#   --n_resamples 100 \
#   --test_size 0.2 \
#   --n_permutations 10000 \
#   --n_estimators 1000 \
#   --run_boruta \
#   --run_ancombc2 \
#   --run_aldex2 \
#   --alpha 0.05 \
#   --random_seed 123 \
#   --n_jobs -1


# ============================================================================
# HIGH-THROUGHPUT USAGE - Faster execution, fewer resamples
# ============================================================================

# For quick exploration with fewer iterations:
# python microbiome_validate_fuzzy_taxa.py \
#   --otu_file your_otu_table.tsv \
#   --metadata_file your_metadata.tsv \
#   --clinical_col disease_status \
#   --top_taxa_file your_fuzzy_taxa.tsv \
#   --output_dir ./results_fast \
#   --n_resamples 20 \
#   --n_permutations 1000 \
#   --n_estimators 200 \
#   --alpha 0.05


# ============================================================================
# PUBLICATION-QUALITY USAGE - Intensive, reproducible analysis
# ============================================================================

# For final publication-quality results with intensive validation:
# python microbiome_validate_fuzzy_taxa.py \
#   --otu_file your_otu_table.tsv \
#   --metadata_file your_metadata.tsv \
#   --metadata_sample_col sample_id \
#   --clinical_col disease_status \
#   --top_taxa_file your_fuzzy_taxa.tsv \
#   --output_dir ./results_publication \
#   --covariates age,sex,batch,treatment \
#   --min_prevalence 0.05 \
#   --min_total_count 10 \
#   --n_resamples 500 \
#   --test_size 0.2 \
#   --n_permutations 50000 \
#   --n_estimators 2000 \
#   --run_boruta \
#   --alpha 0.05 \
#   --random_seed 42 \
#   --n_jobs -1


# ============================================================================
# MULTICLASS CLASSIFICATION - For >2 clinical categories
# ============================================================================

# The pipeline automatically detects multiclass and uses:
# - ANCOM-BC2 with multi-group contrasts
# - ALDEx2 Kruskal-Wallis test
# - RF macro-averaged metrics and one-vs-rest ROC-AUC

# python microbiome_validate_fuzzy_taxa.py \
#   --otu_file otu_multiclass.tsv \
#   --metadata_file metadata_multiclass.tsv \
#   --clinical_col disease_stage \
#   --top_taxa_file fuzzy_taxa.tsv \
#   --output_dir ./results_multiclass \
#   --n_resamples 100 \
#   --alpha 0.05


# ============================================================================
# WITH TAB-DELIMITED INPUT AUTO-DETECTION
# ============================================================================

# Pipeline auto-detects delimiters and orientation:

# python microbiome_validate_fuzzy_taxa.py \
#   --otu_file otu_table.csv \
#   --metadata_file metadata.csv \
#   --sep comma \
#   --otu_orientation auto \
#   --clinical_col condition \
#   --top_taxa_file selected_taxa.txt \
#   --output_dir ./results_auto_detect


# ============================================================================
# INTERACTIVE EXAMPLE - Step by step
# ============================================================================

# 1. First, prepare your input files:
#    - otu_table.tsv: Samples x Taxa (or Taxa x Samples, auto-detected)
#    - metadata.tsv: Sample metadata with sample_id column
#    - fuzzy_taxa.tsv: Contains 'taxon' column with selected taxa

# 2. Do a test run with small sample size to check for errors:

# python microbiome_validate_fuzzy_taxa.py \
#   --otu_file ./test_data/otu_table.tsv \
#   --metadata_file ./test_data/metadata.tsv \
#   --metadata_sample_col sample_id \
#   --clinical_col disease_status \
#   --top_taxa_file ./test_data/fuzzy_taxa.tsv \
#   --output_dir ./test_output \
#   --n_resamples 10 \
#   --n_permutations 100

# 3. Check for errors in ./test_output/pipeline.log

# 4. If successful, run full analysis:

# python microbiome_validate_fuzzy_taxa.py \
#   --otu_file ./data/otu_table.tsv \
#   --metadata_file ./data/metadata.tsv \
#   --metadata_sample_col sample_id \
#   --clinical_col disease_status \
#   --top_taxa_file ./data/fuzzy_taxa.tsv \
#   --output_dir ./results_final \
#   --n_resamples 100 \
#   --n_permutations 10000 \
#   --run_boruta

# 5. Review results:
#    - Check report.md for summary
#    - Open comparison/fuzzy_taxa_evidence_score.tsv in Excel/spreadsheet
#    - View plots in plots/ directory
#    - Examine pipeline.log for any warnings


# ============================================================================
# DEBUGGING TIPS
# ============================================================================

# If you encounter errors:

# 1. Check pipeline.log:
#    tail -100 ./results_xyz/pipeline.log

# 2. For ANCOM-BC2 errors, check R stderr:
#    cat ./results_xyz/differential_abundance/ancombc2/ancombc2_stderr.log

# 3. For ALDEx2 errors:
#    cat ./results_xyz/differential_abundance/aldex2/aldex2_stderr.log

# 4. Check input file formats:
#    head -3 your_otu_table.tsv
#    head -3 your_metadata.tsv
#    head -3 your_fuzzy_taxa.tsv

# 5. Test with a small subset first to diagnose issues quickly


# ============================================================================
# PERFORMANCE TUNING
# ============================================================================

# For large datasets or limited resources:

# FASTER (30 min):
# --n_resamples 20 --n_permutations 1000 --n_estimators 200 --n_jobs -1

# BALANCED (2-4 hours):
# --n_resamples 50 --n_permutations 5000 --n_estimators 1000 --n_jobs -1

# THOROUGH (4-8+ hours):
# --n_resamples 200 --n_permutations 20000 --n_estimators 2000 --n_jobs -1

# PUBLICATION (overnight+):
# --n_resamples 500 --n_permutations 50000 --n_estimators 3000 --n_jobs -1


# ============================================================================
# EXPECTED OUTPUT INTERPRETATION
# ============================================================================

# After running, check results in this order:

# 1. pipeline.log - any warnings or errors?
# 2. report.md - executive summary
# 3. qc/input_summary.json - sample/taxa counts
# 4. comparison/overlap_summary.tsv - statistical significance of overlaps
# 5. comparison/fuzzy_taxa_evidence_score.tsv - KEY OUTPUT
#    - Sort by evidence_score (descending)
#    - Strong (score ≥5): Multiple methods support
#    - Moderate (3-4): Good evidence
#    - Weak (1-2): Single method
#    - Unsupported (0): No external support

# 6. ablation/ablation_metrics_summary.tsv
#    - Compare fuzzy_taxa_only vs all_taxa
#    - Compare all_taxa_except_fuzzy vs all_taxa
#    - Large difference = fuzzy taxa important

# 7. plots/ - visualization of results


echo "Example usage configurations provided above."
echo "Uncomment and modify any configuration for your specific use case."

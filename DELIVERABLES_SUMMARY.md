# Microbiome Fuzzy Taxa Validation Pipeline - Complete Deliverables

## Summary

A production-quality Python pipeline for validating microbiome taxa selected by fuzzy clustering against multiple independent methods (differential abundance and machine learning). The pipeline provides comprehensive evidence scoring and reproducible comparative analysis.

**Created**: May 17, 2026
**Location**: `/scratch/rv2340/Archetypal_Analysis_Project/Archetypal-Analysis/`

---

## Delivered Files

### 1. Main Script
- **`microbiome_validate_fuzzy_taxa.py`** (executable)
  - 1800+ lines of production-quality Python
  - Full pipeline implementation with all methods
  - Comprehensive error handling and logging
  - Type hints and docstrings throughout

### 2. Documentation

#### User Documentation
- **`VALIDATE_FUZZY_TAXA_README.md`** (START HERE)
  - Overview and quick start
  - Argument reference
  - Input/output file formats
  - Key output descriptions
  - Troubleshooting basics

- **`EXAMPLE_USAGE.sh`** (executable)
  - Multiple example command configurations
  - From basic to publication-quality
  - Performance tuning guidance
  - Expected output interpretation

#### Technical Documentation
- **`TECHNICAL_METHODS.md`**
  - Detailed method descriptions
  - Statistical formulas and algorithms
  - ANCOM-BC2 and ALDEx2 implementation details
  - Random Forest and Boruta specifics
  - Overlap analysis mathematics
  - Evidence scoring rationale

- **`TROUBLESHOOTING.md`**
  - Common issues and solutions
  - Debugging workflow
  - Frequently asked questions

### 3. Utilities

- **`generate_test_data.py`** (executable)
  - Generates synthetic microbiome data for testing
  - Creates OTU table, metadata, fuzzy taxa file
  - Validates pipeline installation

### 4. Additional Files

- **`microbiome_validate_fuzzy_taxa.py`** (MAIN SCRIPT - described above)

---

## File Organization

```
Archetypal-Analysis/
├── microbiome_validate_fuzzy_taxa.py        [MAIN SCRIPT]
├── generate_test_data.py                    [TEST DATA GENERATOR]
│
├── VALIDATE_FUZZY_TAXA_README.md            [START HERE - USER GUIDE]
├── EXAMPLE_USAGE.sh                         [USAGE EXAMPLES]
├── TECHNICAL_METHODS.md                     [DETAILED METHODS]
├── TROUBLESHOOTING.md                       [ISSUE RESOLUTION]
│
└── (script generates output_dir with results)
```

---

## Quick Start (3 Steps)

### Step 1: Prepare Data

Create three input files:
1. **OTU table** (TSV/CSV): abundance matrix, samples × taxa or taxa × samples
2. **Metadata** (TSV/CSV): sample annotations with clinical phenotype column
3. **Fuzzy taxa** (TSV/CSV): selected taxa with at minimum a `taxon` column

### Step 2: Generate Test Data (Optional)

```bash
python generate_test_data.py --output_dir ./test_data
```

### Step 3: Run Pipeline

```bash
python microbiome_validate_fuzzy_taxa.py \
  --otu_file your_otu_table.tsv \
  --metadata_file your_metadata.tsv \
  --metadata_sample_col sample_id \
  --clinical_col disease_status \
  --top_taxa_file your_fuzzy_taxa.tsv \
  --output_dir ./results \
  --n_resamples 50 \
  --run_boruta
```

---

## Key Features

### Input Handling
✓ Auto-detect delimiter (tab/comma)
✓ Auto-detect OTU table orientation
✓ Robust sample matching
✓ Comprehensive QC reporting

### Differential Abundance
✓ ANCOM-BC2 (via R/Bioconductor)
✓ ALDEx2 (via R/Bioconductor)
✓ Automatic multiclass handling
✓ FDR-controlled p-values

### Machine Learning
✓ Random Forest with repeated stratified resampling
✓ Permutation importance (held-out test set)
✓ Optional Boruta all-relevant feature selection
✓ Class weight balancing

### Comparative Analysis
✓ Overlap metrics (Jaccard, overlap coefficient)
✓ Statistical significance tests (hypergeometric, permutation)
✓ Rank correlation (Spearman)
✓ Direction agreement (DA vs ML)
✓ External consensus detection

### Predictive Ablation
✓ Compare feature set performance
✓ Paired statistical tests (Wilcoxon)
✓ Confidence intervals and aggregation
✓ Held-out evaluation (no data leakage)

### Evidence Scoring
✓ Integrates all methods into 0-7 point scale
✓ Evidence labels (strong/moderate/weak/unsupported)
✓ Transparent scoring with full audit trail

### Outputs
✓ Comprehensive logging (pipeline.log)
✓ Reproducible R scripts (r_helpers/)
✓ Software versions tracking
✓ Publication-ready plots
✓ Markdown report
✓ Structured TSV/JSON outputs

---

## Output Directory Structure

```
results/
├── pipeline.log                             # Full execution log
├── run_config.json                          # All parameters
├── software_versions.txt                    # Package versions
├── report.md                                # Executive summary

├── qc/
│   ├── input_summary.json
│   ├── sample_matching.tsv
│   └── taxa_filtering_summary.tsv

├── processed/
│   ├── raw_counts_samples_by_taxa.tsv
│   ├── log1p_abundance_samples_by_taxa.tsv
│   └── metadata_matched.tsv

├── fuzzy_taxa/
│   ├── fuzzy_taxa_loaded.tsv
│   ├── fuzzy_taxa_matched.tsv
│   ├── fuzzy_taxa_unmatched_in_top.tsv
│   └── fuzzy_taxa_unmatched_in_otu.tsv

├── differential_abundance/
│   ├── ancombc2/
│   │   ├── ancombc2_results.tsv
│   │   ├── ancombc2_pairwise_results.tsv
│   │   └── ancombc2_results_standardized.tsv
│   └── aldex2/
│       ├── aldex2_results.tsv
│       ├── aldex2_effects.tsv
│       └── aldex2_results_standardized.tsv

├── ml/
│   ├── random_forest/
│   │   ├── rf_resample_metrics.tsv
│   │   └── rf_permutation_importance_summary.tsv
│   └── boruta/
│       ├── boruta_summary.tsv
│       └── boruta_skipped.txt (if not run)

├── comparison/
│   ├── overlap_summary.tsv
│   ├── rank_correlation_summary.tsv
│   ├── direction_agreement.tsv
│   ├── method_consensus_taxa.tsv
│   └── fuzzy_taxa_evidence_score.tsv    [KEY OUTPUT]

├── ablation/
│   ├── ablation_metrics_long.tsv
│   ├── ablation_metrics_summary.tsv
│   └── ablation_paired_comparisons.tsv

├── plots/
│   ├── da_volcano_ancombc2.png
│   ├── da_volcano_aldex2.png
│   ├── rf_top_permutation_importance.png
│   ├── ablation_performance_boxplot.png
│   └── ...

└── r_helpers/
    ├── run_ancombc2.R
    └── run_aldex2.R
```

---

## Key Output: Evidence Scores

The **`comparison/fuzzy_taxa_evidence_score.tsv`** file is the main deliverable.

Columns:
- `taxon`: OTU/ASV identifier
- `cluster`: Fuzzy cluster assignment (if provided)
- `fuzzy_rank`: Rank within fuzzy selection
- `fuzzy_membership_or_importance`: Membership/importance score
- `ancombc2_supported`: Boolean (q < 0.05)
- `aldex2_supported`: Boolean (q < 0.05)
- `rf_supported`: Boolean (in top-N)
- `boruta_supported`: Boolean (if run)
- `direction_consistent`: Boolean (DA vs ML agreement)
- `evidence_score`: 0-7 points from supporting criteria
- `evidence_label`: strong/moderate/weak/unsupported

### Interpretation

| Label | Score | Recommendation |
|-------|-------|-----------------|
| **Strong** | ≥5 | High confidence; suitable for biomarker validation |
| **Moderate** | 3-4 | Good evidence; recommend independent validation |
| **Weak** | 1-2 | Limited evidence; requires further investigation |
| **Unsupported** | 0 | No external support; reconsider inclusion |

---

## Installation & Requirements

### Python Packages
```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn
# Optional: pip install boruta
```

### R/Bioconductor
```R
if (!require("BiocManager")) install.packages("BiocManager")
BiocManager::install(c("ANCOMBC", "ALDEx2", "phyloseq"))
```

### System Requirements
- Python 3.7+
- R 3.6+
- 4GB+ RAM (for typical microbiome datasets)
- 50GB+ disk for 500+ resamples on large datasets

---

## Usage Examples

### Minimal (just required args)
```bash
python microbiome_validate_fuzzy_taxa.py \
  --otu_file otu.tsv \
  --metadata_file metadata.tsv \
  --clinical_col disease \
  --top_taxa_file fuzzy.tsv \
  --output_dir results
```

### Standard (recommended)
```bash
python microbiome_validate_fuzzy_taxa.py \
  --otu_file otu.tsv \
  --metadata_file metadata.tsv \
  --metadata_sample_col sample_id \
  --clinical_col disease \
  --top_taxa_file fuzzy.tsv \
  --output_dir results \
  --covariates age,sex \
  --n_resamples 100 \
  --n_permutations 10000
```

### Publication (comprehensive)
```bash
python microbiome_validate_fuzzy_taxa.py \
  --otu_file otu.tsv \
  --metadata_file metadata.tsv \
  --metadata_sample_col sample_id \
  --clinical_col disease \
  --top_taxa_file fuzzy.tsv \
  --output_dir results \
  --covariates age,sex,batch \
  --min_prevalence 0.05 \
  --min_total_count 10 \
  --n_resamples 500 \
  --n_permutations 50000 \
  --n_estimators 2000 \
  --run_boruta \
  --alpha 0.05 \
  --random_seed 42 \
  --n_jobs -1
```

---

## Reproducibility

To reproduce results:

```bash
# Same seed + single-threaded = bit-exact reproducibility
python microbiome_validate_fuzzy_taxa.py \
  ... \
  --random_seed 123 \
  --n_jobs 1
```

All parameters saved in `run_config.json` for audit trail.

---

## Methods Summary

### Differential Abundance
- **ANCOM-BC2**: Generalized additive log-linear models with bias correction
- **ALDEx2**: Monte-Carlo sampling for compositional data

### Machine Learning
- **Random Forest**: 1000 trees, balanced class weights, repeated stratified resampling
- **Permutation Importance**: Feature importance by performance degradation on held-out test set
- **Boruta**: All-relevant feature selection using random forest importance

### Statistical Tests
- **Hypergeometric test**: Overlap significance (random sampling)
- **Permutation test**: Overlap significance (empirical p-value)
- **Spearman rank correlation**: Feature ranking agreement
- **Wilcoxon signed-rank**: Paired performance comparisons

### Visualization
- Volcano plots (DA methods)
- Feature importance barplots
- Ablation performance boxplots
- Heatmaps of method agreement

---

## Citation

If using this pipeline, cite:
- ANCOM-BC2: Huang et al. (2022) *Nature Communications* 13:6077
- ALDEx2: Fernandes et al. (2013) *PeerJ* 1:e46
- BorutaPy: Kursa & Rudnicki (2010) *J Stat Softw* 36(11)
- scikit-learn: Pedregosa et al. (2011) *JMLR* 12:2825-2830

---

## Support

1. **Read documentation**: Start with `VALIDATE_FUZZY_TAXA_README.md`
2. **Check examples**: See `EXAMPLE_USAGE.sh`
3. **Debug issues**: Consult `TROUBLESHOOTING.md`
4. **Understand methods**: Review `TECHNICAL_METHODS.md`
5. **Generate test data**: Use `generate_test_data.py`

---

## Version & Metadata

- **Script**: `microbiome_validate_fuzzy_taxa.py`
- **Created**: May 17, 2026
- **Python Version**: 3.7+
- **Key Dependencies**: pandas, numpy, scipy, scikit-learn, matplotlib, seaborn, (optionally: boruta)
- **Status**: Production-ready, fully documented

---

## Next Steps

1. ✅ Install dependencies
2. ✅ Prepare input data (OTU table, metadata, fuzzy taxa)
3. ✅ Run `generate_test_data.py` to verify installation
4. ✅ Review `EXAMPLE_USAGE.sh` for your use case
5. ✅ Run pipeline with your data
6. ✅ Review `report.md` and `fuzzy_taxa_evidence_score.tsv`
7. ✅ Validate findings independently

Good luck with your microbiome analysis! 🧬


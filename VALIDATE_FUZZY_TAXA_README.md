# Microbiome Fuzzy Taxa Validation Pipeline

## Overview

A production-quality Python command-line script for validating fuzzy-clustering-selected microbiome taxa against independent differential abundance and machine learning methods.

The script performs:
- **Data QC and filtering**: Sample/taxa matching, prevalence filtering
- **Differential Abundance**: ANCOM-BC2 and ALDEx2 (via R/Bioconductor)
- **Machine Learning**: Random Forest permutation importance, optional Boruta
- **Comparative Analysis**: Overlap significance, rank correlation, direction agreement
- **Predictive Ablation**: Feature set importance via held-out test performance
- **Evidence Scoring**: Integrates all methods into evidence strength scores

## Installation

### Required Python packages:
```bash
pip install pandas numpy scipy scikit-learn statsmodels matplotlib seaborn
```

### Optional:
```bash
pip install boruta  # For Boruta feature selection
```

### Required R/Bioconductor:
The script generates R helper scripts and calls Rscript automatically. You must have:
- R installed
- Bioconductor packages: `ANCOMBC`, `ALDEx2`, `phyloseq`

Install from R console:
```R
if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

BiocManager::install(c("ANCOMBC", "ALDEx2", "phyloseq"))
```

## Quick Start

### Basic example:
```bash
python microbiome_validate_fuzzy_taxa.py \
  --otu_file otu_table.tsv \
  --metadata_file metadata.tsv \
  --metadata_sample_col sample_id \
  --clinical_col disease_status \
  --top_taxa_file fuzzy_selected_taxa.tsv \
  --output_dir results_fuzzy_validation
```

### With all options:
```bash
python microbiome_validate_fuzzy_taxa.py \
  --otu_file otu_table.tsv \
  --metadata_file metadata.tsv \
  --metadata_sample_col sample_id \
  --clinical_col disease_status \
  --top_taxa_file fuzzy_selected_taxa.tsv \
  --output_dir results_fuzzy_validation \
  --covariates age,sex,batch \
  --min_prevalence 0.10 \
  --min_total_count 50 \
  --n_resamples 100 \
  --test_size 0.2 \
  --n_permutations 10000 \
  --n_estimators 1000 \
  --run_boruta \
  --run_ancombc2 \
  --run_aldex2 \
  --alpha 0.05 \
  --random_seed 123 \
  --n_jobs -1
```

## Input File Formats

### OTU/ASV Table
Tab or comma-delimited with taxa as rows or columns, sample IDs as the other dimension. Auto-detected.

Example (samples x taxa):
```
sample_id    taxon_1  taxon_2  taxon_3
sample_001   100      50       200
sample_002   50       25       100
...
```

### Metadata Table
Tab or comma-delimited with sample IDs as row index (or specify column).

Example:
```
sample_id,disease_status,age,sex
sample_001,disease,45,M
sample_002,control,40,F
...
```

### Fuzzy Taxa File
CSV/TSV with at least a `taxon` column. Optional: `cluster`, `fuzzy_rank`, `membership`, `importance`.

Example:
```
cluster,taxon,membership,fuzzy_rank
cluster_1,Firmicutes_OTU_123,0.95,1
cluster_1,Bacteroides_OTU_456,0.92,2
cluster_2,Prevotella_OTU_789,0.88,1
...
```

## Arguments

### Required:
- `--otu_file` PATH: Raw OTU/ASV abundance table
- `--metadata_file` PATH: Metadata with samples and clinical category
- `--clinical_col` NAME: Column containing clinical phenotype/category
- `--top_taxa_file` PATH: File with fuzzy-selected taxa
- `--output_dir` PATH: Output directory

### Optional:
- `--metadata_sample_col` NAME: Column with sample IDs (default: use row index)
- `--otu_orientation` {taxa_rows,samples_rows,auto}: OTU matrix orientation (default: auto)
- `--sep` {auto,comma,tab}: File delimiter (default: auto)
- `--min_prevalence` FLOAT: Minimum prevalence (default: 0.05)
- `--min_total_count` INT: Minimum total count (default: 10)
- `--n_resamples` INT: Repeated resampling iterations (default: 50)
- `--test_size` FLOAT: Test fraction (default: 0.2)
- `--random_seed` INT: Random seed (default: 123)
- `--n_permutations` INT: Permutation test iterations (default: 10000)
- `--n_estimators` INT: Random Forest trees (default: 1000)
- `--covariates` COLS: Comma-separated covariate columns
- `--alpha` FLOAT: Significance threshold (default: 0.05)
- `--run_boruta`: Enable Boruta feature selection
- `--run_ancombc2`: Run ANCOM-BC2 (default: enabled)
- `--run_aldex2`: Run ALDEx2 (default: enabled)
- `--n_jobs` INT: Parallel processes (default: -1, all cores)

## Output Structure

```
output_dir/
├── pipeline.log                          # Detailed execution log
├── run_config.json                       # All arguments as JSON
├── software_versions.txt                 # Package versions
├── report.md                             # Summary report
│
├── qc/
│   ├── input_summary.json               # Counts and metadata
│   ├── sample_matching.tsv              # Sample overlap report
│   └── taxa_filtering_summary.tsv       # Taxa filtering statistics
│
├── processed/
│   ├── raw_counts_samples_by_taxa.tsv   # Raw counts (samples x taxa)
│   ├── log1p_abundance_samples_by_taxa.tsv
│   └── metadata_matched.tsv             # Aligned metadata
│
├── fuzzy_taxa/
│   ├── fuzzy_taxa_loaded.tsv            # Original fuzzy list
│   ├── fuzzy_taxa_matched.tsv           # Matched to OTU
│   └── fuzzy_taxa_unmatched*.tsv        # Unmatched taxa
│
├── differential_abundance/
│   ├── ancombc2/
│   │   ├── ancombc2_results.tsv
│   │   └── ancombc2_pairwise_results.tsv
│   └── aldex2/
│       ├── aldex2_results.tsv
│       └── aldex2_effects.tsv
│
├── ml/
│   ├── random_forest/
│   │   ├── rf_resample_metrics.tsv      # Per-split accuracy, F1, AUC
│   │   └── rf_permutation_importance_summary.tsv
│   └── boruta/
│       ├── boruta_summary.tsv           # Selection frequency per taxon
│       └── boruta_skipped.txt           # If not run
│
├── comparison/
│   ├── overlap_summary.tsv              # Overlap metrics with p-values
│   ├── rank_correlation_summary.tsv
│   ├── direction_agreement.tsv          # Effect direction consistency
│   ├── method_consensus_taxa.tsv        # Taxa by ≥2 methods
│   └── fuzzy_taxa_evidence_score.tsv    # **Key output**
│
├── ablation/
│   ├── ablation_metrics_long.tsv        # Per-split, per-feature-set
│   ├── ablation_metrics_summary.tsv     # Aggregated metrics
│   └── ablation_paired_comparisons.tsv
│
├── plots/
│   ├── da_volcano_ancombc2.png
│   ├── da_volcano_aldex2.png
│   ├── rf_top_permutation_importance.png
│   ├── ablation_performance_boxplot.png
│   └── ...
│
└── r_helpers/
    ├── run_ancombc2.R
    └── run_aldex2.R
```

## Key Output Files

### 1. **fuzzy_taxa_evidence_score.tsv** (Most Important)
Evidence score (0-7) for each fuzzy taxon indicating external support:
- +1 if ANCOM-BC2 significant at q < alpha
- +1 if ALDEx2 significant at q < alpha
- +1 if in RF top-N
- +1 if RF selection frequency ≥ 0.5
- +1 if Boruta selected ≥ 0.5
- +1 if direction consistent between DA and ML
- +1 if in consensus set (≥2 methods agree)

Evidence labels:
- **strong** (≥5): Multiple independent methods support
- **moderate** (3-4): Some evidence from different methods
- **weak** (1-2): Single method support
- **unsupported** (0): No external support

### 2. **overlap_summary.tsv**
Overlap statistics between fuzzy taxa and each external method:
- `n_fuzzy`, `n_method`, `n_overlap`: Set sizes
- `jaccard`, `overlap_coefficient`: Similarity metrics
- `hypergeometric_pval`, `permutation_pval`: Statistical significance

### 3. **ablation_metrics_summary.tsv**
Predictive performance of different feature sets:
- `fuzzy_taxa_only`: Performance using only fuzzy-selected taxa
- `all_taxa`: Performance using all taxa
- `all_taxa_except_fuzzy`: Can fuzzy taxa be replaced?
- `external_consensus_only`: Performance of methods-agreed taxa

### 4. **report.md**
Executive summary with counts, significant taxa, and evidence distribution.

## Important Methodological Notes

### Data Handling
- **Raw counts** used for ANCOM-BC2 and ALDEx2 (compositional methods)
- **Log1p-transformed** (log(1 + abundance)) used for Random Forest, Boruta, permutation importance
- Do NOT use `--force_log1p_for_da` unless specifically required

### ANCOM-BC2
- Bioconductor `ANCOMBC` package, `ancombc2()` function
- Supports multiclass comparisons with pairwise contrasts
- FDR-controlled q-values

### ALDEx2
- Bioconductor `ALDEx2` package
- Binary: t-test + effect sizes
- Multiclass: Kruskal-Wallis + effect sizes

### Random Forest
- StratifiedShuffleSplit with configurable test_size
- Permutation importance computed on held-out test split
- Class weights balanced to handle imbalance

### Boruta
- If `--run_boruta` specified and BorutaPy available
- Repeated within resampling for stable selection
- Returns selection frequency (≥0.5 considered robust)

## Interpreting Results

### Example Workflow:
1. Check `qc/input_summary.json` for sample/taxa counts
2. Review `report.md` for high-level summary
3. Open `comparison/fuzzy_taxa_evidence_score.tsv`:
   - Sort by `evidence_score` descending
   - Examine `evidence_label` (strong/moderate/weak/unsupported)
4. For "strong" taxa, cross-reference with:
   - `differential_abundance/`: Which methods (ANCOM-BC2, ALDEx2)?
   - `ml/random_forest/`: Rank in RF importance?
   - `comparison/direction_agreement.tsv`: Direction consistent?
5. Use `ablation/` to assess importance of fuzzy taxa for prediction:
   - Does `all_taxa_except_fuzzy` performance drop significantly?
   - Compare paired differences via Wilcoxon test (reported)

## Reproducibility

- All random operations seeded by `--random_seed`
- R helper scripts saved in `r_helpers/` for audit
- All arguments logged in `run_config.json`
- Software versions in `software_versions.txt`
- Full log in `pipeline.log`

## Troubleshooting

### ANCOM-BC2 fails:
- Check R is installed: `which Rscript`
- Install ANCOMBC: `R -e "BiocManager::install('ANCOMBC')"`
- Check output in `differential_abundance/ancombc2/ancombc2_stderr.log`

### ALDEx2 fails:
- Install ALDEx2: `R -e "BiocManager::install('ALDEx2')"`
- Check output in `differential_abundance/aldex2/aldex2_stderr.log`

### Boruta not found:
- Install: `pip install boruta`
- Or use `--run_boruta` flag to skip

### Memory issues:
- Reduce `--n_resamples` (default 50)
- Reduce `--n_estimators` (default 1000)
- Reduce `--n_permutations` (default 10000)

### Slow execution:
- Increase `--n_jobs` (default -1 uses all cores)
- Reduce filtering stringency (`--min_prevalence`, `--min_total_count`)

## Citation & References

If using this pipeline, cite:
- ANCOM-BC2: Huang et al., Nature Communications (2021)
- ALDEx2: Fernandes et al., PeerJ (2013)
- scikit-learn: Pedregosa et al., JMLR (2011)
- BorutaPy: Kursa & Rudnicki, Journal of Statistical Software (2010)

## Author

Bioinformatics Engineering Pipeline

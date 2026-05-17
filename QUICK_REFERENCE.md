# Quick Reference Card

## 30-Second Overview

**What**: Validates fuzzy-clustering-selected microbiome taxa against differential abundance (ANCOM-BC2, ALDEx2) and ML methods (Random Forest, Boruta).

**Why**: Determines if fuzzy-selected taxa are robust and clinically relevant using multiple independent approaches.

**Output**: Evidence scores (0-7) for each taxon: strong/moderate/weak/unsupported.

---

## File Locations

| File | Purpose |
|------|---------|
| `microbiome_validate_fuzzy_taxa.py` | **Main script** (2441 lines, 33 functions) |
| `VALIDATE_FUZZY_TAXA_README.md` | **START HERE** - User guide |
| `EXAMPLE_USAGE.sh` | Usage examples & commands |
| `TECHNICAL_METHODS.md` | Statistical methods detail |
| `TROUBLESHOOTING.md` | Common issues & solutions |
| `generate_test_data.py` | Test data generator |
| `DELIVERABLES_SUMMARY.md` | Complete file listing |

---

## 3-Step Quick Start

```bash
# 1. Test installation (optional)
python generate_test_data.py --output_dir test_data

# 2. Run pipeline
python microbiome_validate_fuzzy_taxa.py \
  --otu_file your_otu.tsv \
  --metadata_file your_metadata.tsv \
  --metadata_sample_col sample_id \
  --clinical_col disease_status \
  --top_taxa_file your_fuzzy_taxa.tsv \
  --output_dir ./results \
  --n_resamples 50 \
  --run_boruta

# 3. Review output
cat results/report.md
head results/comparison/fuzzy_taxa_evidence_score.tsv
```

---

## Required Arguments

| Argument | Example |
|----------|---------|
| `--otu_file` | `otu_table.tsv` |
| `--metadata_file` | `metadata.tsv` |
| `--clinical_col` | `disease_status` |
| `--top_taxa_file` | `fuzzy_taxa.tsv` |
| `--output_dir` | `./results` |

---

## Important Optional Arguments

| Argument | Default | Use Case |
|----------|---------|----------|
| `--n_resamples` | 50 | Increase to 100-500 for publication |
| `--n_permutations` | 10000 | Significance testing iterations |
| `--run_boruta` | — | Include Boruta feature selection |
| `--covariates` | — | Include age,sex,batch,etc |
| `--alpha` | 0.05 | FDR threshold |
| `--n_jobs` | -1 | Parallel cores (-1 = all) |

---

## Input File Formats

### OTU Table (TSV/CSV)
```
sample_id    OTU_001  OTU_002  OTU_003
sample_1     100      50       200
sample_2     50       25       100
```
Or taxa as rows, samples as columns (auto-detected).

### Metadata (TSV/CSV)
```
sample_id,disease_status,age,sex
sample_1,disease,45,M
sample_2,control,40,F
```

### Fuzzy Taxa (TSV/CSV)
```
cluster,taxon,membership,fuzzy_rank
cluster_1,OTU_001,0.95,1
cluster_1,OTU_002,0.92,2
```
(Only `taxon` column required; others optional)

---

## Key Outputs

| File | Contents |
|------|----------|
| `report.md` | Executive summary (read first) |
| **`fuzzy_taxa_evidence_score.tsv`** | **Main output** - evidence scores per taxon |
| `overlap_summary.tsv` | Statistical overlap metrics |
| `ablation_metrics_summary.tsv` | Feature set predictive value |
| `pipeline.log` | Full execution log |

---

## Evidence Scores

| Score | Label | Action |
|-------|-------|--------|
| 6-7 | **strong** | ✅ High confidence biomarkers |
| 4-5 | **moderate** | ⚠️ Recommend validation |
| 2-3 | **weak** | ❓ Needs investigation |
| 0-1 | **unsupported** | ❌ Reconsider inclusion |

Each point comes from:
- ✓ ANCOM-BC2 significant (+1)
- ✓ ALDEx2 significant (+1)
- ✓ RF top-N features (+1)
- ✓ RF selected >50% resamples (+1)
- ✓ Boruta selected >50% resamples (+1)
- ✓ Direction consistent DA/ML (+1)
- ✓ Consensus (≥2 methods) (+1)

---

## Installation

```bash
# Python packages
pip install pandas numpy scipy scikit-learn matplotlib seaborn
pip install boruta  # Optional

# R packages (run in R console)
BiocManager::install(c("ANCOMBC", "ALDEx2", "phyloseq"))
```

---

## Timing Guide

| Dataset Size | Time |
|--------------|------|
| 50 samples, 200 taxa, 50 resamples | ~5-10 min |
| 100 samples, 500 taxa, 100 resamples | ~30-60 min |
| 200 samples, 1000 taxa, 500 resamples | 4-8+ hrs |

---

## Common Commands

### Fast test run
```bash
python microbiome_validate_fuzzy_taxa.py \
  --otu_file data.tsv \
  --metadata_file meta.tsv \
  --clinical_col disease \
  --top_taxa_file fuzzy.tsv \
  --output_dir results_test \
  --n_resamples 10 \
  --n_permutations 100
```

### Standard run
```bash
python microbiome_validate_fuzzy_taxa.py \
  --otu_file data.tsv \
  --metadata_file meta.tsv \
  --metadata_sample_col sample_id \
  --clinical_col disease \
  --top_taxa_file fuzzy.tsv \
  --output_dir results \
  --n_resamples 100 \
  --n_permutations 10000 \
  --run_boruta
```

### Publication-quality
```bash
python microbiome_validate_fuzzy_taxa.py \
  --otu_file data.tsv \
  --metadata_file meta.tsv \
  --metadata_sample_col sample_id \
  --clinical_col disease \
  --top_taxa_file fuzzy.tsv \
  --output_dir results_pub \
  --covariates age,sex,batch \
  --n_resamples 500 \
  --n_permutations 50000 \
  --n_estimators 2000 \
  --run_boruta \
  --alpha 0.05 \
  --random_seed 42 \
  --n_jobs -1
```

---

## Debugging Quick Checks

```bash
# 1. Check if script runs
python microbiome_validate_fuzzy_taxa.py --help

# 2. Test data shapes
python generate_test_data.py --output_dir test_data

# 3. Run on test data
python microbiome_validate_fuzzy_taxa.py \
  --otu_file test_data/test_otu_table.tsv \
  --metadata_file test_data/test_metadata.tsv \
  --clinical_col disease_status \
  --top_taxa_file test_data/test_fuzzy_taxa.tsv \
  --output_dir test_output \
  --n_resamples 5

# 4. Check for errors
tail results/pipeline.log

# 5. Inspect results
head results/comparison/fuzzy_taxa_evidence_score.tsv
```

---

## When Results Look Wrong

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| All evidence_score = 0 | Fuzzy taxa don't match OTU taxa | Check `fuzzy_taxa_unmatched_in_otu.tsv` |
| Empty overlap_summary.tsv | No significant taxa from DA methods | Lower `--alpha` to 0.10 |
| Long execution time | Too many resamples | Reduce `--n_resamples` to 20-50 |
| Memory error | Dataset too large | Reduce `--n_estimates` or taxa count |
| "Rscript not found" | R not installed | Install R, ensure Rscript in PATH |

---

## Key Methods at a Glance

| Method | Type | Input | Output |
|--------|------|-------|--------|
| ANCOM-BC2 | DA | Raw counts | Effect size, q-value |
| ALDEx2 | DA | Raw counts | Effect size, q-value |
| Random Forest | ML | log1p counts | Feature importance |
| Boruta | ML | log1p counts | Selected/rejected |
| Permutation Test | Stat | Set overlap | Empirical p-value |
| Spearman Correlation | Stat | Feature ranks | ρ, p-value |

---

## Documentation Map

```
START: VALIDATE_FUZZY_TAXA_README.md
   ↓
USAGE: EXAMPLE_USAGE.sh
   ↓
METHODS: TECHNICAL_METHODS.md
   ↓
ISSUES: TROUBLESHOOTING.md
   ↓
REFERENCE: This quick card
```

---

## Output Interpretation Example

**Scenario**: Your fuzzy taxa analysis selected OTU_123

**Check evidence score** (fuzzy_taxa_evidence_score.tsv):
```
taxon      evidence_score  evidence_label
OTU_123    6               strong
```

**Interpretation**:
- ✅ Strong evidence (score 6/7)
- Likely supported by ANCOM-BC2 + ALDEx2 + RF + direction agreement
- Good candidate for biomarker validation
- Proceed with confidence

**If score was 1 (weak)**:
- ❓ Only single method support (e.g., only in RF top-N)
- Needs independent validation
- Reconsider if selecting small set

---

## Key Papers to Cite

- **ANCOM-BC2**: Huang et al. (2022) *Nat Commun* 13:6077
- **ALDEx2**: Fernandes et al. (2013) *PeerJ* 1:e46
- **Boruta**: Kursa & Rudnicki (2010) *J Stat Softw* 36(11)
- **Random Forest**: Breiman (2001) *Mach Learn* 45:5-32

---

## Technical Details Quick Reference

### Data Transformations
- **DA methods**: Raw counts (compositional)
- **ML methods**: log1p(abundance) = log(1 + x)

### Resampling
- Method: Stratified Shuffle Split
- Default: 50 random train/test splits (80/20)
- Ensures no data leakage

### Multiple Testing Correction
- Method: Benjamini-Hochberg FDR
- Default alpha: 0.05 (5% FDR)

### Multiclass Support
- Binary (2 groups): t-test, standard RF metrics
- Multiclass (>2 groups): Kruskal-Wallis, macro-averaged metrics

---

## Contact & Resources

- **Script location**: `/scratch/rv2340/Archetypal_Analysis_Project/Archetypal-Analysis/microbiome_validate_fuzzy_taxa.py`
- **Full documentation**: See included .md files
- **Test data generator**: `generate_test_data.py`
- **Example commands**: `EXAMPLE_USAGE.sh`

---

*Last updated: May 17, 2026*
*Version: 1.0 (Production)*

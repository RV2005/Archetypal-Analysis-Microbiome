# Troubleshooting Guide: Microbiome Fuzzy Taxa Validation Pipeline

## Common Issues and Solutions

### Installation & Setup

#### Issue 1: "No module named 'pandas'" or similar import error

**Symptom**: Script fails immediately with ModuleNotFoundError

**Solutions**:
```bash
# Install required packages
pip install pandas numpy scipy scikit-learn matplotlib seaborn

# Or install from requirements (if available)
pip install -r requirements.txt

# Verify installation
python -c "import pandas; import sklearn; print('OK')"
```

#### Issue 2: "Rscript: command not found"

**Symptom**: Script fails when running ANCOM-BC2 or ALDEx2
```
ANCOM-BC2 error: Rscript: command not found
```

**Solutions**:
```bash
# Check if R is installed
which Rscript
R --version

# If not installed:
# Ubuntu/Debian:
sudo apt-get install r-base r-base-dev

# macOS (using Homebrew):
brew install r

# CentOS/RedHat:
sudo yum install R
```

#### Issue 3: R package missing (ANCOMBC, ALDEx2, phyloseq)

**Symptom**: R script fails with error like:
```
Error in library(ANCOMBC) : there is no package called 'ANCOMBC'
```

**Solutions**:
```R
# Open R console
R

# Install from Bioconductor
if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

BiocManager::install("ANCOMBC")
BiocManager::install("ALDEx2")
BiocManager::install("phyloseq")

# Verify installation
library(ANCOMBC)
library(ALDEx2)
library(phyloseq)
```

#### Issue 4: BorutaPy not available but --run_boruta specified

**Symptom**: Warning message, Boruta skipped
```
WARNING: BorutaPy not installed. Install with: pip install boruta
```

**Solutions**:
```bash
# Install BorutaPy
pip install boruta

# Then re-run with --run_boruta flag

# Or skip Boruta (it's optional)
# Just remove --run_boruta from command
```

---

### Input Data Issues

#### Issue 5: Sample names don't match between OTU and metadata

**Symptom**: Error message
```
ValueError: Fewer than 2 shared samples between OTU and metadata!
```

**Solutions**:
1. **Check sample ID format**:
   ```bash
   # View first few sample IDs in each file
   head -1 otu_table.tsv | cut -f2-6
   head -2 metadata.tsv | cut -f1,2
   ```

2. **Ensure exact match** (case-sensitive):
   ```bash
   # Fix sample names to match
   # OTU: "Sample_001" vs Metadata: "sample_001"
   # Make them all lowercase or all uppercase
   ```

3. **Check for hidden characters** (spaces, tabs):
   ```bash
   # In Python
   otu_samples = set(pd.read_csv('otu.tsv', sep='\t', index_col=0).index)
   meta_samples = set(pd.read_csv('metadata.tsv', sep='\t', index_col=0).index)
   
   print(f"OTU samples: {list(otu_samples)[:3]}")
   print(f"Meta samples: {list(meta_samples)[:3]}")
   
   # Look for hidden characters
   for s in list(otu_samples)[:1]:
       print(repr(s))  # Shows hidden chars
   ```

4. **Specify correct column** if sample IDs are in a column:
   ```bash
   python microbiome_validate_fuzzy_taxa.py \
     ... \
     --metadata_sample_col sample_id \
     ...
   ```

#### Issue 6: OTU table orientation not detected correctly

**Symptom**: Transpose happens incorrectly, leading to wrong shape results

**Solutions**:
1. **Explicitly specify orientation**:
   ```bash
   python microbiome_validate_fuzzy_taxa.py \
     ... \
     --otu_orientation samples_rows \
     ...
   ```
   Options: `taxa_rows`, `samples_rows`, `auto` (default)

2. **Check your OTU table format**:
   ```bash
   # Should look like one of:
   
   # FORMAT 1 (samples_rows): Samples as rows, taxa as columns
   # sample_id    OTU_0001  OTU_0002  OTU_0003
   # sample_001   100       50        200
   # sample_002   50        25        100
   
   # FORMAT 2 (taxa_rows): Taxa as rows, samples as columns
   # taxon        sample_001  sample_002  sample_003
   # OTU_0001     100         50          200
   # OTU_0002     50          25          100
   ```

#### Issue 7: "Minimum total count" error

**Symptom**: All taxa filtered out
```
Final taxa: 0
```

**Solutions**:
1. **Check your OTU abundance scale**:
   ```python
   import pandas as pd
   otu = pd.read_csv('otu.tsv', sep='\t', index_col=0)
   print(otu.sum(axis=0).describe())  # Column (taxa) sums
   print(otu.min().min())  # Minimum value
   print(otu.max().max())  # Maximum value
   ```

2. **Adjust filtering thresholds** if data is low-abundance:
   ```bash
   python microbiome_validate_fuzzy_taxa.py \
     ... \
     --min_total_count 1 \
     --min_prevalence 0.01 \
     ...
   ```

3. **Check if data is already normalized**:
   - If data is in range [0, 1], it's probably normalized
   - Use `--min_total_count 0` or smaller values

#### Issue 8: Fuzzy taxa file format incorrect

**Symptom**: Error loading fuzzy taxa
```
ValueError: Top taxa file must contain 'taxon' column!
```

**Solutions**:
1. **Ensure 'taxon' column exists**:
   ```bash
   head -1 fuzzy_taxa.tsv  # Should show "taxon" in header
   ```

2. **Correct format examples**:
   ```
   # MINIMAL (required)
   taxon
   OTU_0001
   OTU_0002
   
   # WITH CLUSTERING (optional)
   cluster,taxon,membership,fuzzy_rank
   cluster_1,OTU_0001,0.95,1
   cluster_1,OTU_0002,0.92,2
   cluster_2,OTU_0003,0.88,1
   ```

3. **Check delimiter**:
   ```bash
   # If tab-delimited
   python microbiome_validate_fuzzy_taxa.py \
     ... \
     --sep tab \
     ...
   
   # If comma-delimited
   python microbiome_validate_fuzzy_taxa.py \
     ... \
     --sep comma \
     ...
   ```

---

### Runtime Issues

#### Issue 9: Script takes very long / runs out of memory

**Symptom**: Script runs for hours or crashes with MemoryError

**Solutions**:

1. **Reduce computational load**:
   ```bash
   python microbiome_validate_fuzzy_taxa.py \
     ... \
     --n_resamples 20 \
     --n_permutations 1000 \
     --n_estimators 200 \
     --n_jobs -1 \
     ...
   ```

2. **Test on smaller dataset first**:
   ```bash
   # Generate test data
   python generate_test_data.py --output_dir test_data --n_samples 30 --n_taxa 100
   
   # Run on test data
   python microbiome_validate_fuzzy_taxa.py \
     --otu_file test_data/test_otu_table.tsv \
     --metadata_file test_data/test_metadata.tsv \
     --clinical_col disease_status \
     --top_taxa_file test_data/test_fuzzy_taxa.tsv \
     --output_dir test_output \
     --n_resamples 10
   ```

3. **Monitor memory usage during run**:
   ```bash
   # In separate terminal
   top -p $(pgrep -f microbiome_validate)  # macOS/Linux
   ```

4. **If Boruta causes memory issues**, disable it:
   ```bash
   # Don't use --run_boruta flag
   ```

#### Issue 10: ANCOM-BC2 or ALDEx2 runs forever / timeout

**Symptom**: Script hangs at DA step, then timeout error

**Solutions**:

1. **Check R process**:
   ```bash
   ps aux | grep Rscript
   # If many R processes, kill them: killall Rscript
   ```

2. **Test R separately**:
   ```bash
   # Create small test count file
   R --vanilla < r_helpers/run_ancombc2.R
   ```

3. **Disable DA methods** (if not critical):
   ```bash
   python microbiome_validate_fuzzy_taxa.py \
     ... \
     --run_ancombc2 False \  # This flag doesn't work; instead omit it
     ...
   ```
   Note: There's no "skip" flag; the script runs both by default. Workaround: comment out DA code.

4. **Check R memory limits**:
   ```R
   # In R console
   memory.limit()
   memory.size()
   ```

#### Issue 11: "No convergence" or numerical errors in DA/ML

**Symptom**: Warnings in stderr or NaN values in outputs

**Solutions**:

1. **Check for numerical issues**:
   - NaN effect sizes → likely perfect separation or singular matrix
   - Inf values → numerical overflow

2. **Increase regularization**:
   - This is automatic in RF, but DA methods may need manual adjustment

3. **Filter more aggressively**:
   ```bash
   python microbiome_validate_fuzzy_taxa.py \
     ... \
     --min_prevalence 0.10 \
     --min_total_count 50 \
     ...
   ```

4. **Check for quasi-separation**:
   - One taxon perfectly separates groups
   - May cause logistic regression / LFC estimation to blow up
   - This is actually valuable signal but may cause numerical issues

---

### Output & Interpretation Issues

#### Issue 12: Empty overlap_summary.tsv or few methods represented

**Symptom**: Overlap comparison table missing rows or columns

**Solutions**:

1. **Check if DA methods ran successfully**:
   ```bash
   cat output_dir/differential_abundance/ancombc2/ancombc2_stderr.log
   cat output_dir/differential_abundance/aldex2/aldex2_stderr.log
   ```

2. **Check if any taxa were significant**:
   ```python
   import pandas as pd
   da_results = pd.read_csv('output_dir/differential_abundance/combined_da_results.tsv', sep='\t')
   print(da_results[da_results['significant']].shape)
   ```

3. **Lower significance threshold**:
   ```bash
   python microbiome_validate_fuzzy_taxa.py \
     ... \
     --alpha 0.10 \  # Instead of 0.05
     ...
   ```

#### Issue 13: All evidence scores = 0 (unsupported)

**Symptom**: `fuzzy_taxa_evidence_score.tsv` shows all scores at 0

**Causes & Solutions**:

1. **Fuzzy taxa don't overlap with any method**:
   - Check `comparison/overlap_summary.tsv`
   - Check if fuzzy taxa exist in OTU table: `comparison/fuzzy_taxa_unmatched_in_otu.tsv`

2. **Significance threshold too strict**:
   ```bash
   python microbiome_validate_fuzzy_taxa.py \
     ... \
     --alpha 0.10 \
     ...
   ```

3. **Methods have opposite findings**:
   - Check `direction_agreement.tsv` for consistent directions
   - May indicate true biological variation rather than unsupported taxa

#### Issue 14: Boruta summary empty

**Symptom**: `ml/boruta/boruta_summary.tsv` has no rows

**Solutions**:

1. **Check Boruta was actually run**:
   ```bash
   ls -la output_dir/ml/boruta/
   cat output_dir/ml/boruta/boruta_skipped.txt  # If skipped
   ```

2. **Boruta may find no robust features** (valid outcome):
   - Indicates fuzzy taxa not consistently selected across resamples
   - This is informative: fuzzy selection may be unstable

3. **Increase Boruta iterations** (within resamples):
   - Modify `run_boruta()` function in script, change `max_iter`

---

### Plot Generation Issues

#### Issue 15: Plots missing or empty

**Symptom**: `plots/` directory empty or has 0-byte PNG files

**Solutions**:

1. **Check matplotlib backend**:
   ```bash
   tail -50 output_dir/pipeline.log | grep -i plot
   ```

2. **Verify data exists for plotting**:
   ```bash
   wc -l output_dir/differential_abundance/combined_da_results.tsv
   wc -l output_dir/ml/random_forest/rf_permutation_importance_summary.tsv
   ```

3. **Force matplotlib to use Agg backend** (for headless servers):
   - Script already does this: `matplotlib.use('Agg')`
   - Should work without display

4. **Check disk space**:
   ```bash
   df -h output_dir
   ```

---

### Reproducibility Issues

#### Issue 16: Results differ between runs with same seed

**Symptom**: Different scores or p-values on second run with same --random_seed

**Causes**:

1. **Floating-point non-associativity** in parallel operations
   - Solution: Use `--n_jobs 1` for bit-exact reproducibility

2. **Updated package versions**:
   - Different sklearn version = slightly different RF behavior
   - Check `software_versions.txt`

3. **Different hardware** (CPU):
   - Floating-point arithmetic differs slightly across architectures
   - Expected: tiny differences (< 0.01% of values)

**Solutions**:
```bash
# For exact reproducibility
python microbiome_validate_fuzzy_taxa.py \
  ... \
  --random_seed 123 \
  --n_jobs 1 \
  ...

# Record package versions
pip freeze > requirements_exact.txt
```

---

### Debugging Workflow

#### Step 1: Enable verbose logging

The script logs to both console and `pipeline.log`. Check the log file:
```bash
tail -f output_dir/pipeline.log
```

#### Step 2: Test with minimal data

Generate tiny test dataset:
```bash
python generate_test_data.py --output_dir tiny_test --n_samples 20 --n_taxa 50

python microbiome_validate_fuzzy_taxa.py \
  --otu_file tiny_test/test_otu_table.tsv \
  --metadata_file tiny_test/test_metadata.tsv \
  --clinical_col disease_status \
  --top_taxa_file tiny_test/test_fuzzy_taxa.tsv \
  --output_dir tiny_output \
  --n_resamples 5
```

#### Step 3: Check intermediate files

Verify each step produces expected output:
```bash
# QC
cat output_dir/qc/input_summary.json

# Sample matching
head output_dir/qc/sample_matching.tsv

# Taxa filtering
cat output_dir/qc/taxa_filtering_summary.tsv

# Processed data
wc -l output_dir/processed/raw_counts_samples_by_taxa.tsv

# Fuzzy matching
head output_dir/fuzzy_taxa/fuzzy_taxa_matched.tsv
head output_dir/fuzzy_taxa/fuzzy_taxa_unmatched_in_otu.tsv
```

#### Step 4: Check individual method outputs

```bash
# DA methods
head output_dir/differential_abundance/ancombc2/ancombc2_results.tsv
head output_dir/differential_abundance/aldex2/aldex2_results.tsv

# ML methods
head output_dir/ml/random_forest/rf_permutation_importance_summary.tsv
```

#### Step 5: Use Python to examine data

```python
import pandas as pd
import json

# Load results
evidence = pd.read_csv('output_dir/comparison/fuzzy_taxa_evidence_score.tsv', sep='\t')
print(evidence['evidence_label'].value_counts())

overlaps = pd.read_csv('output_dir/comparison/overlap_summary.tsv', sep='\t')
print(overlaps[['method', 'n_overlap', 'permutation_pval']])

# Check config
with open('output_dir/run_config.json') as f:
    config = json.load(f)
    print(config)
```

---

## FAQ

**Q: Is it okay to use --force_log1p_for_da?**
A: Generally NO. It's a dangerous flag that violates the statistical design of ANCOM-BC2 and ALDEx2. Only use if you have strong reason to believe your data is pre-log-transformed.

**Q: How long should the pipeline take?**
A: 
- 50 samples, 200 taxa, 50 resamples: ~5-10 minutes
- 100 samples, 500 taxa, 100 resamples: ~30-60 minutes
- 200 samples, 1000 taxa, 500 resamples: 4-8+ hours
Timing depends on CPU cores and --n_jobs setting.

**Q: Can I run multiple instances in parallel?**
A: Yes, but use different --output_dir for each and ensure --n_jobs ≤ total_cores / n_instances.

**Q: What if my clinical column has >10 categories?**
A: Script supports any number. Multiclass handling uses per-contrast p-values and macro-averaging for ML metrics.

**Q: How do I handle samples with missing clinical labels?**
A: The script automatically drops them (reports count in pipeline.log). Manually pre-filter metadata if you want control.

---

## Contact & Support

For additional issues:
1. Check the pipeline.log file for full error details
2. Review TECHNICAL_METHODS.md for methodological questions
3. Examine the generated R scripts in r_helpers/ for DA issues
4. Test with generate_test_data.py to ensure basic functionality


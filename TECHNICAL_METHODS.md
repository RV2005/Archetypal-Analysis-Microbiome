# Technical Documentation: Microbiome Fuzzy Taxa Validation Pipeline

## Table of Contents

1. [Overview](#overview)
2. [Data Processing](#data-processing)
3. [Differential Abundance Analysis](#differential-abundance-analysis)
4. [Machine Learning Methods](#machine-learning-methods)
5. [Comparative Analysis](#comparative-analysis)
6. [Predictive Ablation](#predictive-ablation)
7. [Statistical Testing](#statistical-testing)
8. [Reproducibility](#reproducibility)

---

## Overview

The pipeline integrates multiple independent methods to validate microbiome taxa selected by fuzzy clustering. By comparing fuzzy-selected taxa against differential abundance (DA) methods and machine learning (ML) classifiers, we assess whether selected taxa represent robust, clinically-relevant features.

### Design Principles

- **Avoid data leakage**: All model fitting and feature selection occurs within resampling folds
- **Compositional awareness**: Use appropriate methods for compositional microbiome data
- **Complementary approaches**: Different methods (DA, ML, consensus) capture different aspects of feature importance
- **Reproducibility**: Full logging, version tracking, and R script generation

---

## Data Processing

### Input Validation

1. **Load and Orientation Detection**
   - Accept OTU tables in either orientation (taxa x samples or samples x taxa)
   - Auto-detect using metadata sample ID matching
   - Support auto-delimiter detection (tab/comma)

2. **Sample Matching**
   - Identify shared samples between OTU and metadata
   - Report unmatched samples and taxa
   - Align data matrices

3. **Taxa Filtering**
   - **Prevalence filter**: Taxa present in ≥ `min_prevalence` (default 5%) of samples
   - **Abundance filter**: Taxa with total count ≥ `min_total_count` (default 10)
   - **Rationale**: Rare taxa add noise without biological significance
   - Save dropped taxa and filtering statistics

4. **Clinical Label Validation**
   - Drop samples with missing clinical labels
   - Encode categorical labels using LabelEncoder
   - Verify ≥2 clinical classes
   - Support binary and multiclass

### Data Transformations

| Step | Transformation | Used For | Rationale |
|------|----------------|----------|-----------|
| 1 | Raw counts | DA methods (ANCOM-BC2, ALDEx2) | Compositional methods designed for raw counts |
| 2 | log1p(x) = log(1+x) | ML & permutation importance | Stabilize variance, handle zeros, improve ML performance |

**Critical**: Do NOT feed log1p data to ANCOM-BC2 or ALDEx2 unless `--force_log1p_for_da` is explicitly set (with warnings).

---

## Differential Abundance Analysis

### ANCOM-BC2 (Analysis of Composition of Microbiomes with Bias Correction, version 2)

**Reference**: Huang et al., "ANCOM-BC2 improves tests of taxa-wise and category-wise differences," *Nature Communications* 13, 6077 (2022)

#### Method Summary

- **Input**: Raw count matrix (taxa × samples), metadata
- **Model**: Generalized additive log-linear model with bias correction
- **Handles**: 
  - Compositional bias
  - Unbalanced sequencing depth
  - Confounding covariates

#### Implementation

```R
ancombc2(
  data = phyloseq_object,
  fix_formula = "~clinical_group + covariate_1 + ...",
  p_adj_method = "BH",  # Benjamini-Hochberg FDR
  alpha = 0.05,
  pairwise = TRUE,      # Pairwise comparisons if >2 groups
  global = TRUE         # Global test if >2 groups
)
```

#### Output Schema

| Column | Description |
|--------|-------------|
| `taxon` | OTU/ASV identifier |
| `effect_size` | log Fold Change (LFC) of normalized abundance |
| `statistic` | W-statistic |
| `p_value` | Unadjusted p-value |
| `q_value` | FDR-adjusted p-value (Benjamini-Hochberg) |
| `direction` | Sign of effect: positive/negative |
| `significant` | Boolean (q_value < alpha) |

#### Multiclass Handling

For >2 clinical groups:
- Run pairwise contrasts (group A vs group B)
- Include global omnibus test
- Report FDR-controlled q-values across all contrasts

---

### ALDEx2 (ANOVA-Like Differential Expression Tool for HTS Count Data, version 2)

**Reference**: Fernandes et al., "Unifying the analysis of high-throughput sequencing data," *PeerJ* 1, e46 (2013)

#### Method Summary

- **Input**: Raw count matrix, metadata
- **Approach**: Monte-Carlo sampling to account for per-sample uncertainty
- **Compositional handling**: Works with count-based inference rather than normalizing

#### Implementation

**Binary Case (2 groups)**:
```R
x <- aldex(counts, groups, mc.samples=128, test="t.test")
effect_size <- aldex.effect(x)  # Effect sizes, Cohen's d
```

**Multiclass Case (>2 groups)**:
```R
x <- aldex(counts, groups, mc.samples=128, test="kw")  # Kruskal-Wallis
effect_size <- aldex.effect(x)
```

#### Output Schema

| Column | Description |
|--------|-------------|
| `taxon` | OTU/ASV identifier |
| `p_value` | Unadjusted p-value (t-test or Kruskal-Wallis) |
| `q_value` | FDR-adjusted p-value |
| `effect_size` | Cohen's d (effect magnitude) |
| `direction` | Inferred from effect size sign |
| `significant` | Boolean (q_value < alpha) |

---

## Machine Learning Methods

### Random Forest Classification

#### Architecture

```python
RandomForestClassifier(
  n_estimators=1000,           # 1000 trees
  class_weight='balanced',      # Handle class imbalance
  max_depth=20,                # Prevent overfitting
  min_samples_split=5,
  min_samples_leaf=2
)
```

**Input**: log1p-transformed OTU table (samples × taxa)

**Target**: Encoded clinical labels (0, 1, 2, ...)

#### Resampling Strategy

**Repeated Stratified Shuffle Split**:
```python
StratifiedShuffleSplit(
  n_splits=50,        # Default: 50 resamples
  test_size=0.2,      # 20% test / 80% train
  random_state=seed
)
```

Benefits:
- Stratified: Each split maintains class proportions
- Repeated: Multiple random splits provide robust estimates
- Prevents data leakage: Test set completely held out

#### Performance Metrics

Per split, calculate:

| Metric | Formula | Use |
|--------|---------|-----|
| Accuracy | TP+TN / Total | Overall correctness |
| Balanced Accuracy | Mean(TP_rate, TN_rate) | For imbalanced classes |
| Macro F1 | Mean(F1_per_class) | Per-class harmonic mean |
| ROC-AUC | Area under PR curve | Discrimination ability |

For multiclass: macro-averaged or one-vs-rest aggregation

#### Permutation Importance

**Definition**: How much model performance degrades when a feature's values are randomly shuffled?

**Calculation** (on held-out test set):
1. Fit model on training set
2. Record baseline score on test set
3. For each feature:
   - Shuffle feature values in test set
   - Record score with shuffled feature
   - importance = baseline_score - shuffled_score
4. Repeat shuffling (n_repeats=10) to get mean ± std

**Properties**:
- Feature-agnostic (works with any model)
- Captures feature interactions
- Computed on held-out data (no data leakage)

**Output**:
- Mean importance per taxon (aggregated across resamples)
- Std importance (variation across resamples)
- Selection frequency = fraction of resamples where importance > 0

---

### Boruta Feature Selection

**Reference**: Kursa & Rudnicki, "Feature Selection with the Boruta Package," *Journal of Statistical Software* 36(11): 1-13 (2010)

#### Method Summary

- **Type**: All-relevant feature selection
- **Approach**: Compare feature importance to shadow (permuted) features
- **Goal**: Identify all truly predictive features (not just minimal set)

#### Implementation

```python
BorutaPy(
  estimator=RandomForestClassifier(n_estimators=100),
  n_estimators='auto',  # Auto-tune iterations
  max_iter=100,
  random_state=seed
)
boruta.fit(X_train, y_train)
```

#### Output Categories

| Category | Meaning |
|----------|---------|
| **Selected** | Feature importance > shadow consistently |
| **Tentative** | Borderline, may become selected with more data |
| **Rejected** | Feature importance < shadow |

#### Aggregation Across Resamples

- Run Boruta on each resample's training set
- Selection frequency = (# resamples selected) / n_resamples
- Robust selected = frequency ≥ 0.5 (selected in ≥50% of resamples)

---

## Comparative Analysis

### Overlap Analysis

#### Metrics

**1. Overlap Size**
- n_fuzzy = |Fuzzy taxa set|
- n_method = |Method taxa set|
- n_overlap = |Fuzzy ∩ Method|

**2. Similarity Indices**

- **Jaccard Index**:
  $$J = \frac{|A \cap B|}{|A \cup B|}$$
  Range: [0, 1]. Equal to 0 iff no overlap.

- **Overlap Coefficient**:
  $$OC = \frac{|A \cap B|}{\min(|A|, |B|)}$$
  Range: [0, 1]. Emphasizes smallest set.

**3. Statistical Significance**

#### Hypergeometric Test

**Null hypothesis**: Overlap is random given set sizes and universe size.

**Formula** (one-sided survival function):
$$P(X \geq k) = \sum_{i=k}^{\min(n, K)} \frac{\binom{K}{i}\binom{N-K}{n-i}}{\binom{N}{n}}$$

Where:
- N = total taxa (universe)
- K = method taxa count
- n = fuzzy taxa count
- k = observed overlap

**Interpretation**: Low p-value (< 0.05) indicates overlap is more than random chance.

#### Permutation Test

**Procedure**:
1. For each of 10,000 permutations:
   - Randomly sample n_fuzzy taxa from universe
   - Randomly sample n_method taxa from universe
   - Count overlap
2. Empirical p-value = (# permutations with overlap ≥ observed) / 10,000
3. Add pseudocount: p = (1 + count) / (1 + n_permutations)

**Advantages over hypergeometric**:
- No distributional assumptions
- Accounts for universe composition
- More conservative (often higher p-values)

---

### Rank Correlation

#### Method: Spearman Rank Correlation

**Rationale**: Features ranked by importance metrics may not correlate linearly; rank correlation is more robust.

#### Ranking Schemes

| Method | Rank Source |
|--------|------------|
| Fuzzy | fuzzy_rank column or membership score |
| ANCOM-BC2 | -log10(q_value) or abs(effect_size) |
| ALDEx2 | -log10(q_value) or abs(effect_size) |
| Random Forest | mean_permutation_importance |
| Boruta | inverse(ranking) from Boruta |

#### Calculation

For taxa present in both fuzzy and method lists:
1. Rank fuzzy taxa: 1 (top) to n_overlap
2. Rank method taxa: 1 (top) to n_overlap
3. Compute Spearman ρ and p-value

**Interpretation**:
- ρ ≈ 1: Ranks agree (consistent importance)
- ρ ≈ 0: No rank correlation
- ρ ≈ -1: Inverse ranks (unexpected)

**Handling small overlaps**: Report NA if < 3 overlapping taxa (insufficient for correlation).

---

### Direction Agreement

#### For Differential Abundance Methods

**Direction** = sign(effect_size)
- Positive: Elevated in clinical group (or higher covariate level)
- Negative: Reduced in clinical group

#### For Machine Learning

**Approximate direction** via class-wise mean difference:
$$\text{direction} = \text{sign}(\bar{x}_{\text{class+}} - \bar{x}_{\text{class-}})$$

Where $\bar{x}_{\text{class+}}$ is mean abundance in positive/higher class.

#### Agreement Score

- Agreement = True if DA direction == ML direction
- Agreement = False if directions conflict
- Agreement = NaN if direction indeterminate

**Key assumption**: Direction should be consistent if taxa is truly associated with phenotype.

---

## Predictive Ablation

### Feature Set Definitions

| Set | Description | Purpose |
|-----|-------------|---------|
| **fuzzy_taxa_only** | Only fuzzy-selected taxa | Is fuzzy selection sufficient? |
| **all_taxa** | All filtered taxa (baseline) | Best possible with all info |
| **all_taxa_except_fuzzy** | All except fuzzy | Are fuzzy taxa replaceable? |
| **external_consensus_only** | Taxa from ≥2 methods | Robustness of consensus |
| **clinical_covariates_only** | Age, sex, batch, etc. | Non-microbiome baseline |

### Evaluation Protocol

**Identical resampling across sets**:
```python
for split_idx, (train_idx, test_idx) in enumerate(rss.split(X, y)):
  for feature_set_name, features in feature_sets.items():
    X_train = X[features].iloc[train_idx]
    X_test = X[features].iloc[test_idx]
    # Fit and evaluate with same split
```

**Key**: Same train/test indices ensure fair comparison.

### Metrics per Split

- Accuracy, Balanced Accuracy, Macro F1, ROC-AUC

### Aggregation Across Resamples

| Statistic | Use |
|-----------|-----|
| Mean | Central tendency |
| Median | Robust central tendency |
| Std | Variability |
| 95% CI | Uncertainty range |

### Paired Comparisons

**Hypothesis**: 
- H0: Performance(fuzzy_only) = Performance(all_taxa)
- H1: Performance(fuzzy_only) ≠ Performance(all_taxa)

**Test**: Wilcoxon Signed-Rank (non-parametric paired test)
- Uses signed ranks across resamples
- Robust to outliers

**Interpretation**:
- Significant difference: Fuzzy taxa add/remove value
- No significant difference: Fuzzy provides no independent contribution

---

## Statistical Testing

### Multiple Testing Correction

**Global False Discovery Rate (FDR) Control**:

All p-values across methods (DA, ML, overlap) use Benjamini-Hochberg FDR correction:

$$\text{q}_i = p_i \times \frac{m}{\text{rank}(p_i)}$$

Where m = total number of tests, rank in ascending order.

### Significance Thresholds

Default: α = 0.05 (5% FDR)

Interpretation: 5% of discoveries are expected false positives.

---

## Evidence Scoring

### Scoring Scheme

Each fuzzy taxon receives +1 for each supporting criterion:

```
Score = 0-7 points from:
  [+1] ANCOM-BC2 significant (q < α)
  [+1] ALDEx2 significant (q < α)
  [+1] RF top-N (where N = n_fuzzy)
  [+1] RF selection_frequency ≥ 0.5
  [+1] Boruta selected ≥ 0.5 (if Boruta run)
  [+1] Direction agreement consistent
  [+1] External consensus (≥2 methods)
```

### Evidence Labels

| Label | Score | Interpretation |
|-------|-------|-----------------|
| **Strong** | ≥5 | Multiple independent methods support |
| **Moderate** | 3-4 | Consistent evidence from several approaches |
| **Weak** | 1-2 | Single method or weak consensus |
| **Unsupported** | 0 | No external support detected |

### Rationale

- **Diversification**: Points from different method families (DA, ML, consensus)
- **Robustness**: Multiple resamples increase confidence
- **Orthogonality**: DA and ML capture different signal aspects
- **Transparency**: Users can audit decision logic

---

## Reproducibility

### Determinism

All randomness seeded by `--random_seed`:

```python
np.random.seed(seed)
random.seed(seed)
sklearn_model.random_state = seed
```

**Consequence**: Same seed produces identical results on same hardware/versions.

### Audit Trail

1. **pipeline.log**: Full execution log with timestamps
2. **run_config.json**: All parameters as JSON
3. **software_versions.txt**: Python & R package versions
4. **r_helpers/**: Generated R scripts (reproducible R execution)

### Session Reproducibility

To reproduce results:
```bash
# Install exact package versions (if available)
pip install -r requirements.txt

# Run with same seed
python microbiome_validate_fuzzy_taxa.py \
  --otu_file ... \
  --output_dir ./results_repro \
  --random_seed 123 \
  --n_jobs 1  # Single-threaded for bit-exact reproducibility
```

**Note**: Bit-exact reproducibility across machines requires single-threaded execution (`--n_jobs 1`) due to floating-point non-associativity in parallel operations.

---

## References

- **ANCOM-BC2**: Huang et al., *Nature Communications* 13:6077 (2022)
- **ALDEx2**: Fernandes et al., *PeerJ* 1:e46 (2013)
- **Boruta**: Kursa & Rudnicki, *J Stat Softw* 36(11):1-13 (2010)
- **Permutation Importance**: Breiman, *Mach Learn* 45:5-32 (2001)
- **Scikit-learn**: Pedregosa et al., *JMLR* 12:2825-2830 (2011)
- **Microbiome Analysis**: Gloor et al., *Front Microbiol* 8:2224 (2017)

---

## FAQ

### Q: Why use raw counts for DA but log1p for ML?

A: DA methods (ANCOMBC2, ALDEx2) are designed for compositional microbiome data and model raw count distributions directly. ML methods like RF work better with variance-stabilized, log-transformed data to reduce influence of extreme values.

### Q: Should I trust weak evidence (score 1-2)?

A: Use with caution. Weak evidence indicates single-method support, possibly due to chance correlation. Recommended: Validate experimentally or with independent cohort.

### Q: Why permutation test overlaps instead of hypergeometric?

A: Both are valid; permutation test makes fewer assumptions and is more conservative. We report both in output for triangulation.

### Q: How to interpret negative rank correlation?

A: Inverse ranks suggest opposing importance rankings between methods. Investigate if biological phenotype has dual roles or methods capture different aspects (e.g., abundance vs. frequency).

### Q: Can I use categorical covariates?

A: Yes, but encode as numeric (e.g., sex: M=1, F=2). Continuous covariates are preferred for ablation analysis.

### Q: What if I have very few samples?

A: Use smaller n_resamples (e.g., 20) to avoid excessive train set resampling. ANCOM-BC2 and ALDEx2 require ≥5-10 samples per group minimum.


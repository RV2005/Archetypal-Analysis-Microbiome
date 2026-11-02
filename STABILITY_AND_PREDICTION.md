# Predictive and stability analyses

The pipeline now contains two independently resumable stages between `benchmark` and
`analyze`:

```text
load -> filter -> qc -> fit -> evaluate -> select -> benchmark
     -> prediction -> stability -> analyze -> plot
```

`prediction` and `stability` have separate fingerprints and output directories. Changing
one analysis therefore does not cause the other analysis, AA K sweep, or dimensionality
reduction benchmarks to be recomputed. Stability AA fits additionally have per-run
`_SUCCESS.json` files, so an interrupted seed/subsampling sweep resumes completed runs.

## Equal-dimensional metadata prediction

`prediction` now performs **outer-fold refitting**. No AA or dimensionality-reduction
representation that has seen an outer-test sample is used to score that sample. For every outer
fold the pipeline:

1. learns training-dependent AA preprocessing on the training samples only;
2. applies that fitted preprocessing unchanged to the held-out samples;
3. refits AA at the selected K on the training fold;
4. projects held-out samples onto the fixed training-fold archetypes with simplex-constrained
   least squares;
5. refits PCA/PCoA/UMAP on the same training fold and transforms/projects held-out samples with
   those training-fitted models; and
6. trains the supervised linear/logistic predictor on the training representation and scores it
   only on the outer-test representation.

The persisted all-data AA and global benchmark artifacts are **not** used as prediction
representations. Fold-specific representation artifacts under `07_prediction/fold_models/` have
their own fingerprints and are reused when only supervised-model settings change.

### K versus K-1 dimensions

K AA similarity weights sum to one, so the native K-column matrix has rank at most K-1. The
pipeline therefore reports two AA probes:

- `archetypal_analysis_direct`: native K simplex-weight columns; `effective_dimension=K-1`.
- `archetypal_analysis_helmert`: the same weights multiplied by a deterministic orthonormal
  Helmert basis, yielding exactly K-1 independent coordinates.

The Helmert basis spans the subspace orthogonal to the all-ones vector, so it removes only the
direction fixed by the sum-to-one constraint and preserves Euclidean distances between valid
simplex-weight vectors. PCA, PCoA, UMAP, and t-SNE benchmark dimensions are K-1 as well. The
global `benchmark` stage follows the same K-1 rule.

PCA uses its native `transform`, PCoA uses a Gower supplementary-point projection derived from
training distances/eigendecomposition, and UMAP uses its native `transform`. scikit-learn t-SNE
has no out-of-sample transform. Under strict leakage-free prediction, t-SNE therefore receives an
`unavailable`/NA score with an explicit reason rather than being fit jointly with test samples or
using an unrelated KNN surrogate.

### Predictor fitting and regularization

Categorical metadata uses standardized logistic regression. Its inverse regularization strength
`C` is chosen inside each outer training fold by a second, inner cross-validation using the
configured grid. If the outer CV is group-disjoint, the inner C-selection CV is group-disjoint too.
Continuous metadata uses standardized ordinary linear regression and reports R2/MAE; `C` does not
apply to ordinary least-squares regression.

```yaml
prediction:
  enabled: true
  target: null
  raw_baseline: qc_otu      # qc_otu | fit_input (fold-fitted preprocessing)
  group_column: null
  group_fallback_columns: []
  group_auto_detect: true
  group_namespace_columns: [project, Study, StudyID, Cohort, location]
  n_splits: 5
  random_state: 42

  aa_projection:
    max_iter: 2000
    tol: 1.0e-8

  logistic_regression:
    C_values: [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    inner_cv_splits: 4
    scoring: balanced_accuracy   # balanced_accuracy | f1_macro
    max_iter: 5000
    class_weight: null
    solver: lbfgs

  linear_regression:
    fit_intercept: true
```

For repeated measures, set `group_column`. The pipeline uses group-disjoint outer folds and resolves
heterogeneous subject-ID metadata from the canonical standardized metadata file. For categorical
targets it uses `StratifiedGroupKFold` when available; continuous targets use `GroupKFold`.

Outputs are under `07_prediction/`, including `fold_assignments.tsv`, `prediction_scores.tsv`,
per-method out-of-fold predictions/fold metrics, the C chosen in every categorical outer fold, and
the cached fold-specific AA/benchmark representations.

## Initialization stability

The original selected-K AA fit in `03_fit/K_XXX` is the fixed reference and is never
refitted. Additional seeds use 100% of the same already-preprocessed `fit_input` and the
same K/model settings:

```yaml
stability:
  enabled: true
  profile_space: fitted       # fitted | inverse
  matching_metric: cosine     # cosine | correlation
  initialization:
    enabled: true
    seeds: [17, 29, 53, 71, 97]
```

For every candidate run, archetypes are optimally assigned to the reference archetypes
with the Hungarian algorithm using the configured taxonomic-profile similarity. Because
archetype numbering is arbitrary, all comparisons are permutation invariant.

The following values are retained:

- **Per reference archetype:** maximum cosine similarity to any candidate archetype.
- **Per reference archetype:** minimum Euclidean taxonomic-profile distance to any
  candidate archetype.
- Matched cosine/Euclidean values from the one-to-one global assignment.
- **All archetypes:** normalized empirical Wasserstein-1 distance. Archetype profiles are
  L2-normalized, the minimum mean Euclidean assignment cost is computed, and it is divided
  by 2 (the maximum Euclidean distance between unit vectors), producing a scale-free
  set-level distance in `[0, 1]`.

Summary TSVs report medians and 2.5/97.5 percentiles. The plot stage produces per-archetype
cosine/Euclidean violin plots and a set-level normalized-Wasserstein violin.

## Sampling stability

Sampling fits reuse the same selected K and already-preprocessed matrix:

```yaml
stability:
  sampling:
    enabled: true
    fractions: [0.5, 0.6, 0.7, 0.8, 0.9]
    repeats: 5
    random_state: 42
    continuous_stratify_bins: 10
    group_column: null
```

Categorical sampling is stratified by the task category. Continuous sampling is stratified
using quantile bins solely for creating balanced subsamples; age itself remains continuous
for downstream metrics. If a group column is configured, entire subjects are sampled as
units. `sampling.group_column: null` inherits `prediction.group_column`.

Each subsample refits AA once and is compared with the original 100% reference using the
same optimal-assignment metrics above. The existing AA representation metrics are also
computed for every subsample fit.

## Held-out projection

The 80% and 90% sampling fits are reused for held-out projection by default; no extra AA
fit is performed:

```yaml
stability:
  heldout_projection:
    enabled: true
    fractions: [0.8, 0.9]
    max_iter: 2000
    tol: 1.0e-8
```

For omitted samples `X` and fixed subset-derived archetypes `Z`, the pipeline solves

```text
min_A ||X - A Z||_F^2
subject to A_ij >= 0 and every row of A sums to 1.
```

A vectorized projected-gradient solver projects every update onto the probability simplex.
The resulting memberships are saved as `heldout_projection.tsv` and the existing AA
representation metrics plus native projection reconstruction RSS are collected in
`heldout_projection_metrics.tsv`. With grouped sampling, held-out subjects are completely
absent from the subset used to learn the archetypes.

## Parallelism and reuse

Stability runs use a process pool and `SLURM_CPUS_PER_TASK` when `n_jobs: 0`. Each worker
loads the full `fit_input.tsv` only once and reuses it for multiple assigned fits. Every
initialization/subsampling run has its own fingerprint based on K, seed, exact training
sample IDs, fit model settings, and the upstream fit fingerprint.

## Sample-ID round-trip safety

Prediction and stability now read the persisted `sample` column as literal text rather than
allowing pandas to infer its dtype. This prevents identifiers such as `001` from becoming `1`
when a subset happens to contain only numeric-looking IDs. For artifacts produced before this
fix, QC metadata/OTU rows can be reconciled to the existing `fit_input` index using a unique
numeric alias (`1` -> `001`) only when that mapping is unambiguous. AA similarity and benchmark
embeddings with the same row count can also restore the authoritative fit-input row index by
position because those artifacts are written in fit-input order. These repairs are recorded in
`prediction_manifest.json`; ambiguous or stale mismatches still raise an explicit error rather
than silently pairing the wrong samples.

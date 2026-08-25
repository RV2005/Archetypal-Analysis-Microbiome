# Archetypal Analysis Pipeline Refactor

This refactor turns the existing loaders, QC, AA, metrics, and visualization code into one config-driven pipeline with resumable artifacts and a single Slurm launcher.

## Repository layout

```text
archetypal_analysis/
  cli.py                         # aa-pipeline command
  data/
    dataset.py                   # unified otu + metadata loader
    conversion.py                # legacy -> standard schema conversion
    filtering.py                 # legacy DataNormalizer-style filtering/balancing
    quality_control.py           # existing QC implementation
  analysis/
    archetypal.py                # existing AA wrapper + sklearn compatibility
    selection.py                 # best-K policies
    experiments.py               # registry of downstream analyses
    metrics.py                   # existing metric utilities
  visualization/
    simplex.py                   # existing simplex plotting
    stacked_bar.py               # existing similarity bars
    pcoa.py                      # existing PCoA
    embeddings.py                # new UMAP/t-SNE + shared embedding utilities
    pipeline_plots.py            # plot orchestration helpers
  pipeline/
    config.py                    # YAML inheritance/deep merge/validation
    artifacts.py                 # fingerprints and completion markers
    runner.py                    # dependency-aware stage runner
    stages/                      # one module per pipeline stage
configs/
  defaults.yaml                  # global defaults
  datasets/                      # standardized dataset locations only
  experiments/                   # task + per-stage experiment overrides
notebooks/
  convert_dataset_to_standard_schema.ipynb
sbatch/
  run_aa.sh                      # the single Slurm entrypoint
```

## Standard dataset schema

Every dataset has the same physical contract:

```text
datasets/standardized/<dataset>/
  otu.tsv
  metadata.tsv
```

Both files use the same unique `sample` index. `otu.tsv` contains only numeric features/taxa. `metadata.tsv` is a wide table and may contain any columns needed by current or future tasks. The pipeline never branches on dataset name.

For compendium data, use the conversion notebook or `aa-pipeline convert --kind compendium ...` to merge sample metadata, project metadata, and long-form tags into one wide `metadata.tsv`.

## Config strategy

Configs are layered deliberately:

1. `configs/defaults.yaml` owns algorithm/plot defaults.
2. `configs/datasets/*.yaml` owns only the standardized dataset location/schema.
3. `configs/experiments/*.yaml` owns the scientific task and stage overrides.

An experiment can inherit more than one YAML using `extends`. Dictionaries deep-merge; lists/scalars replace their parent values. This keeps dataset paths reusable while allowing every experiment to override filtering, QC, K sweep, selection, downstream analyses, and visualization independently.

The migrated `dog_lifestyle`, `hadza_age`, and `compendium_sample_type` examples preserve the target columns, K ranges, main K choices, colors, label maps, and age buckets represented in the old JSON config.

## Pipeline stages and artifacts

The canonical stage order is:

```text
load -> filter -> qc -> fit -> evaluate -> select -> analyze -> plot
```

Artifacts are written under `results/<experiment.name>/`:

```text
01_load/
02_filter/
03_qc/
04_fit/
  fit_input.tsv
  transform.json
  K_003/{result.json,similarity.tsv,archetypes.tsv,_SUCCESS.json}
  K_004/...
05_evaluate/k_metrics.tsv
06_select/selection.json
07_analyze/K_003/...
08_plot/
  common/
    silhouette_davies_bouldin_over_k.png
    normalized_quality_over_k.png
    simplex_across_k.png
    pcoa.png
    umap.png
    tsne.png
  K_003/
    simplex.png
    similarity_stacked_bar.png
    taxa_bars.png
    taxa_stacked_bar.png
    pcoa.png
    umap.png
    tsne.png
```

Each stage writes `_SUCCESS.json` with a fingerprint of its relevant config plus upstream fingerprints. The fit stage also fingerprints every K directory independently. This lets an interrupted K sweep reuse completed K values.

## Running

Install once from repository root:

```bash
pip install -e .
```

Run the full pipeline locally:

```bash
aa-pipeline run --config configs/experiments/dog_lifestyle.yaml
```

Run one stage (dependencies are checked/run/reused automatically):

```bash
aa-pipeline run --config configs/experiments/dog_lifestyle.yaml --stages fit
```

Run several stages:

```bash
aa-pipeline run --config configs/experiments/dog_lifestyle.yaml --stages evaluate,select,plot
```

Regenerate only an explicitly requested stage while reusing dependencies:

```bash
aa-pipeline run --config configs/experiments/dog_lifestyle.yaml --stages plot --overwrite
```

Inspect resolved inheritance or status:

```bash
aa-pipeline resolve-config --config configs/experiments/dog_lifestyle.yaml
aa-pipeline status --config configs/experiments/dog_lifestyle.yaml
```

On Slurm, the one launcher is:

```bash
sbatch --export=ALL,CONFIG=configs/experiments/dog_lifestyle.yaml,CONDA_ENV=GHG_work sbatch/run_aa.sh
```

`fit.parallel.n_jobs: 0` uses `SLURM_CPUS_PER_TASK`; K values are parallelized inside the single job rather than through a job array.

## Best-K modes

`selection.mode` supports:

- `all`: skip selection; all K values get K-dependent downstream analysis/plots.
- `fixed`: use `selection.fixed_k`.
- `quality`: min-max normalize Silhouette and Davies-Bouldin over K, calculate `silhouette_normalized - davies_bouldin_normalized`, and choose the lowest K within `within_percent` of the maximum score.
- `reconstruction`: reconstruct fitted-space samples as `similarity_degree @ archetypes`, evaluate the configured reconstruction metric, and choose the lowest K within `within_percent` of the minimum error.

Two ready-to-run examples are included: `example_quality_selection.yaml` and `example_reconstruction_selection.yaml`.

## Extending downstream experiments

Add a function decorated with `@register_analysis("name")` in `analysis/experiments.py` (or import a module that registers one), then add the name to `analysis.modules` in YAML. The runner does not need to change.

The built-ins are `sample_scores`, `dominance`, `metadata_association`, and `top_taxa`.

## Notes on behavior retained from the old code

- QC delegates to the existing `QualityController` and uses the same taxonomy-level, prevalence, sequencing-depth, and abundance filters.
- Filtering retains the old `control` + `normalize` balancing semantics, but reads from one combined metadata table instead of dataset-specific projects/tags/sample metadata sources.
- AA retains the `pgd`, `aa_plus_plus`, optimizer kwargs, random-state controls, and scikit-learn compatibility shim.
- Existing simplex, stacked-bar, and PCoA implementations remain in the package; UMAP and t-SNE are added as configurable dataset embeddings.

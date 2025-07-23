# Archetypal Analysis Package

A rich package for microbiome evaluation and visualization through Archetypal Analysis. 

## Overview

This package provides a comprehensive toolkit for performing archetypal analysis on microbiome data, including:

- Data loading and preprocessing (On the Compendium and Hadza Dataset)
- Quality control and normalization
- Archetypal analysis 
- Rich visualization capabilities (simplex plots, composition charts, stacked bar plots)
- Metadata integration and analysis

## Installation

### Prerequisites

```bash
pip install pandas numpy matplotlib scikit-learn scikit-bio tqdm
pip install git+https://github.com/RV2005/archetypes.git
```

### Package Installation

1. Clone or download this package
2. Add the package directory to your Python path, or install in development mode:

```bash
cd archetypal_analysis
pip install -e .
```

## Quick Start

## Key Features

### 1. Data Loading and Preprocessing

Load OTU tables and perform quality control.

```python
from archetypal_analysis.data.loaders import CompendiumDataLoader
from archetypal_analysis.data.quality_control import QualityController
import os

data_loader = CompendiumDataLoader()
otu_file = r"datasets/compendium/taxonomic_table.csv/raw_taxa_110.csv"
projects_file = r"datasets/compendium/projects.csv"
tags_file = r"datasets/compendium/tags.tsv"
meta_file = r"datasets/compendium/sample_metadata.tsv"
otu, projects, tags, meta = data_loader.load_all(otu_file=otu_file, projects_file=projects_file, tags_file=tags_file, meta_file=meta_file)


control = {"region": ["Europe and Northern America"]} 
normalize = {"sample_type": ['cervicovaginal swab', 'appendiceal tissue', 'stool', 'ileal mucosal sample', 'fixed tissue samples', 'esophageal biopsy', 'tumor samples']}
ignore = ["unknown"]
tol = 0 
data_normalizer = DataNormalizer()
otu_norm, norm_meta = data_normalizer.normalize_data(
    control=control,
    normalize=normalize,
    otu=otu,
    meta=meta, projects=projects, tags=tags,
    continuous=False,
    ignore=ignore,
    tol=tol,
    verbose=True
)

# Perform quality control
qc = QualityController()
otu_qc = qc.quality_control(
    otu=otu_norm,
    verbose=True,
    min_taxa_level=3,
    min_nonzero_samples=0.01,
    min_counts=5000,
)
print(otu_qc.shape)
```

### 4. Metadata Parsing

Parse and extract metadata for the samples. This is often a prerequisite for visualization.

```python
from archetypal_analysis.utils.metadata import extract_metadata
from archetypal_analysis.data.loaders import CompendiumDataLoader

# Load metadata files
data_loader = CompendiumDataLoader()
projects_file = r"datasets/compendium/projects.csv"
tags_file = r"datasets/compendium/tags.tsv"
meta_file = r"datasets/compendium/sample_metadata.tsv"
_, projects, tags, meta = data_loader.load_all(otu_file=None, projects_file=projects_file, tags_file=tags_file, meta_file=meta_file)

# Extract specific metadata columns for your samples
# 'otu_aa' is the similarity_degree matrix from an analysis result (see Visualization section)
filter_cols = ["region", "age_months", "sample_type"]
metadata = extract_metadata(otu_aa.index, tags, meta, projects, extract_cols=filter_cols, verbose=False)

print(metadata.head())
```

### 2. Archetypal Analysis

Run archetypal analysis on the preprocessed data.

```python
from archetypal_analysis.analysis.archetypal import ArchetypalAnalysis

# Configuration
FRAC = 1.0
LOG_TRANSFORM = True
N_ARCHETYPES = 5
outDir = r"results"


# Initialize and run analysis
aa = ArchetypalAnalysis(
    n_archetypes=N_ARCHETYPES,
    method="pgd",
    max_iter=100,
)
# otu_qc is from the Data Loading and Preprocessing step
result = aa.fit_transform(otu_qc,
                          apply_log_transform=LOG_TRANSFORM,
                          sample_fraction=FRAC)

# Save results
aa.save_results(result, outDir, frac=FRAC)

# The result object contains:
# result.similarity_degree (pandas.DataFrame): Also known as alphas
# result.archetypes (pandas.DataFrame): Also known as betas
# result.info (dict): Other information like loss
```

### 3. Visualization

Visualize the results of the archetypal analysis. First, load a result if you have one saved.

```python
from archetypal_analysis.data.loaders import ArchetypalResultLoader

filepath = r"results\Compendium_N3_N13_NoPC_NoGlom\Norm_pp_100iter_QC-Region30k\otu_AA_K7_1.0_None_jensenshannon.json" 

loader = ArchetypalResultLoader()
otu_aa, aa_labels, aa_archetypes, info = loader.load_parsed_result(filepath)
```

#### Simplex Plots

Plot the samples on a simplex, colored by metadata.

```python
from archetypal_analysis.visualization.simplex import SimplexPlotter
from archetypal_analysis.utils.metadata import extract_metadata
from archetypal_analysis.data.loaders import CompendiumDataLoader


# Load metadata files (as shown in Metadata Parsing section)
data_loader = CompendiumDataLoader()
projects_file = r"datasets/compendium/projects.csv"
tags_file = r"datasets/compendium/tags.tsv"
meta_file = r"datasets/compendium/sample_metadata.tsv"
_, projects, tags, meta = data_loader.load_all(otu_file=None, projects_file=projects_file, tags_file=tags_file, meta_file=meta_file)


# Extract metadata for the samples in the analysis result
filter_cols = ["region"]
metadata = extract_metadata(otu_aa.index, tags, meta, projects, filter_cols, verbose=True)

# Plot
simplex_plotter = SimplexPlotter()
simplex_plotter.plot_simplex(
    otu_aa=otu_aa,
    metadata=metadata,
    filter_cols=filter_cols,
    cmap_name="tab20",
    legend_type="discrete",
    remove_nan=True,
    alpha = 0.5,
    s=10,
    remove_values={"region": "unknown"},
    figsize=(12, 10)
)
```

#### Composition Plots (Simple Bar and Stacked Bar)

Analyze and visualize the taxonomic composition of the archetypes.

```python
from archetypal_analysis.visualization.composition import CompositionAnalyzer
from archetypal_analysis.data.loaders import HadzaDataLoader

# Load an OTU table to get feature names (taxa)
otu_file = r"datasets\tsv\normalized\hadza_otu_normalized_age.csv"
loader = HadzaDataLoader()
otu = loader.load_otu_table(filename=otu_file)

# Initialize analyzer
composition_analyzer = CompositionAnalyzer()

# Get top taxa for each archetype
NUM_TAXA = 5
top_taxa_archs = composition_analyzer.analyze_top_taxa(
    archetypes=aa_archetypes, # from loaded result
    feature_names=otu.columns,
    num_taxa=NUM_TAXA,
)

# Plot as separate bar charts
composition_analyzer.plot_separate_bar_charts(top_taxa_archetypes=top_taxa_archs)

# Plot as a single stacked bar chart
composition_analyzer.plot_stacked_bar_chart(aa_archetypes=aa_archetypes, feature_names=otu.columns, top_taxa_archetypes=top_taxa_archs, figsize=(12, 6))
```

### Metadata Stacked Bar plot 

Analyze and visualize the distribution of metadata across archetypes using stacked bar plots.

```python
from archetypal_analysis.visualization.stacked_bar import StackedBarPlotter

# Initialize plotter
stacked_bar_plotter = StackedBarPlotter()

# Plot
stacked_bar_plotter.plot_metadata_stacked_bars(
    similarity_degree=otu_aa,
    metadata=metadata,
    perms=info.get('perms', None),
    metadata_column="region",
    discretize=False,
    buckets=None,
)
```

## File Formats

### Input Data Formats

- **OTU Table**: CSV file with samples as rows (with first column index), and taxa as columns. Each column should be named as '.' separated taxonomic levels ("Domain.Kingdom.Phylum.Class.Order.Family.Genus.Species")
- **Metadata**: TSV/CSV file with samples as rows, metadata columns
- **Optional**
    - **Projects**: CSV file with project information (for compendium data)
    - **Tags**: TSV file with (tag, value) pairs for each sample. 


## Contributing

To extend or modify the package:

1. Follow the existing module structure
2. Add comprehensive docstrings
3. Include type hints where possible
4. Add example usage in docstrings
5. Update this README with new features


## License

This package is derived from research code and should be used for academic and research purposes. Please cite the appropriate papers when using this code in publications.

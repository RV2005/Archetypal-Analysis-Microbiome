# Archetypal Analysis Package

A refactored and modularized Python package for microbiome archetypal analysis, extracted from the original Jupyter notebook workflow.

## Overview

This package provides a comprehensive toolkit for performing archetypal analysis on microbiome data, including:

- Data loading and preprocessing
- Quality control and normalization
- Archetypal analysis with multiple algorithms
- Rich visualization capabilities (simplex plots, composition charts)
- Metadata integration and analysis
- BLAST-based taxonomic annotation
- Batch processing for multiple K values

## Installation

### Prerequisites

```bash
pip install pandas numpy matplotlib scikit-learn scikit-bio tqdm
pip install git+https://github.com/RV2005/archetypes.git
```

### Optional dependencies

For BLAST analysis:
```bash
pip install biopython
```

### Package Installation

1. Clone or download this package
2. Add the package directory to your Python path, or install in development mode:

```bash
cd archetypal_analysis
pip install -e .
```

## Package Structure

```
archetypal_analysis/
├── __init__.py                 # Package initialization
├── config.py                   # Configuration settings
├── data/                       # Data handling modules
│   ├── __init__.py
│   ├── loaders.py             # Data loading utilities
│   ├── normalizers.py         # Normalization methods
│   ├── quality_control.py     # Quality control filters
│   └── formatters.py          # Data formatting utilities
├── analysis/                   # Core analysis modules
│   ├── __init__.py
│   ├── archetypal.py          # Main archetypal analysis
│   ├── metrics.py             # Distance and similarity metrics
│   └── clustering.py          # Clustering utilities
├── visualization/              # Plotting modules
│   ├── __init__.py
│   ├── simplex.py             # Simplex plot utilities
│   ├── stacked_bar.py         # Composition visualizations
│   ├── composition.py         # Additional composition plots
│   └── pcoa.py               # PCoA visualizations
├── utils/                      # Utility modules
│   ├── __init__.py
│   ├── metadata.py            # Metadata handling
│   ├── file_io.py             # File I/O utilities
│   └── blast.py               # BLAST analysis tools
└── examples/                   # Example workflows
    ├── example_workflow.py    # Basic example
    └── advanced_workflow.py   # Advanced example
```

## Quick Start

### Basic Usage

```python
import pandas as pd
from archetypal_analysis.data.loaders import DataLoader
from archetypal_analysis.analysis.archetypal import ArchetypalAnalyzer
from archetypal_analysis.visualization.simplex import plot_simplex

# Load data
loader = DataLoader()
otu_data = loader.load_otu_data("your_otu_file.csv")

# Run archetypal analysis
analyzer = ArchetypalAnalyzer(n_archetypes=5, random_state=42)
result = analyzer.fit_transform(otu_data)

# Create simplex plot
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 8))
plot_simplex(result.similarity_degree, metadata, ['group'], ax=ax)
plt.show()
```

### Advanced Workflow

```python
from archetypal_analysis.analysis.archetypal import ArchetypalAnalyzer
from archetypal_analysis.visualization.simplex import plot_multiple_k_simplex
from archetypal_analysis.visualization.stacked_bar import plot_stacked_bar_composition
from archetypal_analysis.utils.metadata import extract_metadata_compendium

# Multi-K analysis
k_values = [3, 5, 7, 8, 10]
results = {}

for k in k_values:
    analyzer = ArchetypalAnalyzer(n_archetypes=k)
    results[k] = analyzer.fit_transform(otu_data)

# Multi-K visualization
fig, axes = plot_multiple_k_simplex(
    results_dir="results/",
    k_values=k_values, 
    metadata=metadata,
    filter_cols=['region']
)

# Composition analysis
result = results[7]  # Choose K=7
fig, ax = plot_stacked_bar_composition(
    result.archetypes,
    otu_data.columns.tolist(),
    num_taxa=8
)
```

## Key Features

### 1. Data Loading and Preprocessing

```python
from archetypal_analysis.data.loaders import DataLoader
from archetypal_analysis.data.normalizers import DataNormalizer
from archetypal_analysis.data.quality_control import QualityController

# Load different data formats
loader = DataLoader()
otu_data = loader.load_otu_data("otu_table.csv")
metadata = loader.load_metadata("sample_metadata.tsv")

# Apply quality control
qc = QualityController()
otu_filtered = qc.filter_low_abundance_taxa(otu_data, min_abundance=0.001)
otu_filtered = qc.filter_low_prevalence_taxa(otu_filtered, min_prevalence=0.1)

# Normalize data
normalizer = DataNormalizer()
otu_normalized = normalizer.total_sum_scaling(otu_filtered)
```

### 2. Archetypal Analysis

```python
from archetypal_analysis.analysis.archetypal import ArchetypalAnalyzer

# Basic analysis
analyzer = ArchetypalAnalyzer(
    n_archetypes=5,
    method='pgd',
    max_iter=100,
    random_state=42
)

result = analyzer.fit_transform(otu_data, apply_log_transform=True)

# Access results
print(f"Loss: {result.loss}")
print(f"Archetype similarities shape: {result.similarity_degree.shape}")
print(f"Archetypes shape: {result.archetypes.shape}")

# Get top contributing taxa
compositions = result.get_archetype_composition(k=10)
```

### 3. Visualization

#### Simplex Plots

```python
from archetypal_analysis.visualization.simplex import plot_simplex

# Basic simplex plot
plot_simplex(
    otu_aa=result.similarity_degree,
    metadata=metadata,
    filter_cols=['region'],
    cmap_name='tab10',
    legend_type='discrete'
)

# Continuous coloring (e.g., age)
plot_simplex(
    otu_aa=result.similarity_degree,
    metadata=metadata, 
    filter_cols=['age_months'],
    cmap_name='viridis',
    legend_type='continuous'
)
```

#### Composition Plots

```python
from archetypal_analysis.visualization.stacked_bar import (
    plot_archetype_composition_bars,
    plot_stacked_bar_composition
)

# Separate bar charts for each archetype
fig, ax = plot_archetype_composition_bars(
    result.archetypes,
    otu_data.columns.tolist(),
    num_taxa=5
)

# Stacked bar chart
fig, ax = plot_stacked_bar_composition(
    result.archetypes,
    otu_data.columns.tolist(),
    num_taxa=8
)
```

### 4. Metadata Integration

```python
from archetypal_analysis.utils.metadata import extract_metadata_compendium

# Extract metadata from multiple sources (compendium format)
metadata = extract_metadata_compendium(
    index=otu_data.index,
    tags=tags_df,
    sample_meta=sample_metadata_df,
    projects=projects_df,
    extract_cols=['region', 'age_months', 'sample_type']
)
```

### 5. BLAST Analysis

```python
from archetypal_analysis.utils.blast import BlastAnalyzer

# Analyze top taxa with BLAST
blast_analyzer = BlastAnalyzer(num_hits=3)
blast_results = blast_analyzer.analyze_archetype_taxa(
    top_taxa_compositions,
    temp_fasta="temp.fasta",
    output_xml="blast_results.xml"
)
```

## Examples

The package includes two comprehensive examples:

1. **`examples/example_workflow.py`**: Basic workflow demonstrating core functionality
2. **`examples/advanced_workflow.py`**: Advanced workflow replicating the original notebook analysis

Run the examples:

```bash
cd archetypal_analysis/examples
python example_workflow.py
python advanced_workflow.py
```

## Configuration

Default settings can be modified in `config.py`:

```python
from archetypal_analysis.config import AnalysisConfig

# Modify default parameters
AnalysisConfig.DEFAULT_N_ARCHETYPES = 7
AnalysisConfig.DEFAULT_MAX_ITER = 200
```

## File Formats

### Input Data Formats

- **OTU/ASV Table**: CSV file with samples as rows, taxa as columns
- **Metadata**: TSV/CSV file with samples as rows, metadata columns
- **Projects**: CSV file with project information (for compendium data)
- **Tags**: TSV file with sample-level tags (for compendium data)

### Output Formats

- **Results**: JSON files with archetypal analysis results
- **Plots**: PNG/PDF files with high-resolution visualizations
- **Reports**: Text summaries of analysis results

## Migration from Notebook

This package refactors the original notebook workflow into modular components:

| Notebook Function | Package Module | 
|---|---|
| Data loading | `data.loaders` |
| OTU normalization | `data.normalizers` |
| Quality control | `data.quality_control` |
| Archetypal analysis | `analysis.archetypal` |
| Simplex plotting | `visualization.simplex` |
| Composition analysis | `visualization.stacked_bar` |
| Metadata extraction | `utils.metadata` |
| BLAST analysis | `utils.blast` |

## Contributing

To extend or modify the package:

1. Follow the existing module structure
2. Add comprehensive docstrings
3. Include type hints where possible
4. Add example usage in docstrings
5. Update this README with new features

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure the package is in your Python path
2. **Missing Dependencies**: Install required packages (pandas, numpy, matplotlib, etc.)
3. **File Not Found**: Check file paths and data format
4. **Memory Issues**: Use data sampling for large datasets
5. **BLAST Errors**: Requires internet connection and BioPython

### Performance Tips

- Use `sample_fraction < 1.0` for large datasets during exploration
- Reduce `max_iter` for faster (but less accurate) results
- Use appropriate `num_taxa` values for composition analysis
- Cache results using the save/load functionality

## License

This package is derived from research code and should be used for academic and research purposes. Please cite the appropriate papers when using this code in publications.

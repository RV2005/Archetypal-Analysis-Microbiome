#!/usr/bin/env python3
"""
Production-quality microbiome fuzzy-taxa validation pipeline.

Validates fuzzy-clustering-selected taxa against multiple independent methods:
- ANCOM-BC2 differential abundance
- ALDEx2 differential abundance
- Random Forest feature importance
- Optional Boruta feature selection
- Repeated resampling / cross-validation

EXAMPLE USAGE:
    python microbiome_validate_fuzzy_taxa.py \
      --otu_file otu_table.tsv \
      --metadata_file metadata.tsv \
      --metadata_sample_col sample_id \
      --clinical_col disease_status \
      --top_taxa_file fuzzy_top_taxa.tsv \
      --output_dir results_validate_fuzzy_taxa \
      --covariates age,sex \
      --n_resamples 50 \
      --n_permutations 10000 \
      --run_boruta

Author: Bioinformatics Pipeline
"""

import argparse
import json
import logging
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.special import comb
from scipy.stats import spearmanr, wilcoxon

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedShuffleSplit,
    cross_val_predict,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from archetypal_analysis.utils.metadata import extract_metadata
from archetypal_analysis.visualization.composition import CompositionAnalyzer
from archetypal_analysis.data.loaders import ArchetypalResultLoader

# Optional imports
try:
    from boruta import BorutaPy
    HAS_BORUTA = True
except ImportError:
    HAS_BORUTA = False

import platform


def json_serializable_default(obj):
    """Convert NumPy/Pandas values that the stdlib JSON encoder cannot handle."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def dump_json(data, path: Path) -> None:
    """Write JSON with support for common scientific Python scalar types."""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=json_serializable_default)


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(output_dir: Path) -> logging.Logger:
    """Configure logging to both console and file."""
    log_path = output_dir / "pipeline.log"
    
    logger = logging.getLogger("microbiome_validate")
    logger.setLevel(logging.DEBUG)
    
    # File handler
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


# ============================================================================
# INPUT VALIDATION & LOADING
# ============================================================================

def infer_delimiter(filepath: Path) -> str:
    """Auto-detect delimiter (tab or comma)."""
    with open(filepath, 'r') as f:
        first_line = f.readline()
    
    if '\t' in first_line:
        return '\t'
    elif ',' in first_line:
        return ','
    else:
        return '\t'  # default


def load_otu_table(
    filepath: Path,
    sep: str = 'auto',
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """Load OTU/ASV table with auto-detection of delimiter."""
    if sep == 'auto':
        sep = infer_delimiter(filepath)
    elif sep == 'comma':
        sep = ','
    elif sep == 'tab':
        sep = '\t'
    
    if logger:
        logger.info(f"Loading OTU table from {filepath} with delimiter='{repr(sep)}'")
    
    otu = pd.read_csv(filepath, sep=sep, index_col=0)
    
    if logger:
        logger.info(f"Loaded OTU table: shape={otu.shape}")
    
    return otu


def load_metadata(
    filepath: Path,
    sep: str = 'auto',
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """Load metadata table."""
    if sep == 'auto':
        sep = infer_delimiter(filepath)
    elif sep == 'comma':
        sep = ','
    elif sep == 'tab':
        sep = '\t'
    
    if logger:
        logger.info(f"Loading metadata from {filepath}")
    
    metadata = pd.read_csv(filepath, sep=sep, index_col=0, low_memory=False)
    
    if logger:
        logger.info(f"Loaded metadata: shape={metadata.shape}")
    
    return metadata


def _resolve_config_path(value: Optional[str], base_dir: Path) -> Optional[str]:
    """Resolve relative config paths against the config file location."""
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def apply_dataset_config(args: argparse.Namespace) -> argparse.Namespace:
    """Populate input arguments from dataset_configs.json when --dataset is used."""
    if not args.dataset:
        return args

    config_path = Path(args.dataset_config_file)
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path

    with open(config_path, 'r') as f:
        configs = json.load(f)

    if args.dataset not in configs:
        available = ', '.join(sorted(configs))
        raise ValueError(
            f"Dataset '{args.dataset}' not found in {config_path}. "
            f"Available datasets: {available}"
        )

    cfg = configs[args.dataset]
    base_dir = config_path.parent

    config_arg_map = {
        'otu_file': 'otu_file',
        'metadata_file': 'meta_file',
        'projects_file': 'projects_file',
        'tags_file': 'tags_file',
        'clinical_col': 'metadata_col',
        'top_taxa_file': 'top_taxa_file',
        'metadata_sample_col': 'metadata_sample_col',
        'metadata_dataset': 'metadata_dataset',
    }
    for arg_name, config_key in config_arg_map.items():
        if getattr(args, arg_name, None) is None and cfg.get(config_key) is not None:
            value = cfg[config_key]
            if arg_name in {
                'otu_file',
                'metadata_file',
                'projects_file',
                'tags_file',
                'top_taxa_file',
            }:
                value = _resolve_config_path(value, base_dir)
            setattr(args, arg_name, value)

    if args.output_dir is None:
        args.output_dir = str(
            (Path.cwd() / "results_validate_fuzzy_taxa" / args.dataset).resolve()
        )
    if args.metadata_dataset is None:
        args.metadata_dataset = 'compendium'

    args.dataset_config = cfg
    return args


def validate_required_inputs(args: argparse.Namespace) -> None:
    """Check required inputs after optional dataset config expansion."""
    if args.metadata_dataset is None:
        args.metadata_dataset = 'compendium'
    required = {
        'otu_file': '--otu_file',
        'metadata_file': '--metadata_file',
        'clinical_col': '--clinical_col',
        'top_taxa_file': '--top_taxa_file',
        'output_dir': '--output_dir',
    }
    dataset_cfg = getattr(args, 'dataset_config', None)
    allow_missing_top_taxa = bool(
        dataset_cfg
        and dataset_cfg.get('result_dir')
        and dataset_cfg.get('otu_file')
        and dataset_cfg.get('main_k') is not None
    )
    missing = [
        flag for attr, flag in required.items()
        if getattr(args, attr, None) is None
    ]
    if allow_missing_top_taxa and '--top_taxa_file' in missing:
        missing.remove('--top_taxa_file')
    if missing:
        config_hint = " or provide --dataset to load them from dataset_configs.json"
        raise ValueError(f"Missing required arguments: {', '.join(missing)}{config_hint}")


def metadata_columns_to_extract(args: argparse.Namespace) -> List[str]:
    """Return phenotype/covariate columns needed from metadata sources."""
    columns = [args.clinical_col]
    if args.covariates:
        columns.extend(c.strip() for c in args.covariates.split(',') if c.strip())
    return list(dict.fromkeys(columns))


def load_optional_metadata_source(
    filepath: Optional[str],
    sep: str = 'auto',
    logger: Optional[logging.Logger] = None,
    label: str = 'metadata source'
) -> Optional[pd.DataFrame]:
    """Load optional projects/tags metadata tables."""
    if not filepath:
        return None
    if logger:
        logger.info(f"Loading {label} from {filepath}")
    return load_metadata(Path(filepath), sep, logger)


def prepare_sample_metadata_for_extraction(
    metadata: pd.DataFrame,
    metadata_sample_col: Optional[str],
) -> pd.DataFrame:
    """Prepare sample metadata for archetypal_analysis.utils.metadata.extract_metadata."""
    if metadata_sample_col is None:
        return metadata
    if metadata_sample_col not in metadata.columns:
        raise ValueError(
            f"metadata_sample_col '{metadata_sample_col}' not found in metadata!"
        )
    metadata = metadata.copy()
    metadata.index = metadata[metadata_sample_col].astype(str)
    return metadata


def extract_validation_metadata(
    sample_index: pd.Index,
    sample_meta: pd.DataFrame,
    projects: Optional[pd.DataFrame],
    tags: Optional[pd.DataFrame],
    extract_cols: List[str],
    metadata_dataset: str = 'compendium',
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Extract metadata aligned to sample_index via the package metadata utility."""
    metadata = extract_metadata(
        index=pd.Index(sample_index),
        tags=tags,
        sample_meta=sample_meta,
        projects=projects,
        extract_cols=extract_cols,
        dataset=metadata_dataset,
        verbose=False,
    )
    if metadata is None:
        raise ValueError(
            f"extract_metadata returned no metadata for dataset='{metadata_dataset}'"
        )
    missing_cols = [col for col in extract_cols if col not in metadata.columns]
    if missing_cols:
        raise ValueError(
            "Extracted metadata is missing required columns: "
            + ', '.join(missing_cols)
        )
    if logger:
        n_with_metadata = metadata[extract_cols].notna().any(axis=1).sum()
        logger.info(
            f"Extracted metadata for {n_with_metadata}/{len(metadata)} samples "
            f"using dataset='{metadata_dataset}'"
        )
    return metadata


def count_metadata_matches(metadata: pd.DataFrame, columns: List[str]) -> int:
    """Count rows with at least one extracted metadata value."""
    return int(metadata[columns].notna().any(axis=1).sum())


def detect_otu_orientation_with_metadata(
    otu: pd.DataFrame,
    sample_meta: pd.DataFrame,
    projects: Optional[pd.DataFrame],
    tags: Optional[pd.DataFrame],
    extract_cols: List[str],
    metadata_dataset: str,
    logger: Optional[logging.Logger] = None,
) -> str:
    """Detect OTU orientation by extracting metadata for both possible sample axes."""
    index_metadata = extract_validation_metadata(
        otu.index,
        sample_meta,
        projects,
        tags,
        extract_cols,
        metadata_dataset,
    )
    column_metadata = extract_validation_metadata(
        otu.columns,
        sample_meta,
        projects,
        tags,
        extract_cols,
        metadata_dataset,
    )
    index_matches = count_metadata_matches(index_metadata, extract_cols)
    column_matches = count_metadata_matches(column_metadata, extract_cols)
    if logger:
        logger.info(
            "Orientation check via extract_metadata: "
            f"index matches={index_matches}, column matches={column_matches}"
        )
    if index_matches >= column_matches:
        return 'samples_rows'
    return 'taxa_rows'


def match_samples_from_extracted_metadata(
    otu: pd.DataFrame,
    metadata: pd.DataFrame,
    extract_cols: List[str],
    logger: Optional[logging.Logger] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Match samples using metadata already aligned by extract_metadata.

    Returns: (otu_matched, metadata_matched, sample_matching_report)
    """
    has_metadata = metadata[extract_cols].notna().any(axis=1)
    matched_index = metadata.index[has_metadata]
    otu_matched = otu.loc[matched_index].copy()
    metadata_matched = metadata.loc[matched_index].copy()

    report = pd.DataFrame({
        'category': [
            'OTU samples',
            'Metadata-extracted rows',
            'Shared with extracted metadata',
            'OTU without extracted metadata',
        ],
        'count': [
            len(otu.index),
            len(metadata.index),
            len(matched_index),
            len(otu.index) - len(matched_index),
        ]
    })

    if len(matched_index) < 2:
        raise ValueError("Fewer than 2 samples with extracted metadata!")

    if logger:
        logger.info(
            f"After metadata extraction: {otu_matched.shape[0]} samples, "
            f"{otu_matched.shape[1]} taxa"
        )

    return otu_matched, metadata_matched, report


def detect_otu_orientation(
    otu: pd.DataFrame,
    metadata_samples: set
) -> str:
    """
    Detect OTU table orientation (taxa_rows or samples_rows).
    
    Returns: 'taxa_rows' or 'samples_rows'
    """
    # Check if most indices match metadata samples
    otu_index_in_metadata = len(set(otu.index) & metadata_samples)
    otu_columns_in_metadata = len(set(otu.columns) & metadata_samples)
    
    if otu_index_in_metadata >= otu_columns_in_metadata:
        return 'samples_rows'
    else:
        return 'taxa_rows'


def transpose_if_needed(
    otu: pd.DataFrame,
    orientation: str,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """Ensure OTU table is samples x taxa."""
    if orientation == 'taxa_rows':
        otu = otu.T
        if logger:
            logger.info("Transposed OTU table to samples x taxa orientation")
    return otu


# ============================================================================
# SAMPLE MATCHING & QC
# ============================================================================

def match_samples(
    otu: pd.DataFrame,
    metadata: pd.DataFrame,
    metadata_sample_col: Optional[str],
    logger: Optional[logging.Logger] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Match samples between OTU and metadata.
    
    Returns: (otu_matched, metadata_matched, sample_matching_report)
    """
    # Get sample IDs
    if metadata_sample_col:
        otu_samples = set(otu.index)
        metadata_samples = set(metadata[metadata_sample_col])
    else:
        otu_samples = set(otu.index)
        metadata_samples = set(metadata.index)
    
    # Find overlaps
    shared = otu_samples & metadata_samples
    otu_only = otu_samples - metadata_samples
    metadata_only = metadata_samples - otu_samples
    
    if logger:
        logger.info(f"OTU samples: {len(otu_samples)}")
        logger.info(f"Metadata samples: {len(metadata_samples)}")
        logger.info(f"Shared samples: {len(shared)}")
        logger.info(f"OTU-only samples: {len(otu_only)}")
        logger.info(f"Metadata-only samples: {len(metadata_only)}")
    
    if len(shared) < 2:
        raise ValueError("Fewer than 2 shared samples between OTU and metadata!")
    
    # Subset
    otu_matched = otu.loc[list(shared)].copy()
    
    if metadata_sample_col:
        metadata_matched = metadata[metadata[metadata_sample_col].isin(shared)].copy()
        metadata_matched.index = metadata_matched[metadata_sample_col]
    else:
        metadata_matched = metadata.loc[list(shared)].copy()
    
    # Reorder metadata to match OTU
    metadata_matched = metadata_matched.loc[otu_matched.index]
    
    # Build report
    report = pd.DataFrame({
        'category': ['OTU samples', 'Metadata samples', 'Shared', 'OTU-only', 'Metadata-only'],
        'count': [len(otu_samples), len(metadata_samples), len(shared), len(otu_only), len(metadata_only)]
    })
    
    if logger:
        logger.info(f"After matching: {otu_matched.shape[0]} samples, {otu_matched.shape[1]} taxa")
    
    return otu_matched, metadata_matched, report


def filter_taxa(
    otu: pd.DataFrame,
    min_prevalence: float = 0.05,
    min_total_count: int = 10,
    logger: Optional[logging.Logger] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter taxa by prevalence and total count.
    
    Returns: (filtered_otu, filtering_report)
    """
    n_samples = otu.shape[0]
    min_n_samples = int(np.ceil(min_prevalence * n_samples))
    
    # Prevalence filter
    prevalence = (otu > 0).sum(axis=0) / n_samples
    prev_pass = prevalence >= min_prevalence
    
    # Total count filter
    total_counts = otu.sum(axis=0)
    count_pass = total_counts >= min_total_count
    
    # Combined filter
    keep = prev_pass & count_pass
    
    filtered_otu = otu.loc[:, keep].copy()
    
    report = pd.DataFrame({
        'filter': [
            'Initial taxa',
            'Fail prevalence',
            'Fail count',
            'Fail either',
            'Final taxa'
        ],
        'count': [
            otu.shape[1],
            (~prev_pass).sum(),
            (~count_pass).sum(),
            (~keep).sum(),
            filtered_otu.shape[1]
        ]
    })
    
    if logger:
        logger.info(f"Taxa filtering: {otu.shape[1]} -> {filtered_otu.shape[1]} taxa")
        logger.info(f"  Prevalence threshold: {min_prevalence} ({min_n_samples} samples)")
        logger.info(f"  Min total count: {min_total_count}")
        logger.info(report.to_string(index=False))
    
    return filtered_otu, report


def validate_clinical_column(
    metadata: pd.DataFrame,
    clinical_col: str,
    logger: Optional[logging.Logger] = None
) -> Tuple[pd.Series, Dict[str, int]]:
    """
    Validate and encode clinical column.
    
    Returns: (encoded_labels, class_counts)
    """
    if clinical_col not in metadata.columns:
        raise ValueError(f"Clinical column '{clinical_col}' not found in metadata!")
    
    clinical = metadata[clinical_col].copy()
    
    # Drop NA
    if clinical.isna().any():
        n_dropped = clinical.isna().sum()
        if logger:
            logger.warning(f"Dropping {n_dropped} samples with missing {clinical_col}")
        clinical = clinical.dropna()
    
    # Get unique classes
    unique_classes = clinical.unique()
    n_classes = len(unique_classes)
    
    if n_classes < 2:
        raise ValueError(f"Fewer than 2 clinical classes in {clinical_col}!")
    
    if logger:
        logger.info(f"Clinical column '{clinical_col}' has {n_classes} classes")
    
    # Encode
    le = LabelEncoder()
    encoded = le.fit_transform(clinical)
    
    class_counts = {
        cls: int(count)
        for cls, count in zip(le.classes_, np.bincount(encoded))
    }
    
    if logger:
        for cls, count in sorted(class_counts.items()):
            logger.info(f"  Class '{cls}': {count} samples")
    
    return pd.Series(encoded, index=clinical.index), class_counts


# ============================================================================
# TOP TAXA LOADING
# ============================================================================

def load_top_taxa(
    filepath: Path,
    sep: str = 'auto',
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """Load fuzzy-selected top taxa file."""
    if sep == 'auto':
        sep = infer_delimiter(filepath)
    elif sep == 'comma':
        sep = ','
    elif sep == 'tab':
        sep = '\t'
    
    if logger:
        logger.info(f"Loading top taxa from {filepath}")
    
    top_taxa = pd.read_csv(filepath, sep=sep)
    
    if logger:
        logger.info(f"Loaded {len(top_taxa)} fuzzy taxa")

    # Normalize common column-name variants used by the composition outputs.
    rename_map = {}
    column_lookup = {col.strip().lower(): col for col in top_taxa.columns}

    for lower_name, original_name in column_lookup.items():
        if lower_name in {'archetype', 'cluster', 'category'}:
            rename_map[original_name] = 'cluster'
        elif lower_name == 'rank':
            rename_map[original_name] = 'fuzzy_rank'

    # Ensure required columns. Some exported files have Taxa/Abundance swapped.
    if 'taxon' not in column_lookup:
        taxon_col = None
        membership_col = None

        taxa_name = column_lookup.get('taxa')
        abundance_name = column_lookup.get('abundance')

        if taxa_name and abundance_name:
            taxa_numeric = pd.to_numeric(top_taxa[taxa_name], errors='coerce').notna().mean()
            abundance_numeric = pd.to_numeric(top_taxa[abundance_name], errors='coerce').notna().mean()

            if taxa_numeric > 0.8 and abundance_numeric < 0.8:
                taxon_col = abundance_name
                membership_col = taxa_name
                if logger:
                    logger.warning(
                        "Detected swapped fuzzy taxa columns: using '%s' as taxon names and '%s' as abundance values",
                        abundance_name,
                        taxa_name,
                    )
            else:
                taxon_col = taxa_name
                membership_col = abundance_name
        elif taxa_name:
            taxon_col = taxa_name
        elif abundance_name:
            taxon_col = abundance_name

        if taxon_col is None:
            raise ValueError("Top taxa file must contain a taxon column such as 'taxon', 'Taxa', or 'Abundance'!")

        rename_map[taxon_col] = 'taxon'
        if membership_col is not None:
            rename_map[membership_col] = 'membership'

    if rename_map:
        top_taxa = top_taxa.rename(columns=rename_map)

    if 'membership' not in top_taxa.columns:
        membership_source = next(
            (col for col in ['abundance', 'importance', 'score'] if col in top_taxa.columns),
            None,
        )
        if membership_source is not None:
            top_taxa = top_taxa.rename(columns={membership_source: 'membership'})

    # Normalize the taxonomy strings so exact matches are more robust
    top_taxa['taxon'] = top_taxa['taxon'].astype(str).str.strip()
    
    return top_taxa


def read_table_auto(filepath: Path) -> pd.DataFrame:
    """Load a CSV/TSV table using the file extension."""
    if filepath.suffix == '.tsv':
        return pd.read_table(filepath, index_col=0)
    return pd.read_csv(filepath, index_col=0)


def resolve_result_path(result_dir: str, k_value: int) -> Path:
    """Build the archetypal result JSON path (matching final_plots)."""
    filename = f"otu_AA_K{k_value}_1.0_None_jensenshannon.json"
    return Path(result_dir) / filename


def reorder_archetypes(
    aa_archetypes: List,
    archetype_order: Optional[List[str]] = None,
    logger: Optional[logging.Logger] = None
) -> Tuple[np.ndarray, Optional[List[str]]]:
    """Reorder archetypes to match a named ordering list."""
    arch_array = np.array(aa_archetypes)
    if not archetype_order:
        return arch_array, None

    try:
        idx = [int(name.split()[-1]) - 1 for name in archetype_order]
    except (ValueError, IndexError) as exc:
        if logger:
            logger.warning("Invalid archetype_order values; using default order: %s", exc)
        return arch_array, None

    if max(idx, default=-1) >= arch_array.shape[0] or min(idx, default=0) < 0:
        if logger:
            logger.warning("archetype_order indices out of range; using default order")
        return arch_array, None

    return arch_array[idx, :], archetype_order


def compute_top_taxa_from_dataset_config(
    dataset_cfg: Dict,
    output_dir: Path,
    logger: Optional[logging.Logger] = None
) -> Path:
    """Compute top taxa from archetypal results using dataset_configs.json."""
    result_dir = dataset_cfg.get('result_dir')
    main_k = dataset_cfg.get('main_k')
    otu_file = dataset_cfg.get('otu_file')

    if not result_dir or main_k is None or not otu_file:
        raise ValueError(
            "Dataset config must include result_dir, main_k, and otu_file to compute top taxa."
        )

    result_path = resolve_result_path(result_dir, int(main_k))
    otu_path = Path(otu_file)
    if not otu_path.is_absolute():
        otu_path = (Path.cwd() / otu_path).resolve()

    loader = ArchetypalResultLoader()
    _, _, aa_archetypes, _ = loader.load_parsed_result(str(result_path))

    otu = read_table_auto(otu_path)
    num_taxa = int(dataset_cfg.get('num_taxa', 5))

    arch_array, arch_names = reorder_archetypes(
        aa_archetypes,
        dataset_cfg.get('archetype_order'),
        logger,
    )
    if arch_names is None:
        arch_names = [f"Archetype {i + 1}" for i in range(len(arch_array))]

    comp = CompositionAnalyzer()
    top_taxa_by_arch = comp.analyze_top_taxa(
        archetypes=arch_array,
        feature_names=otu.columns,
        archetype_names=arch_names,
        num_taxa=num_taxa,
    )

    top_rows = []
    for archetype_name, arch_data in zip(arch_names, top_taxa_by_arch):
        rank = 1
        for abundance, taxon in arch_data:
            if taxon == 'Others':
                continue
            top_rows.append({
                'Archetype': archetype_name,
                'Rank': rank,
                'Taxa': taxon,
                'Abundance': float(abundance),
            })
            rank += 1

    top_taxa_df = pd.DataFrame(top_rows)
    fuzzy_dir = output_dir / "fuzzy_taxa"
    fuzzy_dir.mkdir(parents=True, exist_ok=True)
    output_path = fuzzy_dir / "top_taxa_from_archetypes.tsv"
    top_taxa_df.to_csv(output_path, sep='\t', index=False)

    if logger:
        logger.info("Computed top taxa from archetypes: %s", output_path)

    return output_path


def match_top_taxa(
    top_taxa: pd.DataFrame,
    otu_taxa: List[str],
    logger: Optional[logging.Logger] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Match top taxa to OTU table taxa.
    
    Returns: (matched, unmatched_in_top, unmatched_in_otu)
    """
    otu_set = set(otu_taxa)
    top_set = set(top_taxa['taxon'])
    
    matched_taxa = top_set & otu_set
    unmatched_top = top_set - otu_set
    unmatched_otu = otu_set - top_set
    
    matched_df = top_taxa[top_taxa['taxon'].isin(matched_taxa)].copy()
    
    unmatched_top_df = pd.DataFrame({
        'taxon': list(unmatched_top),
        'status': 'in_fuzzy_not_in_otu'
    })
    
    unmatched_otu_df = pd.DataFrame({
        'taxon': list(unmatched_otu),
        'status': 'in_otu_not_in_fuzzy'
    })
    
    if logger:
        logger.info(f"Top taxa matching:")
        logger.info(f"  Matched: {len(matched_taxa)}")
        logger.info(f"  In fuzzy but not OTU: {len(unmatched_top)}")
        logger.info(f"  In OTU but not fuzzy: {len(unmatched_otu)}")
    
    return matched_df, unmatched_top_df, unmatched_otu_df


def compute_composition_top_taxa(
    fuzzy_matched: pd.DataFrame,
    num_taxa: int,
    logger: Optional[logging.Logger] = None,
    return_per_archetype: bool = False
) -> Union[List[str], List[List[Tuple[float, str]]]]:
    """
    Compute top taxa using the composition analyzer.
    
    Parameters:
    -----------
    fuzzy_matched : pd.DataFrame
        Fuzzy taxa table with 'taxon' and optional 'cluster'/'archetype'/'category' column
    num_taxa : int
        Number of top taxa to extract
    logger : Optional[logging.Logger]
        Logger for messages
    return_per_archetype : bool
        If True, return list of per-archetype tuples; if False, return flattened list of top taxa globally
    
    Returns:
    --------
    If return_per_archetype=True:
        List[List[Tuple[float, str]]]: Top taxa for each archetype with (percentage, taxa_name) tuples
    If return_per_archetype=False:
        List[str]: Top num_taxa taxa globally, ranked by mean importance across archetypes
    """
    cluster_col = next(
        (col for col in ['cluster', 'archetype', 'category'] if col in fuzzy_matched.columns),
        None
    )

    score_col = None
    score_series = None
    for col in ['membership', 'importance', 'fuzzy_rank']:
        if col in fuzzy_matched.columns:
            score_col = col
            score_series = pd.to_numeric(fuzzy_matched[col], errors='coerce')
            break

    if score_col is None:
        raise ValueError(
            "Fuzzy taxa file must contain at least one of 'membership', 'importance', 'fuzzy_rank', or 'abundance' to compute top taxa"
        )

    if score_col == 'fuzzy_rank':
        score_series = -score_series

    working = fuzzy_matched.copy()
    working['_composition_score'] = score_series.fillna(0.0)

    if cluster_col is None:
        matrix = pd.DataFrame([
            working.groupby('taxon')['_composition_score'].max()
        ], index=['Fuzzy taxa'])
    else:
        matrix = (
            working.pivot_table(
                index=cluster_col,
                columns='taxon',
                values='_composition_score',
                aggfunc='max',
                fill_value=0.0,
            )
            .sort_index()
        )

    matrix = matrix.fillna(0.0)
    
    if return_per_archetype:
        comp = CompositionAnalyzer()
        top_taxa_by_archetype = comp.analyze_top_taxa(
            archetypes=matrix.to_numpy(),
            feature_names=list(matrix.columns),
            archetype_names=list(matrix.index),
            num_taxa=num_taxa,
        )
        if logger:
            logger.info(
                "Computed composition-based top-%s taxa per archetype for %s archetype(s) using '%s'",
                num_taxa,
                len(top_taxa_by_archetype),
                score_col,
            )
        return top_taxa_by_archetype
    else:
        taxa_mean_score = matrix.mean(axis=0).sort_values(ascending=False)
        top_k_taxa = taxa_mean_score.head(num_taxa).index.tolist()
        if logger:
            logger.info(
                "Computed globally top-%s taxa (mean score across %s archetype(s)) using '%s'",
                num_taxa,
                len(matrix),
                score_col,
            )
        return top_k_taxa


# ============================================================================
# R HELPER SCRIPTS
# ============================================================================

def generate_ancombc2_script(
    output_dir: Path,
    clinical_col: str,
    covariates: Optional[List[str]],
    alpha: float = 0.05,
    logger: Optional[logging.Logger] = None
) -> Path:
    """Generate R helper script for ANCOM-BC2."""
    script_path = output_dir / "r_helpers" / "run_ancombc2.R"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    
    covariates_str = ""
    if covariates:
        quoted_covariates = ', '.join(f'"{c}"' for c in covariates)
        covariates_str = f", adj_vars = c({quoted_covariates})"
    
    script_content = f'''#!/usr/bin/env Rscript
# ANCOM-BC2 differential abundance analysis

# Check if required packages are installed
required_packages <- c("ANCOMBC", "phyloseq")
for (pkg in required_packages) {{
  if (!require(pkg, character.only = TRUE)) {{
    message(paste("Installing", pkg))
    if (!require("BiocManager", character.only = TRUE)) {{
      install.packages("BiocManager")
    }}
    BiocManager::install(pkg)
  }}
}}

library(ANCOMBC)
library(phyloseq)

# Read command-line arguments
args <- commandArgs(trailingOnly = TRUE)
count_file <- args[1]
metadata_file <- args[2]
clinical_col <- args[3]
output_dir <- args[4]

# Read data
message("Reading count data from ", count_file)
counts <- read.csv(count_file, row.names = 1, check.names = FALSE)

message("Reading metadata from ", metadata_file)
metadata <- read.csv(metadata_file, row.names = 1, check.names = FALSE)

# Ensure matching order
shared_samples <- intersect(rownames(metadata), colnames(counts))
metadata <- metadata[shared_samples, , drop = FALSE]
counts <- counts[, shared_samples, drop = FALSE]

# Create phyloseq object
OTU <- otu_table(as.matrix(counts), taxa_are_rows = TRUE)
META <- sample_data(metadata)
ps <- phyloseq(OTU, META)

# Run ANCOM-BC2
message("Running ANCOM-BC2 with group variable: ", clinical_col)
out <- ancombc2(
  data = ps,
  assay_name = "counts",
  tax_level = NULL,
  fix_formula = paste0("~", clinical_col){covariates_str},
  rand_formula = NULL,
  p_adj_method = "BH",
  prv_cut = 0,
  lib_cut = 0,
  s_min = 0,
  s_max = NULL,
  balance = NULL,
  pseudo = 0,
  pseudo_sens = TRUE,
  alpha = {alpha},
  n_cl = 1,
  verbose = TRUE,
  global = TRUE,
  pairwise = TRUE,
  dunnet = FALSE,
  trend = FALSE,
  iter_control = list(tol = 1e-5, max_iter = 100, verbose = TRUE),
  em_control = list(tol = 1e-5, max_iter = 100),
  lme_control = lmerControl(check.nobs.vs.nlev = "ignore"),
  mcp_adjust = "fdr",
  conserve = TRUE,
  seed = 123,
  verbose = FALSE
)

# Extract and save results
message("Saving ANCOM-BC2 results")

# Main results
res_prim <- out$res
write.csv(res_prim, file.path(output_dir, "ancombc2_results.tsv"), row.names = TRUE, sep = "\\t")

# Pairwise results if available
if (!is.null(out$res_pair)) {{
  res_pair <- out$res_pair
  write.csv(res_pair, file.path(output_dir, "ancombc2_pairwise_results.tsv"), row.names = TRUE, sep = "\\t")
}}

# Global results if available
if (!is.null(out$res_global)) {{
  res_global <- out$res_global
  write.csv(res_global, file.path(output_dir, "ancombc2_global_results.tsv"), row.names = TRUE, sep = "\\t")
}}

message("ANCOM-BC2 analysis completed successfully")
'''
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    if logger:
        logger.info(f"Generated ANCOM-BC2 R script: {script_path}")
    
    return script_path


def generate_aldex2_script(
    output_dir: Path,
    clinical_col: str,
    n_groups: int,
    covariates: Optional[List[str]],
    alpha: float = 0.05,
    logger: Optional[logging.Logger] = None
) -> Path:
    """Generate R helper script for ALDEx2."""
    script_path = output_dir / "r_helpers" / "run_aldex2.R"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Choose ALDEx2 workflow based on number of groups
    if n_groups == 2:
        workflow = "aldex.ttest"
        effect_size_call = "effect_size <- aldex.effect(x, CI.method='boot')"
    else:
        workflow = "aldex.kw"
        effect_size_call = "effect_size <- aldex.effect(x, CI.method='boot')"
    
    script_content = f'''#!/usr/bin/env Rscript
# ALDEx2 differential abundance analysis

# Check if required packages are installed
if (!require("ALDEx2", character.only = TRUE)) {{
  message("Installing ALDEx2")
  if (!require("BiocManager", character.only = TRUE)) {{
    install.packages("BiocManager")
  }}
  BiocManager::install("ALDEx2")
}}

library(ALDEx2)

# Read command-line arguments
args <- commandArgs(trailingOnly = TRUE)
count_file <- args[1]
metadata_file <- args[2]
clinical_col <- args[3]
output_dir <- args[4]

# Read data
message("Reading count data from ", count_file)
counts <- read.csv(count_file, row.names = 1, check.names = FALSE)

message("Reading metadata from ", metadata_file)
metadata <- read.csv(metadata_file, row.names = 1, check.names = FALSE)

# Ensure matching order
shared_samples <- intersect(rownames(metadata), colnames(counts))
metadata <- metadata[shared_samples, , drop = FALSE]
counts <- counts[, shared_samples, drop = FALSE]

# Prepare conditions
message("Preparing ALDEx2 input for clinical column: ", clinical_col)
condition <- factor(metadata[[clinical_col]])
levels_condition <- levels(condition)
message("Condition levels: ", paste(levels_condition, collapse = ", "))

# Run ALDEx2
message("Running ALDEx2")
x <- aldex(counts, condition, mc.samples = 128, test = "{workflow}", verbose = TRUE)

# Calculate effect sizes
message("Calculating effect sizes")
{effect_size_call}

# Save results
message("Saving ALDEx2 results")
write.csv(x, file.path(output_dir, "aldex2_results.tsv"), row.names = TRUE, sep = "\\t")

if (exists("effect_size")) {{
  write.csv(effect_size, file.path(output_dir, "aldex2_effects.tsv"), row.names = TRUE, sep = "\\t")
}}

message("ALDEx2 analysis completed successfully")
'''
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    if logger:
        logger.info(f"Generated ALDEx2 R script: {script_path}")
    
    return script_path


# ============================================================================
# DIFFERENTIAL ABUNDANCE EXECUTION
# ============================================================================

def find_rscript() -> Optional[str]:
    """Return the Rscript executable path if available on PATH."""
    return shutil.which("Rscript")


def write_method_skipped(
    output_dir: Path,
    method_name: str,
    reason: str,
    logger: Optional[logging.Logger] = None
) -> None:
    """Record that an optional method was skipped."""
    output_dir.mkdir(parents=True, exist_ok=True)
    skipped_path = output_dir / f"{method_name}_skipped.txt"
    with open(skipped_path, 'w') as f:
        f.write(reason.rstrip() + "\n")
    if logger:
        logger.warning(f"{method_name} skipped: {reason}")


def run_r_script(
    script_path: Path,
    args: List[str],
    output_dir: Path,
    method_name: str,
    logger: Optional[logging.Logger] = None
) -> Tuple[bool, Path, Path]:
    """
    Execute R script and capture output.
    
    Returns: (success, stdout_path, stderr_path)
    """
    stdout_path = output_dir / f"{method_name}_stdout.log"
    stderr_path = output_dir / f"{method_name}_stderr.log"
    rscript = find_rscript()
    if rscript is None:
        reason = (
            "Rscript was not found on PATH. Install R or load an R module, "
            "or disable R-based differential abundance methods."
        )
        with open(stderr_path, 'w') as f:
            f.write(reason + "\n")
        if logger:
            logger.warning(reason)
        return False, stdout_path, stderr_path
    
    try:
        if logger:
            logger.info(f"Running R script: {script_path}")
        
        result = subprocess.run(
            [rscript, str(script_path)] + args,
            stdout=open(stdout_path, 'w'),
            stderr=open(stderr_path, 'w'),
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode != 0:
            if logger:
                logger.error(f"R script failed with return code {result.returncode}")
                with open(stderr_path) as f:
                    logger.error(f.read())
            return False, stdout_path, stderr_path
        
        if logger:
            logger.info(f"R script completed successfully")
        
        return True, stdout_path, stderr_path
        
    except subprocess.TimeoutExpired:
        if logger:
            logger.error(f"R script timed out after 3600 seconds")
        return False, stdout_path, stderr_path
    except Exception as e:
        if logger:
            logger.error(f"Error running R script: {e}")
        return False, stdout_path, stderr_path


def standardize_da_results(
    results_df: pd.DataFrame,
    method_name: str,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Standardize DA results from different methods into common schema.
    
    Columns: method, taxon, contrast, effect_size, statistic, p_value, q_value, direction, significant
    """
    std_results = []
    
    for idx, row in results_df.iterrows():
        # Extract relevant columns based on method
        if method_name == "ancombc2":
            taxon = idx
            # Try to extract p-value and effect size
            p_val = row.get('p_val', np.nan)
            q_val = row.get('q_val', p_val)
            effect = row.get('lfc', row.get('log_fold_change', np.nan))
            stat = row.get('W', np.nan)
            direction = 'positive' if effect > 0 else 'negative' if effect < 0 else 'unknown'
            
        elif method_name == "aldex2":
            taxon = idx
            # ALDEx2 uses different column names
            p_val = row.get('we.p', row.get('kw.p', np.nan))
            q_val = row.get('we.q', row.get('kw.q', p_val))
            effect = row.get('effect', row.get('rab.all', np.nan))
            stat = p_val  # Use p-value as stat if not available
            direction = 'positive' if effect > 0 else 'negative' if effect < 0 else 'unknown'
        else:
            continue
        
        significant = q_val < 0.05 if not np.isnan(q_val) else False
        
        std_results.append({
            'method': method_name,
            'taxon': taxon,
            'contrast': 'main',
            'effect_size': effect,
            'statistic': stat,
            'p_value': p_val,
            'q_value': q_val,
            'direction': direction,
            'significant': significant
        })
    
    std_df = pd.DataFrame(std_results)
    
    if logger:
        logger.info(f"Standardized {len(std_df)} {method_name} results")
    
    return std_df


DA_RESULT_COLUMNS = [
    'method',
    'taxon',
    'contrast',
    'effect_size',
    'statistic',
    'p_value',
    'q_value',
    'direction',
    'significant',
]

DIRECTION_AGREEMENT_COLUMNS = [
    'taxon',
    'da_direction',
    'da_significant',
    'ml_direction',
    'direction_agrees',
]


def run_ancombc2(
    otu_raw: pd.DataFrame,
    metadata: pd.DataFrame,
    clinical_col: str,
    output_dir: Path,
    covariates: Optional[List[str]] = None,
    alpha: float = 0.05,
    logger: Optional[logging.Logger] = None
) -> Optional[pd.DataFrame]:
    """Run ANCOM-BC2 differential abundance."""
    da_dir = output_dir / "differential_abundance" / "ancombc2"
    da_dir.mkdir(parents=True, exist_ok=True)

    if find_rscript() is None:
        write_method_skipped(
            da_dir,
            "ancombc2",
            "Rscript was not found on PATH. ANCOM-BC2 requires R and was skipped.",
            logger,
        )
        return None
    
    try:
        # Generate script
        script_path = generate_ancombc2_script(
            output_dir, clinical_col, covariates, alpha, logger
        )
        
        # Save data files for R
        count_file = da_dir / "counts_for_ancombc2.csv"
        metadata_file = da_dir / "metadata_for_ancombc2.csv"
        
        otu_raw.to_csv(count_file)
        metadata.to_csv(metadata_file)
        
        # Run R script
        success, stdout_path, stderr_path = run_r_script(
            script_path,
            [str(count_file), str(metadata_file), clinical_col, str(da_dir)],
            da_dir,
            "ancombc2",
            logger
        )
        
        if not success:
            if logger:
                logger.error("ANCOM-BC2 failed. Check logs.")
            return None
        
        # Load results
        results_file = da_dir / "ancombc2_results.tsv"
        if not results_file.exists():
            if logger:
                logger.error("ANCOM-BC2 results file not found")
            return None
        
        results = pd.read_csv(results_file, sep='\t', index_col=0)
        std_results = standardize_da_results(results, "ancombc2", logger)
        
        # Save standardized results
        std_results.to_csv(
            da_dir / "ancombc2_results_standardized.tsv",
            sep='\t',
            index=False
        )
        
        return std_results
        
    except Exception as e:
        if logger:
            logger.error(f"ANCOM-BC2 error: {e}")
        return None


def run_aldex2(
    otu_raw: pd.DataFrame,
    metadata: pd.DataFrame,
    clinical_col: str,
    output_dir: Path,
    class_counts: Dict[str, int],
    covariates: Optional[List[str]] = None,
    alpha: float = 0.05,
    logger: Optional[logging.Logger] = None
) -> Optional[pd.DataFrame]:
    """Run ALDEx2 differential abundance."""
    da_dir = output_dir / "differential_abundance" / "aldex2"
    da_dir.mkdir(parents=True, exist_ok=True)

    if find_rscript() is None:
        write_method_skipped(
            da_dir,
            "aldex2",
            "Rscript was not found on PATH. ALDEx2 requires R and was skipped.",
            logger,
        )
        return None
    
    try:
        # Generate script
        n_groups = len(class_counts)
        script_path = generate_aldex2_script(
            output_dir, clinical_col, n_groups, covariates, alpha, logger
        )
        
        # Save data files for R
        count_file = da_dir / "counts_for_aldex2.csv"
        metadata_file = da_dir / "metadata_for_aldex2.csv"
        
        otu_raw.to_csv(count_file)
        metadata.to_csv(metadata_file)
        
        # Run R script
        success, stdout_path, stderr_path = run_r_script(
            script_path,
            [str(count_file), str(metadata_file), clinical_col, str(da_dir)],
            da_dir,
            "aldex2",
            logger
        )
        
        if not success:
            if logger:
                logger.error("ALDEx2 failed. Check logs.")
            return None
        
        # Load results
        results_file = da_dir / "aldex2_results.tsv"
        if not results_file.exists():
            if logger:
                logger.error("ALDEx2 results file not found")
            return None
        
        results = pd.read_csv(results_file, sep='\t', index_col=0)
        std_results = standardize_da_results(results, "aldex2", logger)
        
        # Save standardized results
        std_results.to_csv(
            da_dir / "aldex2_results_standardized.tsv",
            sep='\t',
            index=False
        )
        
        return std_results
        
    except Exception as e:
        if logger:
            logger.error(f"ALDEx2 error: {e}")
        return None


# ============================================================================
# MACHINE LEARNING: RANDOM FOREST
# ============================================================================

def run_random_forest(
    X_log1p: pd.DataFrame,
    y: pd.Series,
    n_resamples: int = 50,
    test_size: float = 0.2,
    n_estimators: int = 1000,
    random_seed: int = 123,
    n_jobs: int = -1,
    logger: Optional[logging.Logger] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run Random Forest with repeated stratified resampling.
    
    Returns: (rf_metrics, rf_importance)
    """
    ml_dir_name = "random_forest"
    
    if logger:
        logger.info(f"Running Random Forest with {n_resamples} resamples")
    
    rss = StratifiedShuffleSplit(
        n_splits=n_resamples,
        test_size=test_size,
        random_state=random_seed
    )
    
    metrics_list = []
    importance_list = []
    
    for split_idx, (train_idx, test_idx) in enumerate(rss.split(X_log1p, y)):
        if logger and split_idx % 10 == 0:
            logger.info(f"  Resample {split_idx + 1}/{n_resamples}")
        
        X_train, X_test = X_log1p.iloc[train_idx], X_log1p.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Fit RF
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight='balanced',
            random_state=random_seed + split_idx,
            n_jobs=n_jobs,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2
        )
        rf.fit(X_train, y_train)
        
        # Test metrics
        y_pred = rf.predict(X_test)
        y_pred_proba = rf.predict_proba(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        bacc = balanced_accuracy_score(y_test, y_pred)
        f1_mac = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        try:
            if len(np.unique(y)) == 2:
                roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
            else:
                roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
        except:
            roc_auc = np.nan
        
        metrics_list.append({
            'split': split_idx,
            'train_size': len(train_idx),
            'test_size': len(test_idx),
            'accuracy': acc,
            'balanced_accuracy': bacc,
            'macro_f1': f1_mac,
            'roc_auc': roc_auc
        })
        
        # Permutation importance on test set
        perm_imp = permutation_importance(
            rf, X_test, y_test,
            n_repeats=10,
            random_state=random_seed + split_idx,
            n_jobs=n_jobs,
            scoring='balanced_accuracy'
        )
        
        for taxon_idx, taxon in enumerate(X_log1p.columns):
            importance_list.append({
                'split': split_idx,
                'taxon': taxon,
                'importance_mean': perm_imp.importances_mean[taxon_idx],
                'importance_std': perm_imp.importances_std[taxon_idx]
            })
    
    metrics_df = pd.DataFrame(metrics_list)
    importance_long_df = pd.DataFrame(importance_list)
    
    # Aggregate importance
    importance_summary = importance_long_df.groupby('taxon').agg({
        'importance_mean': ['mean', 'median', 'std'],
        'importance_std': 'mean'
    }).reset_index()
    importance_summary.columns = ['taxon', 'mean_importance', 'median_importance', 'std_importance', 'mean_std']
    
    # Selection frequency (importance > 0)
    importance_summary['selection_frequency'] = importance_long_df.groupby('taxon')['importance_mean'].apply(
        lambda x: (x > 0).sum() / len(x)
    ).values
    
    # Rank
    importance_summary = importance_summary.sort_values('mean_importance', ascending=False).reset_index(drop=True)
    importance_summary['rank'] = range(1, len(importance_summary) + 1)
    
    if logger:
        logger.info(f"Random Forest completed. Mean balanced accuracy: {metrics_df['balanced_accuracy'].mean():.3f}")
    
    return metrics_df, importance_summary


# ============================================================================
# MACHINE LEARNING: BORUTA
# ============================================================================

def run_boruta(
    X_log1p: pd.DataFrame,
    y: pd.Series,
    n_resamples: int = 10,
    random_seed: int = 123,
    n_jobs: int = -1,
    logger: Optional[logging.Logger] = None
) -> Optional[pd.DataFrame]:
    """Run BorutaPy feature selection."""
    if not HAS_BORUTA:
        if logger:
            logger.warning("BorutaPy not installed. Install with: pip install boruta")
        return None
    
    if logger:
        logger.info(f"Running Boruta with {n_resamples} resamples")
    
    boruta_results = []
    
    rss = StratifiedShuffleSplit(
        n_splits=n_resamples,
        test_size=0.2,
        random_state=random_seed
    )
    
    for split_idx, (train_idx, test_idx) in enumerate(rss.split(X_log1p, y)):
        if logger:
            logger.info(f"  Boruta resample {split_idx + 1}/{n_resamples}")
        
        X_train, y_train = X_log1p.iloc[train_idx], y.iloc[train_idx]
        
        rf = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=random_seed + split_idx,
            n_jobs=1  # Boruta manages parallelization
        )
        
        boruta = BorutaPy(
            estimator=rf,
            n_estimators='auto',
            max_iter=100,
            random_state=random_seed + split_idx,
            verbose=0
        )
        
        boruta.fit(X_train.values, y_train.values)
        
        for taxon_idx, taxon in enumerate(X_log1p.columns):
            status = 'selected' if boruta.support_[taxon_idx] else (
                'tentative' if boruta.support_weak_[taxon_idx] else 'rejected'
            )
            
            boruta_results.append({
                'split': split_idx,
                'taxon': taxon,
                'status': status,
                'ranking': boruta.ranking_[taxon_idx]
            })
    
    boruta_df = pd.DataFrame(boruta_results)
    
    # Aggregate
    boruta_summary = boruta_df.groupby('taxon').agg({
        'status': lambda x: (x == 'selected').sum() / len(x),
        'ranking': 'mean'
    }).reset_index()
    boruta_summary.columns = ['taxon', 'selection_frequency', 'mean_ranking']
    boruta_summary = boruta_summary.sort_values('selection_frequency', ascending=False)
    
    if logger:
        logger.info(f"Boruta completed. {(boruta_summary['selection_frequency'] >= 0.5).sum()} taxa selected in >= 50% of resamples")
    
    return boruta_summary


# ============================================================================
# MACHINE LEARNING: RF DIRECTION ANNOTATION
# ============================================================================

def compute_rf_direction_annotations(
    X_log1p: pd.DataFrame,
    X_raw: pd.DataFrame,
    y: pd.Series,
    rf_importance: pd.DataFrame,
    abundance_delta_threshold: float = 0.0,
    prevalence_delta_threshold: float = 0.0,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Annotate RF features with direction information.
    
    For each taxon and each category (one-vs-rest):
    - Compute abundance delta: mean_log1p_in_category - mean_log1p_out_category
    - Compute prevalence delta: (count > 0 in cat) - (count > 0 out cat)
    - Assign direction markers based on deltas
    
    Returns DataFrame with columns:
    - taxon, mean_importance, selection_frequency
    - For each unique category: [cat]_abundance_delta, [cat]_prevalence_delta, 
      [cat]_direction_markers
    """
    if logger:
        logger.info("Computing RF direction annotations")
    
    # Get unique categories
    categories = sorted(y.unique())
    
    direction_list = []
    
    # For each taxon in RF importance
    for _, row in rf_importance.iterrows():
        taxon = row['taxon']
        mean_importance = row['mean_importance']
        selection_frequency = row['selection_frequency']
        
        taxon_data = {'taxon': taxon, 'mean_importance': mean_importance, 'selection_frequency': selection_frequency}
        
        # For each category, compute one-vs-rest direction
        for category in categories:
            in_cat_mask = (y == category)
            out_cat_mask = ~in_cat_mask
            
            # Abundance metrics
            mean_log1p_in = X_log1p.loc[in_cat_mask, taxon].mean()
            mean_log1p_out = X_log1p.loc[out_cat_mask, taxon].mean()
            abundance_delta = mean_log1p_in - mean_log1p_out
            
            # Prevalence metrics
            prev_in = (X_raw.loc[in_cat_mask, taxon] > 0).sum() / in_cat_mask.sum()
            prev_out = (X_raw.loc[out_cat_mask, taxon] > 0).sum() / out_cat_mask.sum()
            prevalence_delta = prev_in - prev_out
            
            # Direction markers
            markers = []
            if abundance_delta > abundance_delta_threshold:
                markers.append('positive_abundance_marker')
            elif abundance_delta < -abundance_delta_threshold:
                markers.append('negative_abundance_marker')
            
            if prevalence_delta > prevalence_delta_threshold:
                markers.append('positive_prevalence_marker')
            elif prevalence_delta < -prevalence_delta_threshold:
                markers.append('negative_prevalence_marker')
            
            if not markers:
                markers.append('ambiguous_marker')
            
            # Store results
            cat_str = str(category).replace(' ', '_')
            taxon_data[f'{cat_str}_abundance_delta'] = abundance_delta
            taxon_data[f'{cat_str}_prevalence_delta'] = prevalence_delta
            taxon_data[f'{cat_str}_direction_markers'] = '|'.join(markers)
        
        direction_list.append(taxon_data)
    
    direction_df = pd.DataFrame(direction_list)
    
    if logger:
        logger.info(f"Direction annotations computed for {len(direction_df)} taxa and {len(categories)} categories")
    
    return direction_df


def get_positive_markers(
    direction_annotated_df: pd.DataFrame,
    importance_threshold: float = 0.0,
    abundance_delta_threshold: float = 0.0,
    prevalence_delta_threshold: float = 0.0,
    rule: str = 'abundance_or_prevalence',
    logger: Optional[logging.Logger] = None
) -> Dict[str, List[str]]:
    """
    Extract positive markers for each category from direction-annotated importance.
    
    Returns dict: {category: [taxon1, taxon2, ...]}
    """
    positive_markers = {}
    
    # Extract categories from column names
    categories = set()
    for col in direction_annotated_df.columns:
        if col.endswith('_abundance_delta'):
            cat = col.replace('_abundance_delta', '')
            categories.add(cat)
    
    for category in sorted(categories):
        markers = []
        
        for _, row in direction_annotated_df.iterrows():
            if row['mean_importance'] < importance_threshold:
                continue
            
            direction_col = f'{category}_direction_markers'
            if direction_col not in row or pd.isna(row[direction_col]):
                continue
            
            markers_str = row[direction_col]
            
            # Check if positive marker based on rule
            is_positive = False
            if rule == 'abundance_or_prevalence':
                if 'positive_abundance_marker' in markers_str or 'positive_prevalence_marker' in markers_str:
                    is_positive = True
            elif rule == 'abundance_and_prevalence':
                if 'positive_abundance_marker' in markers_str and 'positive_prevalence_marker' in markers_str:
                    is_positive = True
            
            if is_positive:
                markers.append(row['taxon'])
        
        positive_markers[category] = markers
    
    if logger:
        logger.info(f"Extracted positive markers: {sum(len(m) for m in positive_markers.values())} total across {len(positive_markers)} categories")
    
    return positive_markers


def get_negative_markers(
    direction_annotated_df: pd.DataFrame,
    importance_threshold: float = 0.0,
    logger: Optional[logging.Logger] = None
) -> Dict[str, List[str]]:
    """
    Extract negative markers (depletions/absences) for each category.
    
    Returns dict: {category: [taxon1, taxon2, ...]}
    """
    negative_markers = {}
    
    # Extract categories from column names
    categories = set()
    for col in direction_annotated_df.columns:
        if col.endswith('_abundance_delta'):
            cat = col.replace('_abundance_delta', '')
            categories.add(cat)
    
    for category in sorted(categories):
        markers = []
        
        for _, row in direction_annotated_df.iterrows():
            if row['mean_importance'] < importance_threshold:
                continue
            
            direction_col = f'{category}_direction_markers'
            if direction_col not in row or pd.isna(row[direction_col]):
                continue
            
            markers_str = row[direction_col]
            
            # Check if negative marker
            if 'negative_abundance_marker' in markers_str or 'negative_prevalence_marker' in markers_str:
                markers.append(row['taxon'])
        
        negative_markers[category] = markers
    
    if logger:
        logger.info(f"Extracted negative markers: {sum(len(m) for m in negative_markers.values())} total across {len(negative_markers)} categories")
    
    return negative_markers


# ============================================================================
# COMPARISON: OVERLAP & PERMUTATION TESTS
# ============================================================================

def hypergeometric_pval(
    n_overlap: int,
    n_fuzzy: int,
    n_method: int,
    n_universe: int
) -> float:
    """Calculate hypergeometric p-value for overlap."""
    if n_overlap > min(n_fuzzy, n_method):
        return 1.0
    
    # Hypergeometric survival function: P(X >= observed | hypergeometric)
    # But we use SF which is P(X > observed), so we need to add P(X == observed)
    from scipy.stats import hypergeom
    
    try:
        pval = hypergeom.sf(n_overlap - 1, n_universe, n_method, n_fuzzy)
        return float(pval)
    except:
        return 1.0


def permutation_overlap_test(
    n_overlap_observed: int,
    n_fuzzy: int,
    n_method: int,
    n_universe: int,
    n_permutations: int = 10000,
    random_seed: int = 123
) -> float:
    """Compute permutation p-value for set overlap."""
    np.random.seed(random_seed)
    
    universe_indices = np.arange(n_universe)
    count_greater_or_equal = 0
    
    for _ in range(n_permutations):
        random_fuzzy = np.random.choice(universe_indices, size=n_fuzzy, replace=False)
        random_method = np.random.choice(universe_indices, size=n_method, replace=False)
        random_overlap = len(np.intersect1d(random_fuzzy, random_method))
        
        if random_overlap >= n_overlap_observed:
            count_greater_or_equal += 1
    
    # empirical p-value
    pval = (1 + count_greater_or_equal) / (1 + n_permutations)
    return pval


def compute_overlap_metrics(
    fuzzy_taxa_set: set,
    method_taxa_set: set,
    all_taxa_universe: set,
    n_permutations: int = 10000,
    random_seed: int = 123,
    logger: Optional[logging.Logger] = None
) -> Dict:
    """Compute overlap metrics between fuzzy and method taxa."""
    
    n_fuzzy = len(fuzzy_taxa_set)
    n_method = len(method_taxa_set)
    n_overlap = len(fuzzy_taxa_set & method_taxa_set)
    n_universe = len(all_taxa_universe)
    
    overlap_taxa = list(fuzzy_taxa_set & method_taxa_set)
    
    # Jaccard
    jaccard = n_overlap / len(fuzzy_taxa_set | method_taxa_set) if (len(fuzzy_taxa_set | method_taxa_set) > 0) else 0
    
    # Overlap coefficient
    overlap_coef = n_overlap / min(n_fuzzy, n_method) if min(n_fuzzy, n_method) > 0 else 0
    
    # Hypergeometric p-value
    hyp_pval = hypergeometric_pval(n_overlap, n_fuzzy, n_method, n_universe)
    
    # Permutation p-value
    perm_pval = permutation_overlap_test(
        n_overlap, n_fuzzy, n_method, n_universe,
        n_permutations, random_seed
    )
    
    return {
        'n_fuzzy': n_fuzzy,
        'n_method': n_method,
        'n_overlap': n_overlap,
        'overlap_taxa': ','.join(overlap_taxa),
        'jaccard': jaccard,
        'overlap_coefficient': overlap_coef,
        'hypergeometric_pval': hyp_pval,
        'permutation_pval': perm_pval
    }


# ============================================================================
# COMPARISON: RANK CORRELATION
# ============================================================================

def compute_rank_correlation(
    fuzzy_taxa_dict: Dict[str, float],
    method_taxa_dict: Dict[str, float],
    logger: Optional[logging.Logger] = None
) -> Tuple[float, float, int]:
    """
    Compute Spearman correlation between fuzzy and method taxa ranks.
    
    Returns: (correlation, p_value, n_overlapping)
    """
    
    # Find overlapping taxa
    overlapping = set(fuzzy_taxa_dict.keys()) & set(method_taxa_dict.keys())
    
    if len(overlapping) < 3:
        return np.nan, np.nan, len(overlapping)
    
    # Rank both
    fuzzy_ranks = pd.Series(fuzzy_taxa_dict).rank()
    method_ranks = pd.Series(method_taxa_dict).rank()
    
    # Keep only overlapping
    fuzzy_overlap = fuzzy_ranks[list(overlapping)]
    method_overlap = method_ranks[list(overlapping)]
    
    # Correlate
    corr, pval = spearmanr(fuzzy_overlap, method_overlap)
    
    return corr, pval, len(overlapping)


# ============================================================================
# COMPARISON: DIRECTION AGREEMENT
# ============================================================================

def compute_direction_agreement(
    otu_log1p: pd.DataFrame,
    y_encoded: pd.Series,
    da_results: pd.DataFrame,
    fuzzy_taxa: pd.DataFrame,
    alpha: float = 0.05,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """Compute direction agreement between DA and class-wise differences."""
    
    agreement_list = []
    
    for taxon in fuzzy_taxa['taxon']:
        if taxon not in otu_log1p.columns:
            continue
        
        # DA direction for this taxon
        da_row = da_results[da_results['taxon'] == taxon]
        
        if len(da_row) == 0:
            continue
        
        da_dir = da_row.iloc[0]['direction']
        da_sig = da_row.iloc[0]['significant']
        
        # Compute class-wise mean difference
        class_groups = otu_log1p[taxon].groupby(y_encoded).mean()
        if len(class_groups) >= 2:
            class_groups_sorted = class_groups.sort_index()
            diff = class_groups_sorted.iloc[-1] - class_groups_sorted.iloc[0]
            ml_dir = 'positive' if diff > 0 else 'negative' if diff < 0 else 'unknown'
        else:
            ml_dir = 'unknown'
        
        # Agreement
        if da_dir != 'unknown' and ml_dir != 'unknown':
            agrees = (da_dir == ml_dir)
        else:
            agrees = np.nan
        
        agreement_list.append({
            'taxon': taxon,
            'da_direction': da_dir,
            'da_significant': da_sig,
            'ml_direction': ml_dir,
            'direction_agrees': agrees
        })
    
    return pd.DataFrame(agreement_list)


# ============================================================================
# PREDICTIVE ABLATION
# ============================================================================

def run_ablation_analysis(
    X_log1p: pd.DataFrame,
    y: pd.Series,
    fuzzy_taxa_list: List[str],
    rf_top_taxa: List[str],
    external_consensus: List[str],
    covariates_cols: Optional[List[str]],
    n_resamples: int = 50,
    test_size: float = 0.2,
    random_seed: int = 123,
    n_estimators: int = 1000,
    n_jobs: int = -1,
    logger: Optional[logging.Logger] = None,
    rf_importance: Optional[pd.DataFrame] = None,
    fuzzy_matched: Optional[pd.DataFrame] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run ablation analysis comparing different feature sets.
    
    Returns: (ablation_metrics_long, ablation_summary)
    """
    
    if logger:
        logger.info("Running predictive ablation analysis")
    
    # Define feature sets
    feature_sets = {}
    
    # Extract top-k fuzzy taxa per archetype using composition analyzer, then union
    if fuzzy_matched is not None and not fuzzy_matched.empty:
        for k in [5, 10, 15]:
            top_k_taxa = []
            try:
                top_taxa_by_archetype = compute_composition_top_taxa(
                    fuzzy_matched,
                    k,
                    logger,
                    return_per_archetype=True,
                )
                # Union top-k taxa across all archetypes
                for archetype_taxa in top_taxa_by_archetype:
                    top_k_taxa.extend(
                        [taxon for _, taxon in archetype_taxa if taxon != 'Others']
                    )
                # Remove duplicates while preserving order
                seen = set()
                top_k_taxa = [t for t in top_k_taxa if not (t in seen or seen.add(t))]
            except ValueError as exc:
                if logger:
                    logger.warning(
                        "Composition analyzer fallback unavailable for fuzzy taxa top-k extraction: %s",
                        exc,
                    )
                top_k_taxa = fuzzy_matched['taxon'].head(k).tolist()
            
            if top_k_taxa:
                feature_sets[f'fuzzy_top{k}'] = top_k_taxa
    
    # External consensus
    feature_sets['external_consensus_only'] = external_consensus
    
    # All taxa
    feature_sets['all_taxa'] = list(X_log1p.columns)
    
    # Add clinical covariates if provided
    if covariates_cols:
        feature_sets['clinical_covariates_only'] = covariates_cols
    
    # Ensure all features are in X
    for key in list(feature_sets.keys()):
        feature_sets[key] = [f for f in feature_sets[key] if f in X_log1p.columns]

    feature_set_sizes = {key: len(value) for key, value in feature_sets.items()}
    
    ablation_metrics = []
    
    rss = StratifiedShuffleSplit(
        n_splits=n_resamples,
        test_size=test_size,
        random_state=random_seed
    )
    
    for split_idx, (train_idx, test_idx) in enumerate(rss.split(X_log1p, y)):
        if logger and split_idx % 10 == 0:
            logger.info(f"  Ablation resample {split_idx + 1}/{n_resamples}")
        
        X_train, X_test = X_log1p.iloc[train_idx], X_log1p.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        for feature_set_name, feature_set_list in feature_sets.items():
            if len(feature_set_list) == 0:
                continue
            
            X_train_subset = X_train[feature_set_list]
            X_test_subset = X_test[feature_set_list]
            
            # Fit and evaluate
            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                class_weight='balanced',
                random_state=random_seed + split_idx,
                n_jobs=n_jobs,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2
            )
            rf.fit(X_train_subset, y_train)
            
            y_pred = rf.predict(X_test_subset)
            y_pred_proba = rf.predict_proba(X_test_subset)
            
            acc = accuracy_score(y_test, y_pred)
            bacc = balanced_accuracy_score(y_test, y_pred)
            f1_mac = f1_score(y_test, y_pred, average='macro', zero_division=0)
            
            try:
                if len(np.unique(y)) == 2:
                    roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
                else:
                    roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
            except:
                roc_auc = np.nan
            
            ablation_metrics.append({
                'split': split_idx,
                'feature_set': feature_set_name,
                'n_features': len(feature_set_list),
                'accuracy': acc,
                'balanced_accuracy': bacc,
                'macro_f1': f1_mac,
                'roc_auc': roc_auc
            })
    
    ablation_long = pd.DataFrame(ablation_metrics)

    # Summary with stable feature-set coverage, even when a set has zero usable features.
    summary_columns = [
        'feature_set', 'n_features_first',
        'accuracy_mean', 'accuracy_std', 'accuracy_median',
        'balanced_accuracy_mean', 'balanced_accuracy_std', 'balanced_accuracy_median',
        'macro_f1_mean', 'macro_f1_std', 'macro_f1_median',
        'roc_auc_mean', 'roc_auc_std', 'roc_auc_median'
    ]

    if ablation_long.empty:
        ablation_summary = pd.DataFrame(columns=summary_columns)
    else:
        ablation_summary = ablation_long.groupby('feature_set', as_index=False).agg(
            n_features_first=('n_features', 'first'),
            accuracy_mean=('accuracy', 'mean'),
            accuracy_std=('accuracy', 'std'),
            accuracy_median=('accuracy', 'median'),
            balanced_accuracy_mean=('balanced_accuracy', 'mean'),
            balanced_accuracy_std=('balanced_accuracy', 'std'),
            balanced_accuracy_median=('balanced_accuracy', 'median'),
            macro_f1_mean=('macro_f1', 'mean'),
            macro_f1_std=('macro_f1', 'std'),
            macro_f1_median=('macro_f1', 'median'),
            roc_auc_mean=('roc_auc', 'mean'),
            roc_auc_std=('roc_auc', 'std'),
            roc_auc_median=('roc_auc', 'median')
        )

    expected_feature_sets = list(feature_sets.keys())
    ablation_summary = pd.DataFrame({'feature_set': expected_feature_sets}).merge(
        ablation_summary,
        on='feature_set',
        how='left'
    )
    ablation_summary['n_features_first'] = ablation_summary['feature_set'].map(feature_set_sizes).fillna(0).astype(int)
    ablation_summary = ablation_summary[summary_columns]
    
    if logger:
        logger.info("Ablation analysis completed")
    
    return ablation_long, ablation_summary


# ============================================================================
# PLOTTING
# ============================================================================

def plot_da_volcano(
    da_results: pd.DataFrame,
    method_name: str,
    output_path: Path,
    alpha: float = 0.05,
    logger: Optional[logging.Logger] = None
):
    """Plot volcano plot for DA results."""
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    da_results_plot = da_results.copy()
    da_results_plot['log_p'] = -np.log10(da_results_plot['p_value'].replace(0, 1e-300))
    
    # Colors
    colors = da_results_plot['significant'].map({True: 'red', False: 'gray'})
    
    ax.scatter(
        da_results_plot['effect_size'],
        da_results_plot['log_p'],
        c=colors,
        alpha=0.6,
        s=50
    )
    
    ax.axhline(y=-np.log10(alpha), color='blue', linestyle='--', linewidth=1, label=f'q={alpha}')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    
    ax.set_xlabel('Effect Size (log fold change)', fontsize=12)
    ax.set_ylabel('-log10(p-value)', fontsize=12)
    ax.set_title(f'{method_name} Volcano Plot', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    if logger:
        logger.info(f"Saved volcano plot: {output_path}")


def plot_rf_importance(
    rf_importance: pd.DataFrame,
    top_n: int = 20,
    output_path: Path = None,
    logger: Optional[logging.Logger] = None
):
    """Plot top Random Forest features by importance."""
    
    top_rf = rf_importance.head(top_n).copy()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.barh(top_rf['taxon'], top_rf['mean_importance'], xerr=top_rf['std_importance'])
    ax.set_xlabel('Mean Permutation Importance', fontsize=12)
    ax.set_title(f'Top {top_n} Random Forest Features', fontsize=14)
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    if logger:
        logger.info(f"Saved RF importance plot: {output_path}")


def plot_ablation_comparison(
    ablation_long: pd.DataFrame,
    output_path: Path,
    logger: Optional[logging.Logger] = None
):
    """Plot ablation performance comparison."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    metrics = ['accuracy', 'balanced_accuracy', 'macro_f1', 'roc_auc']
    
    for idx, metric in enumerate(metrics):
        ax = axes.flatten()[idx]
        
        sns.boxplot(
            data=ablation_long,
            x='feature_set',
            y=metric,
            ax=ax
        )
        
        ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=12)
        ax.set_xlabel('Feature Set', fontsize=10)
        ax.set_ylabel(metric.replace("_", " "), fontsize=10)
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    if logger:
        logger.info(f"Saved ablation plot: {output_path}")


# ============================================================================
# EVIDENCE SCORING
# ============================================================================

def compute_evidence_scores(
    fuzzy_taxa: pd.DataFrame,
    da_results_combined: pd.DataFrame,
    rf_importance: pd.DataFrame,
    boruta_summary: Optional[pd.DataFrame],
    consensus_taxa: List[str],
    direction_agreement: pd.DataFrame,
    positive_markers: Optional[Dict[str, List[str]]] = None,
    alpha: float = 0.05,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Compute evidence scores for each fuzzy taxon.
    
    Evidence is awarded for:
    - ANCOM-BC2 significant
    - ALDEx2 significant
    - RF as positive marker for matching category (direction-aware)
    - RF-positive marker selection frequency >= 0.5
    - Boruta selected frequency >= 0.5
    - Direction agreement
    - In external consensus
    
    If positive_markers is provided, uses direction-aware RF evidence.
    Otherwise, falls back to original RF importance-based scoring.
    """
    
    evidence_list = []
    
    for _, fuzzy_row in fuzzy_taxa.iterrows():
        taxon = fuzzy_row['taxon']
        cluster = fuzzy_row.get('cluster', 'NA')
        rank = fuzzy_row.get('fuzzy_rank', np.nan)
        importance = fuzzy_row.get('membership', fuzzy_row.get('importance', np.nan))
        
        score = 0
        
        # ANCOM-BC2
        ancombc2_sig = False
        ancombc2_results = da_results_combined[
            (da_results_combined['method'] == 'ancombc2') &
            (da_results_combined['taxon'] == taxon)
        ]
        if len(ancombc2_results) > 0:
            ancombc2_sig = ancombc2_results.iloc[0]['significant']
            if ancombc2_sig:
                score += 1
        
        # ALDEx2
        aldex2_sig = False
        aldex2_results = da_results_combined[
            (da_results_combined['method'] == 'aldex2') &
            (da_results_combined['taxon'] == taxon)
        ]
        if len(aldex2_results) > 0:
            aldex2_sig = aldex2_results.iloc[0]['significant']
            if aldex2_sig:
                score += 1
        
        # RF - Use direction-aware positive markers if available
        rf_sig = False
        rf_freq = False
        
        if positive_markers is not None and isinstance(cluster, str) and cluster != 'NA':
            # Direction-aware RF scoring
            cat_str = str(cluster).replace(' ', '_')
            if cat_str in positive_markers and taxon in positive_markers[cat_str]:
                rf_sig = True
                score += 1
                
                # Count frequency by checking how many times this taxon appears as marker
                # For now, we count it as high frequency if it appears as a marker
                rf_freq = True
                score += 1
        else:
            # Fallback to original RF importance-based scoring
            rf_results = rf_importance[rf_importance['taxon'] == taxon]
            if len(rf_results) > 0:
                rf_sig = True
                score += 1
                
                # RF selection frequency
                if rf_results.iloc[0]['selection_frequency'] >= 0.5:
                    rf_freq = True
                    score += 1
        
        # Boruta
        boruta_sig = False
        if boruta_summary is not None:
            boruta_results = boruta_summary[boruta_summary['taxon'] == taxon]
            if len(boruta_results) > 0 and boruta_results.iloc[0]['selection_frequency'] >= 0.5:
                boruta_sig = True
                score += 1
        
        # Direction
        dir_agree = False
        dir_results = direction_agreement[direction_agreement['taxon'] == taxon]
        if len(dir_results) > 0 and dir_results.iloc[0]['direction_agrees'] == True:
            dir_agree = True
            score += 1
        
        # Consensus
        in_consensus = taxon in consensus_taxa
        if in_consensus:
            score += 1
        
        # Label
        if score >= 5:
            label = 'strong'
        elif score >= 3:
            label = 'moderate'
        elif score >= 1:
            label = 'weak'
        else:
            label = 'unsupported'
        
        evidence_list.append({
            'cluster': cluster,
            'taxon': taxon,
            'fuzzy_rank': rank,
            'fuzzy_membership_or_importance': importance,
            'ancombc2_supported': ancombc2_sig,
            'aldex2_supported': aldex2_sig,
            'rf_positive_marker': rf_sig,
            'rf_marker_frequency': rf_freq,
            'boruta_supported': boruta_sig,
            'direction_consistent': dir_agree,
            'evidence_score': score,
            'evidence_label': label
        })
    
    evidence_df = pd.DataFrame(evidence_list)
    expected_cols = [
        'taxon', 'cluster', 'fuzzy_rank', 'fuzzy_membership_or_importance',
        'ancombc2_supported', 'aldex2_supported', 'rf_positive_marker', 'rf_marker_frequency',
        'boruta_supported', 'direction_consistent', 'evidence_score', 'evidence_label'
    ]

    # Ensure expected columns exist even if no evidence was found
    for c in expected_cols:
        if c not in evidence_df.columns:
            evidence_df[c] = pd.Series(dtype='object')

    if not evidence_df.empty and 'evidence_score' in evidence_df.columns:
        evidence_df = evidence_df.sort_values('evidence_score', ascending=False)

    # Reorder columns to a stable schema
    evidence_df = evidence_df[[c for c in expected_cols if c in evidence_df.columns]]

    return evidence_df


# ============================================================================
# REPORTING
# ============================================================================

def generate_report(
    output_dir: Path,
    input_summary: Dict,
    sample_matching_report: pd.DataFrame,
    taxa_filtering_report: pd.DataFrame,
    fuzzy_matching_report: Dict,
    da_summary: Dict,
    rf_metrics: pd.DataFrame,
    rf_importance: pd.DataFrame,
    boruta_summary: Optional[pd.DataFrame],
    overlap_summary: pd.DataFrame,
    ablation_summary: pd.DataFrame,
    evidence_scores: pd.DataFrame,
    positive_markers: Optional[Dict[str, List[str]]] = None,
    negative_markers: Optional[Dict[str, List[str]]] = None,
    logger: Optional[logging.Logger] = None
):
    """Generate comprehensive markdown report."""
    
    report_path = output_dir / "report.md"
    
    with open(report_path, 'w') as f:
        f.write("# Microbiome Fuzzy Taxa Validation Report\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Input Summary
        f.write("## 1. Input Summary\n\n")
        f.write(f"- **OTU file**: {input_summary.get('otu_file', 'N/A')}\n")
        f.write(f"- **Metadata file**: {input_summary.get('metadata_file', 'N/A')}\n")
        f.write(f"- **Clinical column**: {input_summary.get('clinical_col', 'N/A')}\n")
        f.write(f"- **Fuzzy taxa file**: {input_summary.get('top_taxa_file', 'N/A')}\n\n")
        
        # Sample Matching
        f.write("## 2. Sample Matching & Filtering\n\n")
        f.write(sample_matching_report.to_markdown(index=False))
        f.write("\n\n")
        
        f.write(taxa_filtering_report.to_markdown(index=False))
        f.write("\n\n")
        
        # Fuzzy Taxa
        f.write("## 3. Fuzzy Taxa Matching\n\n")
        f.write(f"- **Matched**: {fuzzy_matching_report.get('n_matched', 0)}\n")
        f.write(f"- **In fuzzy but not OTU**: {fuzzy_matching_report.get('n_unmatched_fuzzy', 0)}\n")
        f.write(f"- **In OTU but not fuzzy**: {fuzzy_matching_report.get('n_unmatched_otu', 0)}\n\n")
        
        # DA Summary
        f.write("## 4. Differential Abundance\n\n")
        f.write(f"- **ANCOM-BC2 significant**: {da_summary.get('ancombc2_significant', 0)}\n")
        f.write(f"- **ALDEx2 significant**: {da_summary.get('aldex2_significant', 0)}\n")
        f.write(f"- **Fuzzy taxa with ANCOM-BC2 support**: {da_summary.get('fuzzy_ancombc2', 0)}\n")
        f.write(f"- **Fuzzy taxa with ALDEx2 support**: {da_summary.get('fuzzy_aldex2', 0)}\n\n")
        
        # ML Summary
        f.write("## 5. Machine Learning\n\n")
        f.write(f"### Random Forest\n\n")
        f.write(f"- **Mean balanced accuracy**: {rf_metrics['balanced_accuracy'].mean():.3f} ± {rf_metrics['balanced_accuracy'].std():.3f}\n")
        f.write(f"- **Mean ROC-AUC**: {rf_metrics['roc_auc'].mean():.3f} ± {rf_metrics['roc_auc'].std():.3f}\n")
        f.write(f"- **Number of RF-important taxa**: {len(rf_importance)}\n")
        
        # Marker summary if available
        if positive_markers is not None:
            total_positive = sum(len(m) for m in positive_markers.values())
            total_negative = sum(len(m) for m in negative_markers.values()) if negative_markers else 0
            f.write(f"- **RF-positive marker taxa** (abundance/prevalence-enriched): {total_positive}\n")
            for cat, markers in sorted(positive_markers.items()):
                if markers:
                    f.write(f"  - {cat}: {len(markers)} markers\n")
            if negative_markers:
                f.write(f"- **RF-negative marker taxa** (abundance/prevalence-depleted): {total_negative}\n")
                for cat, markers in sorted(negative_markers.items()):
                    if markers:
                        f.write(f"  - {cat}: {len(markers)} markers\n")
            f.write(f"\n**Note**: RF permutation importance is nondirectional (measures discriminative power).\n")
            f.write(f"Positive markers are taxa enriched/abundant in a category (fair comparison for fuzzy clusters).\n")
            f.write(f"Negative markers are taxa depleted/absent in a category (complementary depletion signals).\n\n")
        
        # Get fuzzy taxa set for marking
        fuzzy_taxa_set = set(evidence_scores['taxon'].unique()) if evidence_scores is not None and not evidence_scores.empty else set()
        n_fuzzy = len(fuzzy_taxa_set)
        
        f.write(f"- **Top {n_fuzzy} features** (matched to fuzzy taxa count):\n\n")
        fuzzy_in_top = 0
        for idx, row in rf_importance.head(n_fuzzy).iterrows():
            in_fuzzy = "✓ (in fuzzy list)" if row['taxon'] in fuzzy_taxa_set else ""
            f.write(f"  - {row['taxon']}: {row['mean_importance']:.4f} {in_fuzzy}\n")
            if row['taxon'] in fuzzy_taxa_set:
                fuzzy_in_top += 1
        
        f.write(f"\n- **Fuzzy coverage in top {n_fuzzy} RF features**: {fuzzy_in_top}/{n_fuzzy} ({100*fuzzy_in_top/n_fuzzy:.1f}%)\n")
        
        if boruta_summary is not None:
            f.write(f"\n### Boruta\n\n")
            n_selected = (boruta_summary['selection_frequency'] >= 0.5).sum()
            f.write(f"- **Taxa selected (>= 50% resamples)**: {n_selected}\n\n")
        
        # Ablation
        f.write("## 6. Predictive Ablation\n\n")
        f.write(ablation_summary.to_markdown(index=False))
        f.write("\n\n")
        
        # Evidence Scores
        f.write("## 7. Evidence Scores\n\n")
        req_cols = ['taxon', 'cluster', 'ancombc2_supported', 'aldex2_supported', 'rf_supported', 'evidence_score', 'evidence_label']
        if evidence_scores is None or evidence_scores.empty:
            f.write("No evidence scores were generated (empty results).\n\n")
        else:
            missing = [c for c in req_cols if c not in evidence_scores.columns]
            if missing:
                f.write(f"Evidence scores missing expected columns: {missing}.\n\n")
                # Write whatever is available
                f.write(evidence_scores.head(30).to_markdown(index=False))
                f.write("\n\n")
            else:
                f.write(evidence_scores[req_cols].head(30).to_markdown(index=False))
                f.write("\n\n")
        
        f.write("---\n\nEnd of Report\n")
    
    if logger:
        logger.info(f"Report saved: {report_path}")


# ============================================================================
# SOFTWARE VERSIONS
# ============================================================================

def save_software_versions(output_dir: Path, logger: Optional[logging.Logger] = None):
    """Save versions of all software dependencies."""
    
    versions_path = output_dir / "software_versions.txt"
    
    with open(versions_path, 'w') as f:
        f.write(f"Python: {sys.version}\n")
        f.write(f"Platform: {platform.platform()}\n\n")
        
        # Python packages
        import pandas
        import numpy
        import scipy
        import sklearn
        import matplotlib
        import seaborn
        
        f.write(f"pandas: {pandas.__version__}\n")
        f.write(f"numpy: {numpy.__version__}\n")
        f.write(f"scipy: {scipy.__version__}\n")
        f.write(f"scikit-learn: {sklearn.__version__}\n")
        f.write(f"matplotlib: {matplotlib.__version__}\n")
        f.write(f"seaborn: {seaborn.__version__}\n")
        
        if HAS_BORUTA:
            import boruta
            # f.write(f"boruta: {boruta.__version__}\n")
        
        f.write("\nR version info:\n")
        try:
            result = subprocess.run(
                ["Rscript", "-e", "R.version.string"],
                capture_output=True,
                text=True,
                timeout=10
            )
            f.write(result.stdout)
        except:
            f.write("Could not retrieve R version\n")
    
    if logger:
        logger.info(f"Software versions saved: {versions_path}")


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    
    parser = argparse.ArgumentParser(
        description="Validate fuzzy-clustering-selected microbiome taxa against independent methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLE USAGE:
    python microbiome_validate_fuzzy_taxa.py \\
      --otu_file otu_table.tsv \\
      --metadata_file metadata.tsv \\
      --metadata_sample_col sample_id \\
      --clinical_col disease_status \\
      --top_taxa_file fuzzy_taxa.tsv \\
      --output_dir results_fuzzy_validation \\
      --covariates age,sex,batch \\
      --n_resamples 50 \\
      --n_permutations 10000 \\
      --run_boruta
        """
    )
    
    # Required directly, or populated by --dataset from dataset_configs.json
    parser.add_argument('--otu_file', default=None, help='Raw OTU/ASV table')
    parser.add_argument('--metadata_file', default=None, help='Metadata table')
    parser.add_argument('--clinical_col', default=None, help='Clinical phenotype column')
    parser.add_argument('--top_taxa_file', default=None, help='Fuzzy-selected top taxa file')
    parser.add_argument('--output_dir', default=None, help='Output directory')
    parser.add_argument('--dataset', default=None, help='Dataset key in dataset_configs.json')
    parser.add_argument(
        '--dataset_config_file',
        default='dataset_configs.json',
        help='Dataset config JSON used with --dataset'
    )
    parser.add_argument('--projects_file', default=None, help='Compendium projects metadata file')
    parser.add_argument('--tags_file', default=None, help='Compendium tags metadata file')
    parser.add_argument(
        '--metadata_dataset',
        choices=['compendium', 'hadza'],
        default=None,
        help='Metadata extraction mode passed to archetypal_analysis.utils.metadata.extract_metadata'
    )
    
    # Optional sample/delimiter arguments
    parser.add_argument('--metadata_sample_col', default=None, help='Sample ID column in metadata (default: use index)')
    parser.add_argument('--otu_orientation', choices=['taxa_rows', 'samples_rows', 'auto'], default='auto')
    parser.add_argument('--sep', choices=['auto', 'comma', 'tab'], default='auto', help='File delimiter')
    
    # Filtering arguments
    parser.add_argument('--min_prevalence', type=float, default=0.05, help='Minimum prevalence threshold')
    parser.add_argument('--min_total_count', type=int, default=10, help='Minimum total count')
    
    # Resampling/ML arguments
    parser.add_argument('--n_resamples', type=int, default=50, help='Number of resamples')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test fraction')
    parser.add_argument('--cv_folds', type=int, default=5, help='CV folds (if using CV)')
    parser.add_argument('--random_seed', type=int, default=123, help='Random seed')
    parser.add_argument('--n_permutations', type=int, default=10000, help='Permutation test iterations')
    parser.add_argument('--n_estimators', type=int, default=1000, help='RF n_estimators')
    parser.add_argument('--n_jobs', type=int, default=-1, help='Parallel jobs')
    
    # Method flags
    parser.add_argument('--run_boruta', action='store_true', help='Run BorutaPy')
    parser.set_defaults(run_ancombc2=True, run_aldex2=True)
    parser.add_argument('--run_ancombc2', dest='run_ancombc2', action='store_true', help='Run ANCOM-BC2')
    parser.add_argument('--skip_ancombc2', dest='run_ancombc2', action='store_false', help='Skip ANCOM-BC2')
    parser.add_argument('--run_aldex2', dest='run_aldex2', action='store_true', help='Run ALDEx2')
    parser.add_argument('--skip_aldex2', dest='run_aldex2', action='store_false', help='Skip ALDEx2')
    
    # Advanced options
    parser.add_argument('--covariates', default=None, help='Comma-separated covariates')
    parser.add_argument('--class_positive_label', default=None, help='Positive class label for binary classification')
    parser.add_argument('--alpha', type=float, default=0.05, help='Significance threshold')
    parser.add_argument('--force_log1p_for_da', action='store_true', help='Use log1p data for DA methods (not recommended)')
    
    # Direction-aware RF options
    parser.add_argument('--rf_importance_threshold', type=float, default=0, help='Min RF importance for marker consideration')
    parser.add_argument('--direction_abundance_delta_threshold', type=float, default=0, help='Min abundance delta for direction')
    parser.add_argument('--direction_prevalence_delta_threshold', type=float, default=0, help='Min prevalence delta for direction')
    parser.add_argument('--positive_marker_rule', choices=['abundance_or_prevalence', 'abundance_and_prevalence'], default='abundance_or_prevalence', help='Rule for assigning positive markers')
    
    # Experiment skipping options
    parser.set_defaults(skip_if_done=True)
    parser.add_argument('--skip_if_done', dest='skip_if_done', action='store_true', help='Skip experiment if already completed (default: true)')
    parser.add_argument('--force_rerun', dest='skip_if_done', action='store_false', help='Force re-run even if experiment already completed')
    
    return parser.parse_args()


# ============================================================================
# EXPERIMENT COMPLETION CHECK
# ============================================================================

def is_experiment_done(output_dir: Path) -> bool:
    """Check if an experiment has already completed.
    
    Args:
        output_dir: Output directory to check
    
    Returns:
        True if completion marker exists, False otherwise
    """
    completion_marker = output_dir / "COMPLETION_MARKER.txt"
    return completion_marker.exists()


def get_checkpoint_file(output_dir: Path, step_num: int, step_name: str) -> Path:
    """Get the checkpoint marker file for a given step.
    
    Args:
        output_dir: Output directory
        step_num: Step number (1-19)
        step_name: Human-readable step name
    
    Returns:
        Path to checkpoint file
    """
    checkpoints_dir = output_dir / ".checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    return checkpoints_dir / f"step_{step_num:02d}_{step_name.lower().replace(' ', '_')}.done"


def should_skip_step(output_dir: Path, step_num: int, step_name: str, skip_if_done: bool) -> bool:
    """Check if a step should be skipped.
    
    Args:
        output_dir: Output directory
        step_num: Step number
        step_name: Step name
        skip_if_done: Whether to skip if done
    
    Returns:
        True if step should be skipped, False otherwise
    """
    if not skip_if_done:
        return False
    
    checkpoint_file = get_checkpoint_file(output_dir, step_num, step_name)
    if not checkpoint_file.exists():
        return False

    # If checkpoint contains an Outputs: section, verify each listed file exists.
    try:
        with open(checkpoint_file, 'r') as f:
            lines = [l.strip() for l in f.readlines()]
    except Exception:
        return False

    outputs_idx = None
    for i, line in enumerate(lines):
        if line.startswith('Outputs:'):
            outputs_idx = i
            break

    if outputs_idx is None:
        return True

    missing = []
    for out_line in lines[outputs_idx + 1:]:
        if not out_line:
            continue
        # stored paths are relative to output_dir when possible
        out_path = Path(out_line)
        if not out_path.is_absolute():
            out_path = output_dir / out_path
        if not out_path.exists():
            missing.append(str(out_path))

    return len(missing) == 0


def mark_step_done(output_dir: Path, step_num: int, step_name: str, logger, outputs: Optional[list] = None) -> None:
    """Mark a step as completed by writing a checkpoint file.
    
    Args:
        output_dir: Output directory
        step_num: Step number
        step_name: Step name
        logger: Logger instance
    """
    checkpoint_file = get_checkpoint_file(output_dir, step_num, step_name)
    tmp_file = checkpoint_file.with_suffix('.tmp')
    with open(tmp_file, 'w') as f:
        f.write(f"Step {step_num}: {step_name}\nCompleted at: {datetime.now()}\n")
        if outputs:
            f.write("Outputs:\n")
            for p in outputs:
                # store paths relative to output_dir when possible for portability
                try:
                    ppath = Path(p)
                    if not ppath.is_absolute():
                        ppath = Path(p)
                    try:
                        rel = ppath.relative_to(output_dir)
                        f.write(f"{rel}\n")
                    except Exception:
                        f.write(f"{ppath}\n")
                except Exception:
                    f.write(f"{p}\n")
    try:
        tmp_file.replace(checkpoint_file)
    except Exception:
        # fallback to simple write
        with open(checkpoint_file, 'w') as f:
            f.write(open(tmp_file).read())


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main pipeline execution."""
    
    args = parse_arguments()
    args = apply_dataset_config(args)
    validate_required_inputs(args)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir)
    
    # Check if experiment is already done
    if args.skip_if_done and is_experiment_done(output_dir):
        logger.info("="*80)
        logger.info("EXPERIMENT ALREADY COMPLETED")
        logger.info("="*80)
        logger.info(f"Output directory: {output_dir}")
        logger.info("Skipping re-run (use --force_rerun to override)")
        logger.info("="*80)
        return 0
    logger.info("="*80)
    logger.info("MICROBIOME FUZZY TAXA VALIDATION PIPELINE")
    logger.info("="*80)
    logger.info(f"Start time: {datetime.now()}")
    
    # Save run config
    config_path = output_dir / "run_config.json"
    dump_json(vars(args), config_path)
    logger.info(f"Configuration saved: {config_path}")
    
    # Save software versions
    save_software_versions(output_dir, logger)
    
    try:
        # ====================================================================
        # 1. INPUT LOADING
        # ====================================================================
        logger.info("\n" + "="*80)
        logger.info("1. LOADING INPUT DATA")
        logger.info("="*80)
        
        otu_raw = load_otu_table(Path(args.otu_file), args.sep, logger)
        metadata_raw = load_metadata(Path(args.metadata_file), args.sep, logger)
        metadata_raw = prepare_sample_metadata_for_extraction(
            metadata_raw,
            args.metadata_sample_col,
        )
        projects_raw = load_optional_metadata_source(
            args.projects_file,
            args.sep,
            logger,
            label='projects metadata',
        )
        tags_raw = load_optional_metadata_source(
            args.tags_file,
            args.sep,
            logger,
            label='tags metadata',
        )
        extract_cols = metadata_columns_to_extract(args)
        
        input_summary = {
            'otu_file': args.otu_file,
            'metadata_file': args.metadata_file,
            'projects_file': args.projects_file,
            'tags_file': args.tags_file,
            'metadata_dataset': args.metadata_dataset,
            'clinical_col': args.clinical_col,
            'metadata_extract_cols': extract_cols,
            'top_taxa_file': args.top_taxa_file
        }
        
        # ====================================================================
        # 2. SAMPLE MATCHING
        # ====================================================================
        logger.info("\n" + "="*80)
        logger.info("2. SAMPLE MATCHING & QC")
        logger.info("="*80)
        
        # Detect orientation
        if args.otu_orientation == 'auto':
            detected_orientation = detect_otu_orientation_with_metadata(
                otu_raw,
                metadata_raw,
                projects_raw,
                tags_raw,
                extract_cols,
                args.metadata_dataset,
                logger,
            )
            logger.info(f"Detected OTU orientation: {detected_orientation}")
        else:
            detected_orientation = args.otu_orientation
        
        # Transpose if needed
        otu_samples_taxa = transpose_if_needed(otu_raw, detected_orientation, logger)
        
        # Extract metadata and match samples using package metadata utilities.
        metadata_extracted = extract_validation_metadata(
            otu_samples_taxa.index,
            metadata_raw,
            projects_raw,
            tags_raw,
            extract_cols,
            args.metadata_dataset,
            logger,
        )
        otu_matched, metadata_matched, sample_matching_report = (
            match_samples_from_extracted_metadata(
                otu_samples_taxa,
                metadata_extracted,
                extract_cols,
                logger,
            )
        )
        
        # Save matched data
        qc_dir = output_dir / "qc"
        qc_dir.mkdir(exist_ok=True)
        sample_matching_report.to_csv(qc_dir / "sample_matching.tsv", sep='\t', index=False)
        
        # ====================================================================
        # 3. TAXA FILTERING
        # ====================================================================
        logger.info("\n" + "="*80)
        logger.info("3. TAXA FILTERING")
        logger.info("="*80)
        
        otu_filtered, taxa_filtering_report = filter_taxa(
            otu_matched,
            args.min_prevalence,
            args.min_total_count,
            logger
        )
        taxa_filtering_report.to_csv(qc_dir / "taxa_filtering_summary.tsv", sep='\t', index=False)
        
        # ====================================================================
        # 4. CLINICAL COLUMN VALIDATION
        # ====================================================================
        logger.info("\n" + "="*80)
        logger.info("4. CLINICAL PHENOTYPE VALIDATION")
        logger.info("="*80)
        
        # Align metadata to filtered OTU samples
        metadata_aligned = metadata_matched.loc[otu_filtered.index].copy()
        
        y_encoded, class_counts = validate_clinical_column(
            metadata_aligned, args.clinical_col, logger
        )
        
        # Drop samples with missing labels
        keep_idx = ~y_encoded.isna()
        otu_filtered = otu_filtered[keep_idx]
        metadata_aligned = metadata_aligned[keep_idx]
        y_encoded = y_encoded[keep_idx]
        
        # ====================================================================
        # 5. LOG-TRANSFORM FOR ML
        # ====================================================================
        logger.info("\n" + "="*80)
        logger.info("5. DATA TRANSFORMATION")
        logger.info("="*80)
        
        X_log1p = np.log1p(otu_filtered)
        logger.info(f"Created log1p-transformed matrix: {X_log1p.shape}")
        
        # ====================================================================
        # 6. FUZZY TAXA LOADING
        # ====================================================================
        logger.info("\n" + "="*80)
        logger.info("6. FUZZY TAXA LOADING & MATCHING")
        logger.info("="*80)

        if args.dataset and getattr(args, 'dataset_config', None):
            try:
                computed_top_taxa_path = compute_top_taxa_from_dataset_config(
                    args.dataset_config,
                    output_dir,
                    logger,
                )
                args.top_taxa_file = str(computed_top_taxa_path)
            except Exception as exc:
                logger.warning(
                    "Failed to compute top taxa from archetypal results; using top_taxa_file instead: %s",
                    exc,
                )

        if args.top_taxa_file is None:
            raise ValueError("top_taxa_file is required when top taxa cannot be computed from dataset config.")

        fuzzy_taxa = load_top_taxa(Path(args.top_taxa_file), args.sep, logger)
        
        fuzzy_matched, fuzzy_unmatched_top, fuzzy_unmatched_otu = match_top_taxa(
            fuzzy_taxa, list(otu_filtered.columns), logger
        )
        
        fuzzy_dir = output_dir / "fuzzy_taxa"
        fuzzy_dir.mkdir(exist_ok=True)
        fuzzy_taxa.to_csv(fuzzy_dir / "fuzzy_taxa_loaded.tsv", sep='\t', index=False)
        fuzzy_matched.to_csv(fuzzy_dir / "fuzzy_taxa_matched.tsv", sep='\t', index=False)
        fuzzy_unmatched_top.to_csv(fuzzy_dir / "fuzzy_taxa_unmatched_in_top.tsv", sep='\t', index=False)
        fuzzy_unmatched_otu.to_csv(fuzzy_dir / "fuzzy_taxa_unmatched_in_otu.tsv", sep='\t', index=False)
        
        fuzzy_matching_report = {
            'n_matched': len(fuzzy_matched),
            'n_unmatched_fuzzy': len(fuzzy_unmatched_top),
            'n_unmatched_otu': len(fuzzy_unmatched_otu)
        }
        
        # ====================================================================
        # 7. SAVE PROCESSED DATA
        # ====================================================================
        logger.info("\n" + "="*80)
        logger.info("7. SAVING PROCESSED DATA")
        logger.info("="*80)
        
        processed_dir = output_dir / "processed"
        processed_dir.mkdir(exist_ok=True)
        
        otu_filtered.to_csv(processed_dir / "raw_counts_samples_by_taxa.tsv", sep='\t')
        X_log1p.to_csv(processed_dir / "log1p_abundance_samples_by_taxa.tsv", sep='\t')
        metadata_aligned.to_csv(processed_dir / "metadata_matched.tsv", sep='\t')
        
        # Input summary
        input_summary_data = {
            'n_samples_input_otu': otu_raw.shape[0],
            'n_taxa_input_otu': otu_raw.shape[1],
            'n_samples_input_metadata': len(metadata_raw),
            'n_samples_shared': len(otu_filtered),
            'n_taxa_after_filtering': len(otu_filtered.columns),
            'n_classes': len(class_counts),
            'class_counts': class_counts
        }
        dump_json(input_summary_data, qc_dir / "input_summary.json")
        
        # ====================================================================
        # 8. DIFFERENTIAL ABUNDANCE - ANCOM-BC2
        # ====================================================================
        logger.info("\n" + "="*80)
        logger.info("8. DIFFERENTIAL ABUNDANCE: ANCOM-BC2")
        logger.info("="*80)
        
        da_dir_path = output_dir / "differential_abundance"
        combined_da_file = da_dir_path / "combined_da_results.tsv"
        covariates_list = None
        
        if should_skip_step(output_dir, 8, "ANCOM-BC2", args.skip_if_done):
            logger.info("SKIPPING: Step 8 already completed")
            if combined_da_file.exists():
                da_results_combined = pd.read_csv(combined_da_file, sep='\t')
                logger.info(f"Loaded combined DA results from {combined_da_file}")
            else:
                da_results_combined = pd.DataFrame(columns=DA_RESULT_COLUMNS)
        else:
            if args.covariates:
                covariates_list = [c.strip() for c in args.covariates.split(',')]
                covariates_list = [c for c in covariates_list if c in metadata_aligned.columns]
            
            if args.force_log1p_for_da:
                logger.warning("!!! Using log1p-transformed data for DA methods (NOT RECOMMENDED) !!!")
                otu_for_da = X_log1p
            else:
                otu_for_da = otu_filtered
            
            ancombc2_results = None
            if args.run_ancombc2:
                ancombc2_results = run_ancombc2(
                    otu_for_da,
                    metadata_aligned,
                    args.clinical_col,
                    output_dir,
                    covariates_list,
                    args.alpha,
                    logger
                )
            
            # ====================================================================
            # 9. DIFFERENTIAL ABUNDANCE - ALDEx2
            # ====================================================================
            logger.info("\n" + "="*80)
            logger.info("9. DIFFERENTIAL ABUNDANCE: ALDEx2")
            logger.info("="*80)
            
            aldex2_results = None
            if args.run_aldex2:
                aldex2_results = run_aldex2(
                    otu_for_da,
                    metadata_aligned,
                    args.clinical_col,
                    output_dir,
                    class_counts,
                    covariates_list,
                    args.alpha,
                    logger
                )
            
            # Combine DA results
            da_results_combined = []
            if ancombc2_results is not None:
                da_results_combined.append(ancombc2_results)
            if aldex2_results is not None:
                da_results_combined.append(aldex2_results)
            
            if len(da_results_combined) > 0:
                da_results_combined = pd.concat(da_results_combined, ignore_index=True)
            else:
                da_results_combined = pd.DataFrame(columns=DA_RESULT_COLUMNS)
            
            if len(da_results_combined) > 0:
                da_dir_path.mkdir(parents=True, exist_ok=True)
                da_results_combined.to_csv(
                    combined_da_file,
                    sep='\t',
                    index=False
                )
            
            mark_step_done(output_dir, 8, "ANCOM-BC2", logger)
        
        # ====================================================================
        # 10. MACHINE LEARNING: RANDOM FOREST
        # ====================================================================
        logger.info("\n" + "="*80)
        logger.info("10. MACHINE LEARNING: RANDOM FOREST")
        logger.info("="*80)
        
        ml_dir = output_dir / "ml" / "random_forest"
        rf_metrics_file = ml_dir / "rf_resample_metrics.tsv"
        rf_importance_file = ml_dir / "rf_permutation_importance_summary.tsv"
        
        if should_skip_step(output_dir, 10, "Random Forest", args.skip_if_done):
            logger.info("SKIPPING: Step 10 already completed")
            if rf_metrics_file.exists() and rf_importance_file.exists():
                rf_metrics = pd.read_csv(rf_metrics_file, sep='\t')
                rf_importance = pd.read_csv(rf_importance_file, sep='\t')
                logger.info(f"Loaded RF metrics from {rf_metrics_file}")
            else:
                logger.warning("RF output files not found, will re-run step 10")
                rf_metrics, rf_importance = run_random_forest(
                    X_log1p,
                    y_encoded,
                    args.n_resamples,
                    args.test_size,
                    args.n_estimators,
                    args.random_seed,
                    args.n_jobs,
                    logger
                )
                ml_dir.mkdir(parents=True, exist_ok=True)
                rf_metrics.to_csv(rf_metrics_file, sep='\t', index=False)
                rf_importance.to_csv(rf_importance_file, sep='\t', index=False)
                mark_step_done(output_dir, 10, "Random Forest", logger, outputs=[rf_metrics_file, rf_importance_file])
        else:
            rf_metrics, rf_importance = run_random_forest(
                X_log1p,
                y_encoded,
                args.n_resamples,
                args.test_size,
                args.n_estimators,
                args.random_seed,
                args.n_jobs,
                logger
            )
            
            ml_dir.mkdir(parents=True, exist_ok=True)
            
            rf_metrics.to_csv(rf_metrics_file, sep='\t', index=False)
            rf_importance.to_csv(rf_importance_file, sep='\t', index=False)
            
            mark_step_done(output_dir, 10, "Random Forest", logger, outputs=[rf_metrics_file, rf_importance_file])
        
        # ====================================================================
        # 10.5. MACHINE LEARNING: RF DIRECTION ANNOTATION
        # ====================================================================
        logger.info("\n" + "="*80)
        logger.info("10.5. RF DIRECTION ANNOTATION")
        logger.info("="*80)
        
        rf_direction_file = ml_dir / "rf_direction_annotated_importance.tsv"
        rf_positive_markers_file = ml_dir / "rf_positive_markers_by_category.tsv"
        rf_negative_markers_file = ml_dir / "rf_negative_markers_by_category.tsv"
        
        # Compute direction annotations
        rf_direction_annotated = compute_rf_direction_annotations(
            X_log1p,
            otu_filtered,  # raw counts
            y_encoded,
            rf_importance,
            abundance_delta_threshold=args.direction_abundance_delta_threshold,
            prevalence_delta_threshold=args.direction_prevalence_delta_threshold,
            logger=logger
        )
        rf_direction_annotated.to_csv(rf_direction_file, sep='\t', index=False)
        
        # Extract positive and negative markers
        positive_markers = get_positive_markers(
            rf_direction_annotated,
            importance_threshold=args.rf_importance_threshold,
            abundance_delta_threshold=args.direction_abundance_delta_threshold,
            prevalence_delta_threshold=args.direction_prevalence_delta_threshold,
            rule=args.positive_marker_rule,
            logger=logger
        )
        
        negative_markers = get_negative_markers(
            rf_direction_annotated,
            importance_threshold=args.rf_importance_threshold,
            logger=logger
        )
        
        # Save positive markers
        positive_markers_list = []
        for category, markers in positive_markers.items():
            for marker in markers:
                positive_markers_list.append({'category': category, 'taxon': marker})
        positive_markers_df = pd.DataFrame(positive_markers_list)
        positive_markers_df.to_csv(rf_positive_markers_file, sep='\t', index=False)
        
        # Save negative markers
        negative_markers_list = []
        for category, markers in negative_markers.items():
            for marker in markers:
                negative_markers_list.append({'category': category, 'taxon': marker})
        negative_markers_df = pd.DataFrame(negative_markers_list)
        negative_markers_df.to_csv(rf_negative_markers_file, sep='\t', index=False)
        
        logger.info(f"Saved direction annotations to {rf_direction_file}")
        logger.info(f"Saved {len(positive_markers_df)} positive markers to {rf_positive_markers_file}")
        logger.info(f"Saved {len(negative_markers_df)} negative markers to {rf_negative_markers_file}")
        
        # ====================================================================
        # 11. MACHINE LEARNING: BORUTA
        # ====================================================================
        logger.info("\n" + "="*80)
        logger.info("11. MACHINE LEARNING: BORUTA")
        logger.info("="*80)
        
        boruta_dir = output_dir / "ml" / "boruta"
        boruta_file = boruta_dir / "boruta_summary.tsv"
        boruta_skipped_file = boruta_dir / "boruta_skipped.txt"
        
        if should_skip_step(output_dir, 11, "Boruta", args.skip_if_done):
            logger.info("SKIPPING: Step 11 already completed")
            if boruta_file.exists():
                boruta_summary = pd.read_csv(boruta_file, sep='\t')
                logger.info(f"Loaded Boruta results from {boruta_file}")
            else:
                boruta_summary = None
        else:
            boruta_dir.mkdir(parents=True, exist_ok=True)
            
            boruta_summary = None
            if args.run_boruta:
                if HAS_BORUTA:
                    boruta_summary = run_boruta(
                        X_log1p,
                        y_encoded,
                        10,  # n_resamples for Boruta
                        args.random_seed,
                        args.n_jobs,
                        logger
                    )
                    boruta_summary.to_csv(boruta_file, sep='\t', index=False)
                else:
                    logger.warning("BorutaPy not installed. Install with: pip install boruta")
                    with open(boruta_skipped_file, 'w') as f:
                        f.write("BorutaPy not installed.\nInstall with: pip install boruta\n")
            else:
                with open(boruta_skipped_file, 'w') as f:
                    f.write("Boruta was not run (--run_boruta not specified)\n")
            
            mark_step_done(output_dir, 11, "Boruta", logger)
        
        # ====================================================================
        # 12-14. COMPARISONS: OVERLAP, RANK CORRELATION & DIRECTION
        # ====================================================================
        logger.info("\n" + "="*80)
        logger.info("12-14. COMPARISONS: OVERLAP, RANK CORRELATION & DIRECTION")
        logger.info("="*80)
        
        comparison_dir = output_dir / "comparison"
        overlap_file = comparison_dir / "overlap_summary.tsv"
        rank_corr_file = comparison_dir / "rank_correlation_summary.tsv"
        direction_agreement_file = comparison_dir / "direction_agreement.tsv"
        
        if should_skip_step(output_dir, 12, "Comparisons", args.skip_if_done):
            logger.info("SKIPPING: Steps 12-14 already completed")
            overlap_summary = pd.read_csv(overlap_file, sep='\t') if overlap_file.exists() else pd.DataFrame()
            rank_corr_df = pd.read_csv(rank_corr_file, sep='\t') if rank_corr_file.exists() else pd.DataFrame()
            direction_agreement = pd.read_csv(direction_agreement_file, sep='\t') if direction_agreement_file.exists() else pd.DataFrame()
        else:
            comparison_dir.mkdir(parents=True, exist_ok=True)
            
            # ====================================================================
            # 12. COMPARISONS: OVERLAP & PERMUTATION TESTS
            # ====================================================================
            logger.info("\n" + "="*80)
            logger.info("12. OVERLAP & PERMUTATION TESTS")
            logger.info("="*80)
            
            fuzzy_taxa_set = set(fuzzy_matched['taxon'])
            all_taxa_universe = set(otu_filtered.columns)
            
            # Get significant taxa from each method
            ancombc2_sig = set()
            aldex2_sig = set()
            
            if len(da_results_combined) > 0:
                ancombc2_sig = set(
                    da_results_combined[
                        (da_results_combined['method'] == 'ancombc2') &
                        (da_results_combined['significant'])
                    ]['taxon']
                )
                aldex2_sig = set(
                    da_results_combined[
                        (da_results_combined['method'] == 'aldex2') &
                        (da_results_combined['significant'])
                    ]['taxon']
                )
            
            # Compute overlaps with DA and Boruta (nondirectional methods)
            overlap_results = []
            
            methods = {
                'ANCOM-BC2': ancombc2_sig,
                'ALDEx2': aldex2_sig
            }
            
            # Boruta selected
            boruta_selected = set()
            if boruta_summary is not None:
                boruta_selected = set(boruta_summary[boruta_summary['selection_frequency'] >= 0.5]['taxon'])
                if boruta_selected:
                    methods['Boruta'] = boruta_selected
            
            for method_name, method_taxa in methods.items():
                if len(method_taxa) == 0:
                    continue
                
                metrics = compute_overlap_metrics(
                    fuzzy_taxa_set,
                    method_taxa,
                    all_taxa_universe,
                    args.n_permutations,
                    args.random_seed,
                    logger
                )
                metrics['method'] = method_name
                metrics['overlap_type'] = 'differential_abundance'
                overlap_results.append(metrics)
            
            # Add positive marker overlaps (fair comparison) as main RF comparison
            logger.info("\nComputing RF positive marker overlaps (direction-aware)...")
            positive_marker_overlap_results = []
            
            # Get unique categories from fuzzy taxa
            fuzzy_categories = fuzzy_matched['cluster'].unique() if 'cluster' in fuzzy_matched.columns else [None]
            
            for category in fuzzy_categories:
                if pd.isna(category):
                    continue
                
                # Get fuzzy taxa for this category
                fuzzy_in_cat = set(fuzzy_matched[fuzzy_matched['cluster'] == category]['taxon'])
                if len(fuzzy_in_cat) == 0:
                    continue
                
                # Get RF-positive markers for this category
                cat_str = str(category).replace(' ', '_')
                pos_markers_for_cat = set()
                if cat_str in positive_markers:
                    pos_markers_for_cat = set(positive_markers[cat_str])
                
                if len(pos_markers_for_cat) == 0:
                    logger.warning(f"No RF-positive markers found for category: {category}")
                    continue
                
                # Compute overlap
                metrics = compute_overlap_metrics(
                    fuzzy_in_cat,
                    pos_markers_for_cat,
                    all_taxa_universe,
                    args.n_permutations,
                    args.random_seed,
                    logger
                )
                metrics['method'] = f'RF-positive-markers'
                metrics['category'] = category
                metrics['overlap_type'] = 'direction_aware'
                positive_marker_overlap_results.append(metrics)
            
            # Combine overlap results
            overlap_summary = pd.DataFrame(overlap_results)
            if len(positive_marker_overlap_results) > 0:
                positive_marker_overlap_df = pd.DataFrame(positive_marker_overlap_results)
                overlap_summary = pd.concat([overlap_summary, positive_marker_overlap_df], ignore_index=True)
            
            overlap_summary.to_csv(comparison_dir / "overlap_summary.tsv", sep='\t', index=False)
            
            if len(positive_marker_overlap_results) > 0:
                positive_marker_overlap_df.to_csv(comparison_dir / "positive_marker_overlap_summary.tsv", sep='\t', index=False)
            
            # Compute negative marker complementarity
            logger.info("\nComputing negative marker complementarity...")
            complementarity_results = []
            
            for category in fuzzy_categories:
                if pd.isna(category):
                    continue
                
                # Get fuzzy taxa for this category
                fuzzy_in_cat = set(fuzzy_matched[fuzzy_matched['cluster'] == category]['taxon'])
                if len(fuzzy_in_cat) == 0:
                    continue
                
                # Get RF-negative markers for this category
                cat_str = str(category).replace(' ', '_')
                neg_markers_for_cat = set()
                if cat_str in negative_markers:
                    neg_markers_for_cat = set(negative_markers[cat_str])
                
                if len(neg_markers_for_cat) > 0:
                    # Compute complementarity (how different are fuzzy and negative markers)
                    n_overlap = len(fuzzy_in_cat & neg_markers_for_cat)
                    complementarity = 1.0 - (n_overlap / len(fuzzy_in_cat)) if len(fuzzy_in_cat) > 0 else 0
                    
                    complementarity_results.append({
                        'category': category,
                        'n_fuzzy_taxa': len(fuzzy_in_cat),
                        'n_rf_negative_markers': len(neg_markers_for_cat),
                        'n_overlap_with_negative': n_overlap,
                        'complementarity_score': complementarity,
                        'overlap_taxa': ','.join(fuzzy_in_cat & neg_markers_for_cat)
                    })
            
            if len(complementarity_results) > 0:
                complementarity_df = pd.DataFrame(complementarity_results)
                complementarity_df.to_csv(comparison_dir / "negative_marker_complementarity.tsv", sep='\t', index=False)
            
            # ====================================================================
            # 13. COMPARISONS: RANK CORRELATION
            # ====================================================================
            logger.info("\n" + "="*80)
            logger.info("13. RANK CORRELATION")
            logger.info("="*80)
            
            rank_corr_results = []
            
            # Build ranking dictionaries
            fuzzy_rank_dict = {}
            for _, row in fuzzy_matched.iterrows():
                taxon = row['taxon']
                rank = row.get('fuzzy_rank', row.get('membership', row.get('importance', np.nan)))
                if pd.notna(rank):
                    fuzzy_rank_dict[taxon] = rank
                else:
                    fuzzy_rank_dict[taxon] = 1  # Default rank
            
            if len(da_results_combined) > 0:
                for method in ['ancombc2', 'aldex2']:
                    method_data = da_results_combined[da_results_combined['method'] == method]
                    if len(method_data) == 0:
                        continue
                    
                    method_rank_dict = {}
                    for _, row in method_data.iterrows():
                        method_rank_dict[row['taxon']] = abs(row['effect_size']) if pd.notna(row['effect_size']) else np.inf
                    
                    corr, pval, n_overlap = compute_rank_correlation(fuzzy_rank_dict, method_rank_dict, logger)
                    
                    rank_corr_results.append({
                        'method': method.upper(),
                        'correlation': corr,
                        'p_value': pval,
                        'n_overlapping': n_overlap
                    })
            
            if len(rf_importance) > 0:
                rf_rank_dict = {row['taxon']: row['mean_importance'] for _, row in rf_importance.iterrows()}
                corr, pval, n_overlap = compute_rank_correlation(fuzzy_rank_dict, rf_rank_dict, logger)
                rank_corr_results.append({
                    'method': 'RF',
                    'correlation': corr,
                    'p_value': pval,
                    'n_overlapping': n_overlap
                })
            
            if len(rank_corr_results) > 0:
                rank_corr_df = pd.DataFrame(rank_corr_results)
                rank_corr_df.to_csv(comparison_dir / "rank_correlation_summary.tsv", sep='\t', index=False)
            
            # ====================================================================
            # 14. COMPARISONS: DIRECTION AGREEMENT
            # ====================================================================
            logger.info("\n" + "="*80)
            logger.info("14. DIRECTION AGREEMENT")
            logger.info("="*80)
            
            direction_agreement = pd.DataFrame(columns=DIRECTION_AGREEMENT_COLUMNS)
            if len(da_results_combined) > 0:
                direction_agreement = compute_direction_agreement(
                    X_log1p,
                    y_encoded,
                    da_results_combined,
                    fuzzy_matched,
                    args.alpha,
                    logger
                )
                direction_agreement.to_csv(comparison_dir / "direction_agreement.tsv", sep='\t', index=False)
            
            mark_step_done(output_dir, 12, "Comparisons", logger)

        fuzzy_taxa_set = set(fuzzy_matched['taxon'])
        all_taxa_universe = set(otu_filtered.columns)

        ancombc2_sig = set()
        aldex2_sig = set()
        if len(da_results_combined) > 0:
            ancombc2_sig = set(
                da_results_combined[
                    (da_results_combined['method'] == 'ancombc2') &
                    (da_results_combined['significant'])
                ]['taxon']
            )
            aldex2_sig = set(
                da_results_combined[
                    (da_results_combined['method'] == 'aldex2') &
                    (da_results_combined['significant'])
                ]['taxon']
            )

        rf_top_n = int(len(fuzzy_taxa_set))
        rf_top_taxa = set(rf_importance.head(rf_top_n)['taxon'])
        rf_selected = set(rf_importance[rf_importance['selection_frequency'] >= 0.5]['taxon'])

        boruta_selected = set()
        if boruta_summary is not None:
            boruta_selected = set(boruta_summary[boruta_summary['selection_frequency'] >= 0.5]['taxon'])

        if 'overlap_summary' not in locals():
            overlap_summary = pd.DataFrame()
        if 'rank_corr_df' not in locals():
            rank_corr_df = pd.DataFrame()
        if 'direction_agreement' not in locals():
            direction_agreement = pd.DataFrame(columns=DIRECTION_AGREEMENT_COLUMNS)
        
        # ====================================================================
        # 15. EXTERNAL CONSENSUS
        # ====================================================================
        logger.info("\n" + "="*80)
        logger.info("15. EXTERNAL CONSENSUS")
        logger.info("="*80)
        
        # Taxa supported by at least 2 methods
        consensus_votes = {}
        
        for taxon in all_taxa_universe:
            votes = 0
            if taxon in ancombc2_sig:
                votes += 1
            if taxon in aldex2_sig:
                votes += 1
            if taxon in rf_selected:
                votes += 1
            if taxon in boruta_selected:
                votes += 1
            
            if votes >= 2:
                consensus_votes[taxon] = votes
        
        consensus_taxa = list(consensus_votes.keys())
        logger.info(f"External consensus taxa (>= 2 methods): {len(consensus_taxa)}")
        
        consensus_df = pd.DataFrame({
            'taxon': list(consensus_votes.keys()),
            'n_methods': list(consensus_votes.values())
        })
        consensus_df.to_csv(comparison_dir / "method_consensus_taxa.tsv", sep='\t', index=False)
        
        # ====================================================================
        # 16. PREDICTIVE ABLATION
        # ====================================================================
        logger.info("\n" + "="*80)
        logger.info("16. PREDICTIVE ABLATION")
        logger.info("="*80)
        
        ablation_dir = output_dir / "ablation"
        ablation_long_file = ablation_dir / "ablation_metrics_long.tsv"
        ablation_summary_file = ablation_dir / "ablation_metrics_summary.tsv"
        
        if should_skip_step(output_dir, 16, "Ablation", args.skip_if_done):
            logger.info("SKIPPING: Step 16 already completed")
            if ablation_long_file.exists() and ablation_summary_file.exists():
                ablation_long = pd.read_csv(ablation_long_file, sep='\t')
                ablation_summary = pd.read_csv(ablation_summary_file, sep='\t')
                logger.info(f"Loaded ablation results from {ablation_long_file}")
            else:
                logger.warning("Ablation output files not found, will re-run step 16")
                # Re-run ablation
                covariates_for_ablation = None
                if 'covariates_list' in locals() and covariates_list:
                    # Check if covariates are numeric
                    numeric_covariates = []
                    for cov in covariates_list:
                        if cov in X_log1p.columns or cov in metadata_aligned.columns:
                            try:
                                pd.to_numeric(metadata_aligned[cov])
                                numeric_covariates.append(cov)
                            except:
                                pass
                    if numeric_covariates:
                        covariates_for_ablation = numeric_covariates
                
                ablation_long, ablation_summary = run_ablation_analysis(
                    X_log1p,
                    y_encoded,
                    list(fuzzy_taxa_set),
                    list(rf_top_taxa),
                    consensus_taxa,
                    covariates_for_ablation,
                    args.n_resamples,
                    args.test_size,
                    args.random_seed,
                    args.n_estimators,
                    args.n_jobs,
                    logger,
                    rf_importance,
                    fuzzy_matched
                )
                ablation_dir.mkdir(parents=True, exist_ok=True)
                ablation_long.to_csv(ablation_long_file, sep='\t', index=False)
                ablation_summary.to_csv(ablation_summary_file, sep='\t', index=False)
                mark_step_done(output_dir, 16, "Ablation", logger)
        else:
            covariates_for_ablation = None
            if 'covariates_list' in locals() and covariates_list:
                # Check if covariates are numeric
                numeric_covariates = []
                for cov in covariates_list:
                    if cov in X_log1p.columns or cov in metadata_aligned.columns:
                        try:
                            pd.to_numeric(metadata_aligned[cov])
                            numeric_covariates.append(cov)
                        except:
                            pass
                if numeric_covariates:
                    covariates_for_ablation = numeric_covariates
            
            ablation_long, ablation_summary = run_ablation_analysis(
                X_log1p,
                y_encoded,
                list(fuzzy_taxa_set),
                list(rf_top_taxa),
                consensus_taxa,
                covariates_for_ablation,
                args.n_resamples,
                args.test_size,
                args.random_seed,
                args.n_estimators,
                args.n_jobs,
                logger,
                rf_importance,
                fuzzy_matched
            )
            
            ablation_dir.mkdir(parents=True, exist_ok=True)
            
            ablation_long.to_csv(ablation_long_file, sep='\t', index=False)
            ablation_summary.to_csv(ablation_summary_file, sep='\t', index=False)
            
            mark_step_done(output_dir, 16, "Ablation", logger)
        
        # ====================================================================
        # 17. EVIDENCE SCORING
        # ====================================================================
        logger.info("\n" + "="*80)
        logger.info("17. EVIDENCE SCORING")
        logger.info("="*80)
        
        evidence_score_file = comparison_dir / "fuzzy_taxa_evidence_score.tsv"
        
        if should_skip_step(output_dir, 17, "Evidence Scoring", args.skip_if_done):
            logger.info("SKIPPING: Step 17 already completed")
            need_rerun = False
            if not evidence_score_file.exists():
                logger.warning("Evidence score file not found, will re-run step 17")
                need_rerun = True
            else:
                try:
                    if evidence_score_file.stat().st_size == 0:
                        logger.warning("Evidence score file is empty, will re-run step 17")
                        need_rerun = True
                    else:
                        try:
                            evidence_scores = pd.read_csv(evidence_score_file, sep='\t')
                            logger.info(f"Loaded evidence scores from {evidence_score_file}")
                        except pd.errors.EmptyDataError:
                            logger.warning("Evidence score file contains no data columns, will re-run step 17")
                            need_rerun = True
                except Exception as e:
                    logger.warning(f"Could not read evidence score file metadata: {e}; will re-run step 17")
                    need_rerun = True

            if need_rerun:
                evidence_scores = compute_evidence_scores(
                    fuzzy_matched,
                    da_results_combined,
                    rf_importance,
                    boruta_summary,
                    consensus_taxa,
                    direction_agreement,
                    positive_markers=positive_markers,
                    alpha=args.alpha,
                    logger=logger
                )
                evidence_scores.to_csv(evidence_score_file, sep='\t', index=False)
                mark_step_done(output_dir, 17, "Evidence Scoring", logger, outputs=[evidence_score_file])
        else:
            evidence_scores = compute_evidence_scores(
                fuzzy_matched,
                da_results_combined,
                rf_importance,
                boruta_summary,
                consensus_taxa,
                direction_agreement,
                positive_markers=positive_markers,
                alpha=args.alpha,
                logger=logger
            )
            
            evidence_scores.to_csv(evidence_score_file, sep='\t', index=False)
            
            mark_step_done(output_dir, 17, "Evidence Scoring", logger)
        
        # ====================================================================
        # 18. PLOTTING
        # ====================================================================
        logger.info("\n" + "="*80)
        logger.info("18. GENERATING PLOTS")
        logger.info("="*80)
        
        plots_dir = output_dir / "plots"
        plots_complete_marker = plots_dir / ".plots_complete"
        
        if should_skip_step(output_dir, 18, "Plotting", args.skip_if_done):
            logger.info("SKIPPING: Step 18 already completed")
        else:
            plots_dir.mkdir(parents=True, exist_ok=True)
            
            # DA volcano plots
            if len(da_results_combined) > 0:
                for method in ['ancombc2', 'aldex2']:
                    method_data = da_results_combined[da_results_combined['method'] == method]
                    if len(method_data) > 0:
                        plot_da_volcano(
                            method_data,
                            method.upper(),
                            plots_dir / f"da_volcano_{method}.png",
                            args.alpha,
                            logger
                        )
            
            # RF importance
            plot_rf_importance(
                rf_importance,
                min(20, len(rf_importance)),
                plots_dir / "rf_top_permutation_importance.png",
                logger
            )
            
            # Ablation
            plot_ablation_comparison(
                ablation_long,
                plots_dir / "ablation_performance_boxplot.png",
                logger
            )
            
            mark_step_done(output_dir, 18, "Plotting", logger)
        
        # ====================================================================
        # 19. GENERATE REPORT
        # ====================================================================
        logger.info("\n" + "="*80)
        logger.info("19. GENERATING REPORT")
        logger.info("="*80)
        
        da_summary = {
            'ancombc2_significant': len(ancombc2_sig),
            'aldex2_significant': len(aldex2_sig),
            'fuzzy_ancombc2': len(fuzzy_taxa_set & ancombc2_sig),
            'fuzzy_aldex2': len(fuzzy_taxa_set & aldex2_sig)
        }
        
        generate_report(
            output_dir,
            input_summary,
            sample_matching_report,
            taxa_filtering_report,
            fuzzy_matching_report,
            da_summary,
            rf_metrics,
            rf_importance,
            boruta_summary,
            overlap_summary if len(overlap_summary) > 0 else pd.DataFrame(),
            ablation_summary,
            evidence_scores,
            positive_markers=positive_markers if 'positive_markers' in locals() else None,
            negative_markers=negative_markers if 'negative_markers' in locals() else None,
            logger=logger
        )
        
        # ====================================================================
        # COMPLETION
        # ====================================================================
        logger.info("\n" + "="*80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"End time: {datetime.now()}")
        
        # Write completion marker
        completion_marker = output_dir / "COMPLETION_MARKER.txt"
        with open(completion_marker, 'w') as f:
            f.write(f"Experiment completed at {datetime.now()}\n")
        logger.info(f"Completion marker written: {completion_marker}")
        
        return 0
        
    except Exception as e:
        logger.exception(f"Pipeline failed with error: {e}")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)

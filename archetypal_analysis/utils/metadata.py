"""
Metadata extraction and processing utilities.
"""

import pandas as pd
import numpy as np
from typing import List, Optional

def extract_metadata(index: pd.Index = None,
                     tags: Optional[pd.DataFrame] = None,
                     sample_meta: Optional[pd.DataFrame] = None,
                     projects: Optional[pd.DataFrame] = None,
                     extract_cols: List[str] = [],
                     dataset: str ='compendium',
                     verbose: bool = False) -> pd.DataFrame:
    """
    Extract metadata from a specified dataset (compendium or hadza).
    Args:
        dataset: Dataset to extract metadata from ('compendium' or 'hadza')
        index: Sample indices to extract metadata for
        extract_cols: Columns to extract from the metadata
        tags: DataFrame with tag information (optional)
        sample_meta: DataFrame with sample metadata (optional)
        projects: DataFrame with project information (optional)
        verbose: Print debug information
    Returns:
        DataFrame with extracted metadata for the samples
    """
    if dataset in ['compendium', 'hadza']:
        return extract_metadata_compendium(index, tags, sample_meta, projects, extract_cols, verbose)
    elif dataset == 'hadza':
        # Placeholder for Hadza metadata extraction logic
        raise NotImplementedError("Hadza metadata extraction not implemented yet.")


def extract_metadata_compendium(index: pd.Index, 
                               tags: Optional[pd.DataFrame] = None, 
                               sample_meta: Optional[pd.DataFrame] = None, 
                               projects: Optional[pd.DataFrame] = None, 
                               extract_cols: List[str] = [], 
                               verbose: bool = False) -> pd.DataFrame:
    """
    Extract metadata from compendium datasets (projects, sample metadata, and tags).
    
    This function combines information from different metadata sources based on sample indices
    that follow the pattern: project_srr.
    
    Args:
        index: Sample indices (typically project_srr format)
        tags: DataFrame with tag information
        sample_meta: DataFrame with sample metadata 
        projects: DataFrame with project information
        extract_cols: Columns to extract from the metadata
        verbose: Print debug information
        
    Returns:
        DataFrame with extracted metadata for the samples
    """
    # Ensure target index is an Index object to simplify downstream reindex logic.
    index = pd.Index(index)

    # Categorize columns by source
    project_cols = [col for col in extract_cols if col in projects.columns] if projects is not None else []
    sample_cols = [col for col in extract_cols if col in sample_meta.columns] if sample_meta is not None else []
    tags_cols = [col for col in extract_cols if col not in project_cols and col not in sample_cols] if tags is not None else []

    if verbose:
        print(f"Project columns: {project_cols}")
        print(f"Sample columns: {sample_cols}")
        print(f"Tag columns: {tags_cols}")
    
    # Initialize metadata DataFrame
    metadata = pd.DataFrame(index=index, columns=extract_cols)
    
    # Extract project-level metadata
    for col in project_cols:
        projects_local = projects.copy()
        if not projects_local.index.is_unique:
            projects_local = projects_local[~projects_local.index.duplicated(keep="first")]

        metadata[col] = metadata.index.map(
            lambda x: projects_local.loc[x.split("_")[0], col]
            if isinstance(x, str) and x.split("_")[0] in projects_local.index
            else np.nan
        )
    
    # Prepare sample metadata with combined index
    sample_meta_indexed = None
    if sample_meta is not None:
        sample_meta_indexed = sample_meta.copy()
        if "project" in sample_meta_indexed.columns and "srr" in sample_meta_indexed.columns:
            sample_meta_indexed["sample"] = (
                sample_meta_indexed["project"].astype(str) + "_" + sample_meta_indexed["srr"].astype(str)
            )
            sample_meta_indexed.set_index("sample", inplace=True)

        if not sample_meta_indexed.index.is_unique:
            sample_meta_indexed = sample_meta_indexed[~sample_meta_indexed.index.duplicated(keep="first")]
    
    # Extract sample-level metadata
    for col in sample_cols:
        if sample_meta_indexed is None or col not in sample_meta_indexed.columns:
            metadata[col] = np.nan
            continue
        metadata[col] = sample_meta_indexed[col].reindex(metadata.index)
    
    # Extract tag-based metadata
    for col in tags_cols:
        if tags is None or not set(["project", "srr", "tag", "value"]).issubset(tags.columns):
            metadata[col] = np.nan
            continue

        col_tags_df = tags[tags["tag"] == col].copy()
        if len(col_tags_df) == 0:
            metadata[col] = np.nan
            continue

        col_tags_df["sample"] = col_tags_df["project"].astype(str) + "_" + col_tags_df["srr"].astype(str)
        col_tags_df.set_index("sample", inplace=True)
        if not col_tags_df.index.is_unique:
            col_tags_df = col_tags_df[~col_tags_df.index.duplicated(keep="first")]

        metadata[col] = col_tags_df["value"].reindex(metadata.index)
    
    return metadata

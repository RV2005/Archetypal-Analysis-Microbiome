"""
Metadata extraction and processing utilities.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union

from ..config import DatasetConfig


class MetadataExtractor:
    """Extract and process metadata from compendium"""
    
    def extract_metadata_compendium(self,
                                   index: pd.Index,
                                   tags: pd.DataFrame,
                                   sample_meta: pd.DataFrame,
                                   projects: pd.DataFrame,
                                   extract_cols: List[str] = None,
                                   verbose: bool = False) -> pd.DataFrame:
        """
        Extract metadata from compendium datasets.
        
        Parameters:
        -----------
        index : pd.Index
            Sample indices to extract metadata for
        tags : DataFrame
            Tags data with project, srr, tag, and value columns
        sample_meta : DataFrame
            Sample metadata table
        projects : DataFrame
            Projects metadata table
        extract_cols : List[str]
            Columns to extract from metadata
        verbose : bool
            Print extraction details
            
        Returns:
        --------
        metadata : DataFrame
            Extracted metadata for the specified samples
        """
        if extract_cols is None:
            extract_cols = DatasetConfig.PROJECT_DATA_COLS + DatasetConfig.SAMPLE_METADATA_COLS
        
        # Categorize columns by source
        project_cols = [col for col in extract_cols if col in projects.columns]
        sample_cols = [col for col in extract_cols if col in sample_meta.columns]
        tags_cols = [col for col in extract_cols if col not in project_cols and col not in sample_cols]
        
        if verbose:
            print(f"Project columns: {project_cols}")
            print(f"Sample columns: {sample_cols}")
            print(f"Tags columns: {tags_cols}")
        
        # Initialize metadata DataFrame
        metadata = pd.DataFrame(index=index, columns=extract_cols)
        
        # Extract project-level metadata
        for col in project_cols:
            metadata[col] = metadata.index.map(lambda x: self._get_project_value(x, col, projects))
        
        # Prepare sample metadata if needed
        if sample_cols and not sample_meta.index.name == "sample":
            sample_meta = self._prepare_sample_metadata(sample_meta)
        
        # Extract sample-level metadata
        for col in sample_cols:
            if col in sample_meta.columns:
                metadata[col] = sample_meta.reindex(metadata.index)[col]
        
        # Extract tags-based metadata
        for col in tags_cols:
            if not tags.empty:
                metadata[col] = self._extract_tag_values(metadata.index, col, tags)
        
        return metadata
    
    def _get_project_value(self, sample_id: str, column: str, projects: pd.DataFrame) -> Optional[str]:
        """Extract project-level value for a sample"""
        try:
            project_id = sample_id.split("_")[0]
            if project_id in projects.index:
                return projects.loc[project_id, column]
        except (IndexError, KeyError):
            pass
        return None
    
    def _prepare_sample_metadata(self, sample_meta: pd.DataFrame) -> pd.DataFrame:
        """Prepare sample metadata with proper indexing"""
        if "project" in sample_meta.columns and "srr" in sample_meta.columns:
            sample_meta = sample_meta.copy()
            sample_meta["sample"] = sample_meta["project"] + "_" + sample_meta["srr"]
            sample_meta.set_index("sample", inplace=True)
        return sample_meta
    
    def _extract_tag_values(self, index: pd.Index, tag_name: str, tags: pd.DataFrame) -> pd.Series:
        """Extract values for a specific tag"""
        # Filter tags for the specific tag
        col_tags_df = tags[tags["tag"] == tag_name].copy()
        
        if col_tags_df.empty:
            return pd.Series(index=index, dtype=object)
        
        # Create sample identifiers
        col_tags_df["sample"] = col_tags_df["project"] + "_" + col_tags_df["srr"]
        col_tags_df.set_index("sample", inplace=True)
        
        # Map values to the index
        return col_tags_df["value"].reindex(index)
    
    def filter_metadata(self,
                       metadata: pd.DataFrame,
                       filters: Dict[str, Union[str, List[str]]],
                       filter_type: str = "include") -> pd.DataFrame:
        """
        Filter metadata based on specified criteria.
        
        Parameters:
        -----------
        metadata : DataFrame
            Metadata to filter
        filters : Dict[str, Union[str, List[str]]]
            Dictionary of column: value(s) pairs to filter by
        filter_type : str
            Either "include" or "exclude"
            
        Returns:
        --------
        filtered_metadata : DataFrame
            Filtered metadata
        """
        filtered_metadata = metadata.copy()
        
        for column, values in filters.items():
            if column not in filtered_metadata.columns:
                continue
                
            if isinstance(values, str):
                values = [values]
            
            if filter_type == "include":
                filtered_metadata = filtered_metadata[filtered_metadata[column].isin(values)]
            elif filter_type == "exclude":
                filtered_metadata = filtered_metadata[~filtered_metadata[column].isin(values)]
        
        return filtered_metadata
    
    def get_metadata_summary(self, metadata: pd.DataFrame) -> pd.DataFrame:
        """Get summary statistics for metadata columns"""
        summary = []
        
        for column in metadata.columns:
            col_data = metadata[column]
            
            # Basic stats
            stats = {
                'column': column,
                'total_samples': len(col_data),
                'non_null': col_data.count(),
                'null_count': col_data.isnull().sum(),
                'unique_values': col_data.nunique(),
                'dtype': str(col_data.dtype)
            }
            
            # Add value counts for categorical data
            if col_data.nunique() <= 20:  # Only for categorical columns with few unique values
                value_counts = col_data.value_counts().to_dict()
                stats['value_counts'] = value_counts
            
            summary.append(stats)
        
        return pd.DataFrame(summary)


class MetadataFilters:
    """Common metadata filtering utilities"""
    
    @staticmethod
    def filter_by_age_buckets(metadata: pd.DataFrame,
                             age_column: str = "age_months",
                             buckets: List[tuple] = None) -> pd.DataFrame:
        """Filter and categorize samples by age buckets"""
        if buckets is None:
            buckets = DatasetConfig.AGE_BUCKETS
        
        metadata = metadata.copy()
        age_categories = []
        
        for _, row in metadata.iterrows():
            age = row[age_column]
            if pd.isna(age):
                age_categories.append("Unknown")
                continue
                
            category = "Unknown"
            for min_age, max_age in buckets:
                if min_age <= age < max_age:
                    category = f"{min_age}-{max_age if max_age != float('inf') else '∞'}"
                    break
            age_categories.append(category)
        
        metadata['age_category'] = age_categories
        return metadata
    
    @staticmethod
    def filter_healthy_adults(metadata: pd.DataFrame) -> pd.DataFrame:
        """Filter for healthy adult samples"""
        return metadata[
            (metadata["subjects"].isin(DatasetConfig.SUBJECTS_NO_CONDITION_ADULTS)) &
            (metadata["condition"] == "healthy")
        ]
    
    @staticmethod
    def filter_by_region(metadata: pd.DataFrame, 
                        regions: List[str],
                        exclude_unknown: bool = True) -> pd.DataFrame:
        """Filter samples by geographical region"""
        filtered = metadata[metadata["region"].isin(regions)]
        
        if exclude_unknown:
            filtered = filtered[filtered["region"] != "unknown"]
            
        return filtered

# Constants for metadata extraction
PROJECT_DATA_COLS = ['link', 'amplicon', 'avg_length', 'bead_beating', 'kit', 'sample_type', 'condition', 'subjects', 'notes']
SAMPLE_METADATA_COLS = ['project', 'srr', 'library_strategy', 'library_source', 'pubdate', 'total_bases', 'instrument', 'geo_loc_name', 'iso', 'region']


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
        metadata[col] = metadata.index.map(lambda x: projects.loc[x.split("_")[0], col])
    
    # Prepare sample metadata with combined index
    sample_meta_indexed = sample_meta.copy()
    if "project" in sample_meta.columns and "srr" in sample_meta.columns:    
        sample_meta_indexed["sample"] = sample_meta_indexed["project"] + "_" + sample_meta_indexed["srr"]
        sample_meta_indexed.set_index("sample", inplace=True)
    
    # Extract sample-level metadata
    for col in sample_cols:
        metadata[col] = sample_meta_indexed[col]
    
    # Extract tag-based metadata
    for col in tags_cols:
        col_tags_df = tags[tags["tag"] == col]
        col_tags_df["sample"] = col_tags_df["project"] + "_" + col_tags_df["srr"]
        col_tags_df.set_index("sample", inplace=True)
        metadata[col] = col_tags_df["value"]
    
    return metadata

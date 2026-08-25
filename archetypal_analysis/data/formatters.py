"""
Data formatting utilities for archetypal analysis.
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List, Optional

from ..config import TaxonomicLevels


class HadzaDataFormatter:
    """Format Hadza dataset for archetypal analysis"""
    
    def format_taxonomy(self, 
                       otu: pd.DataFrame, 
                       tax: pd.DataFrame, 
                       group: str = "Genus") -> pd.DataFrame:
        """
        Group OTU data by taxonomic level and create formatted column names.
        
        Parameters:
        -----------
        otu : DataFrame
            Raw OTU table with taxonomic IDs as columns
        tax : DataFrame
            Taxonomy table mapping IDs to taxonomic classifications
        group : str
            Taxonomic level to group by (e.g., "Genus", "Family")
            
        Returns:
        --------
        otu_grouped : DataFrame
            OTU table grouped by specified taxonomic level with formatted names
        """
        # Create taxonomic strings in format "Kingdom.Phylum.Class.Order.Family.Genus"
        taxa_strings = tax.apply(self._create_taxa_string, axis=1)
        col_names = taxa_strings.unique()
        
        # Get unique values for the specified grouping level
        grouped_cols = tax[group].unique()
        
        # Initialize grouped OTU table
        otu_grouped = pd.DataFrame(index=otu.index, columns=col_names)
        
        print(f"Grouping {len(otu.columns)} columns by {group} level...")
        
        # Group OTU data by taxonomic level
        for i, c in tqdm(enumerate(otu.columns), total=len(otu.columns)):
            if c not in tax.index:
                continue
                
            grp_col = tax[group][c]
            if pd.isna(grp_col):
                continue
                
            # Find matching column in the new format
            matching_col = next(
                (col for col in col_names if col.split(".")[-1] == grp_col), 
                None
            )
            
            if matching_col is not None:
                if pd.isnull(otu_grouped[matching_col]).all():
                    otu_grouped[matching_col] = otu[c]
                else:
                    otu_grouped[matching_col] = otu_grouped[matching_col] + otu[c]
        
        # Remove columns that are all NaN
        otu_grouped = otu_grouped.dropna(axis=1, how='all')
        
        return otu_grouped
    
    def _create_taxa_string(self, row: pd.Series) -> str:
        """Create taxonomic string from taxonomy row"""
        return ".".join([str(x) for x in row if str(x) != "NA" and pd.notna(x)])
    
    def fix_sample_indices(self, 
                          otu: pd.DataFrame, 
                          meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fix sample indices to match between OTU and metadata tables.
        
        Parameters:
        -----------
        otu : DataFrame
            OTU table
        meta : DataFrame
            Metadata table with potential SampleID2 column
            
        Returns:
        --------
        otu : DataFrame
            OTU table with consistent indices
        meta : DataFrame
            Metadata table with matching indices
        """
        # Fix metadata indices using SampleID2 where needed
        if 'SampleID2' in meta.columns:
            new_index = meta.index
            new_index = new_index.where(new_index.notnull(), meta["SampleID2"])
            meta.index = new_index
        
        # Keep only samples that exist in both datasets
        common_samples = otu.index.intersection(meta.index)
        otu = otu.loc[common_samples]
        meta = meta.loc[common_samples]
        
        return otu, meta
    
    def save_formatted_data(self, 
                           otu: pd.DataFrame, 
                           meta: pd.DataFrame,
                           otu_filepath: str = "datasets/tsv/hadza_otu_formatted.tsv",
                           meta_filepath: str = "datasets/tsv/hadza_meta_formatted.tsv") -> None:
        """Save formatted Hadza data to files"""
        otu.to_csv(otu_filepath, sep="\t")
        meta.to_csv(meta_filepath, sep="\t")
        print(f"Saved formatted OTU data to {otu_filepath}")
        print(f"Saved formatted metadata to {meta_filepath}")


class CompendiumDataFormatter:
    """Format Compendium dataset for archetypal analysis"""
    
    def prepare_sample_metadata(self, meta: pd.DataFrame) -> pd.DataFrame:
        """Prepare sample metadata with consistent sample column"""
        if "project" in meta.columns and "srr" in meta.columns:
            meta["sample"] = meta["project"] + "_" + meta["srr"]
            meta.set_index("sample", inplace=True)
        return meta
    
    def filter_by_subjects(self, 
                          meta: pd.DataFrame, 
                          subject_categories: List[str]) -> pd.DataFrame:
        """Filter metadata by subject categories"""
        return meta[meta["subjects"].isin(subject_categories)]
    
    def filter_by_conditions(self, 
                           meta: pd.DataFrame, 
                           conditions: List[str]) -> pd.DataFrame:
        """Filter metadata by condition categories"""
        return meta[meta["condition"].isin(conditions)]


class DataTransformer:
    """Common data transformations for archetypal analysis"""
    
    @staticmethod
    def log_transform(otu: pd.DataFrame, base: int = 2, pseudocount: float = 1.0) -> pd.DataFrame:
        """Apply log transformation with pseudocount"""
        if base == 2:
            return np.log2(otu + pseudocount)
        elif base == 10:
            return np.log10(otu + pseudocount)
        else:
            return np.log(otu + pseudocount) / np.log(base)
    
    @staticmethod
    def sample_fraction(otu: pd.DataFrame, frac: float = 1.0) -> pd.DataFrame:
        """Sample a fraction of the data"""
        if frac >= 1.0:
            return otu
        return otu.sample(frac=frac)
    
    @staticmethod
    def remove_infinite_values(data: pd.DataFrame) -> pd.DataFrame:
        """Remove infinite values and replace with NaN"""
        return data.replace([np.inf, -np.inf], np.nan)
    
    @staticmethod
    def filter_low_prevalence_samples(data: pd.DataFrame, min_nonzero: int = 1) -> pd.DataFrame:
        """Filter samples with very few non-zero features"""
        return data[(data > 0).sum(axis=1) >= min_nonzero]

    @staticmethod
    def filter_low_abundance_features(data: pd.DataFrame, min_abundance: float = 0.001) -> pd.DataFrame:
        """Filter features with low abundance across samples"""
        return data.loc[:, (data > min_abundance).sum(axis=0) > 0]
    
    @staticmethod
    def remove_zero_rows(data: pd.DataFrame) -> pd.DataFrame:
        """Remove rows that are entirely zero"""
        return data[(data > 0).sum(axis=1) > 0]
    
    @staticmethod
    def remove_nan(data: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with NaN values"""
        return data.dropna(axis=0, how='any')
    
    @staticmethod
    def general_cleanup(data: pd.DataFrame) -> pd.DataFrame:
        """General cleanup: remove zero rows, NaNs, and infinite values"""
        data = DataTransformer.remove_zero_rows(data)
        data = DataTransformer.remove_infinite_values(data)
        data = DataTransformer.remove_nan(data)
        return data

    @staticmethod
    def standard_scale(data: pd.DataFrame) -> pd.DataFrame:
        """Standard scale the data to have mean 0 and std 1 for each feature"""
        return (data - data.mean()) / data.std(), (data.mean(), data.std())
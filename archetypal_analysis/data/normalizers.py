"""
Data normalization utilities for archetypal analysis.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

from ..utils.metadata import MetadataExtractor
from ..config import DatasetConfig


class DataNormalizer:
    """Main data normalization class with control/normalize parameters"""
    
    def __init__(self, metadata_extractor: Optional[MetadataExtractor] = None):
        self.metadata_extractor = metadata_extractor or MetadataExtractor()
    
    def normalize_data(self, 
                      control: Dict[str, Optional[List]], 
                      normalize: Dict[str, Optional[List]], 
                      otu: pd.DataFrame,
                      meta: pd.DataFrame, 
                      tags: pd.DataFrame, 
                      projects: pd.DataFrame,
                      continuous: bool = False,
                      save_dir: Optional[str] = None,
                      save_name: Optional[str] = None,
                      ignore_zero: bool = True,
                      ignore: List = None,
                      tol: float = 0,
                      verbose: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Normalize the data based on control and normalize parameters.
        
        Parameters:
        -----------
        control : dict
            Dictionary with keys as column names and values as the values to filter by.
        normalize : dict
            Dictionary with keys as column names and values as the values to normalize by.
        otu : DataFrame
            The OTU table to normalize.
        meta : DataFrame
            The metadata for the samples.
        tags : DataFrame
            The tags for the samples.
        projects : DataFrame
            The projects for the samples.
        continuous : bool, optional
            Whether to use continuous normalization with buckets.
        save_dir : str, optional
            The directory to save the normalized data to.
        save_name : str, optional
            Custom name for the saved file.
        ignore_zero : bool, optional
            Whether to ignore samples with zero counts in the normalization process.
        ignore : list, optional
            List of values to ignore in the normalization process.
        tol : float, optional
            Tolerance for the number of samples to normalize to (% of minimum label).
        verbose : bool, optional
            Whether to print verbose output during the normalization process.
        
        Returns:
        --------
        normalized_otu : DataFrame
            The normalized OTU table.
        normalized_meta : DataFrame
            The filtered metadata with the normalized samples.
        """
        if ignore is None:
            ignore = []
            
        columns_of_interest = list(control.keys()) + list(normalize.keys())
        filtered_df = self.metadata_extractor.extract_metadata_compendium(
            meta.index, tags, meta, projects, columns_of_interest
        )
        
        if verbose:
            print(f"Number of samples before control filtering: {len(filtered_df)}")
        
        # Apply control filters
        filtered_df = self._apply_control_filters(filtered_df, control, verbose)
        
        if verbose:
            print(f"Number of samples after control filtering: {len(filtered_df)}")
        
        # Calculate minimum counts for normalization
        min_counts, min_val, ignored = self._calculate_min_counts(
            filtered_df, normalize, continuous, ignore_zero, ignore, verbose
        )
        
        if verbose:
            print(f"Minimum counts: {min_counts} for value {min_val}")
            print(f"Ignored {ignored} values")
        
        # Perform normalization
        normalized_meta = self._normalize_by_categories(
            filtered_df, normalize, continuous, ignore, min_counts, tol
        )
        
        # Filter OTU table
        normalized_idx = normalized_meta.index
        normalized_otu = otu.loc[normalized_idx]
        
        # Save if requested
        if save_dir:
            self._save_normalized_data(normalized_otu, normalize, save_dir, save_name)
        
        if verbose:
            self._print_normalization_summary(normalized_meta, normalize, continuous)
        
        return normalized_otu, normalized_meta
    
    def _apply_control_filters(self, 
                              filtered_df: pd.DataFrame, 
                              control: Dict[str, Optional[List]], 
                              verbose: bool = False) -> pd.DataFrame:
        """Apply control filtering to the dataframe"""
        for label, values in control.items():
            if values is None:
                values = filtered_df[label].unique()
                values = values[~pd.isnull(values)]
            filtered_df = filtered_df[filtered_df[label].isin(values)]
        return filtered_df
    
    def _calculate_min_counts(self,
                             filtered_df: pd.DataFrame,
                             normalize: Dict[str, Optional[List]],
                             continuous: bool,
                             ignore_zero: bool,
                             ignore: List,
                             verbose: bool = False) -> Tuple[int, str, int]:
        """Calculate minimum sample counts for normalization"""
        ignored = 0
        min_counts = np.inf
        min_val = ""
        
        for label, value in normalize.items():
            if continuous:
                assert value is not None, "Continuous normalization requires value buckets to be specified."
            if value is None:
                value = filtered_df[label].unique()
                value = value[~pd.isnull(value)]
                
            for val in value:
                if continuous:
                    values = filtered_df[(filtered_df[label] >= val[0]) & (filtered_df[label] <= val[1])]
                else:
                    values = filtered_df[filtered_df[label] == val]
                    
                if ignore_zero and len(values) == 0:
                    if verbose:
                        print(f"Value {val} not found in {label}. Ignoring")
                    ignored += 1
                    continue
                    
                if val in ignore:
                    ignored += 1
                    if verbose:
                        print(f"Value {val} ignored")
                    continue
                
                if len(values) < min_counts:
                    min_counts = len(values)
                    min_val = val
        
        return int(min_counts), min_val, ignored
    
    def _normalize_by_categories(self,
                                filtered_df: pd.DataFrame,
                                normalize: Dict[str, Optional[List]],
                                continuous: bool,
                                ignore: List,
                                min_counts: int,
                                tol: float) -> pd.DataFrame:
        """Perform the actual normalization by categories"""
        normalized_meta = pd.DataFrame()
        
        for label, value in normalize.items():
            if continuous:
                assert value is not None, "Continuous normalization requires value buckets to be specified."
            if value is None:
                value = filtered_df[label].unique()
                value = value[~pd.isnull(value)]
                
            for val in value:
                if continuous:
                    values = filtered_df[(filtered_df[label] >= val[0]) & (filtered_df[label] <= val[1])]
                else:
                    values = filtered_df[filtered_df[label] == val]
                    
                if len(values) == 0 or val in ignore:
                    continue
                    
                sam_num = int(min(len(values), min_counts * (1 + tol)))
                normalized_meta = pd.concat([normalized_meta, values.sample(sam_num)])
        
        return normalized_meta
    
    def _save_normalized_data(self,
                             normalized_otu: pd.DataFrame,
                             normalize: Dict[str, Optional[List]],
                             save_dir: str,
                             save_name: Optional[str]) -> None:
        """Save the normalized data to file"""
        if save_name is None:
            save_name = "compendium_Norm-" + "_".join(normalize.keys()) + ".csv"
        
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, save_name)
        
        if save_name.endswith(".csv"):
            normalized_otu.to_csv(filepath)
        elif save_name.endswith(".tsv"):
            normalized_otu.to_csv(filepath, sep="\t")
        else:
            raise ValueError("File name must end with .csv or .tsv")
        
        print(f"Saved normalized data to {save_name}")
    
    def _print_normalization_summary(self,
                                   normalized_meta: pd.DataFrame,
                                   normalize: Dict[str, Optional[List]],
                                   continuous: bool) -> None:
        """Print summary of normalization results"""
        counts = {}
        for label, value in normalize.items():
            if value is None:
                value = normalized_meta[label].unique()
                value = value[~pd.isnull(value)]
            counts[label] = {}
            for val in value:
                if continuous:
                    values = normalized_meta[(normalized_meta[label] >= val[0]) & (normalized_meta[label] <= val[1])]
                else:
                    values = normalized_meta[normalized_meta[label] == val]
                counts[label][val] = len(values)
        
        counts_df = pd.DataFrame(counts)
        print(counts_df)
        print()
        print("Normalized Data shape: ", normalized_meta.shape)

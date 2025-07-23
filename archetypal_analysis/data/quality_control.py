"""
Quality control utilities for OTU data.
"""

import os
import pandas as pd
import numpy as np
from typing import Optional

from ..config import TaxonomicLevels


class QualityController:
    """Quality control for OTU data"""
    
    def quality_control(self,
                       otu: pd.DataFrame,
                       save_dir: Optional[str] = None,
                       save_name: Optional[str] = None,
                       verbose: bool = True,
                       min_taxa_level: Optional[int] = None,
                       min_nonzero_samples: Optional[float] = None,
                       min_total_counts: Optional[float] = None,
                       std_filter: Optional[float] = None,
                       min_counts: Optional[int] = None) -> pd.DataFrame:
        """
        Perform quality control on the OTU table.
        
        Parameters:
        -----------
        otu : DataFrame 
            The OTU table with samples as rows and taxa as columns.
        save_dir : str, optional
            The directory to save the filtered OTU table.
        save_name : str, optional
            Custom name for the saved file.
        verbose : bool, optional
            Whether to print the number of samples and taxa removed during filtering.
        min_taxa_level : int, optional
            The minimum taxa level to keep. Default is 3, which means Phylum level.
        min_nonzero_samples : float, optional
            The minimum fraction of non-zero samples required to keep a taxa. Default is 0.05 (5%).
        min_total_counts : float, optional
            The minimum fraction of total counts required to keep a taxa. Default is 0.005 (0.5%).
        std_filter : float, optional
            If provided, will filter out taxa with standard deviation below this value.
        min_counts : int, optional
            Minimum total counts required per sample.
            
        Returns:
        --------
        otu : DataFrame
            The quality-controlled OTU table.
        """
        if verbose:
            print("Starting quality control on OTU table...")
            print(f"Initial OTU table shape: {otu.shape}")
        
        # Apply filters in sequence
        if min_taxa_level:
            otu = self._na_filter(otu, min_taxa_level, verbose)
        if min_nonzero_samples:
            otu = self._min_nonzero_filter(otu, min_nonzero_samples, verbose)
        if min_counts:
            otu = self._filter_sample_sequencing_depth(otu, min_counts, verbose)
        if min_total_counts: # Unused
            otu = self._min_total_counts_filter(otu, min_total_counts, std_filter, verbose)
        
        if verbose:
            print(f"Filtered OTU table shape: {otu.shape}")
        
        # Save if requested
        if save_dir:
            self._save_filtered_data(otu, save_dir, save_name, verbose)
        
        return otu
    
    def _na_filter(self, otu: pd.DataFrame, min_taxa_level: int, verbose: bool = False) -> pd.DataFrame:
        """
        Remove columns for which the minimum taxa level is not known (NA).
        """
        otu_shape = otu.shape
        otu_filtered = otu.copy()
        
        available_taxa = [
            otu_filtered.columns[i] for i in range(otu_filtered.shape[1]) 
            if len(otu_filtered.columns[i].split(".")) > min_taxa_level-1 and
            otu_filtered.columns[i].split(".")[min_taxa_level-1] not in ["NA", "nan", ""]
        ]
        
        otu_filtered = otu_filtered[available_taxa]
        
        if verbose:
            taxa_level_name = TaxonomicLevels.get_level_name(min_taxa_level-1)
            removed_count = otu_shape[1] - otu_filtered.shape[1]
            print(f"Removed {removed_count} columns with unknown {taxa_level_name}")
            
        return otu_filtered
    
    def _min_nonzero_filter(self, otu: pd.DataFrame, min_nonzero_samples: float, verbose: bool = False) -> pd.DataFrame:
        """
        Remove columns that have non-zero values for only < min_nonzero_samples % of samples.
        If a column has all zero values, it will be removed regardless of the min_nonzero_samples
        """
        otu_shape = otu.shape
        min_nonzero_samples_abs = int(min_nonzero_samples * otu.shape[0])
        
        # Filter by minimum non-zero samples
        otu_filtered = otu.loc[:, (otu != 0).sum(axis=0) >= min_nonzero_samples_abs]
        
        # Remove columns with all zero values
        otu_filtered = otu_filtered.loc[:, (otu_filtered != 0).sum(axis=0) > 0]
        
        if verbose:
            removed_count = otu_shape[1] - otu_filtered.shape[1]
            print(f"Removed {removed_count} columns with non-zero values in less than "
                  f"{min_nonzero_samples*100}% samples ({min_nonzero_samples_abs} samples)")
        
        return otu_filtered
    
    def _min_total_counts_filter(self, 
                                otu: pd.DataFrame, 
                                min_total_counts: float, 
                                std_filter: Optional[float] = None, 
                                verbose: bool = False) -> pd.DataFrame:
        """
        Calculate the mean of total counts over all columns, remove columns that have total counts < min_total_counts*mean.
        If std_filter is provided, remove columns that have total counts < mean - std_filter*std.
        """
        otu_shape = otu.shape
        mean_total_counts = otu.sum(axis=0).mean()
        
        if std_filter is not None:
            std_total_counts = otu.sum(axis=0).std()
            threshold = mean_total_counts - std_filter * std_total_counts
            otu_filtered = otu.loc[:, otu.sum(axis=0) >= threshold]
            if verbose:
                removed_count = otu_shape[1] - otu_filtered.shape[1]
                print(f"Removed {removed_count} columns with total counts less than "
                      f"{threshold:.2f} (mean - std_filter * std)")
        else:
            threshold = mean_total_counts * min_total_counts
            otu_filtered = otu.loc[:, otu.sum(axis=0) >= threshold]
            if verbose:
                removed_count = otu_shape[1] - otu_filtered.shape[1]
                print(f"Removed {removed_count} columns with total counts less than "
                      f"{min_total_counts*100}% of mean total counts ({threshold:.2f})")
        
        return otu_filtered
    
    def _filter_sample_sequencing_depth(self, 
                                       otu: pd.DataFrame, 
                                       min_counts: int, 
                                       verbose: bool = False) -> pd.DataFrame:
        """
        Remove samples that have less than min_counts total counts.
        """
        otu_shape = otu.shape
        otu_filtered = otu.loc[otu.sum(axis=1) >= min_counts]
        
        if verbose:
            removed_count = otu_shape[0] - otu_filtered.shape[0]
            print(f"Removed {removed_count} samples with total counts less than {min_counts}")
            
        return otu_filtered
    
    def _save_filtered_data(self, 
                           otu: pd.DataFrame, 
                           save_dir: str, 
                           save_name: Optional[str], 
                           verbose: bool = False) -> None:
        """Save the quality-controlled data"""
        os.makedirs(save_dir, exist_ok=True)
        
        if save_name is None:
            save_name = "compendium_QC.csv"
        
        filepath = os.path.join(save_dir, save_name)
        
        if save_name.endswith(".csv"):
            otu.to_csv(filepath)
        elif save_name.endswith(".tsv"):
            otu.to_csv(filepath, sep="\t")
        else:
            raise ValueError("File name must end with .csv or .tsv")
        
        if verbose:
            print(f"Saved filtered data to {save_name}")

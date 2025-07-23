"""
PCoA (Principal Coordinate Analysis) visualization utilities.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Patch
from typing import List, Optional, Tuple, Dict, Any
import scipy.spatial.distance as distance
from skbio.diversity import beta_diversity
from skbio.stats.ordination import pcoa
from ..data.formatters import DataTransformer

class PCoAVisualizer:
    """PCoA visualization utilities"""
    
    def plot_pcoa(self,
                 pcoa_coordinates: pd.DataFrame,
                 pcoa_results: Any,
                 metadata: pd.DataFrame,
                 filter_col: str,
                 legend_type: str = "discrete",
                 figsize: Tuple[int, int] = (18, 6),
                 alpha: float = 0.5,
                 s: int = 10,
                 pc_combinations: Optional[List[Tuple[str, str]]] = None) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create PCoA visualization with metadata coloring.
        
        Parameters:
        -----------
        pcoa_coordinates : DataFrame
            PCoA coordinates for samples
        pcoa_results : PCoAResults
            Full PCoA results object with proportion explained
        metadata : DataFrame
            Sample metadata
        filter_col : str
            Column name for coloring samples
        legend_type : str
            Type of legend ("discrete" or "continuous")
        figsize : Tuple[int, int]
            Figure size
        alpha : float
            Point transparency
        s : int
            Point size
        pc_combinations : List[Tuple[str, str]], optional
            PC combinations to plot
            
        Returns:
        --------
        fig : Figure
            Matplotlib figure
        axs : ndarray
            Array of axes objects
        """
        if pc_combinations is None:
            pc_combinations = [('PC1', 'PC2'), ('PC2', 'PC3'), ('PC1', 'PC3')]
        
        # Find common samples
        common_samples = pcoa_coordinates.index.intersection(metadata.index)
        
        if len(common_samples) == 0:
            raise ValueError("No common samples found between PCoA coordinates and metadata")
        
        # Extract data for common samples
        plot_coords = pcoa_coordinates.loc[common_samples]
        plot_metadata = metadata.loc[common_samples]
        
        # Prepare colors
        filter_labels = plot_metadata[filter_col]
        color, unique_colors, cmap_name = self._prepare_colors(
            filter_labels, legend_type
        )
        
        # Create subplots
        fig, axs = plt.subplots(1, len(pc_combinations), figsize=figsize)
        if len(pc_combinations) == 1:
            axs = [axs]
        
        # Plot each PC combination
        for i, (x_pc, y_pc) in enumerate(pc_combinations):
            if x_pc not in plot_coords.columns or y_pc not in plot_coords.columns:
                print(f"Warning: {x_pc} or {y_pc} not found in coordinates")
                continue
            
            # Create scatter plot
            scatter = axs[i].scatter(
                plot_coords[x_pc],
                plot_coords[y_pc],
                s=s,
                c=color,
                cmap=cmap_name,
                alpha=alpha
            )
            
            # Set labels with proportion explained
            x_explained = pcoa_results.proportion_explained[int(x_pc[-1])-1]
            y_explained = pcoa_results.proportion_explained[int(y_pc[-1])-1]
            
            axs[i].set_xlabel(f"{x_pc} ({x_explained:.2%} variance explained)")
            axs[i].set_ylabel(f"{y_pc} ({y_explained:.2%} variance explained)")
            axs[i].set_title(f"{x_pc} vs {y_pc}")
            
            # Add legend or colorbar to first plot
            if i == 0:
                self._add_legend_or_colorbar(
                    axs[i], scatter, legend_type, unique_colors, 
                    filter_col, cmap_name, filter_labels
                )
        
        plt.tight_layout()
        return fig, axs
    
    def _prepare_colors(self, 
                       filter_labels: pd.Series,
                       legend_type: str) -> Tuple[pd.Series, Dict, str]:
        """Prepare colors for PCoA plotting"""
        if legend_type == "discrete":
            unique_colors = {
                c: i for i, c in enumerate(sorted(filter_labels.unique(), reverse=True))
            }
            color = filter_labels.map(unique_colors)
            cmap_name = "tab20"
        
        elif legend_type == "continuous":
            unique_colors = {}
            color = pd.to_numeric(filter_labels, errors='coerce')
            cmap_name = "viridis"
        
        else:
            raise ValueError(f"Unknown legend_type: {legend_type}")
        
        return color, unique_colors, cmap_name
    
    def _add_legend_or_colorbar(self,
                               ax: plt.Axes,
                               scatter: Any,
                               legend_type: str,
                               unique_colors: Dict,
                               filter_col: str,
                               cmap_name: str,
                               filter_labels: pd.Series) -> None:
        """Add legend or colorbar to PCoA plot"""
        if legend_type == "continuous":
            cbar = plt.colorbar(
                cm.ScalarMappable(cmap=cmap_name), 
                ax=ax, 
                label=filter_col
            )
        else:
            # Create discrete legend
            cmap = plt.get_cmap(cmap_name, len(unique_colors))
            legend_patches = [
                Patch(color=cmap(unique_colors[c]), label=c)
                for c in unique_colors
            ]
            ax.legend(
                handles=legend_patches, 
                title=filter_col, 
                loc='lower left'
            )
    
    def create_pcoa_from_otu(self,
                           otu_data: pd.DataFrame,
                           distance_metric: str = "jensenshannon",
                           n_dimensions: int = 5,
                           apply_log_transform: bool = True,
                           clean_data: bool = True,
                           verbose: bool = True) -> Tuple[pd.DataFrame, Any]:
        """
        Create PCoA from OTU data.
        
        Parameters:
        -----------
        otu_data : DataFrame
            OTU data with samples as rows
        distance_metric : str
            Distance metric for beta diversity calculation
        n_dimensions : int
            Number of PCoA dimensions to compute
        apply_log_transform : bool
            Whether to apply log2 transformation
        verbose : bool
            Print progress information
            
        Returns:
        --------
        - pcoa_coordinates : DataFrame
            PCoA coordinates for samples
        - pcoa_results : PCoAResults
            Full PCoA results object
        """
        if verbose:
            print(f"Creating PCoA from OTU data...")
            print(f"Initial data shape: {otu_data.shape}")
        
        # Prepare data
        pcoa_data = otu_data.copy()
        
        if apply_log_transform:
            pcoa_data = pcoa_data.map(lambda x: np.log2(x + 1))
        
        if clean_data:  
            pcoa_data = DataTransformer.general_cleanup(pcoa_data)

        if verbose:
            print(f"Processed data shape: {pcoa_data.shape}")
        
        # Calculate beta diversity
        if distance_metric == "jensenshannon":
            distance_func = distance.jensenshannon
        else:
            distance_func = distance_metric
        
        try:
            bdiv = beta_diversity(
                distance_func,
                pcoa_data,
                ids=pcoa_data.index,
                pairwise_func=distance.pdist
            )
            
            if verbose:
                nan_count = np.isnan(bdiv.data).sum()
                inf_count = np.isinf(bdiv.data).sum()
                print(f"Beta diversity computed. NaN: {nan_count}, Inf: {inf_count}")
            
            # Perform PCoA
            pcoa_results = pcoa(bdiv, number_of_dimensions=n_dimensions)
            pcoa_coordinates = pcoa_results.samples
            
            if verbose:
                print(f"PCoA completed. Coordinates shape: {pcoa_coordinates.shape}")
                print(f"Proportion explained by first 3 PCs: {pcoa_results.proportion_explained[:3]}")
            
            return pcoa_coordinates, pcoa_results
            
        except Exception as e:
            print(f"Error in PCoA calculation: {e}")
            raise
    
    

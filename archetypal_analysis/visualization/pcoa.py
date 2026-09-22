"""
PCoA (Principal Coordinate Analysis) visualization utilities.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import re
from matplotlib.patches import Patch
from typing import List, Optional, Tuple, Dict, Any
import scipy.spatial.distance as distance
from skbio.diversity import beta_diversity
from skbio.stats.ordination import pcoa
from ..data.formatters import DataTransformer

class PCoAVisualizer:
    """PCoA visualization utilities"""

    @staticmethod
    def _natural_sort_key(value: Any) -> List[Any]:
        """Sort labels alphabetically with numeric chunks in numeric order."""
        return [int(chunk) if chunk.isdigit() else chunk.lower() for chunk in re.split(r"(\d+)", str(value))]

    @staticmethod
    def _axis_label(component: str) -> str:
        match = re.fullmatch(r"PC(\d+)", str(component))
        if match:
            return f"PiCo {match.group(1)}"
        return str(component)

    @staticmethod
    def _component_index(component: str) -> int:
        match = re.fullmatch(r"PC(\d+)", str(component))
        if not match:
            raise ValueError(f"PCoA component must look like PC1, PC2, etc.; got {component!r}")
        return int(match.group(1)) - 1
    
    def plot_pcoa(self,
                 pcoa_coordinates: pd.DataFrame,
                 pcoa_results: Any,
                 metadata: pd.DataFrame,
                 filter_col: str,
                 legend_type: str = "discrete",
                 figsize: Tuple[int, int] = (18, 6),
                 alpha: float = 0.5,
                 s: int = 10,
                 pc_combinations: Optional[List[Tuple[str, str]]] = None,
                 cmap_name: Optional[str] = None
                 ) -> Tuple[plt.Figure, plt.Axes]:
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
        cmap_name : str, optional
            Name of colormap to use (default: tab20 for discrete, viridis for continuous)
        
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
        color, unique_colors, used_cmap_name = self._prepare_colors(
            filter_labels, legend_type, cmap_name
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
                cmap=used_cmap_name,
                alpha=alpha
            )
            
            # Set labels with proportion explained
            x_explained = pcoa_results.proportion_explained.iloc[self._component_index(x_pc)]
            y_explained = pcoa_results.proportion_explained.iloc[self._component_index(y_pc)]
            x_label = self._axis_label(x_pc)
            y_label = self._axis_label(y_pc)
            
            axs[i].set_xlabel(f"{x_label} ({x_explained:.2%} variance explained)")
            axs[i].set_ylabel(f"{y_label} ({y_explained:.2%} variance explained)")
            axs[i].set_title(f"{x_label} vs {y_label}")
            
        # Add legend or colorbar to last plot
        self._add_legend_or_colorbar(
            axs[i], scatter, legend_type, unique_colors, 
            filter_col, used_cmap_name, filter_labels
        )
    
        plt.tight_layout()
        return fig, axs
    
    def _prepare_colors(self, 
                       filter_labels: pd.Series,
                       legend_type: str,
                       cmap_name: Optional[str] = None) -> Tuple[pd.Series, Dict, str]:
        """Prepare colors for PCoA plotting"""
        if legend_type == "discrete":
            unique_colors = {
                c: i for i, c in enumerate(sorted(filter_labels.unique(), key=self._natural_sort_key))
            }
            color = filter_labels.map(unique_colors)
            used_cmap_name = cmap_name if cmap_name is not None else "tab20"
        
        elif legend_type == "continuous":
            unique_colors = {}
            color = pd.to_numeric(filter_labels, errors='coerce')
            used_cmap_name = cmap_name if cmap_name is not None else "viridis"
        
        else:
            raise ValueError(f"Unknown legend_type: {legend_type}")
        
        return color, unique_colors, used_cmap_name
    
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
            numeric_labels = pd.to_numeric(filter_labels, errors='coerce')
            finite_labels = numeric_labels[np.isfinite(numeric_labels)]
            if finite_labels.empty:
                return

            min_val = float(finite_labels.min())
            max_val = float(finite_labels.max())
            scatter.set_clim(vmin=min_val, vmax=max_val)

            cbar = plt.colorbar(
                scatter,
                ax=ax
            )
            tick_min = int(np.rint(min_val))
            tick_max = int(np.rint(max_val))
            if tick_min > tick_max:
                tick_min, tick_max = tick_max, tick_min
            if tick_min == tick_max:
                tick_vals = np.array([tick_min], dtype=int)
            else:
                n_intervals = min(6, tick_max - tick_min)
                raw_ticks = np.linspace(tick_min, tick_max, n_intervals + 1)
                tick_vals = np.unique(np.rint(raw_ticks).astype(int))
                if tick_vals[0] != tick_min:
                    tick_vals = np.insert(tick_vals, 0, tick_min)
                if tick_vals[-1] != tick_max:
                    tick_vals = np.append(tick_vals, tick_max)

            cbar.set_ticks(tick_vals)
            cbar.set_ticklabels([f"{int(v)}" for v in tick_vals])
            cbar.set_label("Age in months")
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
                loc='lower left',
                frameon=True,
                facecolor='white',
                framealpha=0.85
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
            pcoa_data = np.log2(pcoa_data + 1)
        
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

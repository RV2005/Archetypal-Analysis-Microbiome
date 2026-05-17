"""
Simplex plotting utilities for archetypal analysis visualization.
"""

from unittest import loader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from matplotlib.patches import Patch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from typing import Optional, List, Dict, Any, Union, Tuple
import os
from tqdm import tqdm
import json

from archetypes.visualization import simplex

from archetypal_analysis.data.loaders import ArchetypalResultLoader
from ..config import AnalysisConfig
from ..utils.metadata import extract_metadata_compendium


class SimplexPlotter:
    """Simplex plotting utilities"""

    @staticmethod
    def _natural_sort_key(value: Any) -> List[Any]:
        """Sort labels alphabetically with numeric chunks in numeric order."""
        return [int(chunk) if chunk.isdigit() else chunk.lower() for chunk in re.split(r"(\d+)", str(value))]

    def plot_simplex(
        self,
        otu_aa: pd.DataFrame,
        metadata: pd.DataFrame,
                    filter_cols: List[str],
                    cmap_name: str = "tab20",
                    legend_type: str = "discrete",
                    remove_nan: bool = False,
                    further_filter: Optional[Dict[str, Any]] = None,
                    figsize: Tuple[int, int] = (15, 15),
                    remove_values: Dict[str, str] = None,
                    alpha: float = 0.5,
                    s: int = 5,
                    ax: Optional[plt.Axes] = None,
                    legend_fontsize: int = 12,
                    verbose: bool = False,
                    custom_colors: Optional[Dict[str, Union[int, float]]] = None,
                    make_arrows=True, 
                    **kwargs) -> plt.Axes:
        """
        Plot simplex visualization with metadata coloring.
        
        Parameters:
        -----------
        otu_aa : DataFrame
            The archetypal analysis similarity degree data
        metadata : DataFrame
            The metadata
        filter_cols : list
            The columns to filter the data by (must contain exact column names from metadata)
        cmap_name : str, optional
            The name of the colormap to use
        legend_type : str, optional
            The type of legend to use. Can be "discrete" or "continuous"
        remove_nan : bool, optional
            Whether to remove NaN values from the data
        further_filter : Dict, optional
            Further filter the data by the values in this dict
        figsize : tuple, optional
            The size of the figure
        remove_values : dict, optional
            Dictionary of column: value pairs to remove
        alpha : float, optional
            Transparency of points
        s : int, optional
            Size of points
        ax : Axes, optional
            Matplotlib axes to plot on
        legend_fontsize : int, optional
            Font size for legend
        verbose : bool, optional
            Print diagnostic information
        custom_colors : dict, optional
            A dictionary of custom colors to use for the legend
        **kwargs
            Additional arguments passed to simplex plotting function
            
        Returns:
        --------
        ax : Axes
            Matplotlib axes object
        """
        if remove_values is None:
            remove_values = {}
        
        # Find samples present in both datasets
        present_idx = otu_aa.index.intersection(metadata.index)
        present_points = otu_aa.loc[present_idx]
        
        if verbose:
            print(f"Dropped {len(otu_aa) - len(present_points)} data points: Index not found in metadata")
        
        # Extract filter labels
        filter_labels = self._extract_filter_labels(
            present_points.index, metadata, filter_cols
        )
        
        # Apply further filtering
        if further_filter is not None:
            present_points, filter_labels = self._apply_further_filter(
                present_points, filter_labels, further_filter
            )
        
        # Remove NaN values
        if remove_nan:
            present_points, filter_labels = self._remove_nan_values(
                present_points, filter_labels
            )
        
        # Remove specified values
        if remove_values:
            present_points, filter_labels = self._remove_specified_values(
                present_points, filter_labels, remove_values
            )
        
        # Prepare colors
        color, unique_colors, cmap = self._prepare_colors(
            filter_labels, legend_type, cmap_name, custom_colors
        )
        
        # Create plot
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Plot simplex
        # Ensure simplex direction arrows are not rendered.
        kwargs.pop("show_direction", None)
        simplex(
            present_points,
            alpha=alpha,
            ax=ax,
            c=color,
            cmap=cmap,
            s=s,
            show_direction=False,
            vertices_labels=otu_aa.columns,
            make_arrows=make_arrows,
            **kwargs
        )
        
        # Add legend
        self._add_legend(ax, unique_colors, filter_labels, legend_type, 
                        cmap_name, filter_cols, legend_fontsize)
        
        return ax
    
    def _extract_filter_labels(self, 
                              index: pd.Index,
                              metadata: pd.DataFrame,
                              filter_cols: List[str]) -> pd.Series:
        """Extract filter labels from metadata"""
        filter_labels = pd.Series(index=index, dtype=str)
        
        for col in filter_cols:
            if col in metadata.columns:
                filter_labels = filter_labels.fillna(metadata.loc[index, col])
        
        return filter_labels
    
    def _apply_further_filter(self, 
                             points: pd.DataFrame,
                             labels: pd.Series,
                             further_filter: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply additional filtering based on further_filter dict"""
        filter_type = further_filter.get("filter_type", "equals")
        filter_value = further_filter.get("value", [])
        
        if filter_type == "equals":
            mask = labels.isin(filter_value)
        elif filter_type == "not_equals":
            mask = ~labels.isin(filter_value)
        elif filter_type == "greater_than":
            mask = pd.to_numeric(labels, errors='coerce') > filter_value
        elif filter_type == "less_than":
            mask = pd.to_numeric(labels, errors='coerce') < filter_value
        else:
            mask = pd.Series([True] * len(labels), index=labels.index)
        
        return points[mask], labels[mask]
    
    def _remove_nan_values(self, 
                          points: pd.DataFrame,
                          labels: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Remove NaN values from data"""
        nan_idx = labels.index[labels.isnull()]
        points = points.drop(nan_idx)
        labels = labels.drop(nan_idx)
        return points, labels
    
    def _remove_specified_values(self, 
                                points: pd.DataFrame,
                                labels: pd.Series,
                                remove_values: Dict[str, str]) -> Tuple[pd.DataFrame, pd.Series]:
        """Remove specified values from data"""
        for col, value in remove_values.items():
            rem_idx = labels.index[labels == value]
            points = points.drop(rem_idx)
            labels = labels.drop(rem_idx)
        return points, labels
    
    def _prepare_colors(self, 
                       filter_labels: pd.Series,
                       legend_type: str,
                       cmap_name: str,
                       custom_colors: Optional[Dict]) -> Tuple[pd.Series, Dict, Any]:
        """Prepare colors for plotting"""
        if legend_type == "discrete":
            if custom_colors is not None:
                unique_colors = {
                    c: custom_colors[c] if c in custom_colors else i 
                    for i, c in enumerate(filter_labels.unique())
                }
            else:
                unique_colors = {
                    c: i for i, c in enumerate(sorted(filter_labels.unique(), key=self._natural_sort_key))
                }
            color = filter_labels.map(unique_colors)
            cmap = plt.get_cmap(cmap_name, len(unique_colors))
        
        elif legend_type == "continuous":
            unique_colors = filter_labels
            color = pd.to_numeric(filter_labels, errors='coerce')
            cmap = plt.get_cmap(cmap_name)
        
        else:
            raise ValueError(f"Unknown legend_type: {legend_type}")
        
        return color, unique_colors, cmap
    
    def _add_legend(self, 
                   ax: plt.Axes,
                   unique_colors: Union[Dict, pd.Series],
                   filter_labels: pd.Series,
                   legend_type: str,
                   cmap_name: str,
                   filter_cols: List[str],
                   legend_fontsize: int) -> None:
        """Add legend to the plot"""
        if legend_type == "discrete":
            cmap = plt.get_cmap(cmap_name, len(unique_colors))
            legend_patches = [
                Patch(color=cmap(unique_colors[c]), label=c)
                for c in unique_colors
            ]
            ax.legend(
                handles=legend_patches, 
                title=", ".join(filter_cols), 
                loc='upper right', 
                fontsize=legend_fontsize, 
                bbox_to_anchor=(1.2, 1)
            )
        
        elif legend_type == "continuous":
            numeric_labels = pd.to_numeric(filter_labels, errors='coerce')
            finite_labels = numeric_labels[np.isfinite(numeric_labels)]
            if finite_labels.empty:
                return

            min_val = float(finite_labels.min())
            max_val = float(finite_labels.max())
            if min_val == max_val:
                max_val = min_val + 1e-9

            cbar = plt.colorbar(
                ScalarMappable(
                    cmap=plt.get_cmap(cmap_name),
                    norm=Normalize(vmin=min_val, vmax=max_val)
                ), 
                ax=ax, 
                orientation='vertical'
            )
            cbar.set_label(f"{', '.join(filter_cols)}: min: {min_val}; max: {max_val}")
    
    def plot_group_simplex(self, 
                          directory: str,
                          k_values: List[int],
                          filter_cols: List[str],
                          projects: pd.DataFrame,
                          meta: pd.DataFrame,
                          tags: pd.DataFrame,
                          legend_type: str = "discrete",
                          swapped_cols: Optional[Dict[int, List[str]]] = None,
                          remove_values: Dict[str, str] = None,
                          cmap_name: str = "tab10",
                          alpha: float = 0.5,
                          s: int = 15,
                          plot_size: Tuple[int, int] = (8, 8),
                          legend_fontsize: int = 10) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create subplot grid for multiple K values.
        
        Parameters:
        -----------
        directory : str
            Directory containing result JSON files
        k_values : List[int]
            List of K values to plot
        filter_cols : List[str]
            Columns to use for filtering/coloring
        legend_type : str
            Type of legend ("discrete" or "continuous")
        swapped_cols : Dict[int, List[str]], optional
            Dictionary mapping K values to reordered column names
        projects : DataFrame, optional
            Projects metadata
        meta : DataFrame, optional
            Sample metadata
        tags : DataFrame, optional
            Tags data
        remove_values : Dict[str, str], optional
            Values to remove from visualization
        cmap_name : str
            Colormap name
        alpha : float
            Point transparency
        s : int
            Point size
        plot_size : Tuple[int, int]
            Size of individual subplots
        legend_fontsize : int
            Legend font size
            
        Returns:
        --------
        fig : Figure
            Matplotlib figure
        ax : ndarray
            Array of axes objects
        """
        if remove_values is None:
            remove_values = {}
        
        # Calculate subplot layout
        num_plots = len(k_values)
        num_cols = 2
        num_rows = int(np.ceil(num_plots / num_cols))
        
        fig_size = (plot_size[0] * num_cols, (plot_size[1] * 0.8) * num_rows)
        
        # Create figure
        fig, ax = plt.subplots(num_rows, num_cols, figsize=fig_size, tight_layout=True)
        
        # Handle single subplot case
        if num_plots == 1:
            ax = np.array([ax])
        elif num_rows == 1:
            ax = ax.reshape(1, -1)
        
        for i, k in tqdm(enumerate(k_values), total=num_plots):
            # Load result
            filename = f"otu_AA_K{k}_1.0_None_jensenshannon.json"
            filepath = os.path.join(directory, filename)
            
            try:
                loader = ArchetypalResultLoader()
                otu_aa, _, _, _ = loader.load_parsed_result(filepath)

            except FileNotFoundError:
                print(f"Warning: File not found: {filepath}")
                continue
            
            # Apply column swapping if specified
            if swapped_cols and k in swapped_cols:
                otu_aa = otu_aa[swapped_cols[k]]
            
            # Extract metadata
            if projects is not None and meta is not None and tags is not None:
                metadata = extract_metadata_compendium(
                    otu_aa.index, tags, meta, projects, filter_cols
                )
            else:
                metadata = pd.DataFrame(index=otu_aa.index)
            
            # Get subplot
            if num_plots > 1:
                axis = ax[i // num_cols, i % num_cols]
            else:
                axis = ax[0]
            # Plot simplex
            self.plot_simplex(
                otu_aa,
                metadata,
                filter_cols,
                cmap_name=cmap_name,
                legend_type=legend_type,
                remove_nan=True,
                alpha=alpha,
                s=s,
                remove_values=remove_values,
                figsize=plot_size,
                ax=axis,
                legend_fontsize=legend_fontsize
            )
            
            # Set title and border
            axis.set_title(f"K = {k}")
            xlim, ylim = axis.get_xlim(), axis.get_ylim()
            rect = plt.Rectangle(
                (xlim[0], ylim[0]),
                xlim[1] - xlim[0],
                ylim[1] - ylim[0],
                linewidth=2,
                edgecolor='black',
                facecolor='none'
            )
            axis.add_patch(rect)
        
        # Hide unused subplots
        for j in range(len(k_values), num_rows * num_cols):
            if num_plots > 1:
                fig.delaxes(ax[j // num_cols, j % num_cols])
        
        fig.tight_layout()
        return fig, ax


def filter_data_by_criteria(data: pd.Series, 
                           criteria_value: Any, 
                           filter_type: str = "equals") -> pd.Series:
    """
    Apply filtering criteria to data series.
    
    Args:
        data: Data series to filter
        criteria_value: Value(s) to filter by
        filter_type: Type of filter to apply
        
    Returns:
        Filtered data series
    """
    if filter_type == "equals":
        return data[data.isin(criteria_value if isinstance(criteria_value, list) else [criteria_value])]
    elif filter_type == "not_equals":
        return data[~data.isin(criteria_value if isinstance(criteria_value, list) else [criteria_value])]
    elif filter_type == "greater_than":
        return data[data > criteria_value]
    elif filter_type == "less_than":
        return data[data < criteria_value]
    else:
        return data
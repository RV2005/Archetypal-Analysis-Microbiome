"""
Stacked bar chart visualization for archetypal analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import colorsys
from matplotlib.patches import Patch
from typing import List, Optional, Dict, Tuple, Any

from archetypes.datasets import permute_dataset
from archetypes.visualization import stacked_bar

from ..analysis.archetypal import ArchetypalAnalysisResult
from ..config import DatasetConfig


class StackedBarPlotter:
    """Enhanced stacked bar plotting with metadata integration"""
    
    def stacked_bar(self,
                   points: np.ndarray,
                   perms: Optional[List[int]] = None,
                   ax: Optional[plt.Axes] = None,
                   labels: Optional[np.ndarray] = None,
                   vertices_labels: Optional[List[str]] = None,
                   cmap_name: str = "tab20",
                   pad: float = 0.03,
                   figsize: Optional[Tuple[int, int]] = None,
                   discretize: bool = False,
                   buckets: Optional[List[Tuple[float, float]]] = None,
                   alpha: float = 0.8,
                   **kwargs) -> plt.Axes:
        """
        Enhanced stacked bar method with metadata integration.
        
        Parameters:
        -----------
        points : np.ndarray
            Similarity degree matrix (samples x archetypes)
        perms : List[int], optional
            Permutation indices for ordering samples
        ax : Axes, optional
            Matplotlib axes to plot on
        labels : np.ndarray, optional
            Sample labels for grouping
        vertices_labels : List[str], optional
            Labels for archetypes
        cmap_name : str
            Name of colormap to use
        pad : float
            Padding between groups
        figsize : Tuple[int, int], optional
            Figure size
        discretize : bool
            Whether to discretize continuous labels into buckets
        buckets : List[Tuple[float, float]], optional
            Bucket ranges for discretization
        alpha : float
            Transparency of bars
        **kwargs
            Additional arguments passed to bar plotting
            
        Returns:
        --------
        ax : Axes
            Matplotlib axes object
        """
        if not ax:
            fig, ax = plt.subplots(figsize=figsize)
        
        points = np.asarray(points)
        
        # Apply permutation if provided
        if perms is not None:
            perms_array = np.array([perms])
            points, _ = permute_dataset(points, perms_array)
            if labels is not None:
                labels, _ = permute_dataset(labels, perms_array)
                labels = labels.ravel()
        
        m, n = points.shape
        
        # Set default labels
        if vertices_labels is None:
            vertices_labels = [f"A{i}" for i in range(n)]
        
        if labels is None:
            show_labels = False
            labels = np.arange(m)
        else:
            show_labels = True
        
        if show_labels:
            # Process labels for grouping
            if discretize and buckets is not None:
                processed_labels, label_groups = self._discretize_and_group(
                    labels, points, buckets
                )
            else:
                processed_labels, label_groups = self._group_by_labels(
                    labels, points
                )
            
            # Plot grouped bars
            self._plot_grouped_bars(
                ax, label_groups, processed_labels, vertices_labels,
                cmap_name, pad, alpha, **kwargs
            )
            
        else:
            # Plot simple bars without grouping
            self._plot_simple_bars(
                ax, points, labels, vertices_labels, **kwargs
            )
        
        # Set axis properties
        self._set_axis_properties(ax, show_labels)
        
        return ax
    
    def _discretize_and_group(self,
                             labels: np.ndarray,
                             points: np.ndarray,
                             buckets: List[Tuple[float, float]]) -> Tuple[List[str], Dict]:
        """Discretize continuous labels and group points"""
        label_groups = {}
        processed_labels = []
        
        for bucket_min, bucket_max in buckets:
            if bucket_max == float('inf'):
                bucket_label = f"{bucket_min}+"
            else:
                bucket_label = f"{bucket_min}-{bucket_max}"
            
            # Find points in this bucket
            if bucket_max == float('inf'):
                mask = labels >= bucket_min
            else:
                mask = (labels >= bucket_min) & (labels < bucket_max)
            
            bucket_points = points[mask]
            
            if len(bucket_points) > 0:
                label_groups[bucket_label] = bucket_points
                processed_labels.append(bucket_label)
        
        return processed_labels, label_groups
    
    def _group_by_labels(self,
                        labels: np.ndarray,
                        points: np.ndarray) -> Tuple[List, Dict]:
        """Group points by unique labels"""
        unique_labels, idx = np.unique(labels, return_inverse=True)
        label_groups = {}
        
        for i, label in enumerate(unique_labels):
            mask = idx == i
            label_groups[label] = points[mask]
        
        return unique_labels.tolist(), label_groups
    
    def _plot_grouped_bars(self,
                          ax: plt.Axes,
                          label_groups: Dict,
                          processed_labels: List,
                          vertices_labels: List[str],
                          cmap_name: str,
                          pad: float,
                          alpha: float,
                          **kwargs) -> None:
        """Plot grouped stacked bars"""
        # Calculate group positions
        group_sizes = [len(label_groups[label]) for label in processed_labels]
        x_centers = np.cumsum(group_sizes) - np.array(group_sizes) // 2
        group_padding = pad * sum(group_sizes)
        
        # Create colormap
        cmap = plt.get_cmap(cmap_name, len(vertices_labels))
        colors = [cmap(i) for i in range(len(vertices_labels))]
        
        # Plot bars for each group
        current_x = 0
        for i, label in enumerate(processed_labels):
            grp_points = label_groups[label]
            n_points = len(grp_points)
            
            if n_points == 0:
                continue
            
            bar_x = np.arange(n_points) + current_x
            
            # Plot stacked bars
            bottom = np.zeros(n_points)
            for j, vertex_label in enumerate(vertices_labels):
                ax.bar(
                    bar_x,
                    grp_points[:, j],
                    color=colors[j],
                    bottom=bottom,
                    alpha=alpha,
                    **kwargs
                )
                bottom += grp_points[:, j]
            
            # Add group separator
            if i < len(processed_labels) - 1:
                sep_x = current_x + n_points
                ax.axvline(x=sep_x - 0.5, color='gray', linestyle='--', alpha=0.5)
            
            # Add group label
            group_center = current_x + n_points // 2
            ax.text(
                group_center, -0.05, label,
                ha="center", va="top",
                fontsize=10, rotation=45
            )
            
            current_x += n_points + int(group_padding)
        
        # Create legend
        legend_patches = [
            Patch(color=colors[i], label=vertices_labels[i])
            for i in range(len(vertices_labels))
        ]
        ax.legend(handles=legend_patches, loc="upper left", bbox_to_anchor=(1, 1))
    
    def _plot_simple_bars(self,
                         ax: plt.Axes,
                         points: np.ndarray,
                         labels: np.ndarray,
                         vertices_labels: List[str],
                         **kwargs) -> None:
        """Plot simple stacked bars without grouping"""
        bottom = np.zeros(len(points))
        
        for j, vertex_label in enumerate(vertices_labels):
            ax.bar(
                labels,
                points[:, j],
                label=vertex_label,
                bottom=bottom,
                **kwargs
            )
            bottom += points[:, j]
        
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    
    def _set_axis_properties(self, ax: plt.Axes, show_labels: bool) -> None:
        """Set axis properties and styling"""
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(True)
        ax.spines["bottom"].set_visible(True)
        
        ax.set_ylabel("Similarity degree")
        
        if not show_labels:
            ax.set_xlabel("Observations")
            ax.set_xticks([])
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("")
            ax.set_xticks([])
            ax.set_xticklabels([])
        
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.axis("on")
        ax.autoscale()
    
    def plot_metadata_stacked_bars(self,
                                  similarity_degree: pd.DataFrame,
                                  metadata: pd.DataFrame,
                                  metadata_column: Optional[str] = None,
                                  perms: Optional[List[int]] = None,
                                  discretize: bool = False,
                                  buckets: Optional[List[Tuple[float, float]]] = None,
                                  figsize: Tuple[int, int] = (15, 5),
                                  **kwargs) -> plt.Axes:
        """
        Plot stacked bars with metadata-based grouping.
        
        Parameters:
        -----------
        similarity_degree : DataFrame
            Similarity degree matrix from archetypal analysis
        metadata : DataFrame
            Sample metadata
        metadata_column : str
            Column name to use for grouping/labeling, if None no grouping is done
        perms : List[int], optional
            Permutation indices
        discretize : bool
            Whether to discretize continuous metadata
        buckets : List[Tuple[float, float]], optional
            Bucket ranges for discretization
        figsize : Tuple[int, int]
            Figure size
        **kwargs
            Additional arguments passed to stacked_bar
            
        Returns:
        --------
        ax : Axes
            Matplotlib axes object
        """
        # Get common samples
        common_samples = similarity_degree.index.intersection(metadata.index)
        
        if len(common_samples) == 0:
            raise ValueError("No common samples found between similarity matrix and metadata")
        
        # Extract data for common samples
        plot_similarity = similarity_degree.loc[common_samples]
        plot_metadata = metadata.loc[common_samples]
        
        # Get labels
        labels = plot_metadata[metadata_column].values if metadata_column else None
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        return self.stacked_bar(
            points=plot_similarity.to_numpy(),
            perms=perms,
            ax=ax,
            labels=labels,
            vertices_labels=list(plot_similarity.columns),
            discretize=discretize,
            buckets=buckets,
            **kwargs
        )


class MetadataStackedBarHelper:
    """Helper class for metadata-based stacked bar visualizations"""
    
    @staticmethod
    def prepare_age_buckets(age_data: pd.Series,
                           custom_buckets: Optional[List[Tuple[float, float]]] = None) -> List[Tuple[float, float]]:
        """Prepare age buckets for discretization"""
        if custom_buckets is not None:
            return custom_buckets
        
        # Use default age buckets from config
        return DatasetConfig.AGE_BUCKETS
    
    @staticmethod
    def prepare_region_labels(region_data: pd.Series,
                             exclude_unknown: bool = True) -> pd.Series:
        """Prepare region labels for visualization"""
        if exclude_unknown:
            return region_data[region_data != "unknown"]
        return region_data
    
    @staticmethod
    def get_optimal_figure_size(n_samples: int,
                               n_groups: int,
                               base_width: int = 15,
                               base_height: int = 5) -> Tuple[int, int]:
        """Calculate optimal figure size based on data dimensions"""
        # Adjust width based on number of samples
        width_factor = max(1.0, n_samples / 100)  # Scale up for many samples
        width = int(base_width * width_factor)
        
        # Adjust height slightly based on number of groups
        height_factor = max(1.0, n_groups / 10)
        height = int(base_height * height_factor)
        
        return (min(width, 25), min(height, 12))  # Cap maximum size


def simplify_taxa_name(taxa_name: str) -> str:
    """Extract the most informative part of the taxonomic name"""
    if taxa_name == "Others":
        return "Others"
    
    # Split by dots and get the last meaningful part
    parts = taxa_name.split('.')
    parts = [part for part in parts if part and part not in ["NA", "unknown", "unclassified", "."]]
    taxa_name = parts[-1] if parts else taxa_name
    
    return taxa_name[:20] + "..." if len(taxa_name) > 20 else taxa_name


def generate_distinct_colors(n_colors: int) -> List[Tuple[float, float, float]]:
    """Generate n distinct colors using a combination of colormaps and HSV space"""
    if n_colors <= 12:
        # For small numbers, use qualitative colormap
        return plt.cm.Set3(np.linspace(0, 1, n_colors))
    elif n_colors <= 20:
        # For medium numbers, use tab20
        return plt.cm.tab20(np.linspace(0, 1, n_colors))
    else:
        # For large numbers, generate colors in HSV space for maximum distinction
        colors = []
        for i in range(n_colors):
            hue = i / n_colors
            saturation = 0.7 + (i % 3) * 0.1  # Vary saturation slightly
            value = 0.7 + (i % 2) * 0.2       # Vary brightness slightly
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            colors.append(rgb)
        return colors


def plot_archetype_composition_bars(archetypes: np.ndarray,
                                   otu_columns: List[str],
                                   num_taxa: int = 5,
                                   vertical: bool = False,
                                   figsize: Tuple[int, int] = (15, 8)) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot separate bar charts showing top taxa composition for each archetype.
    
    Args:
        archetypes: Array of archetype compositions
        otu_columns: Column names (taxa) from original OTU data
        num_taxa: Number of top taxa to show per archetype
        vertical: Whether to arrange plots vertically
        figsize: Figure size
        
    Returns:
        Figure and axes objects
    """
    n_archs = len(archetypes)
    
    # Create archetype DataFrame
    archs = pd.DataFrame(
        archetypes, 
        columns=otu_columns, 
        index=[f"Archetype {i+1}" for i in range(n_archs)]
    )
    
    # Extract top taxa for each archetype
    top_taxa_archs = []
    for i in range(archs.shape[0]):
        # Sort taxa by contribution
        arch_sorted = archs.iloc[i].sort_values(ascending=False)
        top_taxa = arch_sorted.iloc[:num_taxa]
        
        # Calculate percentages
        total = archs.loc[f"Archetype {i+1}"].sum()
        top_taxa_list = [(val/total, taxa) for taxa, val in top_taxa.items()]
        
        # Add "Others" category
        others_percentage = 1 - sum([val for val, taxa in top_taxa_list])
        top_taxa_list.append((others_percentage, "Others"))
        
        top_taxa_archs.append(top_taxa_list)
    
    # Find max value for consistent y-axis
    max_val = max([max([val for val, taxa in arch][:-1]) for arch in top_taxa_archs])
    
    # Set up subplot layout
    if vertical: 
        fig, ax = plt.subplots((n_archs+1) // 2, 2, figsize=(8, 15))
    else: 
        fig, ax = plt.subplots(2, (n_archs+1) // 2, figsize=figsize)
    
    # Plot each archetype
    for i, arch in enumerate(top_taxa_archs):
        vals = [val for val, taxa in arch]
        labels = [taxa for val, taxa in arch]
        
        # Simplify labels
        plot_labels = []
        for k, label in enumerate(labels[:-1]):
            plot_labels.append(simplify_taxa_name(label))
        plot_labels += ["Others"]
        
        # Determine subplot position
        x, y = (i//2, i % 2) if vertical else (i % 2, i // 2)
        
        # Plot bars (excluding "Others")
        ax[x, y].bar(plot_labels[:-1], vals[:-1])
        ax[x, y].set_title(f"Archetype {i+1}")
        ax[x, y].set_ylim(0, max_val)
        ax[x, y].set_xticklabels(ax[x, y].get_xticklabels(), rotation=30, ha='right')
    
    fig.tight_layout()
    return fig, ax


def plot_stacked_bar_composition(archetypes: np.ndarray,
                               otu_columns: List[str],
                               num_taxa: int = 5,
                               figsize: Tuple[int, int] = (12, 8)) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot stacked bar chart showing taxa composition across all archetypes.
    
    Args:
        archetypes: Array of archetype compositions
        otu_columns: Column names (taxa) from original OTU data
        num_taxa: Number of top taxa per archetype to include
        figsize: Figure size
        
    Returns:
        Figure and axes objects
    """
    n_archs = len(archetypes)
    archetype_names = [f"Archetype {i+1}" for i in range(n_archs)]
    
    # Create archetype DataFrame
    archs = pd.DataFrame(
        archetypes, 
        columns=otu_columns, 
        index=archetype_names
    )
    
    # Extract top taxa for each archetype (similar to separate bars)
    top_taxa_archs = []
    for i in range(n_archs):
        arch_sorted = archs.iloc[i].sort_values(ascending=False)
        top_taxa = arch_sorted.iloc[:num_taxa]
        
        total = archs.loc[f"Archetype {i+1}"].sum()
        top_taxa_list = [(val/total, taxa) for taxa, val in top_taxa.items()]
        top_taxa_list.append((1 - sum([val for val, taxa in top_taxa_list]), "Others"))
        
        top_taxa_archs.append(top_taxa_list)
    
    # Get union of all top contributing taxa across all archetypes
    all_top_taxa = set()
    for arch in top_taxa_archs:
        for val, taxa in arch[:-1]:  # Exclude "Others"
            all_top_taxa.add(taxa)
    
    all_top_taxa = sorted(list(all_top_taxa))
    
    # Create data structure for stacked bar chart
    taxa_percentages = {}
    
    # Initialize all taxa percentages to 0 for each archetype
    for taxa in all_top_taxa:
        taxa_percentages[taxa] = [0] * n_archs
    
    # Fill in the actual percentages for ALL taxa in all_top_taxa
    for arch_idx in range(n_archs):
        arch_name = f"Archetype {arch_idx+1}"
        total = archs.loc[arch_name].sum()
        
        for taxa in all_top_taxa:
            if taxa in archs.columns:
                taxa_percentages[taxa][arch_idx] = (archs.loc[arch_name, taxa] / total) * 100
            else:
                taxa_percentages[taxa][arch_idx] = 0
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Generate colors
    colors = generate_distinct_colors(len(all_top_taxa))
    
    # Set up the bar positions
    x_pos = np.arange(len(archetype_names))
    legend_added = set()
    
    # For each archetype, sort taxa by their contribution and create stacked segments
    for arch_idx in range(len(archetype_names)):
        # Get taxa and their contributions for this archetype, sorted by contribution (descending)
        arch_taxa_contrib = []
        for taxa in all_top_taxa:
            contrib = taxa_percentages[taxa][arch_idx]
            if contrib > 0:  # Only include taxa with non-zero contributions
                arch_taxa_contrib.append((taxa, contrib))
        
        # Sort by contribution (highest first, will be at bottom of stack)
        arch_taxa_contrib.sort(key=lambda x: x[1], reverse=True)
        
        # Create the stacked segments for this archetype
        bottom_val = 0
        for taxa, contrib in arch_taxa_contrib:
            taxa_idx = all_top_taxa.index(taxa)  # Get original index for color consistency
            simplified_name = simplify_taxa_name(taxa)
            
            # Add to legend only if we haven't seen this taxa before
            label = simplified_name if taxa not in legend_added else ""
            if taxa not in legend_added:
                legend_added.add(taxa)
            
            # Plot this segment
            ax.bar(arch_idx, contrib, bottom=bottom_val, color=colors[taxa_idx], label=label)
            bottom_val += contrib
    
    # Customize the plot
    ax.set_xlabel('Archetypes')
    ax.set_ylabel('Percentage Contribution (%)')
    ax.set_title(f'Stacked Bar Chart: Top {num_taxa} Taxa Contribution by Archetype')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(archetype_names)
    
    # Calculate max height for y-axis limit
    max_height = max([sum([taxa_percentages[taxa][arch_idx] for taxa in all_top_taxa 
                          if taxa_percentages[taxa][arch_idx] > 0]) 
                      for arch_idx in range(len(archetype_names))])
    ax.set_ylim(0, max_height * 1.1 if max_height > 0 else 100)
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    plt.tight_layout()
    
    # Print summary information
    print(f"\nSummary:")
    print(f"Number of archetypes: {n_archs}")
    print(f"Number of top taxa per archetype: {num_taxa}")
    print(f"Total unique taxa across all archetypes: {len(all_top_taxa)}")
    
    # Print detailed breakdown for each archetype
    print(f"\nDetailed breakdown:")
    for arch_idx, arch_name in enumerate(archetype_names):
        print(f"\n{arch_name}:")
        # Get non-zero taxa for this archetype
        arch_taxa = []
        for taxa in all_top_taxa:
            if taxa_percentages[taxa][arch_idx] > 0:
                simplified = simplify_taxa_name(taxa)
                arch_taxa.append((simplified, taxa_percentages[taxa][arch_idx], taxa))
        
        # Sort by percentage (descending)
        arch_taxa.sort(key=lambda x: x[1], reverse=True)
        
        for simplified, percentage, original in arch_taxa:
            if percentage > 0.1:  # Only show taxa with >0.1% contribution
                print(f"  {simplified}: {percentage:.2f}%")
    
    return fig, ax

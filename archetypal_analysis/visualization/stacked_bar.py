"""
Stacked bar chart visualization for archetypal analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from typing import List, Optional, Dict, Tuple, Any, Sequence

def _gradient_order(points: np.ndarray) -> np.ndarray:
    """Sort rows by dominant archetype, then by dominance strength."""
    if points.shape[0] == 0:
        return np.array([], dtype=int)
    clean = np.nan_to_num(points, nan=-np.inf)
    dominant = np.argmax(clean, axis=1)
    strength = np.max(clean, axis=1)
    return np.lexsort((-strength, dominant))


def _label_sort_key(value: Any) -> tuple[bool, str]:
    return (bool(pd.isna(value)), str(value))


def _sorted_unique_labels(labels: np.ndarray) -> list[Any]:
    return sorted(pd.unique(labels).tolist(), key=_label_sort_key)


def _discretized_labels(labels: np.ndarray, buckets: Sequence[Tuple[float, float]]) -> tuple[np.ndarray, list[str]]:
    bucketed = np.full(labels.shape[0], None, dtype=object)
    ordered: list[str] = []
    numeric = labels.astype(float)
    for bucket_min, bucket_max in buckets:
        bucket_label = f"{bucket_min}+" if bucket_max == float("inf") else f"{bucket_min}-{bucket_max}"
        if bucket_max == float("inf"):
            mask = numeric >= bucket_min
        else:
            mask = (numeric >= bucket_min) & (numeric < bucket_max)
        if np.any(mask):
            bucketed[mask] = bucket_label
            ordered.append(bucket_label)
    return bucketed, ordered


def similarity_gradient_permutation(
    points: np.ndarray,
    labels: Optional[np.ndarray] = None,
    label_order: Optional[Sequence[Any]] = None,
) -> list[int]:
    """Return a row permutation that reveals archetype gradients.

    When labels are provided, samples stay inside their metadata-category block and
    are sorted within that block by dominant archetype and similarity strength.
    """
    points = np.asarray(points)
    if labels is None:
        return _gradient_order(points).astype(int).tolist()

    labels = np.asarray(labels).ravel()
    if len(labels) != points.shape[0]:
        raise ValueError("labels must have one entry per similarity row")

    if label_order is None:
        label_order = _sorted_unique_labels(labels)

    perms: list[int] = []
    for label in label_order:
        if pd.isna(label):
            mask = pd.isna(labels)
        else:
            mask = labels == label
        indices = np.flatnonzero(mask)
        if len(indices) == 0:
            continue
        local = _gradient_order(points[indices])
        perms.extend(indices[local].astype(int).tolist())
    return perms


def stacked_bar_gradient_permutation(
    points: np.ndarray,
    labels: Optional[np.ndarray] = None,
    discretize: bool = False,
    buckets: Optional[Sequence[Tuple[float, float]]] = None,
) -> list[int]:
    """Compute the sample order used by the stacked-bar gradient visualization."""
    grouping_labels = None if labels is None else np.asarray(labels).ravel()
    label_order = None
    if grouping_labels is not None and discretize and buckets is not None:
        grouping_labels, label_order = _discretized_labels(grouping_labels, buckets)
    return similarity_gradient_permutation(points, grouping_labels, label_order)


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
                   vertex_colors: Optional[Dict[str, float]] = None,
                   sort_within_label_blocks: bool = True,
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
        vertex_colors : Dict[str, float], optional
            Mapping from vertex label to a float (0-1) for colormap color
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
        
        if labels is not None:
            labels = np.asarray(labels).ravel()

        if perms is None and sort_within_label_blocks:
            perms = stacked_bar_gradient_permutation(points, labels, discretize, buckets)

        if perms is not None:
            perms_array = np.asarray(perms, dtype=int)
            points = points[perms_array]
            if labels is not None:
                labels = labels[perms_array]
            ax._aa_stacked_bar_perms = perms_array.astype(int).tolist()
        
        m, n = points.shape
        
        # Set default labels
        if vertices_labels is None:
            vertices_labels = [f"A{i}" for i in range(n)]
        
        if labels is None:
            show_labels = False
            labels = np.arange(m)
        else:
            show_labels = True
        
        # Prepare colormap and color assignment
        if vertex_colors is not None:
            cmap = plt.get_cmap(cmap_name)
            # Support either numeric cmap positions (legacy) or direct colors (e.g., hex strings).
            colors = []
            for i, label in enumerate(vertices_labels):
                color_value = vertex_colors.get(label, i / max(1, len(vertices_labels) - 1))
                if isinstance(color_value, (int, float, np.integer, np.floating)):
                    colors.append(cmap(float(color_value)))
                else:
                    colors.append(color_value)
        else:
            cmap = plt.get_cmap(cmap_name, len(vertices_labels))
            colors = [cmap(i) for i in range(len(vertices_labels))]
        
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
                cmap_name, pad, alpha, colors=colors, **kwargs
            )
            
        else:
            # Plot simple bars without grouping
            self._plot_simple_bars(
                ax, points, labels, vertices_labels, colors=colors, **kwargs
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
        label_groups = {}

        unique_labels = _sorted_unique_labels(labels)
        for label in unique_labels:
            if pd.isna(label):
                mask = pd.isna(labels)
            else:
                mask = labels == label
            label_groups[label] = points[mask]
        
        return unique_labels, label_groups
    
    def _plot_grouped_bars(self,
                          ax: plt.Axes,
                          label_groups: Dict,
                          processed_labels: List,
                          vertices_labels: List[str],
                          cmap_name: str,
                          pad: float,
                          alpha: float,
                          colors: Optional[List[Any]] = None,
                          **kwargs) -> None:
        """Plot grouped stacked bars"""
        # Calculate group positions
        group_sizes = [len(label_groups[label]) for label in processed_labels]
        x_centers = np.cumsum(group_sizes) - np.array(group_sizes) // 2
        group_padding = pad * sum(group_sizes)
        
        # Use provided colors or default
        if colors is None:
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
                         colors: Optional[List[Any]] = None,
                         **kwargs) -> None:
        """Plot simple stacked bars without grouping"""
        bottom = np.zeros(len(points))
        if colors is None:
            colors = [None] * len(vertices_labels)
        for j, vertex_label in enumerate(vertices_labels):
            ax.bar(
                labels,
                points[:, j],
                label=vertex_label,
                bottom=bottom,
                color=colors[j],
                **kwargs
            )
            bottom += points[:, j]

        legend_patches = [
            Patch(color=colors[i], label=vertices_labels[i])
            for i in range(len(vertices_labels))
        ]
        ax.legend(handles=legend_patches, loc="upper left", bbox_to_anchor=(1, 1))
    
    def _set_axis_properties(self, ax: plt.Axes, show_labels: bool) -> None:
        """Set axis properties and styling"""
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
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
                                  vertex_colors: Optional[Dict[str, float]] = None,
                                  vertices_labels: Optional[List[str]] = None,
                                  sort_within_label_blocks: bool = True,
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

        if vertices_labels is None:
            vertices_labels = list(plot_similarity.columns)

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        return self.stacked_bar(
            points=plot_similarity.to_numpy(),
            perms=perms,
            ax=ax,
            labels=labels,
            vertex_colors=vertex_colors,
            discretize=discretize,
            buckets=buckets,
            vertices_labels=vertices_labels,
            sort_within_label_blocks=sort_within_label_blocks,
            **kwargs
        )
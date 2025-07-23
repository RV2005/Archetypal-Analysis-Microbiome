"""
Archetype composition analysis and visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Any
import colorsys

from ..utils.blast import BlastAnalyzer


class CompositionAnalyzer:
    """Archetype composition analysis and visualization"""
    
    def __init__(self, blast_analyzer: Optional[BlastAnalyzer] = None):
        self.blast_analyzer = blast_analyzer or BlastAnalyzer()
    
    def analyze_top_taxa(self, 
                        archetypes: np.ndarray,
                        feature_names: List[str],
                        archetype_names: Optional[List[str]] = None,
                        num_taxa: int = 5) -> List[List[Tuple[float, str]]]:
        """
        Extract top contributing taxa per archetype.
        
        Parameters:
        -----------
        archetypes : np.ndarray
            Archetype matrix (n_archetypes x n_features)
        feature_names : List[str]
            Names of features (taxa)
        archetype_names : List[str], optional
            Names of archetypes
        num_taxa : int
            Number of top taxa to extract per archetype
            
        Returns:
        --------
        top_taxa_archetypes : List[List[Tuple[float, str]]]
            Top taxa for each archetype with (percentage, taxa_name) tuples
        """
        n_archetypes = len(archetypes)
        if archetype_names is None:
            archetype_names = [f"Archetype {i+1}" for i in range(n_archetypes)]
        
        # Create DataFrame for easier manipulation
        archs_df = pd.DataFrame(
            archetypes,
            columns=feature_names,
            index=archetype_names
        )
        
        top_taxa_archetypes = []
        
        for i, arch_name in enumerate(archetype_names):
            # Sort taxa by contribution (descending)
            arch_values = archs_df.loc[arch_name].sort_values(ascending=False)
            top_taxa = arch_values.head(num_taxa)
            
            # Calculate percentages
            total = archs_df.loc[arch_name].sum()
            top_taxa_tuples = [(val/total, taxa) for taxa, val in top_taxa.items()]
            
            # Add "Others" category
            others_percentage = 1 - sum([val for val, _ in top_taxa_tuples])
            top_taxa_tuples.append((others_percentage, "Others"))
            
            top_taxa_archetypes.append(top_taxa_tuples)
        
        return top_taxa_archetypes
    
    def simplify_taxa_names(self, taxa_names: List[str], max_length: int = 20) -> Dict[str, str]:
        """
        Simplify taxonomic names for better readability.
        
        Parameters:
        -----------
        taxa_names : List[str]
            Original taxonomic names
        max_length : int
            Maximum length for simplified names
            
        Returns:
        --------
        simplified_names : Dict[str, str]
            Mapping from original to simplified names
        """
        simplified_names = {}
        
        for taxa_name in taxa_names:
            if taxa_name == "Others":
                simplified_names[taxa_name] = "Others"
                continue
            
            # Split by dots and get the last meaningful part
            parts = taxa_name.split('.')
            parts = [part for part in parts if part and part not in ["NA", "unknown", "unclassified", "."]]
            
            if parts:
                simplified = parts[-1]
            else:
                simplified = taxa_name
            
            # Truncate if too long
            if len(simplified) > max_length:
                simplified = simplified[:max_length] + "..."
            
            simplified_names[taxa_name] = simplified
        
        return simplified_names
    
    def plot_separate_bar_charts(self,
                                top_taxa_archetypes: List[List[Tuple[float, str]]],
                                archetype_names: Optional[List[str]] = None,
                                figsize: Tuple[int, int] = (15, 8),
                                vertical: bool = False,
                                simplify_names: bool = True,
                                max_name_length: int = 20) -> Tuple[plt.Figure, np.ndarray]:
        """
        Individual bar charts per archetype.
        
        Parameters:
        -----------
        top_taxa_archetypes : List[List[Tuple[float, str]]]
            Top taxa data from analyze_top_taxa
        archetype_names : List[str], optional
            Names for archetypes
        figsize : Tuple[int, int]
            Figure size
        vertical : bool
            Whether to arrange subplots vertically
        simplify_names : bool
            Whether to simplify taxonomic names
        max_name_length : int
            Maximum length for taxa names
            
        Returns:
        --------
        fig : Figure
            Matplotlib figure
        axes : ndarray
            Array of axes objects
        """
        n_archetypes = len(top_taxa_archetypes)
        
        if archetype_names is None:
            archetype_names = [f"Archetype {i+1}" for i in range(n_archetypes)]
        
        # Calculate subplot layout
        if vertical:
            nrows, ncols = (n_archetypes + 1) // 2, 2
        else:
            nrows, ncols = 2, (n_archetypes + 1) // 2
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = axes.flatten() if n_archetypes > 1 else [axes]
        
        # Find maximum percentage for consistent y-axis
        max_val = max([
            max([val for val, taxa in arch][:-1]) 
            for arch in top_taxa_archetypes
        ])
        
        for i, arch_data in enumerate(top_taxa_archetypes):
            if i >= len(axes):
                break
            
            # Extract data (excluding "Others")
            vals = [val for val, taxa in arch_data[:-1]]
            taxa_names = [taxa for val, taxa in arch_data[:-1]]
            
            # Simplify names if requested
            if simplify_names:
                simplified_mapping = self.simplify_taxa_names(taxa_names, max_name_length)
                plot_labels = [simplified_mapping[name] for name in taxa_names]
            else:
                plot_labels = taxa_names
            
            # Create bar chart
            axes[i].bar(plot_labels, vals)
            axes[i].set_title(archetype_names[i])
            axes[i].set_ylim(0, max_val)
            axes[i].set_xticklabels(
                axes[i].get_xticklabels(), 
                rotation=30, 
                ha='right'
            )
        
        # Hide unused subplots
        for j in range(n_archetypes, len(axes)):
            axes[j].set_visible(False)
        
        fig.tight_layout()
        return fig, axes
    
    def plot_stacked_bar_chart(self,
                               aa_archetypes: pd.DataFrame,
                               feature_names: List[str],
                               top_taxa_archetypes: List[List[Tuple[float, str]]],
                               archetype_names: Optional[List[str]] = None,
                               figsize: Tuple[int, int] = (12, 8),
                               sort_by_contribution: bool = True,
                               simplify_names: bool = True,
                              max_name_length: int = 20) -> Tuple[plt.Figure, plt.Axes]:
        """
        Combined stacked bar chart for all archetypes.
        
        Parameters:
        -----------
        aa_archetypes : pd.DataFrame
            Archetype data
        top_taxa_archetypes : List[List[Tuple[float, str]]]
            Top taxa data from analyze_top_taxa
        archetype_names : List[str], optional
            Names for archetypes
        figsize : Tuple[int, int]
            Figure size
        sort_by_contribution : bool
            Whether to sort taxa by contribution within each archetype
        simplify_names : bool
            Whether to simplify taxonomic names
        max_name_length : int
            Maximum length for taxa names
            
        Returns:
        --------
        fig : Figure
            Matplotlib figure
        ax : Axes
            Axes object
        """
        n_archetypes = len(top_taxa_archetypes)
        if archetype_names is None:
            archetype_names = [f"Archetype {i+1}" for i in range(n_archetypes)]

        archs_df = pd.DataFrame(
            aa_archetypes,
            columns=feature_names,
            index=archetype_names
        )
        
        # Get union of all top taxa (excluding "Others")
        all_top_taxa = set()
        for arch_data in top_taxa_archetypes:
            for val, taxa in arch_data[:-1]:  # Exclude "Others"
                all_top_taxa.add(taxa)
        
        all_top_taxa = list(all_top_taxa)
        
        # Simplify names first if needed, then sort by display names
        if simplify_names:
            simplified_mapping = self.simplify_taxa_names(all_top_taxa, max_name_length)
            # Sort by simplified names
            all_top_taxa = sorted(all_top_taxa, key=lambda x: simplified_mapping[x])
        else:
            # Sort by original names
            all_top_taxa = sorted(all_top_taxa)
        
        # Create taxa percentages matrix
        taxa_percentages = self._create_taxa_percentage_matrix(
            archs_df, all_top_taxa, archetype_names
        )
        
        # Generate distinct colors
        colors = self._generate_distinct_colors(len(all_top_taxa))
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot stacked bars
        simplified_mapping_to_pass = simplified_mapping if simplify_names else None
        self._plot_stacked_bars(
            ax, taxa_percentages, all_top_taxa, archetype_names, 
            colors, sort_by_contribution, simplify_names, max_name_length,
            simplified_mapping_to_pass
        )
        
        # Customize plot
        ax.set_xlabel('Archetypes')
        ax.set_ylabel('Percentage Contribution (%)')
        ax.set_title(f'Stacked Bar Chart: Top Taxa Contribution by Archetype')
        ax.set_xticks(range(len(archetype_names)))
        ax.set_xticklabels(archetype_names)
        
        # Calculate and set y-axis limit
        max_height = max([
            sum([taxa_percentages[taxa][arch_idx] for taxa in all_top_taxa])
            for arch_idx in range(len(archetype_names))
        ])
        ax.set_ylim(0, max_height * 1.1 if max_height > 0 else 100)
        
        # Add legend
        if simplify_names:
            legend_labels = [simplified_mapping[taxa] for taxa in all_top_taxa]
        else:
            legend_labels = all_top_taxa

        # Create proxy artists for legend
        from matplotlib.patches import Patch
        handles = [Patch(facecolor=colors[i], label=legend_labels[i]) for i in range(len(legend_labels))]
        # Sort legend by label
        sorted_handles_labels = sorted(zip(handles, legend_labels), key=lambda x: x[1])
        sorted_handles, sorted_labels = zip(*sorted_handles_labels)
        ax.legend(sorted_handles, sorted_labels, loc='upper left', bbox_to_anchor=(1, 1))

        plt.tight_layout()
        return fig, ax
    
    def _create_taxa_percentage_matrix(self,
                                        archs: pd.DataFrame,
                                        all_top_taxa: List[str],
                                        archetype_names: List[str]) -> Dict[str, List[float]]:
        """Create taxa percentage matrix for all archetypes"""
        n_archs = len(archetype_names)
        taxa_percentages = {}

        # Initialize all taxa percentages to 0 for each archetype
        for taxa in all_top_taxa:
            taxa_percentages[taxa] = [0] * n_archs

        # Fill in the actual percentages for ALL taxa in all_top_taxa (not just top N per archetype)
        for arch_idx in range(n_archs):
            arch_name = f"Archetype {arch_idx+1}"
            total = archs.loc[arch_name].sum()
            
            for taxa in all_top_taxa:
                if taxa in archs.columns:
                    # Get the actual percentage of this taxa in this archetype
                    taxa_percentages[taxa][arch_idx] = (archs.loc[arch_name, taxa] / total) * 100
                else:
                    taxa_percentages[taxa][arch_idx] = 0
        
        return taxa_percentages
    
    def _generate_distinct_colors(self, n_colors: int) -> List[str]:
        """Generate n distinct colors using different strategies based on count"""
        if n_colors <= 12:
            # Use qualitative colormap for small numbers
            cmap = plt.cm.Set3
            return [cmap(i/12) for i in range(n_colors)]
        elif n_colors <= 20:
            # Use tab20 for medium numbers
            cmap = plt.cm.tab20
            return [cmap(i/20) for i in range(n_colors)]
        else:
            # Generate colors in HSV space for large numbers
            colors = []
            for i in range(n_colors):
                hue = i / n_colors
                saturation = 0.7 + (i % 3) * 0.1  # Vary saturation slightly
                value = 0.7 + (i % 2) * 0.2       # Vary brightness slightly
                rgb = colorsys.hsv_to_rgb(hue, saturation, value)
                colors.append(rgb)
            return colors
    
    def _plot_stacked_bars(self,
                          ax: plt.Axes,
                          taxa_percentages: Dict[str, List[float]],
                          all_top_taxa: List[str],
                          archetype_names: List[str],
                          colors: List,
                          sort_by_contribution: bool,
                          simplify_names: bool,
                          max_name_length: int,
                          simplified_mapping: Optional[Dict[str, str]] = None) -> None:
        """Plot the stacked bars"""
        legend_added = set()
        
        for arch_idx in range(len(archetype_names)):
            # Get taxa and their contributions for this archetype
            if sort_by_contribution:
                arch_taxa_contrib = []
                for taxa in all_top_taxa:
                    contrib = taxa_percentages[taxa][arch_idx]
                    if contrib > 0:
                        arch_taxa_contrib.append((taxa, contrib))
                
                # Sort by contribution (highest first, will be at bottom of stack)
                arch_taxa_contrib.sort(key=lambda x: x[1], reverse=True)
            else:
                arch_taxa_contrib = [
                    (taxa, taxa_percentages[taxa][arch_idx]) 
                    for taxa in all_top_taxa 
                    if taxa_percentages[taxa][arch_idx] > 0
                ]
            
            # Create the stacked segments for this archetype
            bottom_val = 0
            for taxa, contrib in arch_taxa_contrib:
                taxa_idx = all_top_taxa.index(taxa)  # Get original index for color consistency
                
                # Simplify name for legend
                if simplify_names and simplified_mapping:
                    display_name = simplified_mapping[taxa]
                else:
                    display_name = taxa
                
                # Add to legend only if we haven't seen this taxa before
                label = display_name if taxa not in legend_added else ""
                if taxa not in legend_added:
                    legend_added.add(taxa)
                
                # Plot this segment
                ax.bar(arch_idx, contrib, bottom=bottom_val, 
                      color=colors[taxa_idx], label=label)
                bottom_val += contrib
    
    def create_composition_summary(self,
                                 top_taxa_archetypes: List[List[Tuple[float, str]]],
                                 archetype_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Create a summary DataFrame of archetype compositions.
        
        Parameters:
        -----------
        top_taxa_archetypes : List[List[Tuple[float, str]]]
            Top taxa data from analyze_top_taxa
        archetype_names : List[str], optional
            Names for archetypes
            
        Returns:
        --------
        summary : DataFrame
            Summary of top taxa for each archetype
        """
        n_archetypes = len(top_taxa_archetypes)
        
        if archetype_names is None:
            archetype_names = [f"Archetype {i+1}" for i in range(n_archetypes)]
        
        summary_data = []
        
        for arch_idx, arch_data in enumerate(top_taxa_archetypes):
            arch_name = archetype_names[arch_idx]
            
            for rank, (percentage, taxa) in enumerate(arch_data, 1):
                if taxa == "Others":
                    continue  # Skip "Others" category
                
                summary_data.append({
                    'archetype': arch_name,
                    'rank': rank,
                    'taxa': taxa,
                    'percentage': percentage * 100,  # Convert to percentage
                    'simplified_taxa': self.simplify_taxa_names([taxa])[taxa]
                })
        
        return pd.DataFrame(summary_data)
    

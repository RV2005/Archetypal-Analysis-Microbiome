"""
Advanced example replicating the notebook workflow.

This example demonstrates the complete workflow from the original notebook:
1. Load compendium data and metadata
2. Run archetypal analysis for multiple K values
3. Create simplex plots colored by metadata
4. Generate composition visualizations
5. Perform group analysis
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add the package to Python path
sys.path.append('..')

from archetypal_analysis.data.loaders import DataLoader
from archetypal_analysis.analysis.archetypal import ArchetypalAnalysis
from archetypal_analysis.visualization.simplex import plot_simplex, plot_multiple_k_simplex
from archetypal_analysis.visualization.stacked_bar import plot_archetype_composition_bars, plot_stacked_bar_composition
from archetypal_analysis.utils.metadata import extract_metadata_compendium


def load_compendium_data():
    """Load the compendium datasets."""
    print("Loading compendium data...")
    
    # Data paths (modify as needed)
    data_paths = {
        'otu': r"datasets\compendium\normalized\compendium_Norm_QC-region30k.csv",
        'projects': "datasets/compendium/projects.csv", 
        'metadata': "datasets/compendium/sample_metadata.tsv",
        'tags': "datasets/compendium/tags.tsv"
    }
    
    datasets = {}
    
    # Load OTU data
    try:
        datasets['otu'] = pd.read_csv(data_paths['otu'], index_col=0)
        print(f"   OTU data loaded: {datasets['otu'].shape}")
    except FileNotFoundError:
        print(f"   Warning: {data_paths['otu']} not found, creating sample data")
        # Create sample data
        np.random.seed(42)
        n_samples, n_taxa = 200, 100
        datasets['otu'] = pd.DataFrame(
            np.random.lognormal(0, 1, (n_samples, n_taxa)),
            index=[f"project{i//20}_sample{i%20}" for i in range(n_samples)],
            columns=[f"Bacteria.Phylum{i//10}.Genus{i}" for i in range(n_taxa)]
        )
    
    # Load metadata
    for name, path in data_paths.items():
        if name == 'otu':
            continue
        try:
            if name == 'projects':
                datasets[name] = pd.read_csv(path, index_col=0)
            else:
                datasets[name] = pd.read_csv(path, sep='\t' if path.endswith('.tsv') else ',')
            print(f"   {name.capitalize()} loaded: {datasets[name].shape}")
        except FileNotFoundError:
            print(f"   Warning: {path} not found, creating sample {name} data")
            # Create sample metadata
            if name == 'projects':
                datasets[name] = pd.DataFrame({
                    'sample_type': np.random.choice(['stool', 'saliva', 'skin'], 20),
                    'condition': np.random.choice(['healthy', 'disease'], 20)
                }, index=[f"project{i}" for i in range(20)])
            elif name == 'metadata':
                datasets[name] = pd.DataFrame({
                    'project': np.repeat([f"project{i}" for i in range(20)], 10),
                    'srr': [f"sample{i}" for i in range(10)] * 20,
                    'region': np.random.choice(['North America', 'Europe', 'Asia'], 200)
                })
            else:  # tags
                datasets[name] = pd.DataFrame({
                    'project': np.repeat([f"project{i}" for i in range(20)], 10),
                    'srr': [f"sample{i}" for i in range(10)] * 20,
                    'tag': np.random.choice(['age_months', 'bmi'], 200),
                    'value': np.random.normal(30, 10, 200)
                })
    
    return datasets


def run_multiple_k_analysis(otu_data, k_values, output_dir):
    """Run archetypal analysis for multiple K values."""
    print(f"\nRunning archetypal analysis for K values: {k_values}")
    
    results = {}
    
    for k in k_values:
        print(f"\n--- Analyzing K = {k} ---")
        
        analyzer = ArchetypalAnalysis(
            n_archetypes=k,
            method="pgd", 
            max_iter=100,
            random_state=42
        )
        
        result = analyzer.fit_transform(otu_data, verbose=False)
        results[k] = result
        
        # Save result
        output_file = f"otu_AA_K{k}_1.0_None_jensenshannon.json"
        analyzer.save_results(result, os.path.join(output_dir, output_file))
        
        print(f"   K={k} completed. Loss: {result.loss:.6f}")
    
    return results


def analyze_simplex_visualizations(results, datasets, output_dir):
    """Create simplex visualizations with metadata coloring."""
    print("\nCreating simplex visualizations...")
    
    # Extract metadata for plotting
    sample_indices = list(results[7].similarity_degree.index)  # Use K=7 as reference
    
    extract_cols = ['region', 'sample_type', 'age_months']
    metadata = extract_metadata_compendium(
        sample_indices,
        datasets['tags'], 
        datasets['metadata'],
        datasets['projects'],
        extract_cols
    )
    
    # Plot for region (discrete)
    print("   Creating region-based simplex plot...")
    result = results[7]  # Use K=7
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    plot_simplex(
        otu_aa=result.similarity_degree,
        metadata=metadata,
        filter_cols=['region'],
        cmap_name='tab10',
        legend_type='discrete',
        remove_nan=True,
        alpha=0.6,
        s=15,
        remove_values={'region': 'unknown'},
        ax=ax
    )
    ax.set_title('Archetypal Analysis - Colored by Region')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'simplex_region.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot for age (continuous, if available)
    if 'age_months' in metadata.columns and metadata['age_months'].notna().any():
        print("   Creating age-based simplex plot...")
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        plot_simplex(
            otu_aa=result.similarity_degree,
            metadata=metadata,
            filter_cols=['age_months'],
            cmap_name='viridis',
            legend_type='continuous',
            remove_nan=True,
            alpha=0.75,
            s=20,
            ax=ax
        )
        ax.set_title('Archetypal Analysis - Colored by Age')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'simplex_age.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    # Group plotting for multiple K values
    print("   Creating multi-K simplex visualization...")
    k_subset = [3, 5, 7, 8]  # Subset for demonstration
    
    try:
        fig, axes = plot_multiple_k_simplex(
            results_dir=output_dir,
            k_values=k_subset,
            metadata=metadata,
            filter_cols=['region'],
            legend_type='discrete',
            cmap_name='tab10',
            alpha=0.6,
            s=15,
            remove_values={'region': 'unknown'}
        )
        fig.suptitle('Archetypal Analysis - Multiple K Values', fontsize=16)
        plt.savefig(os.path.join(output_dir, 'simplex_multi_k.png'), dpi=300, bbox_inches='tight')
        plt.show()
    except Exception as e:
        print(f"   Multi-K plotting failed: {e}")


def analyze_compositions(results, otu_data, output_dir):
    """Analyze and visualize archetype compositions."""
    print("\nAnalyzing archetype compositions...")
    
    # Use K=7 for detailed composition analysis
    result = results[7]
    
    # Get compositions
    top_compositions = result.get_archetype_composition(k=10)
    
    print("Top taxa contributions per archetype:")
    for arch_name in result.archetype_names:
        print(f"\n{arch_name}:")
        arch_comp = top_compositions.loc[arch_name]
        top_5 = arch_comp.nlargest(5)
        for taxa, contrib in top_5.items():
            taxa_short = taxa.split('.')[-1] if '.' in taxa else taxa
            print(f"  {taxa_short[:20]}: {contrib:.4f}")
    
    # Separate bar charts
    print("   Creating separate bar charts...")
    fig_bars, ax_bars = plot_archetype_composition_bars(
        result.archetypes,
        otu_data.columns.tolist(),
        num_taxa=5,
        vertical=False,
        figsize=(16, 10)
    )
    fig_bars.suptitle('Top Contributing Taxa per Archetype', fontsize=16)
    plt.savefig(os.path.join(output_dir, 'composition_separate_bars.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Stacked bar chart
    print("   Creating stacked bar chart...")
    fig_stack, ax_stack = plot_stacked_bar_composition(
        result.archetypes,
        otu_data.columns.tolist(),
        num_taxa=8,
        figsize=(14, 8)
    )
    plt.savefig(os.path.join(output_dir, 'composition_stacked_bar.png'), dpi=300, bbox_inches='tight')
    plt.show()


def analyze_loss_curves(results, output_dir):
    """Analyze loss curves across K values."""
    print("\nAnalyzing loss curves...")
    
    k_values = sorted(results.keys())
    losses = [results[k].loss for k in k_values]
    
    # Plot loss curve
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(k_values, losses, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Archetypes (K)')
    ax.set_ylabel('Loss')
    ax.set_title('Archetypal Analysis Loss Curve')
    ax.grid(True, alpha=0.3)
    
    # Calculate and show loss differences
    loss_diffs = [losses[i] - losses[i-1] for i in range(1, len(losses))]
    
    ax2 = ax.twinx()
    ax2.plot(k_values[1:], loss_diffs, 'r^--', alpha=0.7, label='Loss Difference')
    ax2.set_ylabel('Loss Difference', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print loss analysis
    print("Loss analysis:")
    for i, k in enumerate(k_values):
        loss_diff = loss_diffs[i-1] if i > 0 else 0
        print(f"  K={k}: Loss={losses[i]:.6f}, Diff={loss_diff:.6f}")


def main():
    """Main analysis pipeline."""
    print("=== Advanced Archetypal Analysis Workflow ===\n")
    
    # Setup output directory
    output_dir = "results_advanced"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load data
    datasets = load_compendium_data()
    otu_data = datasets['otu']
    
    # Sample data if too large (for demonstration)
    if otu_data.shape[0] > 500:
        otu_data = otu_data.sample(n=500, random_state=42)
        print(f"Sampled OTU data to: {otu_data.shape}")
    
    # 2. Run multiple K analysis
    k_values = [3, 5, 7, 8, 10]  # Reduced for faster demo
    results = run_multiple_k_analysis(otu_data, k_values, output_dir)
    
    # 3. Analyze loss curves
    analyze_loss_curves(results, output_dir)
    
    # 4. Create simplex visualizations
    analyze_simplex_visualizations(results, datasets, output_dir)
    
    # 5. Analyze compositions
    analyze_compositions(results, otu_data, output_dir)
    
    # 6. Summary
    print(f"\n=== Analysis Complete ===")
    print(f"Results saved in: {output_dir}")
    print(f"Analyzed K values: {k_values}")
    print(f"Best K based on loss curve: {min(results.keys(), key=lambda k: results[k].loss)}")
    
    # Optional: Create summary report
    summary_file = os.path.join(output_dir, "analysis_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("Archetypal Analysis Summary\n")
        f.write("==========================\n\n")
        f.write(f"Data shape: {otu_data.shape}\n")
        f.write(f"K values analyzed: {k_values}\n\n")
        f.write("Loss values:\n")
        for k in sorted(results.keys()):
            f.write(f"  K={k}: {results[k].loss:.6f}\n")
    
    print(f"Summary report saved: {summary_file}")


if __name__ == "__main__":
    main()

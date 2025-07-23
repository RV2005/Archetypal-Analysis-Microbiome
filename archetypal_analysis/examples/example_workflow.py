"""
Example workflow for archetypal analysis using the refactored package.

This example demonstrates:
1. Loading and preprocessing microbiome data
2. Running archetypal analysis
3. Creating visualizations
4. Analyzing archetype compositions
5. BLAST analysis of key taxa
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add the package to Python path
sys.path.append('..')

from archetypal_analysis.data.normalizers import DataNormalizer
from archetypal_analysis.data.quality_control import QualityController
from archetypal_analysis.analysis.archetypal import ArchetypalAnalysis
from archetypal_analysis.visualization.simplex import SimplexPlotter
from archetypal_analysis.visualization.stacked_bar import plot_archetype_composition_bars, plot_stacked_bar_composition
from archetypal_analysis.utils.metadata import extract_metadata_compendium
# from archetypal_analysis.utils.blast import BlastAnalyzer
from archetypal_analysis.config import AnalysisConfig


def main():
    """Main analysis workflow."""
    
    print("=== Archetypal Analysis Workflow ===\n")
    
    # 1. Load data
    print("1. Loading data...")
    loader = DataLoader()
    
    # Load OTU data (modify path as needed)
    otu_file = "normalized_otu_Norm-sample_type.csv"
    if os.path.exists(otu_file):
        otu_data = loader.load_otu_data(otu_file)
        print(f"   Loaded OTU data: {otu_data.shape}")
    else:
        print(f"   Warning: OTU file {otu_file} not found. Creating sample data...")
        # Create sample data for demonstration
        np.random.seed(42)
        n_samples, n_taxa = 100, 50
        otu_data = pd.DataFrame(
            np.random.lognormal(0, 1, (n_samples, n_taxa)),
            index=[f"sample_{i}" for i in range(n_samples)],
            columns=[f"taxa_{i}" for i in range(n_taxa)]
        )
        print(f"   Created sample OTU data: {otu_data.shape}")
    
    # 2. Quality control and preprocessing
    print("\n2. Quality control and preprocessing...")
    qc = QualityController()
    
    # Apply basic QC filters
    otu_filtered = qc.filter_low_abundance_taxa(otu_data, min_abundance=0.001)
    otu_filtered = qc.filter_low_prevalence_taxa(otu_filtered, min_prevalence=0.1)
    
    print(f"   After QC: {otu_filtered.shape}")
    
    # Optional: Apply additional normalization
    normalizer = DataNormalizer()
    otu_normalized = normalizer.total_sum_scaling(otu_filtered)
    
    # Sample subset for faster computation (optional)
    frac = 1.0  # Use all data, or set to < 1.0 for subset
    if frac < 1.0:
        otu_sample = otu_normalized.sample(frac=frac, random_state=42)
        print(f"   Sampled {frac*100}% of data: {otu_sample.shape}")
    else:
        otu_sample = otu_normalized
    
    # 3. Run archetypal analysis
    print("\n3. Running archetypal analysis...")
    
    n_archetypes = 5
    analyzer = ArchetypalAnalysis(
        n_archetypes=n_archetypes,
        method="pgd",
        max_iter=10,
        random_state=42
    )
    
    # Fit the model
    result = analyzer.fit_transform(otu_sample, verbose=True)
    
    print(f"   Analysis completed. Final loss: {result.loss:.6f}")
    print(f"   Found {result.n_archetypes} archetypes for {result.n_samples} samples")
    
    # 4. Save results
    print("\n4. Saving results...")
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    analyzer.save_results(result, output_dir, group="example", frac=frac)
    print(f"   Results saved to {output_dir}")
    
    # 5. Basic simplex visualization
    print("\n5. Creating basic simplex visualization...")
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Plot basic simplex colored by archetype labels
    plotter = SimplexPlotter()
    
    # Create simple metadata for demonstration
    demo_metadata = pd.DataFrame({
        'group': np.random.choice(['A', 'B', 'C'], size=len(result.similarity_degree)),
        'age': np.random.normal(30, 10, size=len(result.similarity_degree))
    }, index=result.similarity_degree.index)
    
    # Plot colored by group
    ax = plot_simplex(
        otu_aa=result.similarity_degree,
        metadata=demo_metadata,
        filter_cols=['group'],
        cmap_name='tab10',
        legend_type='discrete',
        alpha=0.7,
        s=20,
        ax=ax
    )
    ax.set_title('Archetypal Analysis - Simplex Plot')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'simplex_basic.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 6. Archetype composition analysis
    print("\n6. Analyzing archetype compositions...")
    
    # Get top contributing taxa for each archetype
    top_compositions = result.get_archetype_composition(k=10)
    
    print("Top taxa per archetype:")
    for i, (arch_name, composition) in enumerate(top_compositions.iterrows()):
        print(f"\n{arch_name}:")
        top_taxa = composition.nlargest(5)
        for taxa, value in top_taxa.items():
            print(f"  {taxa}: {value:.4f}")
    
    # 7. Composition visualizations
    print("\n7. Creating composition visualizations...")
    
    # Separate bar charts
    fig_bars, ax_bars = plot_archetype_composition_bars(
        result.archetypes,
        otu_sample.columns.tolist(),
        num_taxa=5,
        figsize=(15, 10)
    )
    fig_bars.suptitle('Top Taxa by Archetype (Separate Bars)')
    plt.savefig(os.path.join(output_dir, 'composition_bars.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Stacked bar chart
    fig_stack, ax_stack = plot_stacked_bar_composition(
        result.archetypes,
        otu_sample.columns.tolist(),
        num_taxa=5,
        figsize=(12, 8)
    )
    plt.savefig(os.path.join(output_dir, 'composition_stacked.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 8. Optional: BLAST analysis (requires internet connection and BioPython)
    print("\n8. BLAST analysis (optional)...")
    try:
        from Bio import Blast
        
        # Get archetype compositions in the format expected by BLAST analyzer
        archetype_compositions = []
        for i in range(result.n_archetypes):
            arch_data = pd.Series(result.archetypes[i], index=otu_sample.columns)
            top_taxa = arch_data.nlargest(5)
            total = arch_data.sum()
            
            composition = [(val/total, taxa) for taxa, val in top_taxa.items()]
            # Add "Others"
            others_pct = 1 - sum([val for val, _ in composition])
            composition.append((others_pct, "Others"))
            
            archetype_compositions.append(composition)
        
        blast_analyzer = BlastAnalyzer(num_hits=3)
        
        print("   Running BLAST analysis... (this may take a while)")
        blast_results = blast_analyzer.analyze_archetype_taxa(
            archetype_compositions,
            temp_fasta=os.path.join(output_dir, "temp_blast.fasta"),
            output_xml=os.path.join(output_dir, "blast_results.xml")
        )
        
        print("   BLAST analysis completed!")
        print(f"   Found annotations for {len(blast_results)} taxa")
        
        # Get best hit descriptions
        from archetypal_analysis.utils.blast import get_best_hit_descriptions
        best_hits = get_best_hit_descriptions(blast_results)
        
        print("\nBest BLAST hits:")
        for taxa, description in list(best_hits.items())[:10]:
            print(f"  {taxa} -> {description}")
            
    except ImportError:
        print("   BioPython not available, skipping BLAST analysis")
    except Exception as e:
        print(f"   BLAST analysis failed: {e}")
    
    print(f"\n=== Analysis completed! Results saved in '{output_dir}' ===")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Generate synthetic test data for microbiome_validate_fuzzy_taxa.py

This script creates minimal synthetic data to test the pipeline installation and functionality.

Usage:
    python generate_test_data.py --output_dir ./test_data

This creates:
    - test_otu_table.tsv: Synthetic OTU/ASV abundance table
    - test_metadata.tsv: Synthetic sample metadata
    - test_fuzzy_taxa.tsv: Synthetic fuzzy-selected top taxa
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd


def generate_test_data(output_dir: Path, n_samples: int = 50, n_taxa: int = 200):
    """Generate synthetic test data."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating synthetic test data in {output_dir}")
    
    # ====================================================================
    # 1. Generate OTU table (samples x taxa)
    # ====================================================================
    
    print("Generating OTU table...")
    
    # Create sample IDs
    sample_ids = [f"sample_{i:03d}" for i in range(1, n_samples + 1)]
    
    # Create taxon IDs
    taxon_ids = [f"OTU_{i:04d}" for i in range(1, n_taxa + 1)]
    
    # Generate abundance counts with overdispersion (realistic)
    otu_data = np.zeros((n_samples, n_taxa))
    
    for tax_idx in range(n_taxa):
        # Simulate with negative binomial distribution (realistic for microbiome)
        mu = np.random.exponential(scale=1000, size=1)[0]
        size = np.random.uniform(0.5, 5)
        counts = np.random.negative_binomial(size, size / (mu + size), n_samples)
        otu_data[:, tax_idx] = counts
    
    # Create DataFrame
    otu_df = pd.DataFrame(otu_data, index=sample_ids, columns=taxon_ids)
    
    # Save OTU table
    otu_file = output_dir / "test_otu_table.tsv"
    otu_df.to_csv(otu_file, sep='\t')
    print(f"  Saved OTU table: {otu_file}")
    print(f"    Shape: {otu_df.shape} (samples x taxa)")
    
    # ====================================================================
    # 2. Generate metadata
    # ====================================================================
    
    print("Generating metadata...")
    
    # Create clinical groups (binary for simplicity, but supports multiclass)
    n_case = n_samples // 2
    n_control = n_samples - n_case
    
    disease_status = ['disease'] * n_case + ['control'] * n_control
    random.shuffle(disease_status)
    
    # Generate covariates
    ages = np.random.randint(20, 80, n_samples)
    sexes = np.random.choice(['M', 'F'], n_samples)
    batches = np.random.choice([1, 2, 3], n_samples)
    
    metadata_df = pd.DataFrame({
        'sample_id': sample_ids,
        'disease_status': disease_status,
        'age': ages,
        'sex': sexes,
        'batch': batches
    })
    metadata_df.index = sample_ids
    
    # Save metadata
    metadata_file = output_dir / "test_metadata.tsv"
    metadata_df.to_csv(metadata_file, sep='\t')
    print(f"  Saved metadata: {metadata_file}")
    print(f"    Disease: {disease_status.count('disease')}, Control: {disease_status.count('control')}")
    
    # ====================================================================
    # 3. Generate fuzzy-selected top taxa
    # ====================================================================
    
    print("Generating fuzzy-selected taxa...")
    
    # Select some taxa as "fuzzy-selected", including a few that don't exist (for testing)
    n_fuzzy = min(50, n_taxa // 4)
    
    # Ensure some taxa have high counts (enriched in one group)
    selected_taxa = []
    
    # Group 1 enriched taxa
    for i in range(n_fuzzy // 2):
        taxon = taxon_ids[i * 2]  # Biased selection
        if i % 3 == 0:  # Add some with high membership
            membership = np.random.uniform(0.8, 0.99)
        else:
            membership = np.random.uniform(0.7, 0.9)
        
        selected_taxa.append({
            'cluster': 'cluster_1',
            'taxon': taxon,
            'membership': membership,
            'fuzzy_rank': i + 1
        })
    
    # Group 2 enriched taxa
    for i in range(n_fuzzy // 2, n_fuzzy):
        taxon = taxon_ids[n_taxa - 1 - (i - n_fuzzy // 2) * 2]  # Biased selection
        membership = np.random.uniform(0.7, 0.9)
        
        selected_taxa.append({
            'cluster': 'cluster_2',
            'taxon': taxon,
            'membership': membership,
            'fuzzy_rank': i - n_fuzzy // 2 + 1
        })
    
    # Add a few non-existent taxa to test unmatched handling
    selected_taxa.append({
        'cluster': 'cluster_1',
        'taxon': 'NonExistent_OTU_99999',
        'membership': 0.85,
        'fuzzy_rank': n_fuzzy + 1
    })
    
    selected_taxa.append({
        'cluster': 'cluster_2',
        'taxon': 'AlsoNotPresent_OTU_88888',
        'membership': 0.82,
        'fuzzy_rank': n_fuzzy + 2
    })
    
    fuzzy_df = pd.DataFrame(selected_taxa)
    
    # Save fuzzy taxa
    fuzzy_file = output_dir / "test_fuzzy_taxa.tsv"
    fuzzy_df.to_csv(fuzzy_file, sep='\t', index=False)
    print(f"  Saved fuzzy taxa: {fuzzy_file}")
    print(f"    {len(fuzzy_df)} taxa total")
    print(f"    Matched to OTU: {len(fuzzy_df[fuzzy_df['taxon'].isin(taxon_ids)])}")
    print(f"    Unmatched (for testing): {len(fuzzy_df[~fuzzy_df['taxon'].isin(taxon_ids)])}")
    
    # ====================================================================
    # Summary
    # ====================================================================
    
    print("\n" + "="*70)
    print("TEST DATA GENERATED SUCCESSFULLY")
    print("="*70)
    print(f"\nFiles created in {output_dir}:")
    print(f"  1. test_otu_table.tsv")
    print(f"     - Shape: {otu_df.shape}")
    print(f"     - Format: Samples (rows) x Taxa (columns)")
    print(f"\n  2. test_metadata.tsv")
    print(f"     - Columns: sample_id, disease_status, age, sex, batch")
    print(f"     - Disease: {disease_status.count('disease')}, Control: {disease_status.count('control')}")
    print(f"\n  3. test_fuzzy_taxa.tsv")
    print(f"     - Columns: cluster, taxon, membership, fuzzy_rank")
    print(f"     - {len(fuzzy_df)} taxa (includes test for unmatched taxa)")
    
    print("\n" + "="*70)
    print("QUICK TEST COMMAND:")
    print("="*70)
    print(f"""
python microbiome_validate_fuzzy_taxa.py \\
  --otu_file {otu_file} \\
  --metadata_file {metadata_file} \\
  --metadata_sample_col sample_id \\
  --clinical_col disease_status \\
  --top_taxa_file {fuzzy_file} \\
  --output_dir ./test_output \\
  --n_resamples 10 \\
  --n_permutations 100 \\
  --n_estimators 100 \\
  --alpha 0.05
""")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic test data for microbiome validation pipeline"
    )
    parser.add_argument(
        '--output_dir',
        default='./test_data',
        help='Output directory for test files (default: ./test_data)'
    )
    parser.add_argument(
        '--n_samples',
        type=int,
        default=50,
        help='Number of samples (default: 50)'
    )
    parser.add_argument(
        '--n_taxa',
        type=int,
        default=200,
        help='Number of taxa (default: 200)'
    )
    
    args = parser.parse_args()
    
    generate_test_data(args.output_dir, args.n_samples, args.n_taxa)


if __name__ == '__main__':
    main()

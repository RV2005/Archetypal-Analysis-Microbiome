"""
Clustering analysis utilities for archetypal analysis results.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import scipy.spatial.distance as distance
from skbio.diversity import beta_diversity
from skbio.stats.ordination import pcoa

from ..config import AnalysisConfig


class ClusteringAnalyzer:
    """Clustering analysis on archetypal results"""
    
    def cluster_samples_by_archetypes(self, 
                                    similarity_degree: pd.DataFrame,
                                    n_clusters: Optional[int] = None,
                                    method: str = "kmeans",
                                    random_state: int = 42) -> pd.Series:
        """
        Cluster samples based on their archetype similarity patterns.
        
        Parameters:
        -----------
        similarity_degree : DataFrame
            Similarity degree matrix from archetypal analysis
        n_clusters : int, optional
            Number of clusters (defaults to number of archetypes)
        method : str
            Clustering method to use
        random_state : int
            Random state for reproducibility
            
        Returns:
        --------
        cluster_labels : Series
            Cluster assignments for each sample
        """
        if n_clusters is None:
            n_clusters = similarity_degree.shape[1]  # Use number of archetypes
        
        if method == "kmeans":
            clusterer = KMeans(
                n_clusters=n_clusters, 
                random_state=random_state,
                n_init=10
            )
            labels = clusterer.fit_predict(similarity_degree)
        else:
            raise ValueError(f"Unsupported clustering method: {method}")
        
        return pd.Series(labels, index=similarity_degree.index, name="cluster")
    
    def analyze_archetype_dominance(self, 
                                  similarity_degree: pd.DataFrame,
                                  threshold: float = 0.5) -> pd.DataFrame:
        """
        Analyze which archetype dominates each sample.
        
        Parameters:
        -----------
        similarity_degree : DataFrame
            Similarity degree matrix
        threshold : float
            Minimum similarity threshold for dominance
            
        Returns:
        --------
        dominance_analysis : DataFrame
            Analysis of archetype dominance patterns
        """
        analysis_data = []
        
        for sample in similarity_degree.index:
            sample_similarities = similarity_degree.loc[sample]
            
            # Find dominant archetype
            max_similarity = sample_similarities.max()
            dominant_archetype = sample_similarities.idxmax()
            
            # Check if dominance is strong enough
            is_dominant = max_similarity > threshold
            
            # Calculate entropy (measure of archetype mixing)
            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            similarities_norm = sample_similarities + epsilon
            entropy = -np.sum(similarities_norm * np.log(similarities_norm))
            
            analysis_data.append({
                'sample': sample,
                'dominant_archetype': dominant_archetype,
                'max_similarity': max_similarity,
                'is_dominant': is_dominant,
                'entropy': entropy,
                'n_archetypes_above_threshold': (sample_similarities > threshold).sum()
            })
        
        return pd.DataFrame(analysis_data).set_index('sample')
    
    def find_pure_samples(self, 
                         similarity_degree: pd.DataFrame,
                         purity_threshold: float = 0.8) -> Dict[str, List[str]]:
        """
        Find samples that are 'pure' (strongly associated with one archetype).
        
        Parameters:
        -----------
        similarity_degree : DataFrame
            Similarity degree matrix
        purity_threshold : float
            Minimum similarity for a sample to be considered pure
            
        Returns:
        --------
        pure_samples : Dict[str, List[str]]
            Dictionary mapping archetype names to lists of pure samples
        """
        pure_samples = {}
        
        for archetype in similarity_degree.columns:
            pure_samples[archetype] = []
            
            archetype_similarities = similarity_degree[archetype]
            pure_indices = archetype_similarities[archetype_similarities > purity_threshold].index
            pure_samples[archetype] = pure_indices.tolist()
        
        return pure_samples
    
    def calculate_sample_distances(self, 
                                 similarity_degree: pd.DataFrame,
                                 metric: str = "euclidean") -> pd.DataFrame:
        """
        Calculate pairwise distances between samples in archetype space.
        
        Parameters:
        -----------
        similarity_degree : DataFrame
            Similarity degree matrix
        metric : str
            Distance metric to use
            
        Returns:
        --------
        distance_matrix : DataFrame
            Pairwise distance matrix
        """
        # Calculate distance matrix
        distances = distance.pdist(similarity_degree.values, metric=metric)
        distance_matrix = distance.squareform(distances)
        
        return pd.DataFrame(
            distance_matrix,
            index=similarity_degree.index,
            columns=similarity_degree.index
        )


class PCoAAnalyzer:
    """Principal Coordinate Analysis for microbiome data"""
    
    def perform_pcoa(self, 
                    otu_data: pd.DataFrame,
                    distance_metric: str = "braycurtis",
                    n_dimensions: int = 5) -> Tuple[pd.DataFrame, Any]:
        """
        Perform Principal Coordinate Analysis on OTU data.
        
        Parameters:
        -----------
        otu_data : DataFrame
            OTU data with samples as rows
        distance_metric : str
            Distance metric for beta diversity
        n_dimensions : int
            Number of PCoA dimensions to compute
            
        Returns:
        --------
        pcoa_coordinates : DataFrame
            PCoA coordinates for samples
        pcoa_results : PCoAResults
            Full PCoA results object
        """
        # Prepare data
        pcoa_data = otu_data.copy()
        
        # Apply log transformation
        pcoa_data = pcoa_data.map(lambda x: np.log2(x + 1))
        
        # Remove infinite values
        pcoa_data = pcoa_data.replace([np.inf, -np.inf], np.nan)
        pcoa_data = pcoa_data.dropna()
        
        # Filter samples with no features
        pcoa_data = pcoa_data[(pcoa_data > 0).sum(axis=1) > 0]
        
        print(f"PCoA input shape: {pcoa_data.shape}")
        
        # Calculate beta diversity
        try:
            # Get distance function
            if distance_metric == "jensenshannon":
                distance_func = distance.jensenshannon
            else:
                distance_func = distance_metric
            
            bdiv = beta_diversity(
                distance_func,
                pcoa_data,
                ids=pcoa_data.index,
                pairwise_func=distance.pdist
            )
            
            print(f"Beta diversity matrix computed. NaN count: {np.isnan(bdiv.data).sum()}")
            
            # Perform PCoA
            pcoa_results = pcoa(bdiv, number_of_dimensions=n_dimensions)
            
            # Extract coordinates
            pcoa_coordinates = pcoa_results.samples
            
            return pcoa_coordinates, pcoa_results
            
        except Exception as e:
            print(f"Error in PCoA analysis: {e}")
            raise


class ArchetypeComparator:
    """Compare archetypal analysis results across different conditions"""
    
    def compare_archetype_distributions(self, 
                                      results_dict: Dict[str, pd.DataFrame],
                                      statistical_test: str = "ks") -> pd.DataFrame:
        """
        Compare archetype distributions across different conditions.
        
        Parameters:
        -----------
        results_dict : Dict[str, DataFrame]
            Dictionary mapping condition names to similarity degree matrices
        statistical_test : str
            Statistical test to use for comparison
            
        Returns:
        --------
        comparison_results : DataFrame
            Statistical comparison results
        """
        from scipy import stats
        
        comparison_data = []
        conditions = list(results_dict.keys())
        
        # Compare all pairs of conditions
        for i, cond1 in enumerate(conditions):
            for j, cond2 in enumerate(conditions[i+1:], i+1):
                sim1 = results_dict[cond1]
                sim2 = results_dict[cond2]
                
                # Compare each archetype
                for archetype in sim1.columns:
                    if archetype in sim2.columns:
                        dist1 = sim1[archetype].values
                        dist2 = sim2[archetype].values
                        
                        if statistical_test == "ks":
                            statistic, pvalue = stats.ks_2samp(dist1, dist2)
                            test_name = "Kolmogorov-Smirnov"
                        elif statistical_test == "mannwhitney":
                            statistic, pvalue = stats.mannwhitneyu(dist1, dist2)
                            test_name = "Mann-Whitney U"
                        else:
                            raise ValueError(f"Unsupported test: {statistical_test}")
                        
                        comparison_data.append({
                            'condition1': cond1,
                            'condition2': cond2,
                            'archetype': archetype,
                            'test': test_name,
                            'statistic': statistic,
                            'pvalue': pvalue,
                            'mean1': np.mean(dist1),
                            'mean2': np.mean(dist2),
                            'std1': np.std(dist1),
                            'std2': np.std(dist2),
                            'n1': len(dist1),
                            'n2': len(dist2)
                        })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Add multiple testing correction
        if len(comparison_df) > 0:
            from statsmodels.stats.multitest import multipletests
            _, pvals_corrected, _, _ = multipletests(
                comparison_df['pvalue'], 
                method='fdr_bh'
            )
            comparison_df['pvalue_corrected'] = pvals_corrected
        
        return comparison_df
    
    def calculate_archetype_stability(self, 
                                    similarity_matrices: List[pd.DataFrame]) -> Dict[str, float]:
        """
        Calculate stability of archetypes across multiple analyses.
        
        Parameters:
        -----------
        similarity_matrices : List[DataFrame]
            List of similarity degree matrices from multiple runs
            
        Returns:
        --------
        stability_metrics : Dict[str, float]
            Stability metrics for the archetype solution
        """
        if len(similarity_matrices) < 2:
            return {"stability": 1.0, "n_matrices": len(similarity_matrices)}
        
        # Calculate pairwise correlations between matrices
        correlations = []
        
        for i in range(len(similarity_matrices)):
            for j in range(i + 1, len(similarity_matrices)):
                # Find common samples
                common_samples = similarity_matrices[i].index.intersection(
                    similarity_matrices[j].index
                )
                
                if len(common_samples) > 0:
                    mat1 = similarity_matrices[i].loc[common_samples]
                    mat2 = similarity_matrices[j].loc[common_samples]
                    
                    # Calculate correlation for each archetype
                    for arch in mat1.columns:
                        if arch in mat2.columns:
                            corr = np.corrcoef(mat1[arch], mat2[arch])[0, 1]
                            if not np.isnan(corr):
                                correlations.append(corr)
        
        if correlations:
            return {
                "mean_stability": np.mean(correlations),
                "min_stability": np.min(correlations),
                "max_stability": np.max(correlations),
                "std_stability": np.std(correlations),
                "n_comparisons": len(correlations),
                "n_matrices": len(similarity_matrices)
            }
        else:
            return {"stability": 0.0, "n_matrices": len(similarity_matrices)}

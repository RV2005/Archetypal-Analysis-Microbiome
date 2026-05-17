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


class ClusteringAnalyzer: # UNTESTED
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


"""
Analysis metrics and evaluation utilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from tqdm import tqdm

from ..config import DatasetConfig

METRIC_FUNCTIONS = {
    "silhouette": silhouette_score,
    "davies_bouldin": davies_bouldin_score,
    "calinski_harabasz": calinski_harabasz_score,
    "final_loss": lambda x,y: np.nan,
}

class AnalysisMetrics:
    """Calculate various analysis metrics"""
    
    def calc_scores(self, 
                   otu: Union[pd.DataFrame, np.ndarray], 
                   labels: Union[pd.Series, np.ndarray], 
                   discretize: bool = False, 
                   buckets: Optional[List[Tuple[float, float]]] = None, 
                   metrics: List[str] = None) -> Dict[str, float]:
        """
        Calculate clustering evaluation metrics.
        
        Parameters:
        -----------
        otu : DataFrame or ndarray
            The data matrix (samples x features)
        labels : Series or ndarray
            Sample labels for evaluation
        discretize : bool
            Whether to discretize continuous labels into buckets
        buckets : List[Tuple[float, float]], optional
            Bucket ranges for discretization
            
        Returns:
        --------
        scores : Dict[str, float]
            Dictionary containing evaluation metrics
        """
        # Convert to numpy arrays
        if isinstance(otu, pd.DataFrame):
            otu_array = otu.to_numpy()
        else:
            otu_array = otu
            
        if isinstance(labels, pd.Series):
            labels_array = labels.to_numpy()
        else:
            labels_array = labels.copy()
        
        # Discretize labels if requested
        if discretize:
            assert buckets is not None, "If discretizing, tuple buckets must be provided."
            labels_array = self._discretize_labels(labels_array, buckets)
        
        # Handle string labels
        if labels_array.dtype == object or labels_array.dtype.kind == 'U':
            # Convert string labels to numeric
            unique_labels = np.unique(labels_array)
            label_map = {label: i for i, label in enumerate(unique_labels)}
            labels_numeric = np.array([label_map[label] for label in labels_array])
        else:
            labels_numeric = labels_array
        
        if metrics is None:
            metrics = list(METRIC_FUNCTIONS.keys())
        
        # Calculate metrics
        scores = {}

        for metric in metrics:
            try:
                scores[metric] = METRIC_FUNCTIONS[metric](otu_array, labels_numeric)
            except KeyError as e:
                print(f"Metric {metric} is not recognized. Available metrics: {list(METRIC_FUNCTIONS.keys())}")
                scores[metric] = np.nan
            except Exception as e:
                print(f"Warning: Error calculating {metric}: {e}")
                scores[metric] = np.nan
        
        return scores
    
    def _discretize_labels(self, 
                          labels: np.ndarray, 
                          buckets: List[Tuple[float, float]]) -> np.ndarray:
        """Discretize continuous labels into buckets"""
        new_labels = []
        
        for label in labels:
            if pd.isna(label):
                new_labels.append("Unknown")
                continue
                
            bucket_found = False
            for bucket_min, bucket_max in buckets:
                if bucket_min <= label < bucket_max:
                    if bucket_max == float('inf'):
                        new_labels.append(f"{bucket_min}+")
                    else:
                        new_labels.append(f"{bucket_min}-{bucket_max}")
                    bucket_found = True
                    break
            
            if not bucket_found:
                new_labels.append("Unknown")
        
        return np.array(new_labels)
    
    def compare_archetype_quality(self, 
                                 results_dict: Dict[int, Dict],
                                 metadata: pd.DataFrame,
                                 metadata_column: str,
                                 discretize: bool = False,
                                 buckets: Optional[List[Tuple[float, float]]] = None, 
                                 metrics: List[str] = None, 
                                 drop_nan=False) -> pd.DataFrame:
        """
        Compare archetype quality across different K values.
        
        Parameters:
        -----------
        results_dict : Dict[int, Dict]
            Dictionary with K values as keys and AA results as values
        otu_data : DataFrame
            Original OTU data
        metadata : DataFrame
            Sample metadata
        metadata_column : str
            Column to use for evaluation
        discretize : bool
            Whether to discretize labels
        buckets : List[Tuple[float, float]], optional
            Bucket ranges for discretization
        metrics : List[str], optional
            Metrics to calculate: from 'calinski_harabasz', 'silhouette', 'davies_bouldin', 'final_loss' (default: all)

        Returns:
        --------
        comparison : DataFrame
            Comparison metrics for different K values
        """
        comparison_data = []
        
        for k, result_dict in tqdm(results_dict.items(), desc="Evaluating K"):
            # Get similarity degree data
            similarity_degree = pd.DataFrame(
                result_dict["similarity_degree"],
                index=result_dict["index"],
                columns=result_dict["columns"]
            )
            
            # Get labels for evaluation
            common_samples = similarity_degree.index.intersection(metadata.index)
            # Drop samples with NaN labels if requested
            if drop_nan:
                common_samples = common_samples[~metadata.loc[common_samples, metadata_column].isna()]
            
            eval_similarity = similarity_degree.loc[common_samples]
            eval_labels = metadata.loc[common_samples, metadata_column]
            
            # Calculate scores
            scores = self.calc_scores(
                eval_similarity,
                eval_labels,
                discretize=discretize,
                buckets=buckets,
                metrics=metrics
            )
            
            scores['k'] = k
            if 'final_loss' in metrics:
                scores['final_loss'] = result_dict.get('loss', np.nan)[-1]

            comparison_data.append(scores)
        
        comparison_df = pd.DataFrame(comparison_data)
        return comparison_df.sort_values('k')

    def plot_comparison(self, 
                     comparison_df: pd.DataFrame, 
                     metrics: List[str] = None, 
                     figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Plot comparison metrics for different K values. Creates a separate graph for each metric in subplots.
        
        Parameters:
        -----------
        comparison_df : DataFrame
            Comparison metrics DataFrame
        metrics : List[str], optional
            Metrics to plot (default: all)
        figsize : Tuple[int, int]
            Figure size for the plot
        """
        import matplotlib.pyplot as plt
        if metrics is None:
            metrics = [col for col in comparison_df.columns if col not in ['k']]
        n_metrics = len(metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=figsize, sharex=True)
        if n_metrics == 1:
            axes = [axes]

        for ax, metric in zip(axes, metrics):
            ax.plot(comparison_df['k'], comparison_df[metric], marker='o', label=metric)
            ax.set_ylabel('Score')
            ax.set_title(f'Archetype Quality Comparison - {metric}')
            ax.legend()
            ax.grid()

        plt.xlabel('Number of Archetypes (K)')
        plt.tight_layout()
        plt.show()

class LossAnalyzer:
    """Analyze loss curves and optimization behavior"""
    
    def analyze_loss_curves(self, 
                           results_directory: str, 
                           k_values: List[int],
                           file_pattern: str = "otu_AA_K{k}_*.json") -> pd.DataFrame:
        """
        Analyze loss curves across different K values from saved results.
        
        Parameters:
        -----------
        results_directory : str
            Directory containing result files
        k_values : List[int]
            K values to analyze
        file_pattern : str
            Pattern for result filenames
            
        Returns:
        --------
        loss_analysis : DataFrame
            Loss analysis results
        """
        import glob
        import json
        import os
        
        loss_data = []
        
        for k in k_values:
            pattern = file_pattern.format(k=k)
            file_path = os.path.join(results_directory, pattern)
            matching_files = glob.glob(file_path)
            
            if not matching_files:
                print(f"Warning: No files found for K={k} with pattern {pattern}")
                continue
            
            # Use the first matching file
            file_path = matching_files[0]
            
            try:
                with open(file_path, 'r') as f:
                    result_dict = json.load(f)
                
                loss_data.append({
                    'k': k,
                    'loss': result_dict.get('loss', np.nan),
                    'n_samples': len(result_dict.get('index', [])),
                    'n_features': len(result_dict.get('archetypes', [[]])[0]) if result_dict.get('archetypes') else 0,
                    'filename': os.path.basename(file_path)
                })
                
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                continue
        
        loss_df = pd.DataFrame(loss_data).sort_values('k')
        
        # Calculate additional metrics
        if len(loss_df) > 1:
            loss_df['loss_diff'] = loss_df['loss'].diff()
            loss_df['loss_pct_change'] = loss_df['loss'].pct_change() * 100
            
            # Calculate relative improvement
            loss_df['relative_improvement'] = (loss_df['loss'].shift(1) - loss_df['loss']) / loss_df['loss'].shift(1) * 100
        
        return loss_df
    
    def find_optimal_k(self, 
                      loss_df: pd.DataFrame,
                      method: str = "elbow") -> int:
        """
        Find optimal K value using different methods.
        
        Parameters:
        -----------
        loss_df : DataFrame
            Loss analysis DataFrame from analyze_loss_curves
        method : str
            Method to use: "elbow", "min_improvement", or "min_loss"
            
        Returns:
        --------
        optimal_k : int
            Optimal K value
        """
        if method == "min_loss":
            return int(loss_df.loc[loss_df['loss'].idxmin(), 'k'])
        
        elif method == "elbow":
            return self._find_elbow_point(loss_df)
        
        elif method == "min_improvement":
            # Find where improvement becomes minimal
            if 'relative_improvement' in loss_df.columns:
                # Find K where improvement drops below threshold (e.g., 5%)
                threshold = 5.0
                below_threshold = loss_df[loss_df['relative_improvement'] < threshold]
                if not below_threshold.empty:
                    return int(below_threshold.iloc[0]['k'])
            
            # Fallback to elbow method
            return self._find_elbow_point(loss_df)
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _find_elbow_point(self, loss_df: pd.DataFrame) -> int:
        """Find elbow point in loss curve"""
        if len(loss_df) < 3:
            return int(loss_df.iloc[0]['k'])
        
        # Calculate second derivatives
        loss_values = loss_df['loss'].values
        k_values = loss_df['k'].values
        
        second_derivatives = []
        for i in range(1, len(loss_values) - 1):
            second_deriv = loss_values[i-1] - 2*loss_values[i] + loss_values[i+1]
            second_derivatives.append(second_deriv)
        
        # Find K with maximum second derivative (sharpest bend)
        if second_derivatives:
            max_idx = np.argmax(second_derivatives)
            return int(k_values[max_idx + 1])  # +1 because second derivatives start at index 1
        else:
            return int(k_values[0])


class ArchetypeStabilityAnalyzer:
    """Analyze stability of archetypal solutions across runs"""
    
    def analyze_stability(self, 
                         results_list: List[Dict],
                         similarity_threshold: float = 0.95) -> Dict[str, float]:
        """
        Analyze stability of archetypal solutions across multiple runs.
        
        Parameters:
        -----------
        results_list : List[Dict]
            List of archetypal analysis results from multiple runs
        similarity_threshold : float
            Threshold for considering archetypes as stable
            
        Returns:
        --------
        stability_metrics : Dict[str, float]
            Dictionary containing stability metrics
        """
        if len(results_list) < 2:
            return {"stability_score": 1.0, "n_runs": len(results_list)}
        
        # Extract archetypes from all runs
        all_archetypes = [np.array(result["archetypes"]) for result in results_list]
        
        # Calculate pairwise correlations
        correlations = []
        for i in range(len(all_archetypes)):
            for j in range(i + 1, len(all_archetypes)):
                corr_matrix = self._calculate_archetype_correlation(
                    all_archetypes[i], 
                    all_archetypes[j]
                )
                # Get maximum correlation for each archetype (best match)
                max_correlations = np.max(corr_matrix, axis=1)
                correlations.extend(max_correlations)
        
        # Calculate stability metrics
        stability_score = np.mean(correlations)
        stable_fraction = np.mean(np.array(correlations) > similarity_threshold)
        
        return {
            "stability_score": stability_score,
            "stable_fraction": stable_fraction,
            "min_correlation": np.min(correlations),
            "max_correlation": np.max(correlations),
            "n_runs": len(results_list),
            "n_comparisons": len(correlations)
        }
    
    def _calculate_archetype_correlation(self, 
                                       archetypes1: np.ndarray, 
                                       archetypes2: np.ndarray) -> np.ndarray:
        """Calculate correlation matrix between two sets of archetypes"""
        n_archetypes1, n_features = archetypes1.shape
        n_archetypes2 = archetypes2.shape[0]
        
        corr_matrix = np.zeros((n_archetypes1, n_archetypes2))
        
        for i in range(n_archetypes1):
            for j in range(n_archetypes2):
                corr_matrix[i, j] = np.corrcoef(archetypes1[i], archetypes2[j])[0, 1]
        
        return corr_matrix

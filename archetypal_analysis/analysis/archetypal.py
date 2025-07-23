"""
Core archetypal analysis workflow and utilities.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union

from archetypes import AA
from archetypes.datasets import sort_by_archetype_similarity

from ..config import AnalysisConfig, FilePatterns
from ..utils.file_io import ResultsManager
from ..data.formatters import DataTransformer


class ArchetypalAnalysisResult:
    """Container for archetypal analysis results"""
    
    def __init__(self,
                 similarity_degree: pd.DataFrame,
                 archetypes: np.ndarray,
                 labels: np.ndarray,
                 loss: float,
                 info: Dict[str, Any],
                 original_data: Optional[pd.DataFrame] = None):
        self.similarity_degree = similarity_degree
        self.archetypes = archetypes
        self.labels = labels
        self.loss = loss
        self.info = info
        self.original_data = original_data
    
    @property
    def n_archetypes(self) -> int:
        """Number of archetypes"""
        return len(self.archetypes)
    
    @property
    def n_samples(self) -> int:
        """Number of samples"""
        return len(self.similarity_degree)
    
    @property
    def archetype_names(self) -> List[str]:
        """Get archetype column names"""
        return list(self.similarity_degree.columns)
    
    def get_archetype_composition(self, k: int = 5) -> List[Tuple[float, str]]:
        """Get top K contributing features for each archetype
        Returns: 
        top_taxa_archs : List[Tuple[float, str]]
            List of top contributing taxa for each archetype as [(contribution, taxon), ...], where contribution is 0-1
        """
        if self.original_data is None:
            raise ValueError("Original data not available for composition analysis")
        
        # Get top K taxa for each archetype
        top_taxa_archs = []
        for arch_name in self.archetype_names:
            arch_values = self.similarity_degree[arch_name]
            top_k = arch_values.nlargest(k)
            for taxon, contribution in top_k.items():
                top_taxa_archs.append((contribution, taxon))

        return top_taxa_archs

class ArchetypalAnalysis:
    """Main archetypal analysis workflow"""
    
    def __init__(self,
                 n_archetypes: int = AnalysisConfig.DEFAULT_N_ARCHETYPES,
                 method: str = AnalysisConfig.DEFAULT_METHOD,
                 max_iter: int = AnalysisConfig.DEFAULT_MAX_ITER,
                 random_state: int = AnalysisConfig.DEFAULT_RANDOM_STATE,
                 **method_kwargs):
        """
        Initialize archetypal analyzer.
        
        Parameters:
        -----------
        n_archetypes : int
            Number of archetypes to find
        method : str
            Optimization method for archetypal analysis
        max_iter : int
            Maximum iterations for optimization
        random_state : int
            Random seed for reproducibility
        **method_kwargs : dict
            Additional arguments passed to the AA method
        """
        self.n_archetypes = n_archetypes
        self.method = method
        self.max_iter = max_iter
        self.random_state = random_state
        
        # Prepare method kwargs
        self.method_kwargs = method_kwargs.copy()
        self.method_kwargs.setdefault("max_iter_optimizer", max_iter)
        
        # Initialize components
        self.results_manager = ResultsManager()
        self.data_transformer = DataTransformer()
        
        # Analysis state
        self._aa_model = None
        self._last_result = None
    
    def fit_transform(self, 
                     otu_data: pd.DataFrame,
                     apply_log_transform: bool = True,
                     sample_fraction: float = 1.0,
                     verbose: bool = True) -> ArchetypalAnalysisResult:
        """
        Fit archetypal analysis and return results.
        
        Parameters:
        -----------
        otu_data : pd.DataFrame
            OTU data with samples as rows, features as columns
        apply_log_transform : bool
            Whether to apply log2 transformation
        sample_fraction : float
            Fraction of samples to use (for large datasets)
        verbose : bool
            Print progress information
            
        Returns:
        --------
        result : ArchetypalAnalysisResult
            Analysis results container
        """
        if verbose:
            print(f"Starting archetypal analysis with {self.n_archetypes} archetypes...")
            print(f"Input data shape: {otu_data.shape}")
        
        # Prepare data
        processed_data = self._prepare_data(otu_data, apply_log_transform, sample_fraction)
        
        if verbose:
            print(f"Processed data shape: {processed_data.shape}")
        
        # Fit archetypal analysis
        self._fit_aa_model(processed_data, verbose)
        
        # Create results
        result = self._create_result_object(processed_data, otu_data)
        self._last_result = result
        
        if verbose:
            if len(result.loss) > 0:
                print(f"Analysis completed. Final loss: {result.loss[-1]:.6f}")
            else:
                print("Analysis completed.")
        return result
    
    def _prepare_data(self, 
                     otu_data: pd.DataFrame,
                     apply_log_transform: bool,
                     sample_fraction: float) -> pd.DataFrame:
        """Prepare data for archetypal analysis"""
        # Sample data if needed
        if sample_fraction < 1.0:
            otu_data = self.data_transformer.sample_fraction(otu_data, sample_fraction)
        
        # Apply log transformation
        if apply_log_transform:
            otu_data = self.data_transformer.log_transform(otu_data, base=2, pseudocount=1.0)
        
        # Handle infinite values
        otu_data = self.data_transformer.remove_infinite_values(otu_data)
        
        return otu_data
    
    def _fit_aa_model(self, data: pd.DataFrame, verbose: bool = True) -> None:
        """Fit the archetypal analysis model"""
        # Convert to numpy array
        aa_data = data.to_numpy()
        
        # Initialize model
        self._aa_model = AA(
            n_archetypes=self.n_archetypes,
            method=self.method,
            method_kwargs=self.method_kwargs,
            random_state=self.random_state
        )
        
        # Fit model
        if verbose:
            print("Fitting archetypal analysis model...")
        
        self._aa_model.fit(aa_data)
        
        if verbose:
            print("Model fitting completed.")
    
    def _create_result_object(self, 
                             processed_data: pd.DataFrame,
                             original_data: pd.DataFrame) -> ArchetypalAnalysisResult:
        """Create result object from fitted model"""
        # Create archetype names
        archetype_names = [f"Archetype {i+1}" for i in range(self.n_archetypes)]
        
        # Create similarity degree DataFrame
        similarity_degree = pd.DataFrame(
            self._aa_model.similarity_degree_,
            index=processed_data.index,
            columns=archetype_names
        )
        
        # Sort by archetype similarity for better visualization
        data_sorted, info = sort_by_archetype_similarity(
            processed_data.to_numpy(),
            [self._aa_model.similarity_degree_],
            self._aa_model.archetypes_
        )
        
        return ArchetypalAnalysisResult(
            similarity_degree=similarity_degree,
            archetypes=self._aa_model.archetypes_,
            labels=self._aa_model.labels_,
            loss=self._aa_model.loss_,
            info={
                "perms": info["perms"][0].tolist() if info["perms"] else [],
                "scores": info["scores"][0].tolist() if info["scores"] else [],
                "alphas": info["alphas"][0].tolist() if info["alphas"] else []
            },
            original_data=original_data
        )
    
    def save_results(self, 
                    result: ArchetypalAnalysisResult,
                    filepath: str,
                    group: str = "None",
                    method_name: Optional[str] = None,
                    frac: float = 1.0) -> None:
        """
        Save results to JSON format.
        
        Parameters:
        -----------
        result : ArchetypalAnalysisResult
            Results to save
        filepath : str
            Path to save results
        group : str
            Grouping identifier for filename
        method_name : str, optional
            Method name for filename (uses distance method if available)
        frac : float
            Fraction used in analysis for filename
        """
        if method_name is None:
            method_name = "jensenshannon"
        
        # Generate filename if only directory provided
        if os.path.isdir(filepath):
            filename = FilePatterns.get_aa_filename(
                k=self.n_archetypes,
                frac=frac,
                group=group,
                method=method_name
            )
            filepath = os.path.join(filepath, filename)
        
        self.results_manager.save_archetypal_results(
            similarity_degree=result.similarity_degree.to_numpy(),
            archetypes=result.archetypes,
            labels=result.labels,
            index=result.similarity_degree.index.tolist(),
            columns=result.archetype_names,
            loss=result.loss,
            info=result.info,
            filepath=filepath
        )
    
    def load_results(self, filepath: str) -> ArchetypalAnalysisResult:
        """Load results from JSON format"""
        data = self.results_manager.load_archetypal_results(filepath)
        
        similarity_degree = pd.DataFrame(
            data["similarity_degree"],
            index=data["index"],
            columns=data["columns"]
        )
        
        info = {
            "perms": data.get("info_perms", []),
            "scores": data.get("info_scores", []),
            "alphas": data.get("info_alphas", [])
        }
        
        return ArchetypalAnalysisResult(
            similarity_degree=similarity_degree,
            archetypes=np.array(data["archetypes"]),
            labels=np.array(data["labels"]),
            loss=data["loss"],
            info=info
        )
    
    def fit_multiple_k(self,
                      otu_data: pd.DataFrame,
                      k_values: List[int],
                      output_dir: Optional[str] = None,
                      **fit_kwargs) -> Dict[int, ArchetypalAnalysisResult]:
        """
        Fit archetypal analysis for multiple K values.
        
        Parameters:
        -----------
        otu_data : pd.DataFrame
            OTU data to analyze
        k_values : List[int]
            List of K values to test
        output_dir : str, optional
            Directory to save results
        **fit_kwargs : dict
            Additional arguments for fit_transform
            
        Returns:
        --------
        results : Dict[int, ArchetypalAnalysisResult]
            Results for each K value
        """
        results = {}
        
        for k in k_values:
            print(f"\n--- Fitting K = {k} ---")
            
            # Create new analyzer for this K
            analyzer = ArchetypalAnalysis(
                n_archetypes=k,
                method=self.method,
                max_iter=self.max_iter,
                random_state=self.random_state,
                **self.method_kwargs
            )
            
            # Fit model
            result = analyzer.fit_transform(otu_data, **fit_kwargs)
            results[k] = result
            
            # Save if output directory provided
            if output_dir:
                analyzer.save_results(result, output_dir)
        
        return results


class ArchetypeSelector:
    """Utilities for selecting optimal number of archetypes"""
    
    @staticmethod
    def analyze_loss_curves(results: Dict[int, ArchetypalAnalysisResult]) -> pd.DataFrame:
        """Analyze loss curves across different K values"""
        analysis_data = []
        
        for k, result in results.items():
            analysis_data.append({
                'k': k,
                'loss': result.loss,
                'n_samples': result.n_samples
            })
        
        analysis_df = pd.DataFrame(analysis_data).sort_values('k')
        
        # Calculate loss differences
        analysis_df['loss_diff'] = analysis_df['loss'].diff()
        analysis_df['loss_pct_change'] = analysis_df['loss'].pct_change() * 100
        
        return analysis_df
    
    @staticmethod
    def find_elbow_point(loss_values: List[float], k_values: List[int]) -> int:
        """Find elbow point in loss curve using simple heuristic"""
        # Calculate second derivatives
        if len(loss_values) < 3:
            return k_values[0] if k_values else 3
        
        second_derivatives = []
        for i in range(1, len(loss_values) - 1):
            second_deriv = loss_values[i-1] - 2*loss_values[i] + loss_values[i+1]
            second_derivatives.append(second_deriv)
        
        # Find K with maximum second derivative (sharpest bend)
        max_idx = np.argmax(second_derivatives)
        return k_values[max_idx + 1]  # +1 because second derivatives start at index 1

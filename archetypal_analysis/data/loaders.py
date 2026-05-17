"""
Data loading utilities for archetypal analysis.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

from archetypal_analysis.analysis.archetypal import ArchetypalAnalysisResult

from ..config import PathConfig, FilePatterns


class BaseDataLoader:
    """Base class for data loading utilities"""
    
    def __init__(self, base_path: Optional[str] = None):
        self.path_config = PathConfig(base_path)

    def load_df(self, filename: str, fileDir: Optional[str] = None, index_col=0) -> pd.DataFrame:
        """Load a file from the compendium directory"""
        totPath = os.path.join(fileDir, filename) if fileDir else filename
        return self._load_csv_or_tsv(totPath, index_col=index_col)


    def _load_csv_or_tsv(self, filepath: str, index_col: Union[int, str] = 0, **kwargs) -> pd.DataFrame:
        """Load CSV or TSV file with automatic format detection"""
        if filepath.endswith('.tsv'):
            return pd.read_table(filepath, index_col=index_col, **kwargs)
        elif filepath.endswith('.csv'):
            return pd.read_csv(filepath, index_col=index_col, **kwargs)
        else:
            # Try CSV first, then TSV
            try:
                return pd.read_csv(filepath, index_col=index_col, **kwargs)
            except:
                return pd.read_table(filepath, index_col=index_col, **kwargs)


class CompendiumDataLoader(BaseDataLoader):
    """Load compendium datasets (OTU, metadata, projects, tags)"""
    def load_otu_table(self, filename: str, fileDir: Optional[str] = None) -> pd.DataFrame:
        """Load OTU table from specified file"""
        df = self.load_df(filename=filename, fileDir=fileDir, index_col="sample")
        # Drop unnamed columns if they exist
        unnamed_cols = [col for col in df.columns if col.startswith('Unnamed:')]
        if unnamed_cols:
            df = df.drop(columns=unnamed_cols)
        return df
        
    def load_all(self, rootDir: Optional[str] = None, 
                otu_file: Optional[str] = None, 
                projects_file: Optional[str] = None, 
                tags_file: Optional[str] = None, 
                meta_file: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load all compendium datasets
        Returns:
        - otu: OTU table DataFrame
        - projects: Projects DataFrame
        - tags: Tags DataFrame
        - meta: Metadata DataFrame
        """


        projects = self.load_df(filename=projects_file, fileDir=rootDir, index_col=0) if projects_file else None
        meta = self.load_df(filename=meta_file, fileDir=rootDir, index_col=0) if meta_file else None
        tags = self.load_df(filename=tags_file, fileDir=rootDir) if tags_file else None
        otu = self.load_otu_table(filename=otu_file, fileDir=rootDir) if otu_file else None
        
        return otu, projects, tags, meta


class HadzaDataLoader(BaseDataLoader):

    """Load Hadza-specific datasets"""
    
    def load_otu_table(self, filename: str = "hadza_otu_formatted.tsv", fileDir: Optional[str] = None) -> pd.DataFrame:
        """Load Hadza OTU table"""
        return self.load_df(filename=filename, fileDir=fileDir, index_col=0)
    
    def load_all(self, rootDir: Optional[str] = None, 
                otu_file: Optional[str] = None, 
                meta_file: Optional[str] = None, 
                tax_file: Optional[str] = None, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load all formatted Hadza datasets"""
        otu = self.load_df(filename=otu_file, fileDir=rootDir, index_col=0) if otu_file else None
        meta = self.load_df(filename=meta_file, fileDir=rootDir, index_col=0) if meta_file else None
        tax = self.load_df(filename=tax_file, fileDir=rootDir, index_col=0) if tax_file else None
        return otu, meta, tax


class ArchetypalResultLoader(BaseDataLoader):
    """Load saved archetypal analysis results from JSON"""
    
    def load_result(self, filepath: str) -> Dict:
        """Load archetypal analysis result from JSON file"""
        if not os.path.isabs(filepath):
            # Try different locations
            possible_paths = [
                os.path.join(self.path_config.results_dir, filepath),
                os.path.join(self.path_config.base_path, filepath),
                filepath
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    filepath = path
                    break
            else:
                raise FileNotFoundError(f"Could not find result file: {filepath}")
        
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def load_parsed_result(self, filepath: str) -> Tuple[pd.DataFrame, np.ndarray, List, Dict]:
        """
        Load result and convert to convenient formats
        Returns:
        - otu_aa: DataFrame of similarity degree
        - aa_labels: np.ndarray of sample labels
        - aa_archetypes: np.ndarray of archetype vectors
        - info: Dict with additional information (perms, scores, alphas)
        """
        otu_dict = self.load_result(filepath)
        
        # Convert similarity degree to DataFrame
        otu_aa = pd.DataFrame(
            otu_dict["similarity_degree"], 
            index=otu_dict["index"], 
            columns=otu_dict["columns"]
        )
        
        # Extract other components
        aa_labels = np.array(otu_dict["labels"])
        aa_archetypes = otu_dict["archetypes"]
        info = {
            "perms": otu_dict.get("info_perms", []),
            "scores": otu_dict.get("info_scores", []),
            "alphas": otu_dict.get("info_alphas", [])
        }
        
        return otu_aa, aa_labels, aa_archetypes, info
    
    def find_results_by_pattern(self, pattern: str, directory: Optional[str] = None) -> List[str]:
        """Find result files matching a pattern"""
        import glob
        
        if directory is None:
            directory = self.path_config.results_dir
            
        search_pattern = os.path.join(directory, pattern)
        return glob.glob(search_pattern)
    
    def load_multiple_results(self, k_values: List[int], 
                            directory: str, 
                            frac: float = 1.0,
                            method: str = "jensenshannon") -> Dict[int, Dict]:
        """Load multiple results for different K values"""
        results = {}
        
        for k in k_values:
            filename = FilePatterns.get_aa_filename(k, frac, "None", method)
            filepath = os.path.join(directory, filename)
            
            try:
                results[k] = self.load_result(filepath)
            except FileNotFoundError:
                print(f"Warning: Could not find result for K={k} at {filepath}")
                continue
                
        return results

    def load_result_as_obj(self, filepath: str, original_data: Optional[pd.DataFrame]=None) -> ArchetypalAnalysisResult:
        """Load archetypal analysis result from JSON file and convert to object"""
        otu_aa, aa_labels, aa_archetypes, info = self.load_parsed_result(filepath)
        return ArchetypalAnalysisResult(
            similarity_degree=otu_aa,
            archetypes=aa_archetypes,
            labels=aa_labels,
            loss=info.get("loss", 0.0),
            info=info,
            original_data=original_data
        )
"""
File I/O utilities for archetypal analysis.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import pickle

from ..config import FilePatterns


class ResultsManager:
    """Manage saving and loading of analysis results"""
    
    def __init__(self, base_path: Optional[str] = None):
        self.base_path = base_path or os.getcwd()
    
    def save_archetypal_results(self,
                               similarity_degree: np.ndarray,
                               archetypes: np.ndarray,
                               labels: np.ndarray,
                               index: List[str],
                               columns: List[str],
                               loss: float,
                               info: Dict[str, Any],
                               filepath: str) -> None:
        """
        Save archetypal analysis results to JSON file.
        
        Parameters:
        -----------
        similarity_degree : np.ndarray
            Similarity degree matrix
        archetypes : np.ndarray
            Archetype vectors
        labels : np.ndarray
            Sample labels
        index : List[str]
            Sample indices
        columns : List[str]
            Archetype column names
        loss : float
            Final loss value
        info : Dict[str, Any]
            Additional information (perms, scores, alphas)
        filepath : str
            Path to save the results
        """
        aa_dict = {
            "archetypes": archetypes.tolist(),
            "similarity_degree": similarity_degree.tolist(),
            "labels": labels.tolist(),
            "index": index,
            "columns": columns,
            "loss": loss,
            "info_perms": info.get("perms", []),
            "info_scores": info.get("scores", []),
            "info_alphas": info.get("alphas", [])
        }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(aa_dict, f, indent=2)
        
        print(f"Saved archetypal analysis results to {filepath}")
    
    def load_archetypal_results(self, filepath: str) -> Dict[str, Any]:
        """Load archetypal analysis results from JSON file"""
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def save_dataframe(self, 
                      df: pd.DataFrame, 
                      filepath: str, 
                      format: str = "csv") -> None:
        """Save DataFrame to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if format.lower() == "csv":
            df.to_csv(filepath)
        elif format.lower() == "tsv":
            df.to_csv(filepath, sep="\t")
        elif format.lower() == "pickle":
            df.to_pickle(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Saved DataFrame to {filepath}")
    
    def load_dataframe(self, filepath: str, **kwargs) -> pd.DataFrame:
        """Load DataFrame from file with automatic format detection"""
        if filepath.endswith('.pickle') or filepath.endswith('.pkl'):
            return pd.read_pickle(filepath)
        elif filepath.endswith('.tsv'):
            return pd.read_table(filepath, **kwargs)
        else:
            return pd.read_csv(filepath, **kwargs)


class FASTAWriter:
    """Write sequences in FASTA format"""
    
    @staticmethod
    def write_taxa_fasta(taxa_list: List[str], 
                        output_path: str,
                        mode: str = "w") -> None:
        """
        Write taxa names to FASTA format file.
        
        Parameters:
        -----------
        taxa_list : List[str]
            List of taxa names to write
        output_path : str
            Path to output FASTA file
        mode : str
            File write mode ("w" for write, "a" for append)
        """
        with open(output_path, mode) as f:
            for taxa in taxa_list:
                if taxa != "Others":
                    f.write(f">{taxa}\n")
                    f.write(f"{taxa}\n")
        
        print(f"Written {len([t for t in taxa_list if t != 'Others'])} sequences to {output_path}")


class ConfigManager:
    """Manage analysis configuration files"""
    
    @staticmethod
    def save_config(config: Dict[str, Any], filepath: str) -> None:
        """Save analysis configuration to JSON file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        json_config = ConfigManager._prepare_for_json(config)
        
        with open(filepath, 'w') as f:
            json.dump(json_config, f, indent=2)
        
        print(f"Saved configuration to {filepath}")
    
    @staticmethod
    def load_config(filepath: str) -> Dict[str, Any]:
        """Load analysis configuration from JSON file"""
        with open(filepath, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def _prepare_for_json(obj: Any) -> Any:
        """Recursively prepare object for JSON serialization"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: ConfigManager._prepare_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [ConfigManager._prepare_for_json(item) for item in obj]
        else:
            return obj


class PathUtils:
    """Path manipulation utilities"""
    
    @staticmethod
    def ensure_directory(filepath: str) -> str:
        """Ensure directory exists for the given filepath"""
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
        return filepath
    
    @staticmethod
    def get_available_filename(filepath: str) -> str:
        """Get available filename by adding suffix if file exists"""
        if not os.path.exists(filepath):
            return filepath
        
        base, ext = os.path.splitext(filepath)
        counter = 1
        
        while os.path.exists(f"{base}_{counter}{ext}"):
            counter += 1
        
        return f"{base}_{counter}{ext}"
    
    @staticmethod
    def find_files(directory: str, pattern: str) -> List[str]:
        """Find files matching pattern in directory"""
        import glob
        search_pattern = os.path.join(directory, pattern)
        return glob.glob(search_pattern)


class DataExporter:
    """Export analysis results in various formats"""
    
    def __init__(self, results_manager: Optional[ResultsManager] = None):
        self.results_manager = results_manager or ResultsManager()
    
    def export_metadata_summary(self, 
                               metadata: pd.DataFrame,
                               output_path: str) -> None:
        """Export metadata summary to TSV file"""
        # Generate value counts for each column
        summary_data = []
        
        for column in metadata.columns:
            col_data = metadata[column]
            value_counts = col_data.value_counts()
            
            for value, count in value_counts.items():
                summary_data.append({
                    'column': column,
                    'value': value,
                    'count': count,
                    'percentage': count / len(metadata) * 100
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_path, sep="\t", index=False)
        print(f"Exported metadata summary to {output_path}")
    
    def export_archetype_composition(self,
                                   archetypes: np.ndarray,
                                   feature_names: List[str],
                                   archetype_names: List[str],
                                   output_path: str,
                                   top_n: int = 10) -> None:
        """Export top contributing features per archetype"""
        composition_data = []
        
        for i, (archetype, name) in enumerate(zip(archetypes, archetype_names)):
            # Get top features for this archetype
            top_indices = np.argsort(archetype)[-top_n:][::-1]
            
            for rank, idx in enumerate(top_indices, 1):
                composition_data.append({
                    'archetype': name,
                    'rank': rank,
                    'feature': feature_names[idx],
                    'weight': archetype[idx],
                    'percentage': archetype[idx] / archetype.sum() * 100
                })
        
        composition_df = pd.DataFrame(composition_data)
        composition_df.to_csv(output_path, index=False)
        print(f"Exported archetype composition to {output_path}")

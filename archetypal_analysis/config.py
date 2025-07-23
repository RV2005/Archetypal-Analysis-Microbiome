"""
Configuration and constants for archetypal analysis.
"""

import os
from typing import List, Dict, Tuple, Optional

class AnalysisConfig:
    """Central configuration for analysis parameters"""
    
    # Default archetypal analysis parameters
    DEFAULT_N_ARCHETYPES = 5
    DEFAULT_METHOD = "pgd"
    DEFAULT_MAX_ITER = 10
    DEFAULT_RANDOM_STATE = 3
    
    # Quality control defaults
    DEFAULT_MIN_TAXA_LEVEL = 3
    DEFAULT_MIN_NONZERO_SAMPLES = 0.05
    DEFAULT_MIN_TOTAL_COUNTS = 0.005
    DEFAULT_MIN_COUNTS = 5000
    
    # Visualization defaults
    DEFAULT_FIGSIZE = (15, 15)
    DEFAULT_ALPHA = 0.5
    DEFAULT_S = 5
    DEFAULT_CMAP = "tab20"


class TaxonomicLevels:
    """Taxonomic level constants and utilities"""
    
    LEVELS = ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"]
    
    KINGDOM = 0
    PHYLUM = 1
    CLASS = 2
    ORDER = 3
    FAMILY = 4
    GENUS = 5
    SPECIES = 6
    
    @classmethod
    def get_level_name(cls, level: int) -> str:
        """Get taxonomic level name by index"""
        if 0 <= level < len(cls.LEVELS):
            return cls.LEVELS[level]
        raise ValueError(f"Invalid taxonomic level: {level}")
    
    @classmethod
    def get_level_index(cls, name: str) -> int:
        """Get taxonomic level index by name"""
        try:
            return cls.LEVELS.index(name)
        except ValueError:
            raise ValueError(f"Invalid taxonomic level name: {name}")


class FilePatterns:
    """File naming patterns and paths"""
    
    # Common file extensions
    CSV = ".csv"
    TSV = ".tsv" 
    JSON = ".json"
    
    # Dataset patterns
    OTU_PATTERN = "*otu*.{ext}"
    META_PATTERN = "*meta*.{ext}"
    TAX_PATTERN = "*tax*.{ext}"
    
    # Results patterns
    AA_RESULT_PATTERN = "otu_AA_K{k}_{frac}_{group}_{method}.json"
    
    @classmethod
    def get_aa_filename(cls, k: int, frac: float = 1.0, 
                       group: str = "None", method: str = "jensenshannon") -> str:
        """Generate archetypal analysis result filename"""
        return cls.AA_RESULT_PATTERN.format(k=k, frac=frac, group=group, method=method)


class DatasetConfig:
    """Dataset-specific configuration"""
    
    # Compendium dataset columns
    PROJECT_DATA_COLS = [
        'link', 'amplicon', 'avg_length', 'bead_beating', 'kit', 
        'sample_type', 'condition', 'subjects', 'notes'
    ]
    
    SAMPLE_METADATA_COLS = [
        'project', 'srr', 'library_strategy', 'library_source', 'pubdate',
        'total_bases', 'instrument', 'geo_loc_name', 'iso', 'region'
    ]
    
    # Subject categories
    SUBJECTS_NO_CONDITION_ADULTS = [
        'centenarians & relatives', 'adults aged above 40', 'adults', 
        'adults and elderly', 'elderly', 'healthy adults', 'girls and women', 
        'adolescents', 'women', 'healthy women', 'females', 'healthy humans', 
        'healthy donors', 'healthy individuals', 'healthy chinese ages 3-100+', 
        'all ages', 'healthy western individuals', 'Chinese adults (18-65)', 
        'Indian families', 'Finnish adult men', 
        'long-living group (> 90 yr), elderly, young adults'
    ]
    
    SUBJECTS_NO_CONDITION_CHILDREN = [
        'younger than 2', 'infants', 'children', 'infants and children', 
        'young children', 'babies', 'female infant', 'infants and toddlers', 
        'Children'
    ]
    
    SUBJECTS_NO_CONDITION_MIXED = [
        'children and adults', 'adults + children', 'infants, children and adults', 
        'infants and adults', 'children and adolscents', 'children and adolescents', 
        'childrena and adolescents'
    ]
    
    # Top occurring conditions
    CONDITION_TOP9_OCCURRING = [
        'allogeneic hematopoietic cell transplantation', 'preterm', 'CD', 
        'allo-HCT', 'IBD', 'colorectal cancer', 'asthma', 
        'malnutrition + healthy', 'HIV'
    ]
    
    # Age buckets for analysis
    AGE_BUCKETS = [
        (0, 3), (3, 6), (6, 12), (12, 24), (24, 36), (36, 48), (48, 60), (60, float('inf'))
    ]


class PathConfig:
    """Default paths configuration"""
    
    def __init__(self, base_path: Optional[str] = None):
        self.base_path = base_path or os.getcwd()
        
    @property
    def datasets_dir(self) -> str:
        return os.path.join(self.base_path, "datasets")
    
    @property
    def results_dir(self) -> str:
        return os.path.join(self.base_path, "results")
        
    @property
    def tsv_dir(self) -> str:
        return os.path.join(self.datasets_dir, "tsv")
        
    @property
    def compendium_dir(self) -> str:
        return os.path.join(self.datasets_dir, "compendium")
        
    @property
    def normalized_dir(self) -> str:
        return os.path.join(self.compendium_dir, "normalized")

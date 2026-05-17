"""
Archetypal Analysis Package for Genetic Heritage Group Research

This package provides a comprehensive toolkit for performing archetypal analysis
on microbiome and genomic datasets, with specialized support for the Compendium
and Hadza datasets.
"""

__version__ = "1.0.0"
__author__ = "Genetic Heritage Group"

from .analysis.archetypal import ArchetypalAnalysis
from .data.loaders import CompendiumDataLoader, HadzaDataLoader
from .data.normalizers import DataNormalizer
from .visualization.simplex import SimplexPlotter

__all__ = [
    'ArchetypalAnalysis',
    'CompendiumDataLoader', 
    'HadzaDataLoader',
    'DataNormalizer',
    'SimplexPlotter'
]

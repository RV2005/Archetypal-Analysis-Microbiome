"""Archetypal analysis toolkit and config-driven pipeline.

Heavy numerical/plotting dependencies are imported lazily so configuration and dataset
conversion utilities remain usable in lightweight environments.
"""

__version__ = "2.2.0"
__author__ = "Genetic Heritage Group"

__all__ = [
    "ArchetypalAnalysis",
    "CompendiumDataLoader",
    "HadzaDataLoader",
    "UnifiedDatasetLoader",
    "DataNormalizer",
    "SimplexPlotter",
]


def __getattr__(name):
    if name == "ArchetypalAnalysis":
        from .analysis.archetypal import ArchetypalAnalysis
        return ArchetypalAnalysis
    if name in {"CompendiumDataLoader", "HadzaDataLoader"}:
        from .data.loaders import CompendiumDataLoader, HadzaDataLoader
        return {"CompendiumDataLoader": CompendiumDataLoader, "HadzaDataLoader": HadzaDataLoader}[name]
    if name == "UnifiedDatasetLoader":
        from .data.dataset import UnifiedDatasetLoader
        return UnifiedDatasetLoader
    if name == "DataNormalizer":
        from .data.normalizers import DataNormalizer
        return DataNormalizer
    if name == "SimplexPlotter":
        from .visualization.simplex import SimplexPlotter
        return SimplexPlotter
    raise AttributeError(name)
"""YAML configuration loading, inheritance, validation, and path helpers."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import yaml

from ..data.preprocessing import resolve_preprocessing_mode


class ConfigError(ValueError):
    """Raised when a pipeline configuration is invalid."""


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge dictionaries; lists/scalars are replaced, not concatenated."""
    merged: Dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, Mapping)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _as_path_list(value: Any) -> Iterable[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(x, str) for x in value):
        return value
    raise ConfigError("'extends' must be a string or a list of strings")


def _load_yaml_recursive(path: Path, stack: tuple[Path, ...]) -> Dict[str, Any]:
    path = path.resolve()
    if path in stack:
        cycle = " -> ".join(str(p) for p in (*stack, path))
        raise ConfigError(f"Configuration inheritance cycle detected: {cycle}")
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"Config root must be a mapping: {path}")

    parents = _as_path_list(raw.pop("extends", None))
    merged: Dict[str, Any] = {}
    for parent in parents:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = path.parent / parent_path
        merged = deep_merge(merged, _load_yaml_recursive(parent_path, (*stack, path)))
    merged = deep_merge(merged, raw)
    merged.setdefault("_meta", {})
    merged["_meta"]["config_path"] = str(path)
    return merged


def load_config(path: str | os.PathLike[str]) -> Dict[str, Any]:
    """Load a YAML config with recursive ``extends`` support and validate it."""
    cfg = _load_yaml_recursive(Path(path), ())
    validate_config(cfg)
    return cfg


def validate_config(cfg: Mapping[str, Any]) -> None:
    required_sections = ["experiment", "dataset", "task", "filter", "qc", "fit", "evaluation", "selection", "visualization", "analysis"]
    missing = [section for section in required_sections if section not in cfg]
    if missing:
        raise ConfigError(f"Missing required config sections: {missing}")

    exp_name = cfg.get("experiment", {}).get("name")
    if not exp_name:
        raise ConfigError("experiment.name is required")

    dataset = cfg.get("dataset", {})
    if not dataset.get("root"):
        raise ConfigError("dataset.root is required")
    if not dataset.get("otu_file") or not dataset.get("metadata_file"):
        raise ConfigError("dataset.otu_file and dataset.metadata_file are required")
    validation = dataset.get("validation", {}) or {}
    if not isinstance(validation, (Mapping, bool)):
        raise ConfigError("dataset.validation must be a mapping or boolean")
    if isinstance(validation, Mapping):
        for flag in (
            "enabled",
            "unique_sample_ids",
            "numeric_otu",
            "coerce_numeric_otu",
            "sample_match",
            "target_column",
        ):
            if flag in validation and not isinstance(validation[flag], bool):
                raise ConfigError(f"dataset.validation.{flag} must be true or false")

    reader = dataset.get("reader", {}) or {}
    if not isinstance(reader, Mapping):
        raise ConfigError("dataset.reader must be a mapping")
    for flag in ("low_memory", "memory_map", "otu_low_memory", "metadata_low_memory", "otu_memory_map", "metadata_memory_map"):
        if flag in reader and not isinstance(reader[flag], bool):
            raise ConfigError(f"dataset.reader.{flag} must be true or false")
    for table_name in ("otu", "metadata"):
        table_reader = reader.get(table_name, {}) or {}
        if not isinstance(table_reader, Mapping):
            raise ConfigError(f"dataset.reader.{table_name} must be a mapping")
        for flag in ("low_memory", "memory_map"):
            if flag in table_reader and not isinstance(table_reader[flag], bool):
                raise ConfigError(f"dataset.reader.{table_name}.{flag} must be true or false")

    target = cfg.get("task", {}).get("target")
    if not target:
        raise ConfigError("task.target is required")
    required_metadata = cfg.get("task", {}).get("required_metadata_columns", []) or []
    if not isinstance(required_metadata, list) or not all(isinstance(column, str) for column in required_metadata):
        raise ConfigError("task.required_metadata_columns must be a list of column names")

    k_values = cfg.get("fit", {}).get("k_values")
    if not isinstance(k_values, list) or not k_values:
        raise ConfigError("fit.k_values must be a non-empty list")
    if any((not isinstance(k, int) or k < 2) for k in k_values):
        raise ConfigError("fit.k_values must contain integers >= 2")
    if len(set(k_values)) != len(k_values):
        raise ConfigError("fit.k_values contains duplicates")

    preprocessing = cfg.get("fit", {}).get("preprocessing", {}) or {}
    try:
        preprocessing_mode = resolve_preprocessing_mode(preprocessing)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    if preprocessing_mode == "log":
        log_base = float(preprocessing.get("log_base", 2.0))
        if log_base <= 0 or log_base == 1:
            raise ConfigError("fit.preprocessing.log_base must be positive and != 1")
        if float(preprocessing.get("pseudocount", 1.0)) <= 0:
            raise ConfigError("fit.preprocessing.pseudocount must be > 0 for log mode")
    elif preprocessing_mode == "relative_abundance":
        if float(preprocessing.get("relative_abundance_scale", 1.0)) <= 0:
            raise ConfigError("fit.preprocessing.relative_abundance_scale must be > 0")
    elif preprocessing_mode == "clr":
        if float(preprocessing.get("clr_pseudocount", 1.0)) <= 0:
            raise ConfigError("fit.preprocessing.clr_pseudocount must be > 0")

    mode = str(cfg.get("selection", {}).get("mode", "all")).lower()
    aliases = {"none": "all", "skip": "all"}
    mode = aliases.get(mode, mode)
    allowed_modes = {"all", "fixed", "quality", "reconstruction"}
    if mode not in allowed_modes:
        raise ConfigError(f"selection.mode must be one of {sorted(allowed_modes)}")
    if mode == "fixed":
        fixed_k = cfg.get("selection", {}).get("fixed_k")
        if fixed_k not in k_values:
            raise ConfigError("selection.fixed_k must be present in fit.k_values")

    legend_type = str(cfg.get("task", {}).get("legend_type", "discrete")).lower()
    if legend_type not in {"discrete", "continuous"}:
        raise ConfigError("task.legend_type must be 'discrete' or 'continuous'")

    ordering = cfg.get("visualization", {}).get("archetype_ordering", {}) or {}
    if not isinstance(ordering, Mapping):
        raise ConfigError("visualization.archetype_ordering must be a mapping")
    ordering_method = str(ordering.get("method", "association_similarity")).lower()
    if ordering_method not in {"top_association", "association_similarity"}:
        raise ConfigError(
            "visualization.archetype_ordering.method must be 'top_association' or 'association_similarity'"
        )
    similarity_metric = str(ordering.get("similarity_metric", "cosine")).lower()
    if similarity_metric not in {"cosine", "euclidean"}:
        raise ConfigError(
            "visualization.archetype_ordering.similarity_metric must be 'cosine' or 'euclidean'"
        )
    if int(ordering.get("exact_max_k", 15)) < 2:
        raise ConfigError("visualization.archetype_ordering.exact_max_k must be >= 2")


def project_root(cfg: Mapping[str, Any]) -> Path:
    """Resolve the repository/project root used for relative paths."""
    value = cfg.get("paths", {}).get("project_root", ".")
    p = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if p.is_absolute():
        return p.resolve()
    # Relative project roots are interpreted from the working directory. The sbatch
    # launcher always ``cd``s to repository root, keeping CLI and notebook behavior simple.
    return (Path.cwd() / p).resolve()


def resolve_project_path(cfg: Mapping[str, Any], value: str | os.PathLike[str]) -> Path:
    p = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return p.resolve() if p.is_absolute() else (project_root(cfg) / p).resolve()


def dataset_paths(cfg: Mapping[str, Any]) -> tuple[Path, Path]:
    root = resolve_project_path(cfg, cfg["dataset"]["root"])
    return root / cfg["dataset"]["otu_file"], root / cfg["dataset"]["metadata_file"]


def output_root(cfg: Mapping[str, Any]) -> Path:
    base = resolve_project_path(cfg, cfg.get("paths", {}).get("output_root", "results"))
    return base / str(cfg["experiment"]["name"])


def normalize_selection_mode(cfg: Mapping[str, Any]) -> str:
    mode = str(cfg.get("selection", {}).get("mode", "all")).lower()
    return {"none": "all", "skip": "all"}.get(mode, mode)
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
    required_sections = ["experiment", "dataset", "task", "filter", "qc", "fit", "evaluation", "benchmarks", "selection", "visualization", "analysis"]
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

    benchmarks = cfg.get("benchmarks", {}) or {}
    if not isinstance(benchmarks, Mapping):
        raise ConfigError("benchmarks must be a mapping")
    for method in ("pca", "pcoa", "umap", "tsne"):
        method_cfg = benchmarks.get(method, {}) or {}
        if not isinstance(method_cfg, Mapping):
            raise ConfigError(f"benchmarks.{method} must be a mapping")
        if "enabled" in method_cfg and not isinstance(method_cfg["enabled"], bool):
            raise ConfigError(f"benchmarks.{method}.enabled must be true or false")
    parallel = benchmarks.get("parallel", {}) or {}
    if not isinstance(parallel, Mapping):
        raise ConfigError("benchmarks.parallel must be a mapping")
    backend = str(parallel.get("backend", "process")).lower()
    if backend not in {"process", "sequential"}:
        raise ConfigError("benchmarks.parallel.backend must be 'process' or 'sequential'")
    if int(parallel.get("n_jobs", 0) or 0) < 0:
        raise ConfigError("benchmarks.parallel.n_jobs must be >= 0")
    pcoa_metric = str((benchmarks.get("pcoa", {}) or {}).get("distance_metric", "euclidean")).lower()
    if not pcoa_metric:
        raise ConfigError("benchmarks.pcoa.distance_metric must be non-empty")
    tsne_method = str((benchmarks.get("tsne", {}) or {}).get("method", "auto")).lower()
    if tsne_method not in {"auto", "exact", "barnes_hut"}:
        raise ConfigError("benchmarks.tsne.method must be 'auto', 'exact', or 'barnes_hut'")
    if tsne_method == "barnes_hut" and (max(k_values) - 1) > 3 and (benchmarks.get("tsne", {}) or {}).get("enabled", True):
        raise ConfigError("benchmarks.tsne.method=barnes_hut cannot satisfy K-1 dimensions above 3; use auto or exact")

    prediction = cfg.get("prediction", {}) or {}
    if not isinstance(prediction, Mapping):
        raise ConfigError("prediction must be a mapping")
    if int(prediction.get("n_splits", 5)) < 2:
        raise ConfigError("prediction.n_splits must be >= 2")
    if str(prediction.get("raw_baseline", "qc_otu")).lower() not in {"qc_otu", "fit_input"}:
        raise ConfigError("prediction.raw_baseline must be 'qc_otu' or 'fit_input'")
    if prediction.get("group_column") is not None and not isinstance(prediction.get("group_column"), str):
        raise ConfigError("prediction.group_column must be null or a metadata column name")
    group_fallbacks = prediction.get("group_fallback_columns", []) or []
    if not isinstance(group_fallbacks, list) or not all(isinstance(column, str) for column in group_fallbacks):
        raise ConfigError("prediction.group_fallback_columns must be a list of metadata column names")
    if not isinstance(prediction.get("group_auto_detect", True), bool):
        raise ConfigError("prediction.group_auto_detect must be true or false")
    group_namespaces = prediction.get("group_namespace_columns", []) or []
    if not isinstance(group_namespaces, list) or not all(isinstance(column, str) for column in group_namespaces):
        raise ConfigError("prediction.group_namespace_columns must be a list of metadata column names")
    logistic_cfg = prediction.get("logistic_regression", {}) or {}
    if not isinstance(logistic_cfg, Mapping):
        raise ConfigError("prediction.logistic_regression must be a mapping")
    c_values = logistic_cfg.get("C_values", logistic_cfg.get("C_grid", [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]))
    if not isinstance(c_values, list) or not c_values or any(float(value) <= 0 for value in c_values):
        raise ConfigError("prediction.logistic_regression.C_values must be a non-empty list of positive values")
    if int(logistic_cfg.get("inner_cv_splits", 4)) < 2:
        raise ConfigError("prediction.logistic_regression.inner_cv_splits must be >= 2")
    if str(logistic_cfg.get("scoring", "balanced_accuracy")) not in {"balanced_accuracy", "f1_macro"}:
        raise ConfigError("prediction.logistic_regression.scoring must be 'balanced_accuracy' or 'f1_macro'")
    aa_projection = prediction.get("aa_projection", {}) or {}
    if not isinstance(aa_projection, Mapping):
        raise ConfigError("prediction.aa_projection must be a mapping")
    if int(aa_projection.get("max_iter", 2000)) < 1:
        raise ConfigError("prediction.aa_projection.max_iter must be >= 1")
    if float(aa_projection.get("tol", 1e-8)) <= 0:
        raise ConfigError("prediction.aa_projection.tol must be > 0")

    stability = cfg.get("stability", {}) or {}
    if not isinstance(stability, Mapping):
        raise ConfigError("stability must be a mapping")
    profile_space = str(stability.get("profile_space", "fitted")).lower()
    if profile_space not in {"fitted", "inverse"}:
        raise ConfigError("stability.profile_space must be 'fitted' or 'inverse'")
    matching_metric = str(stability.get("matching_metric", "cosine")).lower()
    if matching_metric not in {"cosine", "correlation"}:
        raise ConfigError("stability.matching_metric must be 'cosine' or 'correlation'")
    init_cfg = stability.get("initialization", {}) or {}
    if not isinstance(init_cfg, Mapping):
        raise ConfigError("stability.initialization must be a mapping")
    seeds = init_cfg.get("seeds", []) or []
    if not isinstance(seeds, list) or any(not isinstance(seed, int) for seed in seeds):
        raise ConfigError("stability.initialization.seeds must be a list of integers")
    sampling_cfg = stability.get("sampling", {}) or {}
    if not isinstance(sampling_cfg, Mapping):
        raise ConfigError("stability.sampling must be a mapping")
    fractions = sampling_cfg.get("fractions", []) or []
    if not isinstance(fractions, list) or any(not 0 < float(value) < 1 for value in fractions):
        raise ConfigError("stability.sampling.fractions must contain values strictly between 0 and 1")
    if int(sampling_cfg.get("repeats", 1)) < 1:
        raise ConfigError("stability.sampling.repeats must be >= 1")
    if int(sampling_cfg.get("continuous_stratify_bins", 10)) < 2:
        raise ConfigError("stability.sampling.continuous_stratify_bins must be >= 2")
    sampling_group = sampling_cfg.get("group_column")
    if sampling_group is not None and not isinstance(sampling_group, str):
        raise ConfigError("stability.sampling.group_column must be null or a metadata column name")
    sampling_fallbacks = sampling_cfg.get("group_fallback_columns")
    if sampling_fallbacks is not None and (
        not isinstance(sampling_fallbacks, list)
        or not all(isinstance(column, str) for column in sampling_fallbacks)
    ):
        raise ConfigError("stability.sampling.group_fallback_columns must be null or a list of metadata columns")
    sampling_auto = sampling_cfg.get("group_auto_detect")
    if sampling_auto is not None and not isinstance(sampling_auto, bool):
        raise ConfigError("stability.sampling.group_auto_detect must be null, true, or false")
    sampling_namespaces = sampling_cfg.get("group_namespace_columns")
    if sampling_namespaces is not None and (
        not isinstance(sampling_namespaces, list)
        or not all(isinstance(column, str) for column in sampling_namespaces)
    ):
        raise ConfigError("stability.sampling.group_namespace_columns must be null or a list of metadata columns")
    projection_cfg = stability.get("heldout_projection", {}) or {}
    if not isinstance(projection_cfg, Mapping):
        raise ConfigError("stability.heldout_projection must be a mapping")
    heldout_fractions = projection_cfg.get("fractions", []) or []
    if not isinstance(heldout_fractions, list) or any(not 0 < float(value) < 1 for value in heldout_fractions):
        raise ConfigError("stability.heldout_projection.fractions must contain values strictly between 0 and 1")
    if not 0 < float(projection_cfg.get("test_frac", 0.2)) < 1:
        raise ConfigError("stability.heldout_projection.test_frac must be strictly between 0 and 1")
    stability_parallel = stability.get("parallel", {}) or {}
    if not isinstance(stability_parallel, Mapping):
        raise ConfigError("stability.parallel must be a mapping")
    if str(stability_parallel.get("backend", "process")).lower() not in {"process", "sequential"}:
        raise ConfigError("stability.parallel.backend must be 'process' or 'sequential'")
    if int(stability_parallel.get("n_jobs", 0) or 0) < 0:
        raise ConfigError("stability.parallel.n_jobs must be >= 0")

    metrics_cfg = cfg.get("analysis", {}).get("metrics", {}) or {}
    if not isinstance(metrics_cfg, Mapping):
        raise ConfigError("analysis.metrics must be a mapping")
    if float(metrics_cfg.get("fuzzy_m", 2.0)) <= 1:
        raise ConfigError("analysis.metrics.fuzzy_m must be > 1")
    if int(metrics_cfg.get("continuous_bins", 10)) < 2:
        raise ConfigError("analysis.metrics.continuous_bins must be >= 2")
    if int(metrics_cfg.get("convex_hull_max_dimension", 6)) < 2:
        raise ConfigError("analysis.metrics.convex_hull_max_dimension must be >= 2")

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

    taxa_heatmap = cfg.get("visualization", {}).get("taxa", {}).get("heatmap", {}) or {}
    if not isinstance(taxa_heatmap, Mapping):
        raise ConfigError("visualization.taxa.heatmap must be a mapping")
    heatmap_scale = str(taxa_heatmap.get("scale", "auto")).lower()
    if heatmap_scale not in {"auto", "manual"}:
        raise ConfigError("visualization.taxa.heatmap.scale must be 'auto' or 'manual'")
    if heatmap_scale == "manual":
        if taxa_heatmap.get("vmin") is None or taxa_heatmap.get("vmax") is None:
            raise ConfigError("manual taxa heatmap scale requires vmin and vmax")
        if float(taxa_heatmap["vmax"]) <= float(taxa_heatmap["vmin"]):
            raise ConfigError("visualization.taxa.heatmap.vmax must be greater than vmin")

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
    benchmarks = cfg.get("benchmarks", {}) or {}
    if not isinstance(benchmarks, Mapping):
        raise ConfigError("benchmarks must be a mapping")
    for method in ("pca", "pcoa", "umap", "tsne"):
        method_cfg = benchmarks.get(method, {}) or {}
        if not isinstance(method_cfg, Mapping):
            raise ConfigError(f"benchmarks.{method} must be a mapping")
        if "enabled" in method_cfg and not isinstance(method_cfg["enabled"], bool):
            raise ConfigError(f"benchmarks.{method}.enabled must be true or false")
    parallel = benchmarks.get("parallel", {}) or {}
    if not isinstance(parallel, Mapping):
        raise ConfigError("benchmarks.parallel must be a mapping")
    backend = str(parallel.get("backend", "process")).lower()
    if backend not in {"process", "sequential"}:
        raise ConfigError("benchmarks.parallel.backend must be 'process' or 'sequential'")
    if int(parallel.get("n_jobs", 0) or 0) < 0:
        raise ConfigError("benchmarks.parallel.n_jobs must be >= 0")
    pcoa_metric = str((benchmarks.get("pcoa", {}) or {}).get("distance_metric", "euclidean")).lower()
    if not pcoa_metric:
        raise ConfigError("benchmarks.pcoa.distance_metric must be non-empty")
    tsne_method = str((benchmarks.get("tsne", {}) or {}).get("method", "auto")).lower()
    if tsne_method not in {"auto", "exact", "barnes_hut"}:
        raise ConfigError("benchmarks.tsne.method must be 'auto', 'exact', or 'barnes_hut'")
    if tsne_method == "barnes_hut" and (max(k_values) - 1) > 3 and (benchmarks.get("tsne", {}) or {}).get("enabled", True):
        raise ConfigError("benchmarks.tsne.method=barnes_hut cannot satisfy K-1 dimensions above 3; use auto or exact")

    prediction = cfg.get("prediction", {}) or {}
    if not isinstance(prediction, Mapping):
        raise ConfigError("prediction must be a mapping")
    if int(prediction.get("n_splits", 5)) < 2:
        raise ConfigError("prediction.n_splits must be >= 2")
    if str(prediction.get("raw_baseline", "qc_otu")).lower() not in {"qc_otu", "fit_input"}:
        raise ConfigError("prediction.raw_baseline must be 'qc_otu' or 'fit_input'")
    if prediction.get("group_column") is not None and not isinstance(prediction.get("group_column"), str):
        raise ConfigError("prediction.group_column must be null or a metadata column name")
    group_fallbacks = prediction.get("group_fallback_columns", []) or []
    if not isinstance(group_fallbacks, list) or not all(isinstance(column, str) for column in group_fallbacks):
        raise ConfigError("prediction.group_fallback_columns must be a list of metadata column names")
    if not isinstance(prediction.get("group_auto_detect", True), bool):
        raise ConfigError("prediction.group_auto_detect must be true or false")
    group_namespaces = prediction.get("group_namespace_columns", []) or []
    if not isinstance(group_namespaces, list) or not all(isinstance(column, str) for column in group_namespaces):
        raise ConfigError("prediction.group_namespace_columns must be a list of metadata column names")
    logistic_cfg = prediction.get("logistic_regression", {}) or {}
    if not isinstance(logistic_cfg, Mapping):
        raise ConfigError("prediction.logistic_regression must be a mapping")
    c_values = logistic_cfg.get("C_values", logistic_cfg.get("C_grid", [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]))
    if not isinstance(c_values, list) or not c_values or any(float(value) <= 0 for value in c_values):
        raise ConfigError("prediction.logistic_regression.C_values must be a non-empty list of positive values")
    if int(logistic_cfg.get("inner_cv_splits", 4)) < 2:
        raise ConfigError("prediction.logistic_regression.inner_cv_splits must be >= 2")
    if str(logistic_cfg.get("scoring", "balanced_accuracy")) not in {"balanced_accuracy", "f1_macro"}:
        raise ConfigError("prediction.logistic_regression.scoring must be 'balanced_accuracy' or 'f1_macro'")
    aa_projection = prediction.get("aa_projection", {}) or {}
    if not isinstance(aa_projection, Mapping):
        raise ConfigError("prediction.aa_projection must be a mapping")
    if int(aa_projection.get("max_iter", 2000)) < 1:
        raise ConfigError("prediction.aa_projection.max_iter must be >= 1")
    if float(aa_projection.get("tol", 1e-8)) <= 0:
        raise ConfigError("prediction.aa_projection.tol must be > 0")

    stability = cfg.get("stability", {}) or {}
    if not isinstance(stability, Mapping):
        raise ConfigError("stability must be a mapping")
    profile_space = str(stability.get("profile_space", "fitted")).lower()
    if profile_space not in {"fitted", "inverse"}:
        raise ConfigError("stability.profile_space must be 'fitted' or 'inverse'")
    matching_metric = str(stability.get("matching_metric", "cosine")).lower()
    if matching_metric not in {"cosine", "correlation"}:
        raise ConfigError("stability.matching_metric must be 'cosine' or 'correlation'")
    init_cfg = stability.get("initialization", {}) or {}
    if not isinstance(init_cfg, Mapping):
        raise ConfigError("stability.initialization must be a mapping")
    seeds = init_cfg.get("seeds", []) or []
    if not isinstance(seeds, list) or any(not isinstance(seed, int) for seed in seeds):
        raise ConfigError("stability.initialization.seeds must be a list of integers")
    sampling_cfg = stability.get("sampling", {}) or {}
    if not isinstance(sampling_cfg, Mapping):
        raise ConfigError("stability.sampling must be a mapping")
    fractions = sampling_cfg.get("fractions", []) or []
    if not isinstance(fractions, list) or any(not 0 < float(value) < 1 for value in fractions):
        raise ConfigError("stability.sampling.fractions must contain values strictly between 0 and 1")
    if int(sampling_cfg.get("repeats", 1)) < 1:
        raise ConfigError("stability.sampling.repeats must be >= 1")
    if int(sampling_cfg.get("continuous_stratify_bins", 10)) < 2:
        raise ConfigError("stability.sampling.continuous_stratify_bins must be >= 2")
    sampling_group = sampling_cfg.get("group_column")
    if sampling_group is not None and not isinstance(sampling_group, str):
        raise ConfigError("stability.sampling.group_column must be null or a metadata column name")
    sampling_fallbacks = sampling_cfg.get("group_fallback_columns")
    if sampling_fallbacks is not None and (
        not isinstance(sampling_fallbacks, list)
        or not all(isinstance(column, str) for column in sampling_fallbacks)
    ):
        raise ConfigError("stability.sampling.group_fallback_columns must be null or a list of metadata columns")
    sampling_auto = sampling_cfg.get("group_auto_detect")
    if sampling_auto is not None and not isinstance(sampling_auto, bool):
        raise ConfigError("stability.sampling.group_auto_detect must be null, true, or false")
    sampling_namespaces = sampling_cfg.get("group_namespace_columns")
    if sampling_namespaces is not None and (
        not isinstance(sampling_namespaces, list)
        or not all(isinstance(column, str) for column in sampling_namespaces)
    ):
        raise ConfigError("stability.sampling.group_namespace_columns must be null or a list of metadata columns")
    projection_cfg = stability.get("heldout_projection", {}) or {}
    if not isinstance(projection_cfg, Mapping):
        raise ConfigError("stability.heldout_projection must be a mapping")
    heldout_fractions = projection_cfg.get("fractions", []) or []
    if not isinstance(heldout_fractions, list) or any(not 0 < float(value) < 1 for value in heldout_fractions):
        raise ConfigError("stability.heldout_projection.fractions must contain values strictly between 0 and 1")
    if not 0 < float(projection_cfg.get("test_frac", 0.2)) < 1:
        raise ConfigError("stability.heldout_projection.test_frac must be strictly between 0 and 1")
    stability_parallel = stability.get("parallel", {}) or {}
    if not isinstance(stability_parallel, Mapping):
        raise ConfigError("stability.parallel must be a mapping")
    if str(stability_parallel.get("backend", "process")).lower() not in {"process", "sequential"}:
        raise ConfigError("stability.parallel.backend must be 'process' or 'sequential'")
    if int(stability_parallel.get("n_jobs", 0) or 0) < 0:
        raise ConfigError("stability.parallel.n_jobs must be >= 0")

    metrics_cfg = cfg.get("analysis", {}).get("metrics", {}) or {}
    if not isinstance(metrics_cfg, Mapping):
        raise ConfigError("analysis.metrics must be a mapping")
    if float(metrics_cfg.get("fuzzy_m", 2.0)) <= 1:
        raise ConfigError("analysis.metrics.fuzzy_m must be > 1")
    if int(metrics_cfg.get("continuous_bins", 10)) < 2:
        raise ConfigError("analysis.metrics.continuous_bins must be >= 2")
    if int(metrics_cfg.get("convex_hull_max_dimension", 6)) < 2:
        raise ConfigError("analysis.metrics.convex_hull_max_dimension must be >= 2")

    mode = str(cfg.get("selection", {}).get("mode", "all")).lower()
    return {"none": "all", "skip": "all"}.get(mode, mode)

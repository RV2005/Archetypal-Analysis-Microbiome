"""Dimensionality-reduction benchmark computation and artifact helpers.

Benchmarks intentionally consume ``fit/fit_input.tsv`` directly. This keeps every
method on the exact same samples, features, and preprocessing transform as archetypal
analysis instead of silently applying visualization-specific preprocessing a second time.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from ..data.indexed_io import read_sample_indexed_tsv
from ..pipeline.artifacts import atomic_write_json


@dataclass(frozen=True)
class BenchmarkTask:
    method: str
    dimension: int

    @property
    def directory_name(self) -> str:
        return f"D_{self.dimension:03d}"


def _benchmark_dimensions(k_values: int | Sequence[int]) -> list[int]:
    """Return benchmark dimensions matched to AA's intrinsic K-1 simplex dimension."""
    if isinstance(k_values, int):
        ks = list(range(2, int(k_values) + 1))
    else:
        ks = sorted({int(k) for k in k_values})
    if not ks or min(ks) < 2:
        raise ValueError("Benchmarking requires AA K values >= 2")
    return sorted({k - 1 for k in ks})


def benchmark_tasks(k_values: int | Sequence[int], cfg: Mapping[str, Any]) -> list[BenchmarkTask]:
    """Return the exact computations required by the benchmark contract.

    AA similarity weights lie on a K-simplex and therefore have K-1 independent
    dimensions. PCA and PCoA are fitted once at the largest requested K-1 dimension
    and sliced as needed; UMAP and t-SNE are fitted independently at each K-1 dimension.
    """
    dimensions = _benchmark_dimensions(k_values)
    max_dimension = max(dimensions)
    if not bool(cfg.get("enabled", True)):
        return []

    tasks: list[BenchmarkTask] = []
    if bool((cfg.get("pca", {}) or {}).get("enabled", True)):
        tasks.append(BenchmarkTask("pca", max_dimension))
    if bool((cfg.get("pcoa", {}) or {}).get("enabled", True)):
        tasks.append(BenchmarkTask("pcoa", max_dimension))
    if bool((cfg.get("umap", {}) or {}).get("enabled", True)):
        tasks.extend(BenchmarkTask("umap", dimension) for dimension in dimensions)
    if bool((cfg.get("tsne", {}) or {}).get("enabled", True)):
        tasks.extend(BenchmarkTask("tsne", dimension) for dimension in dimensions)
    return tasks


def _component_columns(prefix: str, dimension: int) -> list[str]:
    return [f"{prefix}{i}" for i in range(1, dimension + 1)]


def _validate_dimension(data: pd.DataFrame, method: str, dimension: int) -> None:
    n_samples, n_features = data.shape
    if dimension < 1:
        raise ValueError(f"{method} benchmark dimension must be >= 1")
    if method == "pca" and dimension > min(n_samples, n_features):
        raise ValueError(
            f"PCA requested {dimension} dimensions but fit input shape is {data.shape}; "
            "n_components must be <= min(n_samples, n_features)"
        )
    if method == "pcoa" and dimension > n_samples - 1:
        raise ValueError(
            f"PCoA requested {dimension} dimensions for {n_samples} samples; "
            "at most n_samples - 1 coordinates are available"
        )
    if method in {"umap", "tsne"} and dimension >= n_samples:
        raise ValueError(
            f"{method.upper()} requested {dimension} dimensions for only {n_samples} samples"
        )


def _compute_pca(
    data: pd.DataFrame,
    dimension: int,
    cfg: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, pd.DataFrame]]:
    model = PCA(
        n_components=dimension,
        svd_solver=str(cfg.get("svd_solver", "auto")),
        random_state=cfg.get("random_state"),
        **dict(cfg.get("kwargs", {})),
    )
    values = model.fit_transform(data.to_numpy(dtype=float))
    columns = _component_columns("PC", dimension)
    coords = pd.DataFrame(values, index=data.index, columns=columns)
    extras = {
        "components.tsv": pd.DataFrame(model.components_, index=columns, columns=data.columns),
        "mean.tsv": pd.DataFrame([model.mean_], index=["mean"], columns=data.columns),
        "explained_variance.tsv": pd.DataFrame(
            {
                "component": columns,
                "explained_variance": model.explained_variance_,
                "explained_variance_ratio": model.explained_variance_ratio_,
                "singular_value": model.singular_values_,
            }
        ),
    }
    meta = {
        "method": "pca",
        "dimension": dimension,
        "explained_variance_ratio": model.explained_variance_ratio_.tolist(),
        "total_explained_variance_ratio": float(np.sum(model.explained_variance_ratio_)),
    }
    return coords, meta, extras


def _pcoa_distance_metric(name: str):
    normalized = str(name).strip().lower()
    if normalized in {"jensen-shannon", "jensen_shannon", "js", "jensenshannon"}:
        return distance.jensenshannon, "jensenshannon"
    if normalized in {"bray-curtis", "bray_curtis", "bray curtis", "braycurtis"}:
        return "braycurtis", "braycurtis"
    return normalized, normalized


def _compute_pcoa(
    data: pd.DataFrame,
    dimension: int,
    cfg: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, pd.DataFrame]]:
    """Classical PCoA/MDS on a configurable pairwise distance matrix."""
    metric, metric_name = _pcoa_distance_metric(cfg.get("distance_metric", "euclidean"))
    values = data.to_numpy(dtype=float)
    if metric_name in {"jensenshannon", "braycurtis"} and np.any(values < 0):
        raise ValueError(
            f"{metric_name} PCoA requires non-negative fit input. "
            "Use benchmarks.pcoa.distance_metric=euclidean for CLR/standard-scaled data."
        )

    condensed = distance.pdist(values, metric=metric)
    if not np.isfinite(condensed).all():
        raise ValueError("PCoA pairwise distances contain NaN or infinite values")
    distances = distance.squareform(condensed)
    n = len(values)
    centering = np.eye(n) - np.ones((n, n), dtype=float) / n
    gram = -0.5 * centering @ (distances ** 2) @ centering
    eigenvalues_all, eigenvectors_all = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues_all)[::-1]
    eigenvalues_all = np.real_if_close(eigenvalues_all[order]).astype(float)
    eigenvectors_all = np.real_if_close(eigenvectors_all[:, order]).astype(float)

    tolerance = (
        max(1e-12, float(np.max(np.abs(eigenvalues_all))) * 1e-12)
        if eigenvalues_all.size
        else 1e-12
    )
    positive = eigenvalues_all > tolerance
    positive_count = int(np.sum(positive))
    if positive_count < dimension:
        raise ValueError(
            f"PCoA produced only {positive_count} positive axes, fewer than requested dimension={dimension}. "
            "Choose a Euclidean distance or lower max K."
        )
    eigenvalues = eigenvalues_all[positive][:dimension]
    eigenvectors = eigenvectors_all[:, positive][:, :dimension]
    coords_values = eigenvectors * np.sqrt(eigenvalues)[None, :]
    columns = _component_columns("PC", dimension)
    coords = pd.DataFrame(coords_values, index=data.index, columns=columns)
    total_positive = float(np.sum(eigenvalues_all[eigenvalues_all > 0]))
    explained = eigenvalues / total_positive if total_positive > 0 else np.full(dimension, np.nan)
    extras = {
        "eigenvalues.tsv": pd.DataFrame(
            {
                "component": columns,
                "eigenvalue": eigenvalues,
                "proportion_explained": explained,
            }
        )
    }
    meta = {
        "method": "pcoa",
        "dimension": dimension,
        "distance_metric": metric_name,
        "positive_axes": positive_count,
        "proportion_explained": explained.tolist(),
        "total_proportion_explained": float(np.nansum(explained)),
    }
    return coords, meta, extras


def _compute_umap(
    data: pd.DataFrame,
    dimension: int,
    cfg: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, pd.DataFrame]]:
    try:
        import umap
    except ImportError as exc:
        raise ImportError(
            "UMAP benchmarks require 'umap-learn'. Install it in the pipeline environment."
        ) from exc

    n_neighbors = min(int(cfg.get("n_neighbors", 15)), max(2, len(data) - 1))
    kwargs = dict(cfg.get("kwargs", {}))
    # Benchmarks are parallelized at the process level. Avoid nested worker pools unless
    # the user deliberately overrides n_jobs in kwargs.
    kwargs.setdefault("n_jobs", 1)
    model = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=dimension,
        min_dist=float(cfg.get("min_dist", 0.1)),
        metric=str(cfg.get("metric", "euclidean")),
        random_state=int(cfg.get("random_state", 42)),
        **kwargs,
    )
    values = model.fit_transform(data.to_numpy(dtype=float))
    columns = _component_columns("UMAP", dimension)
    coords = pd.DataFrame(values, index=data.index, columns=columns)
    meta = {
        "method": "umap",
        "dimension": dimension,
        "n_neighbors": n_neighbors,
        "min_dist": float(cfg.get("min_dist", 0.1)),
        "metric": str(cfg.get("metric", "euclidean")),
    }
    return coords, meta, {}


def _compute_tsne(
    data: pd.DataFrame,
    dimension: int,
    cfg: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, pd.DataFrame]]:
    perplexity = float(cfg.get("perplexity", 30.0))
    if perplexity >= len(data):
        raise ValueError(f"t-SNE perplexity ({perplexity}) must be smaller than n_samples ({len(data)})")

    configured_method = str(cfg.get("method", "auto")).lower()
    if configured_method == "auto":
        method = "barnes_hut" if dimension <= 3 else "exact"
    else:
        method = configured_method
    if dimension > 3 and method == "barnes_hut":
        raise ValueError(
            "scikit-learn Barnes-Hut t-SNE supports n_components <= 3; use method=auto or exact"
        )

    kwargs = dict(cfg.get("kwargs", {}))
    tsne_args: dict[str, Any] = {
        "n_components": dimension,
        "perplexity": perplexity,
        "learning_rate": cfg.get("learning_rate", "auto"),
        "init": str(cfg.get("init", "pca")),
        "metric": str(cfg.get("metric", "euclidean")),
        "random_state": int(cfg.get("random_state", 42)),
        "method": method,
        **kwargs,
    }
    iteration_key = "max_iter" if "max_iter" in inspect.signature(TSNE).parameters else "n_iter"
    tsne_args[iteration_key] = int(cfg.get("max_iter", 1000))
    model = TSNE(**tsne_args)
    values = model.fit_transform(data.to_numpy(dtype=float))
    columns = _component_columns("tSNE", dimension)
    coords = pd.DataFrame(values, index=data.index, columns=columns)
    meta = {
        "method": "tsne",
        "dimension": dimension,
        "perplexity": perplexity,
        "metric": str(cfg.get("metric", "euclidean")),
        "algorithm": method,
        "kl_divergence": float(getattr(model, "kl_divergence_", np.nan)),
    }
    return coords, meta, {}


def compute_benchmark(data: pd.DataFrame, method: str, dimension: int, cfg: Mapping[str, Any]):
    method = str(method).lower()
    _validate_dimension(data, method, dimension)
    if method == "pca":
        return _compute_pca(data, dimension, cfg)
    if method == "pcoa":
        return _compute_pcoa(data, dimension, cfg)
    if method == "umap":
        return _compute_umap(data, dimension, cfg)
    if method == "tsne":
        return _compute_tsne(data, dimension, cfg)
    raise ValueError(f"Unknown benchmark method: {method}")



@dataclass(frozen=True)
class TrainTestBenchmark:
    train: pd.DataFrame
    test: pd.DataFrame | None
    metadata: dict[str, Any]


def _validate_nonnegative_distance_input(values: np.ndarray, metric_name: str) -> None:
    if metric_name in {"jensenshannon", "braycurtis"} and np.any(values < 0):
        raise ValueError(
            f"{metric_name} PCoA requires non-negative fit input. "
            "Use Euclidean PCoA when the fold preprocessing can produce negative values."
        )


def _fit_pcoa_train_test(
    train: pd.DataFrame,
    test: pd.DataFrame,
    dimension: int,
    cfg: Mapping[str, Any],
) -> TrainTestBenchmark:
    """Fit classical PCoA on train only and Gower-project held-out samples."""
    metric, metric_name = _pcoa_distance_metric(cfg.get("distance_metric", "euclidean"))
    x_train = train.to_numpy(dtype=float)
    x_test = test.to_numpy(dtype=float)
    _validate_nonnegative_distance_input(x_train, metric_name)
    _validate_nonnegative_distance_input(x_test, metric_name)

    condensed = distance.pdist(x_train, metric=metric)
    if not np.isfinite(condensed).all():
        raise ValueError("Training PCoA pairwise distances contain NaN or infinite values")
    d_train = distance.squareform(condensed)
    d2 = d_train ** 2
    n = len(train)
    centering = np.eye(n) - np.ones((n, n), dtype=float) / n
    gram = -0.5 * centering @ d2 @ centering
    eigenvalues_all, eigenvectors_all = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues_all)[::-1]
    eigenvalues_all = np.real_if_close(eigenvalues_all[order]).astype(float)
    eigenvectors_all = np.real_if_close(eigenvectors_all[:, order]).astype(float)
    tolerance = max(1e-12, float(np.max(np.abs(eigenvalues_all))) * 1e-12) if eigenvalues_all.size else 1e-12
    positive = eigenvalues_all > tolerance
    positive_count = int(np.sum(positive))
    if positive_count < dimension:
        raise ValueError(
            f"Training-fold PCoA produced only {positive_count} positive axes; "
            f"{dimension} are required for the K-1 comparison."
        )
    eigenvalues = eigenvalues_all[positive][:dimension]
    eigenvectors = eigenvectors_all[:, positive][:, :dimension]
    train_values = eigenvectors * np.sqrt(eigenvalues)[None, :]

    # Gower interpolation. b contains inner products between each supplementary point
    # and the centered training configuration, derived solely from train-test distances.
    if len(test):
        d_test = distance.cdist(x_test, x_train, metric=metric)
        if not np.isfinite(d_test).all():
            raise ValueError("PCoA train-test distances contain NaN or infinite values")
        d_test2 = d_test ** 2
        train_row_mean = d2.mean(axis=1)
        train_grand_mean = float(d2.mean())
        test_row_mean = d_test2.mean(axis=1)
        b = -0.5 * (
            d_test2
            - train_row_mean[None, :]
            - test_row_mean[:, None]
            + train_grand_mean
        )
        test_values = (b @ eigenvectors) / np.sqrt(eigenvalues)[None, :]
    else:
        test_values = np.empty((0, dimension), dtype=float)

    columns = _component_columns("PC", dimension)
    total_positive = float(np.sum(eigenvalues_all[eigenvalues_all > 0]))
    explained = eigenvalues / total_positive if total_positive > 0 else np.full(dimension, np.nan)
    return TrainTestBenchmark(
        train=pd.DataFrame(train_values, index=train.index, columns=columns),
        test=pd.DataFrame(test_values, index=test.index, columns=columns),
        metadata={
            "method": "pcoa",
            "dimension": int(dimension),
            "distance_metric": metric_name,
            "positive_axes": positive_count,
            "proportion_explained": explained.tolist(),
            "out_of_sample_method": "gower_supplementary_projection",
        },
    )


def fit_benchmark_train_test(
    train: pd.DataFrame,
    test: pd.DataFrame,
    method: str,
    dimension: int,
    cfg: Mapping[str, Any],
) -> TrainTestBenchmark:
    """Fit one benchmark on outer-training samples and transform held-out samples.

    PCA, PCoA, and UMAP have defensible train-only out-of-sample mappings.  scikit-learn
    t-SNE does not expose one; strict prediction therefore reports t-SNE as unavailable
    rather than fitting it on the test samples or introducing a surrogate KNN mapping.
    """
    method = str(method).lower()
    dimension = int(dimension)
    if dimension < 1:
        raise ValueError("Prediction benchmark dimension must be >= 1")
    if list(train.columns) != list(test.columns):
        raise ValueError("Prediction benchmark train/test feature columns must match")
    if dimension > min(len(train), train.shape[1]) and method == "pca":
        raise ValueError("PCA K-1 dimension exceeds the training-fold matrix rank bound")
    if method == "pca":
        model = PCA(
            n_components=dimension,
            svd_solver=str(cfg.get("svd_solver", "auto")),
            random_state=cfg.get("random_state"),
            **dict(cfg.get("kwargs", {})),
        )
        train_values = model.fit_transform(train.to_numpy(dtype=float))
        test_values = model.transform(test.to_numpy(dtype=float))
        columns = _component_columns("PC", dimension)
        return TrainTestBenchmark(
            pd.DataFrame(train_values, index=train.index, columns=columns),
            pd.DataFrame(test_values, index=test.index, columns=columns),
            {
                "method": "pca",
                "dimension": dimension,
                "explained_variance_ratio": model.explained_variance_ratio_.tolist(),
                "out_of_sample_method": "pca_transform",
            },
        )
    if method == "pcoa":
        return _fit_pcoa_train_test(train, test, dimension, cfg)
    if method == "umap":
        try:
            import umap
        except ImportError as exc:
            raise ImportError("UMAP prediction requires 'umap-learn'.") from exc
        n_neighbors = min(int(cfg.get("n_neighbors", 15)), max(2, len(train) - 1))
        kwargs = dict(cfg.get("kwargs", {}))
        kwargs.setdefault("n_jobs", 1)
        model = umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=dimension,
            min_dist=float(cfg.get("min_dist", 0.1)),
            metric=str(cfg.get("metric", "euclidean")),
            random_state=int(cfg.get("random_state", 42)),
            **kwargs,
        )
        train_values = model.fit_transform(train.to_numpy(dtype=float))
        test_values = model.transform(test.to_numpy(dtype=float))
        columns = _component_columns("UMAP", dimension)
        return TrainTestBenchmark(
            pd.DataFrame(train_values, index=train.index, columns=columns),
            pd.DataFrame(test_values, index=test.index, columns=columns),
            {
                "method": "umap",
                "dimension": dimension,
                "n_neighbors": n_neighbors,
                "metric": str(cfg.get("metric", "euclidean")),
                "out_of_sample_method": "umap_native_transform",
            },
        )
    if method == "tsne":
        return TrainTestBenchmark(
            train=pd.DataFrame(index=train.index),
            test=None,
            metadata={
                "method": "tsne",
                "dimension": dimension,
                "available": False,
                "out_of_sample_method": None,
                "reason": (
                    "scikit-learn TSNE has no transform for held-out samples. Strict outer-fold "
                    "prediction leaves t-SNE scores NA instead of fitting on test data or using "
                    "an ad-hoc surrogate projection."
                ),
            },
        )
    raise ValueError(f"Unknown benchmark method: {method}")

def run_benchmark_task(
    method: str,
    dimension: int,
    fit_input_path: str,
    method_cfg: dict[str, Any],
    output_dir: str,
    fingerprint: str,
    overwrite: bool,
) -> dict[str, Any]:
    """Process worker entrypoint with per-task resumability."""
    started = perf_counter()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    marker_path = out / "_SUCCESS.json"
    embedding_path = out / "embedding.tsv"
    metadata_path = out / "benchmark.json"
    required_extras = {
        "pca": [out / "components.tsv", out / "mean.tsv", out / "explained_variance.tsv"],
        "pcoa": [out / "eigenvalues.tsv"],
    }.get(method, [])

    if (
        not overwrite
        and marker_path.exists()
        and embedding_path.exists()
        and metadata_path.exists()
        and all(path.exists() for path in required_extras)
    ):
        try:
            marker = json.loads(marker_path.read_text())
            if marker.get("fingerprint") == fingerprint:
                return {
                    "method": method,
                    "dimension": dimension,
                    "status": "reused",
                    "embedding_path": str(embedding_path),
                    "metadata_path": str(metadata_path),
                    "extra_paths": [str(path) for path in required_extras],
                }
        except (OSError, json.JSONDecodeError):
            pass

    data = read_sample_indexed_tsv(fit_input_path)
    coords, metadata, extras = compute_benchmark(data, method, dimension, method_cfg)
    coords.to_csv(embedding_path, sep="\t", index_label="sample")
    extra_paths: list[str] = []
    for filename, frame in extras.items():
        path = out / filename
        index = filename in {"components.tsv", "mean.tsv"}
        frame.to_csv(path, sep="\t", index=index)
        extra_paths.append(str(path))

    metadata = {
        **metadata,
        "n_samples": int(coords.shape[0]),
        "input_features": int(data.shape[1]),
        "embedding_columns": coords.columns.tolist(),
        "extras": extra_paths,
    }
    atomic_write_json(metadata_path, metadata)
    duration = perf_counter() - started
    atomic_write_json(
        marker_path,
        {
            "fingerprint": fingerprint,
            "method": method,
            "dimension": dimension,
            "embedding": str(embedding_path),
            "metadata": str(metadata_path),
            "duration_seconds": round(float(duration), 6),
        },
    )
    return {
        "method": method,
        "dimension": dimension,
        "status": "ran",
        "embedding_path": str(embedding_path),
        "metadata_path": str(metadata_path),
        "extra_paths": extra_paths,
        "duration_seconds": round(float(duration), 6),
    }

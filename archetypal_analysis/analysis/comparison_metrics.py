"""Comparable quantitative metrics for AA and dimensionality-reduction benchmarks.

The functions in this module deliberately separate three ideas:

* representation quality: Silhouette, Davies-Bouldin, fuzzy silhouette, and convex-hull
  coverage are calculated in the method's representation coordinates;
* metadata association: PERMANOVA-style pseudo-F and fuzzy normalized mutual
  information quantify how strongly the representation/fuzzy partition relates to the
  configured metadata target;
* reconstruction: a common least-squares decoder supplies one RSS that is defined for
  every representation. Native reconstruction RSS is reported separately where the
  method has a genuine inverse model (AA and PCA).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, QhullError
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_samples, silhouette_score


@dataclass
class FuzzyPartition:
    memberships: np.ndarray
    labels: np.ndarray
    centers: np.ndarray
    iterations: int


def normalize_memberships(values: np.ndarray) -> np.ndarray:
    memberships = np.asarray(values, dtype=float)
    if memberships.ndim != 2:
        raise ValueError("Membership matrix must be two-dimensional")
    memberships = np.clip(memberships, a_min=0.0, a_max=None)
    row_sums = memberships.sum(axis=1, keepdims=True)
    n_clusters = memberships.shape[1]
    if n_clusters == 0:
        return memberships
    zero = row_sums[:, 0] <= 0
    out = np.divide(
        memberships,
        row_sums,
        out=np.zeros_like(memberships),
        where=row_sums > 0,
    )
    if np.any(zero):
        out[zero] = 1.0 / n_clusters
    return out


def _memberships_from_distances(distances: np.ndarray, fuzzifier: float) -> np.ndarray:
    if fuzzifier <= 1:
        raise ValueError("Fuzzy c-means fuzzifier must be > 1")
    distances = np.asarray(distances, dtype=float)
    n_samples, n_clusters = distances.shape
    memberships = np.zeros((n_samples, n_clusters), dtype=float)
    exponent = 2.0 / (fuzzifier - 1.0)
    tiny = np.finfo(float).eps

    for i in range(n_samples):
        row = distances[i]
        zero = row <= tiny
        if np.any(zero):
            memberships[i, zero] = 1.0 / int(np.sum(zero))
            continue
        ratios = (row[:, None] / row[None, :]) ** exponent
        memberships[i] = 1.0 / ratios.sum(axis=1)
    return normalize_memberships(memberships)


def fuzzy_c_means(
    points: np.ndarray,
    n_clusters: int,
    *,
    fuzzifier: float = 2.0,
    max_iter: int = 300,
    tol: float = 1e-5,
    random_state: int = 42,
) -> FuzzyPartition:
    """Deterministic fuzzy c-means initialized with KMeans centers."""
    x = np.asarray(points, dtype=float)
    if x.ndim != 2:
        raise ValueError("points must be a 2D array")
    if n_clusters < 2:
        raise ValueError("n_clusters must be >= 2")
    if n_clusters >= len(x):
        raise ValueError("n_clusters must be smaller than the number of samples")
    if not np.isfinite(x).all():
        raise ValueError("Fuzzy c-means input contains NaN or infinite values")

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    kmeans.fit(x)
    centers = np.asarray(kmeans.cluster_centers_, dtype=float)
    memberships = _memberships_from_distances(
        np.linalg.norm(x[:, None, :] - centers[None, :, :], axis=2),
        fuzzifier,
    )

    iterations = 0
    for iterations in range(1, max_iter + 1):
        weighted = memberships ** fuzzifier
        denom = weighted.sum(axis=0)[:, None]
        if np.any(denom <= 0):
            break
        new_centers = (weighted.T @ x) / denom
        distances = np.linalg.norm(x[:, None, :] - new_centers[None, :, :], axis=2)
        new_memberships = _memberships_from_distances(distances, fuzzifier)
        delta = float(np.max(np.abs(new_memberships - memberships)))
        centers = new_centers
        memberships = new_memberships
        if delta <= tol:
            break

    return FuzzyPartition(
        memberships=memberships,
        labels=np.argmax(memberships, axis=1),
        centers=centers,
        iterations=iterations,
    )


def safe_silhouette_and_db(points: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    x = np.asarray(points, dtype=float)
    labels = np.asarray(labels)
    unique = np.unique(labels)
    if len(x) < 3 or len(unique) < 2 or len(unique) >= len(x):
        return np.nan, np.nan
    try:
        silhouette = float(silhouette_score(x, labels))
    except Exception:
        silhouette = np.nan
    try:
        db = float(davies_bouldin_score(x, labels))
    except Exception:
        db = np.nan
    return silhouette, db


def fuzzy_silhouette_score(
    points: np.ndarray,
    memberships: np.ndarray,
    *,
    alpha: float = 1.0,
) -> float:
    """Campello/Hruschka-style fuzzy silhouette.

    A crisp silhouette is first calculated from the dominant fuzzy cluster. Each sample
    is then weighted by ``(u_max - u_second)^alpha`` so ambiguous samples contribute less.
    """
    x = np.asarray(points, dtype=float)
    memberships = normalize_memberships(memberships)
    if memberships.shape[0] != len(x) or memberships.shape[1] < 2:
        return np.nan
    labels = np.argmax(memberships, axis=1)
    if len(np.unique(labels)) < 2 or len(np.unique(labels)) >= len(labels):
        return np.nan
    try:
        crisp = silhouette_samples(x, labels)
    except Exception:
        return np.nan
    ordered = np.sort(memberships, axis=1)
    weights = np.maximum(ordered[:, -1] - ordered[:, -2], 0.0) ** float(alpha)
    total = float(weights.sum())
    if total <= 0:
        return np.nan
    return float(np.dot(weights, crisp) / total)


def _categorical_permanova_pseudo_f(points: np.ndarray, labels: pd.Series) -> tuple[float, int, int]:
    x = np.asarray(points, dtype=float)
    y = labels.astype(str).to_numpy()
    groups = pd.unique(y)
    n = len(x)
    g = len(groups)
    if n < 3 or g < 2 or g >= n:
        return np.nan, max(g - 1, 0), max(n - g, 0)

    grand = x.mean(axis=0)
    ss_between = 0.0
    ss_within = 0.0
    for group in groups:
        subset = x[y == group]
        mean = subset.mean(axis=0)
        ss_between += len(subset) * float(np.sum((mean - grand) ** 2))
        ss_within += float(np.sum((subset - mean) ** 2))
    df_between = g - 1
    df_within = n - g
    if df_between <= 0 or df_within <= 0 or ss_within <= 0:
        return np.nan, df_between, df_within
    score = (ss_between / df_between) / (ss_within / df_within)
    return float(score), df_between, df_within


def _continuous_permanova_pseudo_f(points: np.ndarray, values: pd.Series) -> tuple[float, int, int]:
    """Euclidean distance-based linear-model pseudo-F for one continuous predictor."""
    x = np.asarray(points, dtype=float)
    y = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    n = len(x)
    if n < 3 or not np.isfinite(y).all() or np.isclose(np.std(y), 0):
        return np.nan, 1, max(n - 2, 0)
    z = (y - y.mean()) / y.std(ddof=0)
    design = np.column_stack([np.ones(n), z])
    coef, *_ = np.linalg.lstsq(design, x, rcond=None)
    fitted = design @ coef
    intercept_only = np.repeat(x.mean(axis=0, keepdims=True), n, axis=0)
    ss_model = float(np.sum((fitted - intercept_only) ** 2))
    ss_residual = float(np.sum((x - fitted) ** 2))
    df_model = 1
    df_residual = n - 2
    if df_residual <= 0 or ss_residual <= 0:
        return np.nan, df_model, df_residual
    return float((ss_model / df_model) / (ss_residual / df_residual)), df_model, df_residual


def metadata_permanova_score(
    points: pd.DataFrame,
    metadata: pd.Series,
    legend_type: str,
) -> tuple[float, str, int, int, int]:
    common = points.index.intersection(metadata.index, sort=False)
    labels = metadata.reindex(common)
    if str(legend_type).lower() == "continuous":
        numeric = pd.to_numeric(labels, errors="coerce")
        valid = numeric.notna()
        score, df1, df2 = _continuous_permanova_pseudo_f(
            points.loc[valid].to_numpy(dtype=float), numeric.loc[valid]
        )
        return score, "continuous_distance_based_pseudo_f", int(valid.sum()), df1, df2

    valid = labels.notna()
    score, df1, df2 = _categorical_permanova_pseudo_f(
        points.loc[valid].to_numpy(dtype=float), labels.loc[valid]
    )
    return score, "categorical_one_way_pseudo_f", int(valid.sum()), df1, df2


def _discretize_with_buckets(values: pd.Series, buckets: list[list[float]]) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    out = pd.Series(index=values.index, dtype=object)
    for lo, hi in buckets:
        if hi is None or (isinstance(hi, float) and np.isinf(hi)):
            mask = numeric >= lo
            label = f"{lo}+"
        else:
            mask = (numeric >= lo) & (numeric < hi)
            label = f"{lo}-{hi}"
        out.loc[mask] = label
    return out


def metadata_categories_for_mi(
    values: pd.Series,
    task_cfg: Mapping[str, Any],
    evaluation_cfg: Mapping[str, Any],
    metric_cfg: Mapping[str, Any],
) -> tuple[pd.Series, str]:
    if str(task_cfg.get("legend_type", "discrete")).lower() != "continuous":
        return values.astype(object), "categorical"

    buckets = evaluation_cfg.get("buckets")
    if evaluation_cfg.get("discretize_target", False) and buckets:
        return _discretize_with_buckets(values, buckets), "continuous_configured_buckets"

    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return pd.Series(index=values.index, dtype=object), "continuous_quantile_bins"
    requested = max(2, int(metric_cfg.get("continuous_bins", 10)))
    q = min(requested, int(valid.nunique()))
    if q < 2:
        return pd.Series(index=values.index, dtype=object), "continuous_quantile_bins"
    binned = pd.qcut(valid, q=q, duplicates="drop")
    out = pd.Series(index=values.index, dtype=object)
    out.loc[binned.index] = binned.astype(str)
    return out, f"continuous_quantile_bins_{binned.nunique()}"


def fuzzy_normalized_mutual_information(
    memberships: np.ndarray,
    labels: pd.Series,
) -> tuple[float, int]:
    memberships = normalize_memberships(memberships)
    labels = labels.reset_index(drop=True)
    valid = labels.notna().to_numpy()
    memberships = memberships[valid]
    labels = labels.loc[valid].astype(str).to_numpy()
    n = len(labels)
    if n == 0 or memberships.shape[1] < 2:
        return np.nan, n

    categories = sorted(pd.unique(labels).tolist())
    if len(categories) < 2:
        return np.nan, n
    joint = np.zeros((len(categories), memberships.shape[1]), dtype=float)
    for index, category in enumerate(categories):
        joint[index] = memberships[labels == category].sum(axis=0)
    joint /= float(n)
    p_category = joint.sum(axis=1)
    p_cluster = joint.sum(axis=0)

    positive = joint > 0
    denom = p_category[:, None] * p_cluster[None, :]
    mutual_information = float(
        np.sum(joint[positive] * np.log(joint[positive] / denom[positive]))
    )
    h_category = float(-np.sum(p_category[p_category > 0] * np.log(p_category[p_category > 0])))
    h_cluster = float(-np.sum(p_cluster[p_cluster > 0] * np.log(p_cluster[p_cluster > 0])))
    normalizer = np.sqrt(h_category * h_cluster)
    if normalizer <= 0:
        return np.nan, n
    return float(np.clip(mutual_information / normalizer, 0.0, 1.0)), n


def linear_decoder_rss(representation: pd.DataFrame, original: pd.DataFrame) -> tuple[float, int]:
    common = representation.index.intersection(original.index, sort=False)
    z = representation.loc[common].to_numpy(dtype=float)
    x = original.loc[common].to_numpy(dtype=float)
    if len(common) == 0:
        return np.nan, 0
    design = np.column_stack([np.ones(len(z)), z])
    coefficients, *_ = np.linalg.lstsq(design, x, rcond=None)
    reconstructed = design @ coefficients
    return float(np.sum((x - reconstructed) ** 2)), int(len(common))


def convex_hull_coverage_ratio(
    points: np.ndarray,
    *,
    max_dimension: int = 6,
) -> tuple[float, int, int]:
    """Hull volume divided by the axis-aligned bounding-box volume.

    The metric is scale invariant because the retained intrinsic coordinates are
    min-max normalized first. To keep exact QHull computation practical at large K,
    data with intrinsic rank above ``max_dimension`` are deterministically projected
    onto their leading SVD axes before the ratio is calculated.
    """
    x = np.asarray(points, dtype=float)
    x = x[np.isfinite(x).all(axis=1)]
    if len(x) < 3 or x.shape[1] == 0:
        return np.nan, 0, 0
    centered = x - x.mean(axis=0, keepdims=True)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    if singular.size == 0:
        return np.nan, 0, 0
    tol = max(centered.shape) * np.finfo(float).eps * float(singular[0])
    intrinsic_rank = int(np.sum(singular > tol))
    if intrinsic_rank <= 0:
        return np.nan, 0, intrinsic_rank
    used = min(intrinsic_rank, int(max_dimension))
    projected = centered @ vt[:used].T
    lo = projected.min(axis=0)
    hi = projected.max(axis=0)
    span = hi - lo
    keep = span > np.finfo(float).eps
    projected = projected[:, keep]
    span = span[keep]
    used = projected.shape[1]
    if used == 0:
        return np.nan, 0, intrinsic_rank
    scaled = (projected - projected.min(axis=0)) / span
    if used == 1:
        return 1.0, 1, intrinsic_rank
    if len(scaled) <= used:
        return np.nan, used, intrinsic_rank
    try:
        hull = ConvexHull(scaled, qhull_options="QJ")
        ratio = float(np.clip(hull.volume, 0.0, 1.0))
    except QhullError:
        ratio = np.nan
    return ratio, used, intrinsic_rank


def evaluate_representation(
    *,
    method: str,
    dimension: int,
    representation: pd.DataFrame,
    memberships: np.ndarray,
    original_input: pd.DataFrame,
    metadata: pd.Series,
    task_cfg: Mapping[str, Any],
    evaluation_cfg: Mapping[str, Any],
    metric_cfg: Mapping[str, Any],
    native_reconstruction_rss: float = np.nan,
    native_reconstruction_method: str = "not_available",
) -> dict[str, Any]:
    common = representation.index.intersection(original_input.index, sort=False)
    representation = representation.loc[common]
    memberships = normalize_memberships(np.asarray(memberships, dtype=float))
    if memberships.shape[0] != len(representation):
        raise ValueError(
            f"Membership rows ({memberships.shape[0]}) do not match representation rows ({len(representation)})"
        )
    hard_labels = np.argmax(memberships, axis=1)
    silhouette, db = safe_silhouette_and_db(representation.to_numpy(dtype=float), hard_labels)
    fuzzy_silhouette = fuzzy_silhouette_score(
        representation.to_numpy(dtype=float),
        memberships,
        alpha=float(metric_cfg.get("fuzzy_silhouette_alpha", 1.0)),
    )

    metadata_aligned = metadata.reindex(representation.index)
    permanova, permanova_mode, permanova_n, permanova_df1, permanova_df2 = metadata_permanova_score(
        representation,
        metadata_aligned,
        str(task_cfg.get("legend_type", "discrete")),
    )
    mi_categories, mi_encoding = metadata_categories_for_mi(
        metadata_aligned,
        task_cfg,
        evaluation_cfg,
        metric_cfg,
    )
    fuzzy_nmi, fuzzy_nmi_n = fuzzy_normalized_mutual_information(
        memberships,
        mi_categories.reset_index(drop=True),
    )
    rss, reconstruction_n = linear_decoder_rss(representation, original_input)
    hull, hull_dimension, hull_rank = convex_hull_coverage_ratio(
        representation.to_numpy(dtype=float),
        max_dimension=int(metric_cfg.get("convex_hull_max_dimension", 6)),
    )

    return {
        "method": method,
        "dimension": int(dimension),
        "n_samples": int(len(representation)),
        "n_clusters": int(memberships.shape[1]),
        "silhouette": silhouette,
        "davies_bouldin": db,
        "fuzzy_silhouette": fuzzy_silhouette,
        "metadata_permanova": permanova,
        "metadata_permanova_mode": permanova_mode,
        "metadata_permanova_n": permanova_n,
        "metadata_permanova_df1": permanova_df1,
        "metadata_permanova_df2": permanova_df2,
        "metadata_fuzzy_nmi": fuzzy_nmi,
        "metadata_mi_encoding": mi_encoding,
        "metadata_fuzzy_nmi_n": fuzzy_nmi_n,
        "reconstruction_rss": rss,
        "reconstruction_samples": reconstruction_n,
        "reconstruction_method": "least_squares_linear_decoder",
        "native_reconstruction_rss": native_reconstruction_rss,
        "native_reconstruction_method": native_reconstruction_method,
        "convex_hull_coverage_ratio": hull,
        "convex_hull_dimension_used": hull_dimension,
        "convex_hull_intrinsic_rank": hull_rank,
    }

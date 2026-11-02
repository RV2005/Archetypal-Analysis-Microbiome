"""Archetypal-analysis stability, matching, subsampling, and held-out projection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ..data.indexed_io import read_sample_indexed_tsv
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split

from ..pipeline.artifacts import atomic_write_json
from .experiments import inverse_archetypes


@dataclass(frozen=True)
class StabilityTask:
    kind: str
    k: int
    seed: int
    sample_ids: tuple[str, ...]
    output_dir: str
    fingerprint: str
    fraction: float | None = None
    repeat: int | None = None


def _safe_row_normalize_l2(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, norms, out=np.zeros_like(x), where=norms > 0)


def archetype_profile_space(archetypes: np.ndarray, transform: Mapping[str, Any], space: str) -> np.ndarray:
    space = str(space).lower()
    if space == "fitted":
        return np.asarray(archetypes, dtype=float)
    if space == "inverse":
        return inverse_archetypes(np.asarray(archetypes, dtype=float), transform)
    raise ValueError("stability.profile_space must be 'fitted' or 'inverse'")


def archetype_pairwise_similarity(reference: np.ndarray, candidate: np.ndarray, metric: str) -> np.ndarray:
    metric = str(metric).lower()
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    if metric == "cosine":
        sim = 1.0 - cdist(reference, candidate, metric="cosine")
    elif metric == "correlation":
        sim = 1.0 - cdist(reference, candidate, metric="correlation")
    else:
        raise ValueError("stability.matching_metric must be 'cosine' or 'correlation'")
    return np.nan_to_num(sim, nan=-1.0, posinf=-1.0, neginf=-1.0)


def normalized_wasserstein_distance(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Uniform empirical Wasserstein-1 distance on L2-normalized archetype profiles.

    With equal K and uniform weights, Wasserstein-1 under Euclidean ground cost is the
    minimum mean assignment cost. L2-normalized profiles lie on/in the unit sphere, so
    Euclidean ground distances are bounded by 2; dividing by 2 gives a scale-free [0, 1]
    quantity. Zero profiles remain at the origin.
    """
    ref = _safe_row_normalize_l2(reference)
    cand = _safe_row_normalize_l2(candidate)
    if len(ref) != len(cand) or len(ref) == 0:
        return np.nan
    costs = cdist(ref, cand, metric="euclidean")
    rows, cols = linear_sum_assignment(costs)
    return float(np.mean(costs[rows, cols]) / 2.0)


def compare_archetypes(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    matching_metric: str = "cosine",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Match archetypes globally and report requested per-archetype/set distances."""
    ref = np.asarray(reference, dtype=float)
    cand = np.asarray(candidate, dtype=float)
    if ref.ndim != 2 or cand.ndim != 2 or ref.shape != cand.shape:
        raise ValueError(f"Archetype matrices must have equal 2D shape; got {ref.shape} and {cand.shape}")

    similarity = archetype_pairwise_similarity(ref, cand, matching_metric)
    assignment_rows, assignment_cols = linear_sum_assignment(-similarity)
    assignment = {int(i): int(j) for i, j in zip(assignment_rows, assignment_cols)}

    cosine = 1.0 - cdist(ref, cand, metric="cosine")
    cosine = np.nan_to_num(cosine, nan=np.nan)
    euclidean = cdist(ref, cand, metric="euclidean")
    rows: list[dict[str, Any]] = []
    for i in range(len(ref)):
        j = assignment[i]
        finite_cosine = cosine[i][np.isfinite(cosine[i])]
        rows.append(
            {
                "reference_archetype": f"Archetype {i + 1}",
                "matched_archetype": f"Archetype {j + 1}",
                "matching_similarity": float(similarity[i, j]),
                "matched_cosine_similarity": float(cosine[i, j]) if np.isfinite(cosine[i, j]) else np.nan,
                "matched_euclidean_distance": float(euclidean[i, j]),
                "max_cosine_similarity": float(np.max(finite_cosine)) if finite_cosine.size else np.nan,
                "min_euclidean_distance": float(np.min(euclidean[i])),
            }
        )
    table = pd.DataFrame(rows)
    overall = {
        "matching_metric": str(matching_metric),
        "median_matched_similarity": float(table["matching_similarity"].median()),
        "median_max_cosine_similarity": float(table["max_cosine_similarity"].median()),
        "median_min_euclidean_distance": float(table["min_euclidean_distance"].median()),
        "normalized_wasserstein_distance": normalized_wasserstein_distance(ref, cand),
    }
    return table, overall


def project_rows_to_simplex(values: np.ndarray) -> np.ndarray:
    """Euclidean projection of each row onto the probability simplex."""
    v = np.asarray(values, dtype=float)
    if v.ndim == 1:
        v = v[None, :]
    u = np.sort(v, axis=1)[:, ::-1]
    cssv = np.cumsum(u, axis=1) - 1.0
    ind = np.arange(1, v.shape[1] + 1, dtype=float)
    cond = u - cssv / ind > 0
    rho = cond.sum(axis=1) - 1
    theta = cssv[np.arange(len(v)), rho] / (rho + 1.0)
    return np.maximum(v - theta[:, None], 0.0)


def project_onto_archetypes(
    samples: np.ndarray,
    archetypes: np.ndarray,
    *,
    max_iter: int = 2000,
    tol: float = 1e-8,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Project new samples onto fixed archetypes with simplex-constrained least squares.

    Solves ``min_A ||X - A Z||_F^2`` subject to every row of A being non-negative and
    summing to one using vectorized projected gradient descent.
    """
    x = np.asarray(samples, dtype=float)
    z = np.asarray(archetypes, dtype=float)
    if x.ndim != 2 or z.ndim != 2 or x.shape[1] != z.shape[1]:
        raise ValueError("samples and archetypes must be 2D and share feature dimension")
    if len(z) < 2:
        raise ValueError("Held-out projection requires at least two archetypes")
    gram = z @ z.T
    lipschitz = 2.0 * float(np.linalg.norm(gram, ord=2))
    step = 1.0 / max(lipschitz, np.finfo(float).eps)
    alpha = np.full((len(x), len(z)), 1.0 / len(z), dtype=float)
    previous = np.inf
    iterations = 0
    for iterations in range(1, int(max_iter) + 1):
        gradient = 2.0 * ((alpha @ z - x) @ z.T)
        updated = project_rows_to_simplex(alpha - step * gradient)
        delta = float(np.max(np.abs(updated - alpha))) if updated.size else 0.0
        alpha = updated
        if iterations % 25 == 0 or delta <= tol:
            residual = x - alpha @ z
            objective = float(np.sum(residual * residual))
            if delta <= tol or (
                np.isfinite(previous)
                and abs(previous - objective) <= tol * max(1.0, abs(previous))
            ):
                previous = objective
                break
            previous = objective
    residual = x - alpha @ z
    rss = float(np.sum(residual * residual))
    return alpha, {"iterations": int(iterations), "rss": rss, "converged": bool(iterations < int(max_iter))}


def _strata_for_sampling(
    metadata: pd.DataFrame,
    target: str,
    legend_type: str,
    continuous_bins: int,
    groups: pd.Series | None = None,
) -> pd.Series:
    values = metadata[target]
    if groups is None:
        if str(legend_type).lower() != "continuous":
            return values.astype(str)
        numeric = pd.to_numeric(values, errors="coerce")
        q = min(max(2, int(continuous_bins)), int(numeric.nunique(dropna=True)))
        return pd.qcut(numeric, q=q, duplicates="drop").astype(str)

    frame = pd.DataFrame({"target": values, "group": groups}).dropna(subset=["group"])
    if str(legend_type).lower() != "continuous":
        group_target = frame.groupby("group")["target"].agg(
            lambda s: s.astype(str).value_counts().sort_index().idxmax()
        )
        return group_target.astype(str)
    numeric = pd.to_numeric(frame["target"], errors="coerce")
    group_target = numeric.groupby(frame["group"]).median()
    q = min(max(2, int(continuous_bins)), int(group_target.nunique(dropna=True)))
    return pd.qcut(group_target, q=q, duplicates="drop").astype(str)


def stratified_subsample_ids(
    metadata: pd.DataFrame,
    target: str,
    legend_type: str,
    fraction: float,
    random_state: int,
    *,
    group_column: str | None = None,
    continuous_bins: int = 10,
) -> tuple[list[str], list[str]]:
    """Return deterministic train/held-out IDs, stratifying samples or whole groups."""
    if not 0 < float(fraction) < 1:
        raise ValueError("sampling fractions must be between 0 and 1")
    if target not in metadata.columns:
        raise KeyError(f"Sampling target '{target}' missing from metadata")
    local = metadata.loc[metadata[target].notna()].copy()
    if group_column:
        if group_column not in local.columns:
            raise KeyError(f"Sampling group_column '{group_column}' missing from metadata")
        local = local.loc[local[group_column].notna()]
        groups = local[group_column].astype(str)
        strata = _strata_for_sampling(local, target, legend_type, continuous_bins, groups)
        rng = np.random.RandomState(int(random_state))
        selected_groups: list[str] = []
        for _stratum, group_ids in strata.groupby(strata).groups.items():
            group_ids = np.asarray(list(group_ids), dtype=object)
            n = max(1, min(len(group_ids), int(round(float(fraction) * len(group_ids)))))
            selected_groups.extend(rng.choice(group_ids, size=n, replace=False).tolist())
        selected = set(map(str, selected_groups))
        train = local.index[groups.isin(selected)].astype(str).tolist()
        heldout = local.index[~groups.isin(selected)].astype(str).tolist()
        return train, heldout

    strata = _strata_for_sampling(local, target, legend_type, continuous_bins)
    indices = np.arange(len(local))
    try:
        train_pos, heldout_pos = train_test_split(
            indices,
            train_size=float(fraction),
            random_state=int(random_state),
            stratify=strata.to_numpy(),
        )
    except ValueError:
        # Very small strata can make exact stratification impossible. Falling back to
        # unstratified sampling is preferable to silently dropping categories.
        train_pos, heldout_pos = train_test_split(
            indices,
            train_size=float(fraction),
            random_state=int(random_state),
            stratify=None,
        )
    return local.index[train_pos].astype(str).tolist(), local.index[heldout_pos].astype(str).tolist()


# Process workers load the fit matrix once per process instead of reparsing it for every run.
_WORKER_DATA: pd.DataFrame | None = None


def initialize_stability_worker(fit_input_path: str) -> None:
    global _WORKER_DATA
    _WORKER_DATA = read_sample_indexed_tsv(fit_input_path)


def _serialize_result(result, k: int, feature_names: Sequence[str], fit_cfg: Mapping[str, Any], seed: int) -> dict[str, Any]:
    loss = result.loss
    if isinstance(loss, np.ndarray):
        loss = loss.tolist()
    elif not isinstance(loss, list):
        try:
            loss = list(loss)
        except TypeError:
            loss = [float(loss)]
    return {
        "schema_version": 2,
        "k": int(k),
        "archetypes": np.asarray(result.archetypes).tolist(),
        "similarity_degree": result.similarity_degree.to_numpy().tolist(),
        "labels": np.asarray(result.labels).tolist(),
        "index": result.similarity_degree.index.astype(str).tolist(),
        "columns": result.similarity_degree.columns.tolist(),
        "feature_names": list(feature_names),
        "loss": loss,
        "fit_config": {**dict(fit_cfg), "random_state": int(seed)},
    }


def run_stability_fit_task(
    task: StabilityTask,
    fit_cfg: dict[str, Any],
    overwrite: bool,
) -> dict[str, Any]:
    """Process worker for one initialization/subsampling AA fit."""
    started = perf_counter()
    out = Path(task.output_dir)
    marker = out / "_SUCCESS.json"
    result_path = out / "result.json"
    similarity_path = out / "similarity.tsv"
    archetypes_path = out / "archetypes.tsv"
    samples_path = out / "train_samples.tsv"
    required = [result_path, similarity_path, archetypes_path, samples_path]
    if not overwrite and marker.exists() and all(path.exists() for path in required):
        try:
            payload = json.loads(marker.read_text())
            if payload.get("fingerprint") == task.fingerprint:
                return {
                    "kind": task.kind,
                    "k": task.k,
                    "seed": task.seed,
                    "fraction": task.fraction,
                    "repeat": task.repeat,
                    "status": "reused",
                    "result_path": str(result_path),
                    "output_dir": str(out),
                }
        except (OSError, json.JSONDecodeError):
            pass

    global _WORKER_DATA
    if _WORKER_DATA is None:
        raise RuntimeError("Stability worker was not initialized with fit_input")
    missing = [sample for sample in task.sample_ids if sample not in _WORKER_DATA.index]
    if missing:
        raise KeyError(f"Stability sample IDs are missing from fit_input: {missing[:10]}")
    data = _WORKER_DATA.loc[list(task.sample_ids)]

    from .archetypal import ArchetypalAnalysis

    aa = ArchetypalAnalysis(
        n_archetypes=int(task.k),
        method=str(fit_cfg.get("method", "pgd")),
        max_iter=int(fit_cfg.get("max_iter", 100)),
        random_state=int(task.seed),
        method_kwargs=dict(fit_cfg.get("method_kwargs", {})),
        model_kwargs=dict(fit_cfg.get("model_kwargs", {})),
    )
    result = aa.fit_transform(
        data,
        apply_log_transform=False,
        sample_fraction=1.0,
        std_scale=False,
        verbose=bool(fit_cfg.get("verbose_fit", False)),
    )
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_json(result_path, _serialize_result(result, task.k, data.columns, fit_cfg, task.seed))
    result.similarity_degree.to_csv(similarity_path, sep="\t", index_label="sample")
    pd.DataFrame(
        np.asarray(result.archetypes),
        index=[f"Archetype {i + 1}" for i in range(task.k)],
        columns=data.columns,
    ).to_csv(archetypes_path, sep="\t", index_label="archetype")
    pd.DataFrame({"sample": list(task.sample_ids)}).to_csv(samples_path, sep="\t", index=False)
    duration = perf_counter() - started
    atomic_write_json(
        marker,
        {
            "fingerprint": task.fingerprint,
            "kind": task.kind,
            "k": int(task.k),
            "seed": int(task.seed),
            "fraction": task.fraction,
            "repeat": task.repeat,
            "duration_seconds": round(float(duration), 6),
        },
    )
    return {
        "kind": task.kind,
        "k": task.k,
        "seed": task.seed,
        "fraction": task.fraction,
        "repeat": task.repeat,
        "status": "ran",
        "result_path": str(result_path),
        "output_dir": str(out),
        "duration_seconds": round(float(duration), 6),
    }

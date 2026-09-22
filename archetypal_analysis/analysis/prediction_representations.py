"""Train-only representation fitting for outer-fold prediction experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..data.preprocessing import apply_preprocessing
from .benchmarks import TrainTestBenchmark, fit_benchmark_train_test
from .prediction import helmert_transform_memberships
from .stability import project_onto_archetypes


@dataclass(frozen=True)
class FoldPreprocessing:
    train: pd.DataFrame
    test: pd.DataFrame
    metadata: dict[str, Any]


@dataclass(frozen=True)
class AAFoldRepresentation:
    train_direct: pd.DataFrame
    test_direct: pd.DataFrame
    train_helmert: pd.DataFrame
    test_helmert: pd.DataFrame
    archetypes: pd.DataFrame
    metadata: dict[str, Any]


def fit_fold_preprocessing(
    raw_otu: pd.DataFrame,
    train_ids: pd.Index,
    test_ids: pd.Index,
    fit_cfg: Mapping[str, Any],
) -> FoldPreprocessing:
    """Learn every training-dependent AA preprocessing choice on the outer train fold.

    Abundance transforms such as log/relative abundance/CLR are sample-wise.  Feature
    variance filtering and standard scaling are fitted on training samples only and then
    applied unchanged to the held-out fold.
    """
    prep = dict(fit_cfg.get("preprocessing", {}) or {})
    train_raw = raw_otu.reindex(train_ids).copy()
    test_raw = raw_otu.reindex(test_ids).copy()
    if train_raw.isna().all(axis=1).any() or test_raw.isna().all(axis=1).any():
        raise ValueError("Prediction fold contains sample IDs absent from the post-QC OTU table")

    train, transform = apply_preprocessing(train_raw, prep)
    test, _ = apply_preprocessing(test_raw, prep)
    train = train.replace([np.inf, -np.inf], np.nan)
    test = test.replace([np.inf, -np.inf], np.nan)
    if train.isna().any(axis=1).any() or test.isna().any(axis=1).any():
        bad_train = train.index[train.isna().any(axis=1)].astype(str).tolist()[:10]
        bad_test = test.index[test.isna().any(axis=1)].astype(str).tolist()[:10]
        raise ValueError(
            "Fold-specific preprocessing produced NaN/Inf rows. "
            f"train examples={bad_train}, test examples={bad_test}"
        )

    if prep.get("drop_zero_variance_features", True):
        std = train.std(axis=0)
        keep = std[(std > 0) & np.isfinite(std)].index
        train = train.loc[:, keep]
        test = test.loc[:, keep]
    if train.shape[1] == 0:
        raise ValueError("Fold-specific preprocessing removed every feature")

    standard_scale = bool(prep.get("standard_scale", False))
    if standard_scale:
        mean = train.mean(axis=0)
        std = train.std(axis=0)
        keep = std[(std > 0) & np.isfinite(std)].index
        train = train.loc[:, keep]
        test = test.loc[:, keep]
        mean = mean.loc[keep]
        std = std.loc[keep]
        train = (train - mean) / std
        test = (test - mean) / std
        transform["mean"] = mean.tolist()
        transform["std"] = std.tolist()
    else:
        transform["mean"] = None
        transform["std"] = None

    transform.update(
        {
            "standard_scale": standard_scale,
            "features": train.columns.astype(str).tolist(),
            "training_samples": train.index.astype(str).tolist(),
            "test_samples": test.index.astype(str).tolist(),
            "sample_fraction": 1.0,
            "fold_fitted": True,
        }
    )
    return FoldPreprocessing(train=train.astype(float), test=test.astype(float), metadata=transform)


def fit_aa_train_test(
    train: pd.DataFrame,
    test: pd.DataFrame,
    k: int,
    fit_cfg: Mapping[str, Any],
    projection_cfg: Mapping[str, Any] | None = None,
) -> AAFoldRepresentation:
    """Fit AA on training samples and simplex-project held-out samples onto fixed archetypes."""
    # Lazy import keeps utility/config tests usable without the optional heavy AA dependency.
    from .archetypal import ArchetypalAnalysis

    aa = ArchetypalAnalysis(
        n_archetypes=int(k),
        method=str(fit_cfg.get("method", "pgd")),
        max_iter=int(fit_cfg.get("max_iter", 100)),
        random_state=int(fit_cfg.get("random_state", 3407)),
        method_kwargs=dict(fit_cfg.get("method_kwargs", {})),
        model_kwargs=dict(fit_cfg.get("model_kwargs", {})),
    )
    result = aa.fit_transform(
        train,
        apply_log_transform=False,
        sample_fraction=1.0,
        std_scale=False,
        verbose=bool(fit_cfg.get("verbose_fit", False)),
    )
    train_direct = result.similarity_degree.copy()
    columns = train_direct.columns.tolist()
    archetypes = pd.DataFrame(
        np.asarray(result.archetypes, dtype=float),
        index=columns,
        columns=train.columns,
    )

    projection_cfg = dict(projection_cfg or {})
    test_alpha, projection_info = project_onto_archetypes(
        test.to_numpy(dtype=float),
        archetypes.to_numpy(dtype=float),
        max_iter=int(projection_cfg.get("max_iter", 2000)),
        tol=float(projection_cfg.get("tol", 1e-8)),
    )
    test_direct = pd.DataFrame(test_alpha, index=test.index, columns=columns)
    train_helmert = helmert_transform_memberships(train_direct)
    test_helmert = helmert_transform_memberships(test_direct)
    assert isinstance(train_helmert, pd.DataFrame) and isinstance(test_helmert, pd.DataFrame)
    return AAFoldRepresentation(
        train_direct=train_direct,
        test_direct=test_direct,
        train_helmert=train_helmert,
        test_helmert=test_helmert,
        archetypes=archetypes,
        metadata={
            "k": int(k),
            "native_columns": int(k),
            "intrinsic_dimension": int(k - 1),
            "heldout_projection": projection_info,
            "fit_random_state": int(fit_cfg.get("random_state", 3407)),
        },
    )


def fit_benchmark_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    method: str,
    dimension: int,
    benchmark_cfg: Mapping[str, Any],
) -> TrainTestBenchmark:
    method_cfg = dict((benchmark_cfg.get(method, {}) or {}))
    method_cfg.pop("enabled", None)
    return fit_benchmark_train_test(train, test, method, int(dimension), method_cfg)

"""Cross-validated metadata prediction utilities.

The outer prediction workflow fits every unsupervised representation on the outer-training
samples only.  The supervised predictor is then trained on those training coordinates and
scored on held-out samples projected with the *training-fitted* representation.

For AA, the K simplex weights have only K-1 independent degrees of freedom.  Two AA probes are
therefore reported: the original K membership columns (useful because they are the native AA
output) and a deterministic orthonormal Helmert coordinate system with K-1 columns.  All
conventional dimensionality-reduction benchmarks use K-1 dimensions for comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:  # available in modern scikit-learn, but keep a graceful fallback
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover - exercised only on older sklearn
    StratifiedGroupKFold = None


@dataclass(frozen=True)
class CVPlan:
    index: pd.Index
    folds: np.ndarray
    n_splits: int
    grouped: bool
    group_column: str | None

    def split_indices(self) -> list[tuple[np.ndarray, np.ndarray]]:
        values = np.asarray(self.folds, dtype=int)
        positions = np.arange(len(values))
        return [(positions[values != fold], positions[values == fold]) for fold in range(self.n_splits)]

    def frame(self, groups: pd.Series | None = None) -> pd.DataFrame:
        out = pd.DataFrame({"fold": self.folds}, index=self.index)
        out.index.name = "sample"
        if groups is not None:
            out["group"] = groups.reindex(self.index).to_numpy()
        return out


def _effective_n_splits(
    y: pd.Series,
    requested: int,
    categorical: bool,
    groups: pd.Series | None,
) -> int:
    requested = max(2, int(requested))
    if groups is not None:
        unique_groups = int(groups.nunique(dropna=True))
        if unique_groups < 2:
            raise ValueError("Group-aware cross-validation requires at least two unique groups")
        requested = min(requested, unique_groups)
        if categorical:
            frame = pd.DataFrame({"label": y.astype(str), "group": groups})
            counts = frame.groupby("label")["group"].nunique()
            if len(counts):
                requested = min(requested, int(counts.min()))
    elif categorical:
        counts = y.astype(str).value_counts()
        if len(counts):
            requested = min(requested, int(counts.min()))
    else:
        requested = min(requested, len(y))
    if requested < 2:
        raise ValueError("Not enough independent samples/groups to create at least two CV folds")
    return requested


def _splitter(
    y: pd.Series,
    *,
    categorical: bool,
    groups: pd.Series | None,
    n_splits: int,
    random_state: int,
):
    x_dummy = np.zeros((len(y), 1), dtype=float)
    if groups is not None and categorical and StratifiedGroupKFold is not None:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=int(random_state))
        return list(splitter.split(x_dummy, y.astype(str).to_numpy(), groups.to_numpy()))
    if groups is not None:
        splitter = GroupKFold(n_splits=n_splits)
        return list(splitter.split(x_dummy, y.to_numpy(), groups.to_numpy()))
    if categorical:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(random_state))
        return list(splitter.split(x_dummy, y.astype(str).to_numpy()))
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=int(random_state))
    return list(splitter.split(x_dummy))


def build_cv_plan(
    metadata: pd.DataFrame,
    target: str,
    legend_type: str,
    *,
    n_splits: int = 5,
    random_state: int = 42,
    group_column: str | None = None,
) -> tuple[CVPlan, pd.Series, pd.Series | None]:
    """Build one reusable outer-fold assignment for every compared method."""
    if target not in metadata.columns:
        raise KeyError(f"Prediction target '{target}' is not present in metadata")
    categorical = str(legend_type).lower() != "continuous"
    y = metadata[target]
    if categorical:
        valid = y.notna()
        y_clean = y.loc[valid].astype(str)
    else:
        numeric = pd.to_numeric(y, errors="coerce")
        valid = numeric.notna()
        y_clean = numeric.loc[valid]

    groups: pd.Series | None = None
    if group_column:
        if group_column not in metadata.columns:
            raise KeyError(
                f"Prediction group_column '{group_column}' is not present in retained metadata."
            )
        valid &= metadata[group_column].notna()
        y_clean = (y.astype(str) if categorical else pd.to_numeric(y, errors="coerce")).loc[valid]
        groups = metadata.loc[valid, group_column].astype(str)

    index = y_clean.index
    if len(index) < 3:
        raw_non_missing = int(y.notna().sum())
        details = (
            f"target={target!r}, metadata_rows={len(metadata)}, "
            f"raw_non_missing_target={raw_non_missing}, usable_target_rows={len(index)}"
        )
        if not categorical:
            numeric_total = int(pd.to_numeric(y, errors="coerce").notna().sum())
            details += f", numeric_target_rows={numeric_total}"
        if group_column:
            group_non_missing = (
                int(metadata[group_column].notna().sum()) if group_column in metadata.columns else 0
            )
            details += f", group_column={group_column!r}, non_missing_groups={group_non_missing}"
        raise ValueError(
            "Prediction requires at least three usable samples after target/group filtering; "
            + details
        )

    effective = _effective_n_splits(y_clean, n_splits, categorical, groups)
    splits = _splitter(
        y_clean,
        categorical=categorical,
        groups=groups,
        n_splits=effective,
        random_state=int(random_state),
    )
    fold_ids = np.full(len(index), -1, dtype=int)
    for fold, (_train, test) in enumerate(splits):
        fold_ids[np.asarray(test, dtype=int)] = fold
    if np.any(fold_ids < 0):
        raise RuntimeError("CV splitter did not assign every sample to a test fold")

    return (
        CVPlan(
            index=pd.Index(index, name="sample"),
            folds=fold_ids,
            n_splits=effective,
            grouped=groups is not None,
            group_column=group_column,
        ),
        y_clean,
        groups,
    )


def helmert_basis(k: int) -> np.ndarray:
    """Return a deterministic K x (K-1) orthonormal Helmert basis.

    Columns span the hyperplane orthogonal to the all-ones vector.  Multiplying AA simplex
    weights by this basis is lossless with respect to their K-1 independent degrees of freedom
    (the omitted direction is fixed by the sum-to-one constraint).
    """
    k = int(k)
    if k < 2:
        raise ValueError("Helmert coordinates require at least two archetypes")
    basis = np.zeros((k, k - 1), dtype=float)
    for j in range(1, k):
        scale = np.sqrt(j * (j + 1.0))
        basis[:j, j - 1] = 1.0 / scale
        basis[j, j - 1] = -j / scale
    return basis


def helmert_transform_memberships(memberships: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
    """Map K AA membership columns deterministically to K-1 orthonormal coordinates."""
    if isinstance(memberships, pd.DataFrame):
        values = memberships.to_numpy(dtype=float)
        basis = helmert_basis(values.shape[1])
        columns = [f"Helmert{i}" for i in range(1, values.shape[1])]
        return pd.DataFrame(values @ basis, index=memberships.index, columns=columns)
    values = np.asarray(memberships, dtype=float)
    if values.ndim != 2:
        raise ValueError("AA memberships must be a two-dimensional matrix")
    return values @ helmert_basis(values.shape[1])


def _classifier(cfg: Mapping[str, Any], *, C: float) -> Pipeline:
    model_cfg = dict(cfg.get("logistic_regression", {}) or {})
    kwargs = {
        "max_iter": int(model_cfg.get("max_iter", 5000)),
        "C": float(C),
        "class_weight": model_cfg.get("class_weight"),
        "solver": str(model_cfg.get("solver", "lbfgs")),
        "random_state": int(model_cfg.get("random_state", cfg.get("random_state", 42))),
    }
    if "multi_class" in inspect.signature(LogisticRegression).parameters:
        kwargs["multi_class"] = model_cfg.get("multi_class", "auto")
    model = LogisticRegression(**kwargs)
    return Pipeline([("scale", StandardScaler()), ("model", model)])


def _regressor(cfg: Mapping[str, Any]) -> Pipeline:
    model_cfg = dict(cfg.get("linear_regression", {}) or {})
    model = LinearRegression(fit_intercept=bool(model_cfg.get("fit_intercept", True)))
    return Pipeline([("scale", StandardScaler()), ("model", model)])


def _c_values(cfg: Mapping[str, Any]) -> list[float]:
    model_cfg = dict(cfg.get("logistic_regression", {}) or {})
    raw = model_cfg.get("C_values", model_cfg.get("C_grid"))
    if raw is None:
        raw = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    values = sorted({float(value) for value in raw})
    if not values or any(value <= 0 or not np.isfinite(value) for value in values):
        raise ValueError("prediction.logistic_regression.C_values must contain positive finite values")
    return values


def fit_predictor_train_test(
    train_representation: pd.DataFrame,
    test_representation: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    cfg: Mapping[str, Any],
    *,
    legend_type: str,
    groups_train: pd.Series | None = None,
    outer_fold: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit one supervised predictor on an outer-training representation and predict test.

    Logistic-regression ``C`` is selected by an inner CV confined to the outer-training set.
    When repeated-measures groups are supplied, inner CV is group-disjoint as well.
    """
    if list(train_representation.columns) != list(test_representation.columns):
        raise ValueError("Train/test representation columns must match exactly")
    x_train = train_representation.to_numpy(dtype=float)
    x_test = test_representation.to_numpy(dtype=float)
    categorical = str(legend_type).lower() != "continuous"

    if not categorical:
        estimator = _regressor(cfg)
        estimator.fit(x_train, y_train.to_numpy(dtype=float))
        pred = estimator.predict(x_test)
        return np.asarray(pred, dtype=float), {
            "selected_C": np.nan,
            "inner_cv_splits": 0,
            "inner_cv_score": np.nan,
            "inner_cv_scoring": None,
            "predictor": "linear_regression",
        }

    c_values = _c_values(cfg)
    model_cfg = dict(cfg.get("logistic_regression", {}) or {})
    scoring = str(model_cfg.get("scoring", "balanced_accuracy"))
    inner_requested = int(model_cfg.get("inner_cv_splits", 4))
    random_state = int(model_cfg.get("random_state", cfg.get("random_state", 42))) + int(outer_fold)

    selected_c = c_values[0]
    inner_score = np.nan
    inner_splits = 0
    tuning_status = "single_C"
    if len(c_values) > 1:
        try:
            effective = _effective_n_splits(y_train, inner_requested, True, groups_train)
            cv = _splitter(
                y_train,
                categorical=True,
                groups=groups_train,
                n_splits=effective,
                random_state=random_state,
            )
            search = GridSearchCV(
                _classifier(cfg, C=1.0),
                param_grid={"model__C": c_values},
                scoring=scoring,
                cv=cv,
                refit=True,
                n_jobs=1,
                error_score="raise",
            )
            search.fit(x_train, y_train.astype(str).to_numpy())
            selected_c = float(search.best_params_["model__C"])
            inner_score = float(search.best_score_)
            inner_splits = int(effective)
            tuning_status = "nested_cv"
        except ValueError:
            # Very small outer folds can lack enough independent groups/classes for another
            # split.  Use the C closest to 1 rather than introducing a test-dependent choice.
            selected_c = min(c_values, key=lambda value: (abs(np.log10(value)), value))
            tuning_status = "insufficient_inner_cv"

    estimator = _classifier(cfg, C=selected_c)
    estimator.fit(x_train, y_train.astype(str).to_numpy())
    pred = estimator.predict(x_test)
    return np.asarray(pred, dtype=object), {
        "selected_C": float(selected_c),
        "inner_cv_splits": int(inner_splits),
        "inner_cv_score": inner_score,
        "inner_cv_scoring": scoring,
        "tuning_status": tuning_status,
        "predictor": "logistic_regression",
    }


def score_predictions(
    observed: pd.Series,
    predicted: Sequence[Any],
    *,
    legend_type: str,
    global_classes: Sequence[str] | None = None,
) -> dict[str, float]:
    categorical = str(legend_type).lower() != "continuous"
    if categorical:
        labels = list(global_classes) if global_classes is not None else sorted(observed.astype(str).unique())
        return {
            "balanced_accuracy": float(balanced_accuracy_score(observed.astype(str), predicted)),
            "macro_f1": float(
                f1_score(observed.astype(str), predicted, labels=labels, average="macro", zero_division=0)
            ),
        }
    obs = observed.to_numpy(dtype=float)
    pred = np.asarray(predicted, dtype=float)
    return {
        "r2": float(r2_score(obs, pred)),
        "mae": float(mean_absolute_error(obs, pred)),
    }


def summarize_prediction_folds(
    predictions: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    *,
    legend_type: str,
) -> dict[str, Any]:
    categorical = str(legend_type).lower() != "continuous"
    summary: dict[str, Any] = {
        "n_samples": int(len(predictions)),
        "n_splits": int(fold_metrics["fold"].nunique()) if not fold_metrics.empty else 0,
    }
    if categorical:
        selected_c = pd.to_numeric(fold_metrics.get("selected_C"), errors="coerce") if "selected_C" in fold_metrics else pd.Series(dtype=float)
        finite_c = selected_c[np.isfinite(selected_c)]
        summary.update(
            {
                "selected_C_median": float(finite_c.median()) if not finite_c.empty else np.nan,
                "selected_C_min": float(finite_c.min()) if not finite_c.empty else np.nan,
                "selected_C_max": float(finite_c.max()) if not finite_c.empty else np.nan,
                "balanced_accuracy_mean": float(fold_metrics["balanced_accuracy"].mean()),
                "balanced_accuracy_std": float(fold_metrics["balanced_accuracy"].std(ddof=1)) if len(fold_metrics) > 1 else 0.0,
                "macro_f1_mean": float(fold_metrics["macro_f1"].mean()),
                "macro_f1_std": float(fold_metrics["macro_f1"].std(ddof=1)) if len(fold_metrics) > 1 else 0.0,
                "balanced_accuracy_oof": float(balanced_accuracy_score(predictions["observed"].astype(str), predictions["predicted"].astype(str))),
                "macro_f1_oof": float(
                    f1_score(
                        predictions["observed"].astype(str),
                        predictions["predicted"].astype(str),
                        labels=sorted(predictions["observed"].astype(str).unique()),
                        average="macro",
                        zero_division=0,
                    )
                ),
            }
        )
    else:
        summary.update(
            {
                "r2_mean": float(fold_metrics["r2"].mean()),
                "r2_std": float(fold_metrics["r2"].std(ddof=1)) if len(fold_metrics) > 1 else 0.0,
                "mae_mean": float(fold_metrics["mae"].mean()),
                "mae_std": float(fold_metrics["mae"].std(ddof=1)) if len(fold_metrics) > 1 else 0.0,
                "r2_oof": float(r2_score(predictions["observed"].to_numpy(dtype=float), predictions["predicted"].to_numpy(dtype=float))),
                "mae_oof": float(mean_absolute_error(predictions["observed"].to_numpy(dtype=float), predictions["predicted"].to_numpy(dtype=float))),
            }
        )
    return summary


def cross_validated_prediction(
    representation: pd.DataFrame,
    y: pd.Series,
    plan: CVPlan,
    cfg: Mapping[str, Any],
    *,
    legend_type: str,
    groups: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Backward-compatible fixed-representation CV helper.

    The pipeline's production prediction stage no longer uses this path for AA/benchmarks;
    it refits those representations inside every outer fold.  This helper remains useful for
    unit tests and for the raw-input baseline.
    """
    missing = plan.index.difference(representation.index)
    if len(missing):
        raise ValueError(f"Representation is missing CV samples, including {missing.tolist()[:10]}")
    representation = representation.reindex(plan.index)
    y_aligned = y.reindex(plan.index)
    groups_aligned = groups.reindex(plan.index) if groups is not None else None
    categorical = str(legend_type).lower() != "continuous"
    global_classes = sorted(y_aligned.astype(str).unique().tolist()) if categorical else None

    pred_values: np.ndarray = np.empty(len(plan.index), dtype=object if categorical else float)
    fold_rows: list[dict[str, Any]] = []
    for fold, (train, test) in enumerate(plan.split_indices()):
        train_index = plan.index[train]
        test_index = plan.index[test]
        pred, tuning = fit_predictor_train_test(
            representation.loc[train_index],
            representation.loc[test_index],
            y_aligned.loc[train_index],
            y_aligned.loc[test_index],
            cfg,
            legend_type=legend_type,
            groups_train=(groups_aligned.loc[train_index] if groups_aligned is not None else None),
            outer_fold=fold,
        )
        pred_values[test] = pred
        metrics = score_predictions(
            y_aligned.loc[test_index], pred, legend_type=legend_type, global_classes=global_classes
        )
        fold_rows.append(
            {
                "fold": fold,
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                **metrics,
                **tuning,
            }
        )

    predictions = pd.DataFrame(
        {"observed": y_aligned.to_numpy(), "predicted": pred_values, "fold": plan.folds},
        index=plan.index,
    )
    predictions.index.name = "sample"
    folds = pd.DataFrame(fold_rows)
    summary = summarize_prediction_folds(predictions, folds, legend_type=legend_type)
    summary.update({"grouped_cv": bool(plan.grouped), "group_column": plan.group_column})
    return predictions, folds, summary

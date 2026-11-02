"""Random-forest prediction from taxa ranked by training-fold AA archetypes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from .experiments import archetype_taxa_values
from .prediction import score_predictions, summarize_prediction_folds


def rank_archetype_taxa(
    archetypes: pd.DataFrame, transform: Mapping[str, Any], value_space: str = "fitted"
) -> list[str]:
    """Interleave per-archetype top-taxa lists, keeping each taxon only once."""
    values = archetype_taxa_values(archetypes.to_numpy(dtype=float), transform, value_space)
    orders = [np.argsort(-row, kind="stable") for row in values]
    names = archetypes.columns.astype(str).tolist()
    ranked: list[str] = []
    seen: set[str] = set()
    for rank in range(len(names)):
        for order in orders:
            name = names[int(order[rank])]
            if name not in seen:
                seen.add(name)
                ranked.append(name)
    return ranked


def taxa_counts(n_features: int) -> list[int]:
    if n_features < 1:
        raise ValueError("AA fit input must contain at least one feature")
    return sorted(set(range(10, n_features + 1, 20)) | {int(n_features)})


def evaluate_taxa_count(
    k: int,
    count: int,
    folds: list[dict[str, Any]],
    y: pd.Series,
    legend_type: str,
    rf_cfg: Mapping[str, Any],
    random_state: int,
    *,
    selection_mode: str = "aa",
    random_repeat: int = 0,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit and score one (K, taxa count) task across the fixed outer folds."""
    if selection_mode not in {"aa", "random"}:
        raise ValueError("selection_mode must be 'aa' or 'random'")
    categorical = str(legend_type).lower() != "continuous"
    global_classes = sorted(y.astype(str).unique().tolist()) if categorical else None
    predictions: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for fold in folds:
        fold_id = int(fold["fold"])
        train = fold["train"]
        test = fold["test"]
        if selection_mode == "aa":
            taxa = fold["ranking"][:count]
        else:
            rng = np.random.default_rng(
                np.random.SeedSequence([int(random_state), int(count), int(random_repeat), fold_id])
            )
            taxa = rng.choice(train.columns.to_numpy(), size=min(count, train.shape[1]), replace=False).tolist()
        # A random forest's fixed seed is sensitive to input column order. Sort both
        # selections so a shared feature set produces the same fitted model.
        model_columns = sorted(taxa)
        x_train = train.loc[:, model_columns].to_numpy(dtype=float)
        x_test = test.loc[:, model_columns].to_numpy(dtype=float)
        y_train = y.reindex(train.index)
        y_test = y.reindex(test.index)
        kwargs = {
            "n_estimators": int(rf_cfg.get("n_estimators", 300)),
            "max_depth": rf_cfg.get("max_depth"),
            "min_samples_leaf": int(rf_cfg.get("min_samples_leaf", 1)),
            "max_features": rf_cfg.get("max_features", "sqrt"),
            "random_state": int(random_state) + fold_id,
            "n_jobs": 1,
        }
        if categorical:
            model = RandomForestClassifier(**kwargs, class_weight=rf_cfg.get("class_weight"))
            model.fit(x_train, y_train.astype(str).to_numpy())
        else:
            model = RandomForestRegressor(**kwargs)
            model.fit(x_train, y_train.to_numpy(dtype=float))
        predicted = model.predict(x_test)
        metrics = score_predictions(
            y_test, predicted, legend_type=legend_type, global_classes=global_classes
        )
        fold_rows.append({
            "fold": fold_id, "n_train": len(train), "n_test": len(test),
            "n_taxa": len(taxa), **metrics,
        })
        predictions.append(pd.DataFrame({
            "observed": y_test.to_numpy(), "predicted": predicted, "fold": fold_id,
        }, index=test.index))
        selected_rows.extend(
            {"fold": fold_id, "rank": rank, "taxon": taxon}
            for rank, taxon in enumerate(taxa, 1)
        )
    pred_frame = pd.concat(predictions).reindex(y.index)
    pred_frame.index.name = "sample"
    fold_frame = pd.DataFrame(fold_rows)
    summary = summarize_prediction_folds(pred_frame, fold_frame, legend_type=legend_type)
    summary.update({
        "k": int(k), "num_taxa": int(count),
        "n_taxa_min": int(fold_frame["n_taxa"].min()),
        "n_taxa_max": int(fold_frame["n_taxa"].max()),
    })
    return summary, pred_frame, fold_frame, pd.DataFrame(selected_rows)

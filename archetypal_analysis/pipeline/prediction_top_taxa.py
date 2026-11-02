"""Standalone, resumable top-AA-taxa prediction experiment."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ..analysis.prediction import build_cv_plan
from ..analysis.top_taxa_prediction import evaluate_taxa_count, rank_archetype_taxa, taxa_counts
from ..data.grouping import resolve_group_labels
from ..data.indexed_io import align_source_frame_to_target_index, read_sample_indexed_tsv
from .artifacts import ArtifactStore, atomic_write_json, file_signature, stable_hash
from .config import dataset_paths, output_root
from .stages.prediction import (
    PREDICTION_SCHEMA_VERSION,
    _load_cached_aa,
    _load_cached_preprocessing,
)


def _evaluate_task(task: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    folds = [
        {
            "fold": item["fold"],
            "train": read_sample_indexed_tsv(Path(item["train_path"])),
            "test": read_sample_indexed_tsv(Path(item["test_path"])),
            "ranking": item["ranking"],
        }
        for item in task["folds"]
    ]
    return evaluate_taxa_count(
        task["k"], task["count"], folds, task["y"], task["legend_type"],
        task["rf_cfg"], task["random_state"],
        selection_mode=task["method"], random_repeat=task["random_repeat"],
    )


def _write_task(task: dict[str, Any], result) -> dict[str, Any]:
    summary, predictions, fold_metrics, selected = result
    summary.update({"method": task["method"], "random_repeat": task["random_repeat"]})
    out_dir = Path(task["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out_dir / "predictions.tsv", sep="\t", index_label="sample")
    fold_metrics.to_csv(out_dir / "fold_metrics.tsv", sep="\t", index=False)
    selected.to_csv(out_dir / "selected_taxa.tsv", sep="\t", index=False)
    atomic_write_json(out_dir / "_SUCCESS.json", {"fingerprint": task["fingerprint"], "summary": summary})
    return summary


def _plot_scores(scores: pd.DataFrame, path: Path, categorical: bool) -> None:
    metric = "macro_f1_mean" if categorical else "r2_mean"
    spread = "macro_f1_std" if categorical else "r2_std"
    fig, ax = plt.subplots(figsize=(8, 5))
    for k, group in scores.loc[scores["method"] == "aa"].groupby("k", sort=True):
        group = group.sort_values("num_taxa")
        ax.errorbar(group["num_taxa"], group[metric], yerr=group[spread], marker="o", capsize=3, label=f"AA taxa, K={k}")
    random_scores = scores.loc[scores["method"] == "random"]
    if not random_scores.empty:
        band = random_scores.groupby("num_taxa")[metric].agg(["mean", "min", "max"]).sort_index()
        x = band.index.to_numpy(dtype=int)
        ax.plot(x, band["mean"].to_numpy(), "--s", color="0.25", label="Random taxa")
        if random_scores["random_repeat"].nunique() > 1:
            ax.fill_between(x, band["min"].to_numpy(), band["max"].to_numpy(), color="0.5", alpha=0.18,
                            label="Random draw range")
    ax.set_xlabel("Number of taxa")
    ax.set_ylabel("Macro F1" if categorical else "R²")
    ax.set_title("Random forest prediction: AA taxa vs random taxa")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_top_taxa_prediction(
    config: Mapping[str, Any], *, n_jobs: int | None = None, overwrite: bool = False
) -> Path:
    """Use existing fit/QC/selection artifacts; never invoke the pipeline runner."""
    store = ArtifactStore(output_root(config))
    fit_dir = store.stage_dir("fit")
    qc_dir = store.stage_dir("qc")
    selection_path = store.stage_dir("select") / "selection.json"
    needed = [fit_dir / "fit_input.tsv", qc_dir / "metadata.tsv", qc_dir / "otu.tsv", selection_path]
    missing = [str(path) for path in needed if not path.exists()]
    if missing:
        raise FileNotFoundError("Run the pipeline through select first; missing: " + ", ".join(missing))

    prediction_cfg = dict(config.get("prediction", {}) or {})
    top_cfg = dict(prediction_cfg.get("top_taxa", {}) or {})
    rf_cfg = dict(top_cfg.get("random_forest", {}) or {})
    fit_cfg = dict(config["fit"])
    fit_input = read_sample_indexed_tsv(fit_dir / "fit_input.tsv")
    metadata, _ = align_source_frame_to_target_index(
        read_sample_indexed_tsv(qc_dir / "metadata.tsv"), fit_input.index, source_name="QC metadata"
    )
    otu, _ = align_source_frame_to_target_index(
        read_sample_indexed_tsv(qc_dir / "otu.tsv"), fit_input.index, source_name="QC OTU"
    )
    selected_k = sorted(int(k) for k in json.loads(selection_path.read_text())["selected_k"])
    if not selected_k:
        raise ValueError("Selection artifact does not contain any K values")
    target = str(prediction_cfg.get("target") or config["task"]["target"])
    legend_type = str(config["task"].get("legend_type", "discrete"))
    value_space = str(top_cfg.get("value_space") or config.get("analysis", {}).get(
        "top_taxa_value_space", config.get("analysis", {}).get("taxa_value_space", "fitted")
    )).lower()
    group_column = prediction_cfg.get("group_column")
    cv_group_column = group_column
    group_resolution = None
    if group_column:
        _, source_metadata_path = dataset_paths(config)
        resolution = resolve_group_labels(
            metadata, fit_input.index, preferred_column=str(group_column),
            source_metadata_path=source_metadata_path,
            sample_id_column=str(config.get("dataset", {}).get("sample_id_column", "sample")),
            fallback_columns=prediction_cfg.get("group_fallback_columns", []) or [],
            auto_detect=bool(prediction_cfg.get("group_auto_detect", True)),
            namespace_columns=prediction_cfg.get(
                "group_namespace_columns", ["project", "Study", "StudyID", "Cohort", "location"]
            ) or [],
        )
        cv_group_column = "__resolved_prediction_group__"
        metadata[cv_group_column] = resolution.groups
        group_resolution = resolution.info
    plan, y, groups = build_cv_plan(
        metadata, target, legend_type,
        n_splits=int(prediction_cfg.get("n_splits", 5)),
        random_state=int(prediction_cfg.get("random_state", 42)),
        group_column=cv_group_column,
    )

    out_root = store.stage_dir("prediction") / "top_taxa"
    out_root.mkdir(parents=True, exist_ok=True)
    fold_frame = plan.frame(groups)
    fold_frame[target] = y.reindex(plan.index).to_numpy()
    fold_frame.to_csv(out_root / "fold_assignments.tsv", sep="\t", index_label="sample")
    counts = taxa_counts(fit_input.shape[1])
    random_repeats = int(top_cfg.get("random_repeats", 5))
    if random_repeats < 1:
        raise ValueError("prediction.top_taxa.random_repeats must be >= 1")
    fold_info: dict[int, list[dict[str, Any]]] = {k: [] for k in selected_k}
    cache_stats = {"preprocessing_ran": 0, "preprocessing_reused": 0, "aa_ran": 0, "aa_reused": 0}
    outer_cache = store.stage_dir("prediction") / "fold_models"
    for fold, (train_pos, test_pos) in enumerate(plan.split_indices()):
        train_ids, test_ids = plan.index[train_pos], plan.index[test_pos]
        fold_dir = outer_cache / f"fold_{fold:03d}"
        prep_fp = stable_hash({
            "schema": PREDICTION_SCHEMA_VERSION,
            "qc": store.upstream_fingerprint("qc"),
            "train": train_ids.tolist(), "test": test_ids.tolist(),
            "preprocessing": fit_cfg.get("preprocessing", {}),
        })
        train, test, transform, _, prep_status = _load_cached_preprocessing(
            fold_dir / "preprocessing", prep_fp, otu, train_ids, test_ids, fit_cfg
        )
        cache_stats[f"preprocessing_{prep_status}"] += 1
        for k in selected_k:
            aa_fp = stable_hash({
                "schema": PREDICTION_SCHEMA_VERSION, "preprocessing": prep_fp, "k": k,
                "fit_model": {
                    key: value for key, value in fit_cfg.items()
                    if key not in {"k_values", "parallel", "preprocessing"}
                },
                "projection": prediction_cfg.get("aa_projection", {}) or {},
            })
            aa_dir = fold_dir / f"K_{k:03d}" / "aa"
            frames, _, _, status = _load_cached_aa(
                aa_dir, aa_fp, train, test, k, fit_cfg,
                dict(prediction_cfg.get("aa_projection", {}) or {}),
            )
            cache_stats[f"aa_{status}"] += 1
            ranking = rank_archetype_taxa(frames["archetypes"], transform, value_space)
            fold_info[k].append({
                "fold": fold, "train_path": str(fold_dir / "preprocessing" / "train_fit_input.tsv"),
                "test_path": str(fold_dir / "preprocessing" / "test_fit_input.tsv"),
                "ranking": ranking, "preprocessing_fingerprint": prep_fp, "aa_fingerprint": aa_fp,
            })

    requested_jobs = int(top_cfg.get("parallel", {}).get("n_jobs", 0) or 0) if n_jobs is None else int(n_jobs)
    if requested_jobs < 0:
        raise ValueError("prediction.top_taxa.parallel.n_jobs must be >= 0")
    if requested_jobs == 0:
        requested_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    tasks: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for k in selected_k:
        for count in counts:
            out_dir = out_root / f"K_{k:03d}" / f"N_{count:04d}"
            fingerprint = stable_hash({
                "schema": 1, "folds": [
                    {key: item[key] for key in ("preprocessing_fingerprint", "aa_fingerprint")}
                    for item in fold_info[k]
                ],
                "ranking": [item["ranking"] for item in fold_info[k]],
                "count": count, "k": k, "y": y.to_dict(), "rf": rf_cfg,
                "random_state": prediction_cfg.get("random_state", 42),
                "legend_type": legend_type, "value_space": value_space,
            })
            marker = out_dir / "_SUCCESS.json"
            required = [out_dir / name for name in ("predictions.tsv", "fold_metrics.tsv", "selected_taxa.tsv")]
            if not overwrite and marker.exists() and all(path.exists() for path in required):
                cached = json.loads(marker.read_text())
                if cached.get("fingerprint") == fingerprint:
                    summaries.append({**cached["summary"], "method": "aa", "random_repeat": -1})
                    continue
            tasks.append({
                "k": k, "count": count, "folds": fold_info[k], "y": y,
                "legend_type": legend_type, "rf_cfg": rf_cfg,
                "random_state": int(prediction_cfg.get("random_state", 42)),
                "out_dir": str(out_dir), "fingerprint": fingerprint,
                "method": "aa", "random_repeat": -1,
            })
    # The random baseline depends on preprocessing and the target, but not on K.
    # One shared set of repeated draws is compared with every selected AA model.
    random_folds = fold_info[selected_k[0]]
    for count in counts:
        for repeat in range(random_repeats):
            out_dir = out_root / "random" / f"N_{count:04d}" / f"repeat_{repeat:03d}"
            fingerprint = stable_hash({
                "schema": 2,
                "folds": [item["preprocessing_fingerprint"] for item in random_folds],
                "count": count, "repeat": repeat, "y": y.to_dict(), "rf": rf_cfg,
                "random_state": prediction_cfg.get("random_state", 42),
                "legend_type": legend_type,
            })
            marker = out_dir / "_SUCCESS.json"
            required = [out_dir / name for name in ("predictions.tsv", "fold_metrics.tsv", "selected_taxa.tsv")]
            if not overwrite and marker.exists() and all(path.exists() for path in required):
                cached = json.loads(marker.read_text())
                if cached.get("fingerprint") == fingerprint:
                    summaries.append(cached["summary"])
                    continue
            tasks.append({
                "k": 0, "count": count, "folds": random_folds, "y": y,
                "legend_type": legend_type, "rf_cfg": rf_cfg,
                "random_state": int(prediction_cfg.get("random_state", 42)),
                "out_dir": str(out_dir), "fingerprint": fingerprint,
                "method": "random", "random_repeat": repeat,
            })
    workers = min(max(1, requested_jobs), max(1, len(tasks)))
    if workers == 1 or top_cfg.get("parallel", {}).get("backend", "process") == "sequential":
        for task in tasks:
            summaries.append(_write_task(task, _evaluate_task(task)))
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
            future_to_task = {pool.submit(_evaluate_task, task): task for task in tasks}
            for future in as_completed(future_to_task):
                summaries.append(_write_task(future_to_task[future], future.result()))

    scores = pd.DataFrame(summaries).sort_values(["method", "k", "num_taxa", "random_repeat"])
    scores.to_csv(out_root / "scores.tsv", sep="\t", index=False)
    plot_path = out_root / "score_vs_num_taxa.png"
    _plot_scores(scores, plot_path, legend_type.lower() != "continuous")
    atomic_write_json(out_root / "manifest.json", {
        "selected_k": selected_k, "num_taxa": counts, "target": target,
        "target_type": legend_type, "ranking": "round_robin_per_archetype_top_values_unique",
        "value_space": value_space, "feature_input": "outer_fold_fit_input",
        "selection_scope": "outer_training_fold", "random_forest": rf_cfg,
        "random_repeats": random_repeats,
        "n_jobs": workers, "tasks_ran": len(tasks), "tasks_reused": len(summaries) - len(tasks),
        "cache": cache_stats, "group_resolution": group_resolution,
        "fit_input_signature": file_signature(fit_dir / "fit_input.tsv"),
    })
    return out_root

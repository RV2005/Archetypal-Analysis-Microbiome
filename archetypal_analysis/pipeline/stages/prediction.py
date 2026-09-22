"""Leakage-safe outer-fold metadata prediction from AA and benchmark representations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ...analysis.prediction import (
    build_cv_plan,
    fit_predictor_train_test,
    score_predictions,
    summarize_prediction_folds,
)
from ...analysis.prediction_representations import (
    fit_aa_train_test,
    fit_benchmark_fold,
    fit_fold_preprocessing,
)
from ...data.grouping import resolve_group_labels
from ...data.indexed_io import align_source_frame_to_target_index, read_sample_indexed_tsv
from ..artifacts import atomic_write_json, file_signature, stable_hash
from ..config import dataset_paths
from .base import PipelineStage, StageContext


PREDICTION_SCHEMA_VERSION = 3


def _marker_reusable(path: Path, fingerprint: str, required: list[Path]) -> bool:
    if not path.exists() or not all(item.exists() for item in required):
        return False
    try:
        return json.loads(path.read_text()).get("fingerprint") == fingerprint
    except (OSError, json.JSONDecodeError):
        return False


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index_label="sample")


def _load_cached_preprocessing(
    cache_dir: Path,
    fingerprint: str,
    raw_otu: pd.DataFrame,
    train_ids: pd.Index,
    test_ids: pd.Index,
    fit_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], list[Path], str]:
    train_path = cache_dir / "train_fit_input.tsv"
    test_path = cache_dir / "test_fit_input.tsv"
    transform_path = cache_dir / "transform.json"
    marker_path = cache_dir / "_SUCCESS.json"
    required = [train_path, test_path, transform_path]
    if _marker_reusable(marker_path, fingerprint, required):
        return (
            read_sample_indexed_tsv(train_path),
            read_sample_indexed_tsv(test_path),
            json.loads(transform_path.read_text()),
            [*required, marker_path],
            "reused",
        )
    fitted = fit_fold_preprocessing(raw_otu, train_ids, test_ids, fit_cfg)
    _write_frame(fitted.train, train_path)
    _write_frame(fitted.test, test_path)
    atomic_write_json(transform_path, fitted.metadata)
    atomic_write_json(
        marker_path,
        {
            "fingerprint": fingerprint,
            "n_train": int(len(fitted.train)),
            "n_test": int(len(fitted.test)),
            "n_features": int(fitted.train.shape[1]),
        },
    )
    return fitted.train, fitted.test, fitted.metadata, [*required, marker_path], "ran"


def _load_cached_aa(
    cache_dir: Path,
    fingerprint: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    k: int,
    fit_cfg: dict[str, Any],
    projection_cfg: dict[str, Any],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any], list[Path], str]:
    paths = {
        "train_direct": cache_dir / "train_direct.tsv",
        "test_direct": cache_dir / "test_direct.tsv",
        "train_helmert": cache_dir / "train_helmert.tsv",
        "test_helmert": cache_dir / "test_helmert.tsv",
        "archetypes": cache_dir / "archetypes.tsv",
    }
    metadata_path = cache_dir / "representation.json"
    marker_path = cache_dir / "_SUCCESS.json"
    required = [*paths.values(), metadata_path]
    if _marker_reusable(marker_path, fingerprint, required):
        frames = {name: read_sample_indexed_tsv(path) for name, path in paths.items() if name != "archetypes"}
        archetypes = pd.read_csv(paths["archetypes"], sep="\t", index_col="archetype")
        frames["archetypes"] = archetypes
        return frames, json.loads(metadata_path.read_text()), [*required, marker_path], "reused"

    fitted = fit_aa_train_test(train, test, k, fit_cfg, projection_cfg)
    cache_dir.mkdir(parents=True, exist_ok=True)
    _write_frame(fitted.train_direct, paths["train_direct"])
    _write_frame(fitted.test_direct, paths["test_direct"])
    _write_frame(fitted.train_helmert, paths["train_helmert"])
    _write_frame(fitted.test_helmert, paths["test_helmert"])
    fitted.archetypes.to_csv(paths["archetypes"], sep="\t", index_label="archetype")
    atomic_write_json(metadata_path, fitted.metadata)
    atomic_write_json(marker_path, {"fingerprint": fingerprint, **fitted.metadata})
    frames = {
        "train_direct": fitted.train_direct,
        "test_direct": fitted.test_direct,
        "train_helmert": fitted.train_helmert,
        "test_helmert": fitted.test_helmert,
        "archetypes": fitted.archetypes,
    }
    return frames, fitted.metadata, [*required, marker_path], "ran"


def _load_cached_benchmark(
    cache_dir: Path,
    fingerprint: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    method: str,
    dimension: int,
    benchmark_cfg: dict[str, Any],
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, dict[str, Any], list[Path], str]:
    train_path = cache_dir / "train_embedding.tsv"
    test_path = cache_dir / "test_embedding.tsv"
    metadata_path = cache_dir / "representation.json"
    marker_path = cache_dir / "_SUCCESS.json"
    required = [metadata_path]
    if method != "tsne":
        required.extend([train_path, test_path])
    if _marker_reusable(marker_path, fingerprint, required):
        metadata = json.loads(metadata_path.read_text())
        train_frame = read_sample_indexed_tsv(train_path) if train_path.exists() else None
        test_frame = read_sample_indexed_tsv(test_path) if test_path.exists() else None
        return train_frame, test_frame, metadata, [*required, marker_path], "reused"

    fitted = fit_benchmark_fold(train, test, method, dimension, benchmark_cfg)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if fitted.test is not None:
        _write_frame(fitted.train, train_path)
        _write_frame(fitted.test, test_path)
    atomic_write_json(metadata_path, fitted.metadata)
    atomic_write_json(marker_path, {"fingerprint": fingerprint, **fitted.metadata})
    emitted = [metadata_path, marker_path]
    if fitted.test is not None:
        emitted.extend([train_path, test_path])
    return fitted.train, fitted.test, fitted.metadata, emitted, "ran"


class PredictionStage(PipelineStage):
    name = "prediction"
    # Global all-data benchmark embeddings are intentionally not dependencies: using them in
    # CV would leak held-out samples. Prediction refits representations inside each outer fold.
    dependencies = ("select",)

    def config_fragment(self, cfg):
        return {
            "prediction_schema": PREDICTION_SCHEMA_VERSION,
            "prediction": cfg.get("prediction", {}),
            "benchmarks": cfg.get("benchmarks", {}),
            "task": cfg["task"],
            "fit": cfg["fit"],
        }

    def fingerprint(self, ctx: StageContext) -> str:
        base = super().fingerprint(ctx)
        prediction_cfg = ctx.config.get("prediction", {}) or {}
        if not prediction_cfg.get("group_column"):
            return base
        _otu_path, metadata_path = dataset_paths(ctx.config)
        return stable_hash({"base": base, "group_metadata_source": file_signature(metadata_path)})

    def required_outputs(self, ctx: StageContext) -> list[Path]:
        d = ctx.store.stage_dir(self.name)
        return [d / "prediction_manifest.json", d / "prediction_scores.tsv", d / "fold_assignments.tsv"]

    def run(self, ctx: StageContext, overwrite: bool = False) -> dict[str, Any]:
        fp = self.fingerprint(ctx)
        outputs = self.required_outputs(ctx)
        if not overwrite and ctx.store.is_reusable(self.name, fp, outputs):
            return {"status": "reused", "outputs": outputs}

        cfg = dict(ctx.config.get("prediction", {}) or {})
        stage_dir = ctx.store.stage_dir(self.name)
        stage_dir.mkdir(parents=True, exist_ok=True)
        if not cfg.get("enabled", False):
            pd.DataFrame().to_csv(outputs[1], sep="\t", index=False)
            pd.DataFrame().to_csv(outputs[2], sep="\t", index=False)
            manifest = {"enabled": False, "outputs": [str(p) for p in outputs]}
            atomic_write_json(outputs[0], manifest)
            ctx.store.mark_success(self.name, fp, outputs, manifest)
            return {"status": "ran", **manifest, "outputs": outputs}

        selection = json.loads((ctx.store.stage_dir("select") / "selection.json").read_text())
        selected_k = sorted(int(k) for k in selection["selected_k"])
        if any(k < 2 for k in selected_k):
            raise ValueError("Prediction requires selected K >= 2 so K-1 is at least one dimension")
        comparison_dimensions = {k: k - 1 for k in selected_k}

        fit_dir = ctx.store.stage_dir("fit")
        fit_input = read_sample_indexed_tsv(fit_dir / "fit_input.tsv")
        qc_metadata = read_sample_indexed_tsv(ctx.store.stage_dir("qc") / "metadata.tsv")
        qc_otu = read_sample_indexed_tsv(ctx.store.stage_dir("qc") / "otu.tsv")
        fit_metadata, metadata_id_repairs = align_source_frame_to_target_index(
            qc_metadata, fit_input.index, source_name="QC metadata"
        )
        qc_otu_aligned, otu_id_repairs = align_source_frame_to_target_index(
            qc_otu, fit_input.index, source_name="QC OTU"
        )

        target = str(cfg.get("target") or ctx.config["task"]["target"])
        group_column = cfg.get("group_column")
        legend_type = str(ctx.config["task"].get("legend_type", "discrete"))
        group_resolution = None
        cv_group_column = group_column
        if group_column:
            _otu_path, source_metadata_path = dataset_paths(ctx.config)
            resolution = resolve_group_labels(
                fit_metadata,
                fit_input.index,
                preferred_column=str(group_column),
                source_metadata_path=source_metadata_path,
                sample_id_column=str(ctx.config.get("dataset", {}).get("sample_id_column", "sample")),
                fallback_columns=cfg.get("group_fallback_columns", []) or [],
                auto_detect=bool(cfg.get("group_auto_detect", True)),
                namespace_columns=cfg.get(
                    "group_namespace_columns", ["project", "Study", "StudyID", "Cohort", "location"]
                ) or [],
            )
            cv_group_column = "__resolved_prediction_group__"
            fit_metadata[cv_group_column] = resolution.groups
            group_resolution = resolution.info

        plan, y, groups = build_cv_plan(
            fit_metadata,
            target,
            legend_type,
            n_splits=int(cfg.get("n_splits", 5)),
            random_state=int(cfg.get("random_state", 42)),
            group_column=cv_group_column,
        )
        fold_frame = plan.frame(groups)
        fold_frame[target] = y.reindex(plan.index).to_numpy()
        fold_frame.to_csv(outputs[2], sep="\t", index_label="sample")

        fit_cfg = dict(ctx.config["fit"])
        benchmark_cfg = dict(ctx.config.get("benchmarks", {}) or {})
        projection_cfg = dict(cfg.get("aa_projection", {}) or {})
        raw_source = str(cfg.get("raw_baseline", "qc_otu")).lower()
        global_classes = sorted(y.astype(str).unique().tolist()) if legend_type.lower() != "continuous" else None
        all_outputs: list[Path] = list(outputs)
        cache_stats = {"preprocessing_ran": 0, "preprocessing_reused": 0, "representation_ran": 0, "representation_reused": 0}

        accumulators: dict[tuple[int | None, str], dict[str, Any]] = {}
        unavailable: dict[tuple[int, str], str] = {}

        def accumulate(
            key: tuple[int | None, str],
            *,
            family: str,
            comparison_k: int | None,
            train_rep: pd.DataFrame,
            test_rep: pd.DataFrame,
            train_ids: pd.Index,
            test_ids: pd.Index,
            fold: int,
            representation_dimension: int,
            effective_dimension: int,
            representation_variant: str,
        ) -> None:
            y_train = y.reindex(train_ids)
            y_test = y.reindex(test_ids)
            groups_train = groups.reindex(train_ids) if groups is not None else None
            pred, tuning = fit_predictor_train_test(
                train_rep,
                test_rep,
                y_train,
                y_test,
                cfg,
                legend_type=legend_type,
                groups_train=groups_train,
                outer_fold=fold,
            )
            metrics = score_predictions(
                y_test,
                pred,
                legend_type=legend_type,
                global_classes=global_classes,
            )
            bucket = accumulators.setdefault(
                key,
                {
                    "family": family,
                    "comparison_k": comparison_k,
                    "representation_dimension": int(representation_dimension),
                    "effective_dimension": int(effective_dimension),
                    "representation_variant": representation_variant,
                    "predictions": [],
                    "folds": [],
                },
            )
            pred_frame = pd.DataFrame(
                {
                    "observed": y_test.to_numpy(),
                    "predicted": pred,
                    "fold": int(fold),
                },
                index=test_ids,
            )
            pred_frame.index.name = "sample"
            bucket["predictions"].append(pred_frame)
            bucket["folds"].append(
                {
                    "fold": int(fold),
                    "n_train": int(len(train_ids)),
                    "n_test": int(len(test_ids)),
                    **metrics,
                    **tuning,
                }
            )

        max_benchmark_dimension = max(comparison_dimensions.values())
        outer_cache = stage_dir / "fold_models"

        for fold, (train_pos, test_pos) in enumerate(plan.split_indices()):
            train_ids = plan.index[train_pos]
            test_ids = plan.index[test_pos]
            fold_dir = outer_cache / f"fold_{fold:03d}"
            prep_fp = stable_hash(
                {
                    "schema": PREDICTION_SCHEMA_VERSION,
                    "qc": ctx.store.upstream_fingerprint("qc"),
                    "train": train_ids.tolist(),
                    "test": test_ids.tolist(),
                    "preprocessing": fit_cfg.get("preprocessing", {}),
                }
            )
            train_fit, test_fit, _transform, prep_outputs, prep_status = _load_cached_preprocessing(
                fold_dir / "preprocessing",
                prep_fp,
                qc_otu_aligned,
                train_ids,
                test_ids,
                fit_cfg,
            )
            all_outputs.extend(prep_outputs)
            cache_stats[f"preprocessing_{prep_status}"] += 1

            # Raw baseline is genuinely outer-fold trained. Its scaler lives inside the
            # supervised pipeline and is therefore fitted on train only as well.
            if raw_source == "qc_otu":
                raw_train = qc_otu_aligned.reindex(train_ids)
                raw_test = qc_otu_aligned.reindex(test_ids)
            elif raw_source == "fit_input":
                # 'fit_input' now means the fold-fitted AA preprocessing, not the globally
                # persisted matrix whose scaling/feature decisions used all samples.
                raw_train, raw_test = train_fit, test_fit
            else:
                raise ValueError("prediction.raw_baseline must be 'qc_otu' or 'fit_input'")
            accumulate(
                (None, "raw_input"),
                family="baseline",
                comparison_k=None,
                train_rep=raw_train,
                test_rep=raw_test,
                train_ids=train_ids,
                test_ids=test_ids,
                fold=fold,
                representation_dimension=int(raw_train.shape[1]),
                effective_dimension=int(raw_train.shape[1]),
                representation_variant=raw_source,
            )

            # Fit PCA/PCoA once at the largest required K-1 dimension in this fold, then
            # reuse/slice those train-only embeddings for smaller selected K values.
            linear_benchmarks: dict[str, tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]] = {}
            for method in ("pca", "pcoa"):
                if not bool((benchmark_cfg.get(method, {}) or {}).get("enabled", True)):
                    continue
                method_fp = stable_hash(
                    {
                        "schema": PREDICTION_SCHEMA_VERSION,
                        "preprocessing": prep_fp,
                        "method": method,
                        "dimension": max_benchmark_dimension,
                        "config": benchmark_cfg.get(method, {}),
                    }
                )
                train_emb, test_emb, metadata, emitted, status = _load_cached_benchmark(
                    fold_dir / "benchmarks" / method / f"D_{max_benchmark_dimension:03d}",
                    method_fp,
                    train_fit,
                    test_fit,
                    method,
                    max_benchmark_dimension,
                    benchmark_cfg,
                )
                all_outputs.extend(emitted)
                cache_stats[f"representation_{status}"] += 1
                if train_emb is not None and test_emb is not None:
                    linear_benchmarks[method] = (train_emb, test_emb, metadata)

            for k in selected_k:
                dimension = comparison_dimensions[k]
                aa_fp = stable_hash(
                    {
                        "schema": PREDICTION_SCHEMA_VERSION,
                        "preprocessing": prep_fp,
                        "k": k,
                        "fit_model": {
                            key: value
                            for key, value in fit_cfg.items()
                            if key not in {"k_values", "parallel", "preprocessing"}
                        },
                        "projection": projection_cfg,
                    }
                )
                aa_frames, _aa_meta, emitted, status = _load_cached_aa(
                    fold_dir / f"K_{k:03d}" / "aa",
                    aa_fp,
                    train_fit,
                    test_fit,
                    k,
                    fit_cfg,
                    projection_cfg,
                )
                all_outputs.extend(emitted)
                cache_stats[f"representation_{status}"] += 1
                accumulate(
                    (k, "archetypal_analysis_direct"),
                    family="aa",
                    comparison_k=k,
                    train_rep=aa_frames["train_direct"],
                    test_rep=aa_frames["test_direct"],
                    train_ids=train_ids,
                    test_ids=test_ids,
                    fold=fold,
                    representation_dimension=k,
                    effective_dimension=dimension,
                    representation_variant="native_simplex_weights",
                )
                accumulate(
                    (k, "archetypal_analysis_helmert"),
                    family="aa",
                    comparison_k=k,
                    train_rep=aa_frames["train_helmert"],
                    test_rep=aa_frames["test_helmert"],
                    train_ids=train_ids,
                    test_ids=test_ids,
                    fold=fold,
                    representation_dimension=dimension,
                    effective_dimension=dimension,
                    representation_variant="orthonormal_helmert",
                )

                for method, (train_max, test_max, _metadata) in linear_benchmarks.items():
                    accumulate(
                        (k, method),
                        family="benchmark",
                        comparison_k=k,
                        train_rep=train_max.iloc[:, :dimension].copy(),
                        test_rep=test_max.iloc[:, :dimension].copy(),
                        train_ids=train_ids,
                        test_ids=test_ids,
                        fold=fold,
                        representation_dimension=dimension,
                        effective_dimension=dimension,
                        representation_variant="outer_fold_refit",
                    )

                for method in ("umap", "tsne"):
                    if not bool((benchmark_cfg.get(method, {}) or {}).get("enabled", True)):
                        continue
                    method_fp = stable_hash(
                        {
                            "schema": PREDICTION_SCHEMA_VERSION,
                            "preprocessing": prep_fp,
                            "method": method,
                            "dimension": dimension,
                            "config": benchmark_cfg.get(method, {}),
                        }
                    )
                    train_emb, test_emb, metadata, emitted, status = _load_cached_benchmark(
                        fold_dir / "benchmarks" / method / f"D_{dimension:03d}",
                        method_fp,
                        train_fit,
                        test_fit,
                        method,
                        dimension,
                        benchmark_cfg,
                    )
                    all_outputs.extend(emitted)
                    cache_stats[f"representation_{status}"] += 1
                    if test_emb is None:
                        unavailable[(k, method)] = str(metadata.get("reason", "No held-out transform is available"))
                        continue
                    accumulate(
                        (k, method),
                        family="benchmark",
                        comparison_k=k,
                        train_rep=train_emb,
                        test_rep=test_emb,
                        train_ids=train_ids,
                        test_ids=test_ids,
                        fold=fold,
                        representation_dimension=dimension,
                        effective_dimension=dimension,
                        representation_variant="outer_fold_refit",
                    )

        score_rows: list[dict[str, Any]] = []
        # Write the raw baseline once, then reference the same OOF result from every selected K.
        raw_bucket = accumulators.pop((None, "raw_input"))
        raw_predictions = pd.concat(raw_bucket["predictions"]).reindex(plan.index)
        raw_folds = pd.DataFrame(raw_bucket["folds"]).sort_values("fold")
        raw_summary = summarize_prediction_folds(raw_predictions, raw_folds, legend_type=legend_type)
        raw_dir = stage_dir / "raw_input"
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_prediction_path = raw_dir / "predictions.tsv"
        raw_fold_path = raw_dir / "fold_metrics.tsv"
        raw_predictions.to_csv(raw_prediction_path, sep="\t", index_label="sample")
        raw_folds.to_csv(raw_fold_path, sep="\t", index=False)
        all_outputs.extend([raw_prediction_path, raw_fold_path])
        for k in selected_k:
            score_rows.append(
                {
                    "family": "baseline",
                    "method": "raw_input",
                    "comparison_k": k,
                    "benchmark_dimension": comparison_dimensions[k],
                    "representation_dimension": raw_bucket["representation_dimension"],
                    "effective_dimension": raw_bucket["effective_dimension"],
                    "representation_variant": raw_bucket["representation_variant"],
                    "status": "ok",
                    "reason": None,
                    "target": target,
                    "target_type": legend_type,
                    **raw_summary,
                }
            )

        for (k, method), bucket in sorted(accumulators.items(), key=lambda item: (int(item[0][0]), item[0][1])):
            predictions = pd.concat(bucket["predictions"]).reindex(plan.index)
            folds = pd.DataFrame(bucket["folds"]).sort_values("fold")
            summary = summarize_prediction_folds(predictions, folds, legend_type=legend_type)
            method_dir = stage_dir / f"K_{int(k):03d}" / method
            method_dir.mkdir(parents=True, exist_ok=True)
            prediction_path = method_dir / "predictions.tsv"
            fold_path = method_dir / "fold_metrics.tsv"
            predictions.to_csv(prediction_path, sep="\t", index_label="sample")
            folds.to_csv(fold_path, sep="\t", index=False)
            all_outputs.extend([prediction_path, fold_path])
            score_rows.append(
                {
                    "family": bucket["family"],
                    "method": method,
                    "comparison_k": int(k),
                    "benchmark_dimension": comparison_dimensions[int(k)],
                    "representation_dimension": bucket["representation_dimension"],
                    "effective_dimension": bucket["effective_dimension"],
                    "representation_variant": bucket["representation_variant"],
                    "status": "ok",
                    "reason": None,
                    "target": target,
                    "target_type": legend_type,
                    **summary,
                }
            )

        # Keep unavailable strict-OOS benchmarks visible rather than silently disappearing.
        for (k, method), reason in sorted(unavailable.items()):
            score_rows.append(
                {
                    "family": "benchmark",
                    "method": method,
                    "comparison_k": int(k),
                    "benchmark_dimension": comparison_dimensions[int(k)],
                    "representation_dimension": comparison_dimensions[int(k)],
                    "effective_dimension": comparison_dimensions[int(k)],
                    "representation_variant": "outer_fold_refit",
                    "status": "unavailable",
                    "reason": reason,
                    "target": target,
                    "target_type": legend_type,
                }
            )

        scores = pd.DataFrame(score_rows).sort_values(["comparison_k", "family", "method"]).reset_index(drop=True)
        scores.to_csv(outputs[1], sep="\t", index=False)
        manifest = {
            "enabled": True,
            "schema_version": PREDICTION_SCHEMA_VERSION,
            "target": target,
            "target_type": legend_type,
            "selected_k": selected_k,
            "benchmark_dimensions": {str(k): int(k - 1) for k in selected_k},
            "dimension_rationale": "K AA simplex weights have K-1 independent dimensions; benchmark representations therefore use K-1.",
            "aa_variants": {
                "archetypal_analysis_direct": "native K simplex weights (rank K-1)",
                "archetypal_analysis_helmert": "deterministic orthonormal Helmert coordinates with K-1 columns",
            },
            "representation_scope": "outer_fold_refit_no_test_leakage",
            "preprocessing_scope": "outer_train_fit_then_test_transform",
            "group_column": group_column,
            "resolved_group_column": cv_group_column if group_column else None,
            "group_resolution": group_resolution,
            "grouped_cv": bool(plan.grouped),
            "n_splits": int(plan.n_splits),
            "raw_baseline": raw_source,
            "logistic_C_values": list(map(float, (cfg.get("logistic_regression", {}) or {}).get("C_values", [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]))),
            "logistic_C_selection": "inner cross-validation within each outer training fold",
            "tsne_policy": "NA under strict out-of-sample prediction because sklearn TSNE has no transform",
            "metadata_sample_ids_repaired": int(metadata_id_repairs),
            "otu_sample_ids_repaired": int(otu_id_repairs),
            "cache": cache_stats,
            "outputs": [str(path) for path in all_outputs],
        }
        atomic_write_json(outputs[0], manifest)
        ctx.store.mark_success(
            self.name,
            fp,
            all_outputs,
            {
                "target": target,
                "selected_k": selected_k,
                "grouped_cv": bool(plan.grouped),
                "n_rows": int(len(scores)),
                "schema_version": PREDICTION_SCHEMA_VERSION,
            },
        )
        return {
            "status": "ran",
            "selected_k": selected_k,
            "grouped_cv": bool(plan.grouped),
            "n_rows": int(len(scores)),
            "outputs": all_outputs,
        }

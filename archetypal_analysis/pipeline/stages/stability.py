"""Resumable initialization/subsampling stability and held-out AA projection."""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ...analysis.comparison_metrics import evaluate_representation, normalize_memberships
from ...analysis.experiments import load_result, similarity_frame
from ...data.grouping import resolve_group_labels
from ...data.indexed_io import align_source_frame_to_target_index, read_sample_indexed_tsv
from ...analysis.stability import (
    StabilityTask,
    archetype_profile_space,
    compare_archetypes,
    initialize_stability_worker,
    project_onto_archetypes,
    run_stability_fit_task,
    stratified_subsample_ids,
)
from ..artifacts import atomic_write_json, file_signature, stable_hash
from ..config import dataset_paths
from .base import PipelineStage, StageContext


def _summary_quantiles(frame: pd.DataFrame, group_columns: list[str], metrics: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    grouped = frame.groupby(group_columns, dropna=False, sort=True) if group_columns else [((), frame)]
    for key, group in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        row = {column: value for column, value in zip(group_columns, key)}
        row["n"] = int(len(group))
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_median"] = float(values.median()) if len(values) else np.nan
            row[f"{metric}_q025"] = float(values.quantile(0.025)) if len(values) else np.nan
            row[f"{metric}_q975"] = float(values.quantile(0.975)) if len(values) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


class StabilityStage(PipelineStage):
    name = "stability"
    dependencies = ("select",)

    def config_fragment(self, cfg):
        return {
            "stability": cfg.get("stability", {}),
            "fit_model": {
                key: value
                for key, value in cfg["fit"].items()
                if key not in {"k_values", "parallel", "preprocessing"}
            },
            "task": cfg["task"],
            "evaluation": cfg["evaluation"],
            "analysis_metrics": cfg.get("analysis", {}).get("metrics", {}),
        }

    def fingerprint(self, ctx: StageContext) -> str:
        base = super().fingerprint(ctx)
        stability_cfg = ctx.config.get("stability", {}) or {}
        sampling_cfg = stability_cfg.get("sampling", {}) or {}
        prediction_cfg = ctx.config.get("prediction", {}) or {}
        group_column = sampling_cfg.get("group_column") or prediction_cfg.get("group_column")
        if not group_column:
            return base
        _otu_path, metadata_path = dataset_paths(ctx.config)
        return stable_hash({
            "base": base,
            "group_metadata_source": file_signature(metadata_path),
        })

    def required_outputs(self, ctx: StageContext) -> list[Path]:
        d = ctx.store.stage_dir(self.name)
        return [
            d / "stability_manifest.json",
            d / "initialization_archetype_metrics.tsv",
            d / "initialization_run_metrics.tsv",
            d / "sampling_archetype_metrics.tsv",
            d / "sampling_run_metrics.tsv",
            d / "subsample_fit_metrics.tsv",
            d / "heldout_projection_metrics.tsv",
            d / "initialization_archetype_summary.tsv",
            d / "initialization_set_summary.tsv",
            d / "sampling_archetype_summary.tsv",
            d / "sampling_set_summary.tsv",
        ]

    @staticmethod
    def _native_rss(result: Mapping[str, Any], fit_input: pd.DataFrame) -> float:
        sim = similarity_frame(result)
        common = sim.index.intersection(fit_input.index, sort=False)
        if len(common) == 0:
            return np.nan
        alpha = sim.loc[common].to_numpy(dtype=float)
        archetypes = np.asarray(result["archetypes"], dtype=float)
        x = fit_input.loc[common].to_numpy(dtype=float)
        return float(np.sum((x - alpha @ archetypes) ** 2))

    def _evaluate_aa_representation(
        self,
        ctx: StageContext,
        result: Mapping[str, Any],
        fit_input: pd.DataFrame,
        metadata: pd.DataFrame,
        method: str,
    ) -> dict[str, Any]:
        sim = similarity_frame(result)
        common = sim.index.intersection(fit_input.index, sort=False)
        sim = sim.loc[common]
        target = ctx.config["task"]["target"]
        return evaluate_representation(
            method=method,
            dimension=int(result["k"]),
            representation=sim,
            memberships=normalize_memberships(sim.to_numpy(dtype=float)),
            original_input=fit_input.loc[common],
            metadata=metadata[target].reindex(common),
            task_cfg=ctx.config["task"],
            evaluation_cfg=ctx.config["evaluation"],
            metric_cfg=ctx.config.get("analysis", {}).get("metrics", {}),
            native_reconstruction_rss=self._native_rss(result, fit_input),
            native_reconstruction_method="aa_similarity_times_archetypes",
        )

    def run(self, ctx: StageContext, overwrite: bool = False) -> dict[str, Any]:
        fp = self.fingerprint(ctx)
        outputs = self.required_outputs(ctx)
        if not overwrite and ctx.store.is_reusable(self.name, fp, outputs):
            return {"status": "reused", "outputs": outputs}

        cfg = dict(ctx.config.get("stability", {}) or {})
        stage_dir = ctx.store.stage_dir(self.name)
        stage_dir.mkdir(parents=True, exist_ok=True)
        if not cfg.get("enabled", False):
            for path in outputs[1:]:
                pd.DataFrame().to_csv(path, sep="\t", index=False)
            manifest = {"enabled": False, "outputs": [str(path) for path in outputs]}
            atomic_write_json(outputs[0], manifest)
            ctx.store.mark_success(self.name, fp, outputs, manifest)
            return {"status": "ran", **manifest, "outputs": outputs}

        selection = json.loads((ctx.store.stage_dir("select") / "selection.json").read_text())
        selected_k = [int(k) for k in selection["selected_k"]]
        fit_dir = ctx.store.stage_dir("fit")
        fit_input_path = fit_dir / "fit_input.tsv"
        fit_input = read_sample_indexed_tsv(fit_input_path)
        metadata = read_sample_indexed_tsv(ctx.store.stage_dir("qc") / "metadata.tsv")
        metadata, _metadata_id_repairs = align_source_frame_to_target_index(
            metadata, fit_input.index, source_name="QC metadata"
        )
        transform = json.loads((fit_dir / "transform.json").read_text())
        target = ctx.config["task"]["target"]
        legend_type = ctx.config["task"].get("legend_type", "discrete")
        fit_cfg = dict(ctx.config["fit"])
        fit_cfg.pop("preprocessing", None)
        fit_cfg.pop("parallel", None)
        fit_cfg.pop("k_values", None)
        base_seed = int(ctx.config["fit"].get("random_state", 42))
        profile_space = str(cfg.get("profile_space", "fitted"))
        matching_metric = str(cfg.get("matching_metric", "cosine"))

        prediction_cfg = dict(ctx.config.get("prediction", {}) or {})
        prediction_group = prediction_cfg.get("group_column")
        sampling_cfg = dict(cfg.get("sampling", {}) or {})
        configured_sampling_group = sampling_cfg.get("group_column") or prediction_group
        sampling_group = configured_sampling_group
        group_resolution = None
        if configured_sampling_group:
            _otu_path, source_metadata_path = dataset_paths(ctx.config)
            fallback_columns = sampling_cfg.get("group_fallback_columns")
            if fallback_columns is None:
                fallback_columns = prediction_cfg.get("group_fallback_columns", []) or []
            auto_detect = sampling_cfg.get("group_auto_detect")
            if auto_detect is None:
                auto_detect = prediction_cfg.get("group_auto_detect", True)
            namespace_columns = sampling_cfg.get("group_namespace_columns")
            if namespace_columns is None:
                namespace_columns = prediction_cfg.get(
                    "group_namespace_columns",
                    ["project", "Study", "StudyID", "Cohort", "location"],
                )
            resolution = resolve_group_labels(
                metadata,
                fit_input.index,
                preferred_column=str(configured_sampling_group),
                source_metadata_path=source_metadata_path,
                sample_id_column=str(ctx.config.get("dataset", {}).get("sample_id_column", "sample")),
                fallback_columns=fallback_columns or [],
                auto_detect=bool(auto_detect),
                namespace_columns=namespace_columns or [],
            )
            sampling_group = "__resolved_stability_group__"
            metadata[sampling_group] = resolution.groups
            group_resolution = resolution.info

        tasks: list[StabilityTask] = []
        task_holdouts: dict[str, list[str]] = {}
        fit_upstream = ctx.store.upstream_fingerprint("fit")
        init_cfg = dict(cfg.get("initialization", {}) or {})
        if init_cfg.get("enabled", True):
            seeds = [int(value) for value in init_cfg.get("seeds", [17, 29, 53, 71, 97])]
            for k in selected_k:
                for seed in seeds:
                    # The original selected-K fit is the reference and is reused when the
                    # requested seed equals the original seed; never refit it redundantly.
                    if seed == base_seed:
                        continue
                    out = stage_dir / "initialization" / f"K_{k:03d}" / f"seed_{seed:010d}"
                    task_fp = stable_hash(
                        {
                            "fit": fit_upstream,
                            "kind": "initialization",
                            "k": k,
                            "seed": seed,
                            "samples": stable_hash(fit_input.index.tolist()),
                            "fit_cfg": fit_cfg,
                        }
                    )
                    tasks.append(
                        StabilityTask(
                            kind="initialization",
                            k=k,
                            seed=seed,
                            sample_ids=tuple(fit_input.index.astype(str)),
                            output_dir=str(out),
                            fingerprint=task_fp,
                        )
                    )

        if sampling_cfg.get("enabled", True):
            fractions = [float(value) for value in sampling_cfg.get("fractions", [0.5, 0.6, 0.7, 0.8, 0.9])]
            repeats = int(sampling_cfg.get("repeats", 5))
            sampling_seed = int(sampling_cfg.get("random_state", 42))
            continuous_bins = int(sampling_cfg.get("continuous_stratify_bins", 10))
            split_cache: dict[tuple[float, int], tuple[list[str], list[str], int]] = {}
            for fraction in fractions:
                for repeat in range(repeats):
                    seed = sampling_seed + int(round(fraction * 1000)) * 1000 + repeat
                    train_ids, heldout_ids = stratified_subsample_ids(
                        metadata,
                        target,
                        legend_type,
                        fraction,
                        seed,
                        group_column=sampling_group,
                        continuous_bins=continuous_bins,
                    )
                    split_cache[(fraction, repeat)] = (train_ids, heldout_ids, seed)
                    for k in selected_k:
                        frac_label = int(round(fraction * 100))
                        out = stage_dir / "sampling" / f"K_{k:03d}" / f"fraction_{frac_label:03d}" / f"repeat_{repeat:03d}"
                        task_fp = stable_hash(
                            {
                                "fit": fit_upstream,
                                "kind": "sampling",
                                "k": k,
                                "seed": seed,
                                "fraction": fraction,
                                "repeat": repeat,
                                "samples": stable_hash(train_ids),
                                "fit_cfg": fit_cfg,
                            }
                        )
                        task = StabilityTask(
                            kind="sampling",
                            k=k,
                            seed=seed,
                            fraction=fraction,
                            repeat=repeat,
                            sample_ids=tuple(train_ids),
                            output_dir=str(out),
                            fingerprint=task_fp,
                        )
                        tasks.append(task)
                        task_holdouts[task.output_dir] = heldout_ids

        parallel = dict(cfg.get("parallel", {}) or {})
        configured_jobs = int(parallel.get("n_jobs", 0) or 0)
        if configured_jobs <= 0:
            configured_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
        n_jobs = max(1, min(configured_jobs, len(tasks))) if tasks else 1
        backend = str(parallel.get("backend", "process")).lower()
        run_results: list[dict[str, Any]] = []
        if tasks:
            if n_jobs == 1 or backend == "sequential":
                initialize_stability_worker(str(fit_input_path))
                for task in tasks:
                    run_results.append(run_stability_fit_task(task, fit_cfg, overwrite))
            elif backend == "process":
                with ProcessPoolExecutor(
                    max_workers=n_jobs,
                    initializer=initialize_stability_worker,
                    initargs=(str(fit_input_path),),
                ) as pool:
                    future_map = {pool.submit(run_stability_fit_task, task, fit_cfg, overwrite): task for task in tasks}
                    for future in as_completed(future_map):
                        task = future_map[future]
                        try:
                            run_results.append(future.result())
                        except Exception as exc:
                            raise RuntimeError(
                                f"Stability fit failed: kind={task.kind}, K={task.k}, seed={task.seed}, fraction={task.fraction}"
                            ) from exc
            else:
                raise ValueError("stability.parallel.backend must be 'process' or 'sequential'")

        reference_profiles: dict[int, np.ndarray] = {}
        references: dict[int, dict[str, Any]] = {}
        for k in selected_k:
            reference = load_result(fit_dir / f"K_{k:03d}" / "result.json")
            references[k] = reference
            reference_profiles[k] = archetype_profile_space(
                np.asarray(reference["archetypes"], dtype=float), transform, profile_space
            )

        init_arch_rows: list[dict[str, Any]] = []
        init_run_rows: list[dict[str, Any]] = []
        sampling_arch_rows: list[dict[str, Any]] = []
        sampling_run_rows: list[dict[str, Any]] = []
        subsample_metric_rows: list[dict[str, Any]] = []
        heldout_metric_rows: list[dict[str, Any]] = []
        projection_cfg = dict(cfg.get("heldout_projection", {}) or {})
        heldout_fractions = {round(float(x), 10) for x in projection_cfg.get("fractions", [0.8, 0.9])}
        projection_enabled = bool(projection_cfg.get("enabled", True))
        extra_outputs: list[Path] = []

        result_by_out = {row["output_dir"]: row for row in run_results}
        task_by_out = {task.output_dir: task for task in tasks}
        for output_dir, run_row in sorted(result_by_out.items()):
            task = task_by_out[output_dir]
            candidate = load_result(Path(run_row["result_path"]))
            candidate_profiles = archetype_profile_space(
                np.asarray(candidate["archetypes"], dtype=float), transform, profile_space
            )
            match_table, overall = compare_archetypes(
                reference_profiles[task.k], candidate_profiles, matching_metric=matching_metric
            )
            base = {
                "k": int(task.k),
                "seed": int(task.seed),
                "kind": task.kind,
                "fraction": task.fraction,
                "repeat": task.repeat,
                "profile_space": profile_space,
                "matching_metric": matching_metric,
            }
            enriched = match_table.assign(**base)
            comparison_path = Path(output_dir) / "archetype_comparison.tsv"
            enriched.to_csv(comparison_path, sep="\t", index=False)
            extra_outputs.append(comparison_path)
            overall_row = {**base, **overall, "status": run_row.get("status")}
            if task.kind == "initialization":
                init_arch_rows.extend(enriched.to_dict(orient="records"))
                init_run_rows.append(overall_row)
            else:
                sampling_arch_rows.extend(enriched.to_dict(orient="records"))
                sampling_run_rows.append(overall_row)
                train_input = fit_input.loc[list(task.sample_ids)]
                train_metadata = metadata.loc[list(task.sample_ids)]
                fit_metrics = self._evaluate_aa_representation(
                    ctx, candidate, train_input, train_metadata, "aa_subsample_fit"
                )
                subsample_metric_rows.append({**base, **fit_metrics})

                if projection_enabled and round(float(task.fraction or 0.0), 10) in heldout_fractions:
                    heldout_ids = task_holdouts.get(output_dir, [])
                    if heldout_ids:
                        heldout_input = fit_input.loc[heldout_ids]
                        alpha, projection_info = project_onto_archetypes(
                            heldout_input.to_numpy(dtype=float),
                            np.asarray(candidate["archetypes"], dtype=float),
                            max_iter=int(projection_cfg.get("max_iter", 2000)),
                            tol=float(projection_cfg.get("tol", 1e-8)),
                        )
                        projection = pd.DataFrame(
                            alpha,
                            index=heldout_input.index,
                            columns=[f"Archetype {i + 1}" for i in range(task.k)],
                        )
                        projection_path = Path(output_dir) / "heldout_projection.tsv"
                        projection.to_csv(projection_path, sep="\t", index_label="sample")
                        extra_outputs.append(projection_path)
                        heldout_result = {
                            "k": task.k,
                            "archetypes": candidate["archetypes"],
                            "similarity_degree": alpha.tolist(),
                            "index": heldout_input.index.tolist(),
                            "columns": projection.columns.tolist(),
                            "feature_names": candidate["feature_names"],
                        }
                        heldout_metrics = self._evaluate_aa_representation(
                            ctx,
                            heldout_result,
                            heldout_input,
                            metadata.loc[heldout_ids],
                            "aa_heldout_projection",
                        )
                        heldout_metrics["native_reconstruction_rss"] = projection_info["rss"]
                        heldout_metrics["native_reconstruction_method"] = "fixed_archetype_simplex_projection"
                        heldout_metric_rows.append(
                            {
                                **base,
                                "n_heldout": len(heldout_ids),
                                "projection_iterations": projection_info["iterations"],
                                "projection_converged": projection_info["converged"],
                                **heldout_metrics,
                            }
                        )

        init_arch = pd.DataFrame(init_arch_rows)
        init_runs = pd.DataFrame(init_run_rows)
        sampling_arch = pd.DataFrame(sampling_arch_rows)
        sampling_runs = pd.DataFrame(sampling_run_rows)
        subsample_metrics = pd.DataFrame(subsample_metric_rows)
        heldout_metrics = pd.DataFrame(heldout_metric_rows)
        for frame, path in [
            (init_arch, outputs[1]),
            (init_runs, outputs[2]),
            (sampling_arch, outputs[3]),
            (sampling_runs, outputs[4]),
            (subsample_metrics, outputs[5]),
            (heldout_metrics, outputs[6]),
        ]:
            frame.to_csv(path, sep="\t", index=False)

        init_summary = _summary_quantiles(
            init_arch,
            ["k", "reference_archetype"],
            ["max_cosine_similarity", "min_euclidean_distance", "matched_cosine_similarity"],
        )
        init_set_summary = _summary_quantiles(
            init_runs,
            ["k"],
            ["normalized_wasserstein_distance", "median_matched_similarity"],
        )
        sampling_summary = _summary_quantiles(
            sampling_arch,
            ["k", "fraction", "reference_archetype"],
            ["max_cosine_similarity", "min_euclidean_distance", "matched_cosine_similarity"],
        )
        sampling_set_summary = _summary_quantiles(
            sampling_runs,
            ["k", "fraction"],
            ["normalized_wasserstein_distance", "median_matched_similarity"],
        )
        summary_paths = list(outputs[7:11])
        for frame, path in zip(
            [init_summary, init_set_summary, sampling_summary, sampling_set_summary], summary_paths
        ):
            frame.to_csv(path, sep="\t", index=False)
        extra_outputs.extend(summary_paths)

        run_artifact_paths: list[Path] = []
        for run_row in run_results:
            run_dir = Path(run_row["output_dir"])
            run_artifact_paths.extend(
                [
                    Path(run_row["result_path"]),
                    run_dir / "similarity.tsv",
                    run_dir / "archetypes.tsv",
                    run_dir / "train_samples.tsv",
                    run_dir / "_SUCCESS.json",
                ]
            )
        all_outputs = [*outputs, *extra_outputs, *run_artifact_paths]
        manifest = {
            "enabled": True,
            "selected_k": selected_k,
            "reference_seed": base_seed,
            "profile_space": profile_space,
            "matching_metric": matching_metric,
            "initialization_runs": int(len(init_runs)),
            "sampling_runs": int(len(sampling_runs)),
            "heldout_projection_runs": int(len(heldout_metrics)),
            "sampling_group_column": configured_sampling_group,
            "resolved_sampling_group_column": sampling_group if configured_sampling_group else None,
            "group_resolution": group_resolution,
            "n_jobs": n_jobs,
            "parallel_backend": backend,
            "outputs": [str(path) for path in all_outputs],
        }
        atomic_write_json(outputs[0], manifest)
        ctx.store.mark_success(
            self.name,
            fp,
            all_outputs,
            {
                "selected_k": selected_k,
                "initialization_runs": int(len(init_runs)),
                "sampling_runs": int(len(sampling_runs)),
                "heldout_projection_runs": int(len(heldout_metrics)),
                "n_jobs": n_jobs,
            },
        )
        return {"status": "ran", **manifest, "outputs": all_outputs}

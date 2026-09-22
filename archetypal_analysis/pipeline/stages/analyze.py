"""Run quantitative AA/benchmark comparison metrics and registered downstream analyses."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ...analysis.comparison_metrics import (
    evaluate_representation,
    fuzzy_c_means,
    normalize_memberships,
)
from ...analysis.experiments import ANALYSIS_REGISTRY, load_result
from ..artifacts import atomic_write_json
from .base import PipelineStage, StageContext


METRIC_POLICY = {
    "silhouette": "Hard clusters are dominant fuzzy memberships; score is computed in each method's representation coordinates.",
    "davies_bouldin": "Hard clusters are dominant fuzzy memberships; score is computed in each method's representation coordinates.",
    "fuzzy_silhouette": "Crisp sample silhouettes weighted by the difference between the largest and second-largest membership.",
    "metadata_permanova": "Categorical targets use one-way Euclidean pseudo-F; continuous targets use a one-predictor distance-based linear-model pseudo-F.",
    "metadata_fuzzy_nmi": "Fuzzy normalized MI from soft memberships. Continuous metadata uses configured evaluation buckets or quantile bins.",
    "reconstruction_rss": "Common least-squares linear decoder from each representation back to fit_input; native AA/PCA RSS is also reported separately.",
    "convex_hull_coverage_ratio": "Convex-hull volume divided by its axis-aligned bounding box after intrinsic-rank projection and min-max scaling.",
}


class AnalyzeStage(PipelineStage):
    name = "analyze"
    dependencies = ("select", "benchmark", "prediction", "stability")

    def config_fragment(self, cfg):
        return {
            "analysis": cfg["analysis"],
            "task": cfg["task"],
            "evaluation": cfg["evaluation"],
            "fit_k_values": cfg["fit"]["k_values"],
        }

    def required_outputs(self, ctx: StageContext) -> list[Path]:
        d = ctx.store.stage_dir(self.name)
        return [
            d / "analysis_manifest.json",
            d / "representation_metrics.tsv",
            d / "representation_metrics.json",
            d / "prediction_scores.tsv",
            d / "initialization_stability_summary.tsv",
            d / "sampling_stability_summary.tsv",
            d / "heldout_projection_metrics.tsv",
        ]

    @staticmethod
    def _native_aa_rss(result: dict[str, Any], fit_input: pd.DataFrame) -> float:
        similarity = pd.DataFrame(
            result["similarity_degree"],
            index=pd.Index(result["index"], name="sample"),
            columns=result["columns"],
        )
        common = similarity.index.intersection(fit_input.index, sort=False)
        if len(common) == 0:
            return np.nan
        x = fit_input.loc[common].to_numpy(dtype=float)
        alpha = similarity.loc[common].to_numpy(dtype=float)
        archetypes = np.asarray(result["archetypes"], dtype=float)
        return float(np.sum((x - alpha @ archetypes) ** 2))

    @staticmethod
    def _native_pca_rss(
        representation: pd.DataFrame,
        dimension: int,
        fit_input: pd.DataFrame,
        artifact_dir: Path,
    ) -> float:
        components_path = artifact_dir / "components.tsv"
        mean_path = artifact_dir / "mean.tsv"
        if not components_path.exists() or not mean_path.exists():
            return np.nan
        components = pd.read_csv(components_path, sep="\t", index_col=0)
        mean = pd.read_csv(mean_path, sep="\t", index_col=0).iloc[0]
        common = representation.index.intersection(fit_input.index, sort=False)
        if len(common) == 0:
            return np.nan
        z = representation.loc[common].iloc[:, :dimension].to_numpy(dtype=float)
        basis = components.iloc[:dimension].reindex(columns=fit_input.columns).to_numpy(dtype=float)
        center = mean.reindex(fit_input.columns).to_numpy(dtype=float)
        reconstructed = z @ basis + center
        x = fit_input.loc[common].to_numpy(dtype=float)
        return float(np.sum((x - reconstructed) ** 2))

    def _quantitative_metrics(
        self,
        ctx: StageContext,
        fit_input: pd.DataFrame,
        metadata: pd.DataFrame,
    ) -> pd.DataFrame:
        metric_cfg = dict(ctx.config["analysis"].get("metrics", {}) or {})
        if not metric_cfg.get("enabled", True):
            return pd.DataFrame()

        target = ctx.config["task"]["target"]
        metadata_target = metadata[target]
        fit_dir = ctx.store.stage_dir("fit")
        rows: list[dict[str, Any]] = []

        # AA is evaluated for every fitted K, independent of selected-K downstream plots.
        for k in sorted(int(value) for value in ctx.config["fit"]["k_values"]):
            result = load_result(fit_dir / f"K_{k:03d}" / "result.json")
            representation = pd.DataFrame(
                result["similarity_degree"],
                index=pd.Index(result["index"], name="sample"),
                columns=result["columns"],
            )
            memberships = normalize_memberships(representation.to_numpy(dtype=float))
            row = evaluate_representation(
                method="archetypal_analysis",
                dimension=k,
                representation=representation,
                memberships=memberships,
                original_input=fit_input,
                metadata=metadata_target,
                task_cfg=ctx.config["task"],
                evaluation_cfg=ctx.config["evaluation"],
                metric_cfg=metric_cfg,
                native_reconstruction_rss=self._native_aa_rss(result, fit_input),
                native_reconstruction_method="aa_similarity_times_archetypes",
            )
            row.update({"family": "aa", "k": k, "benchmark_fit_dimension": np.nan})
            rows.append(row)

        benchmark_dir = ctx.store.stage_dir("benchmark")
        manifest_path = benchmark_dir / "benchmark_manifest.json"
        if not manifest_path.exists():
            return pd.DataFrame(rows)
        benchmark_manifest = json.loads(manifest_path.read_text())
        if not benchmark_manifest.get("enabled", True):
            return pd.DataFrame(rows)

        fuzzifier = float(metric_cfg.get("fuzzy_m", 2.0))
        fuzzy_max_iter = int(metric_cfg.get("fuzzy_max_iter", 300))
        fuzzy_tol = float(metric_cfg.get("fuzzy_tol", 1e-5))
        random_state = int(metric_cfg.get("random_state", 42))
        max_k = int(benchmark_manifest["max_k"])
        max_dimension = int(benchmark_manifest.get("max_dimension", max_k))
        k_to_dimension = {
            int(k): int(dimension)
            for k, dimension in (benchmark_manifest.get("aa_k_to_dimension", {}) or {}).items()
        }
        dimension_to_k = {dimension: k for k, dimension in k_to_dimension.items()}
        benchmark_dimensions = [
            int(dimension)
            for dimension in benchmark_manifest.get("dimensions", range(1, max_dimension + 1))
        ]

        for method, method_info in benchmark_manifest.get("methods", {}).items():
            artifacts = method_info.get("artifacts", [])
            if not artifacts:
                continue

            if method in {"pca", "pcoa"}:
                artifact = artifacts[0]
                embedding_path = Path(artifact["embedding_path"])
                max_embedding = pd.read_csv(embedding_path, sep="\t", index_col="sample")
                artifact_dir = embedding_path.parent
                for dimension in benchmark_dimensions:
                    comparison_k = int(dimension_to_k.get(dimension, dimension + 1))
                    representation = max_embedding.iloc[:, :dimension].copy()
                    partition = fuzzy_c_means(
                        representation.to_numpy(dtype=float),
                        comparison_k,
                        fuzzifier=fuzzifier,
                        max_iter=fuzzy_max_iter,
                        tol=fuzzy_tol,
                        random_state=random_state,
                    )
                    native_rss = (
                        self._native_pca_rss(representation, dimension, fit_input, artifact_dir)
                        if method == "pca"
                        else np.nan
                    )
                    row = evaluate_representation(
                        method=method,
                        dimension=dimension,
                        representation=representation,
                        memberships=partition.memberships,
                        original_input=fit_input,
                        metadata=metadata_target,
                        task_cfg=ctx.config["task"],
                        evaluation_cfg=ctx.config["evaluation"],
                        metric_cfg=metric_cfg,
                        native_reconstruction_rss=native_rss,
                        native_reconstruction_method=("pca_inverse_transform" if method == "pca" else "not_available"),
                    )
                    row.update(
                        {
                            "family": "benchmark",
                            "k": np.nan,
                            "comparison_k": comparison_k,
                            "benchmark_fit_dimension": max_dimension,
                            "fuzzy_iterations": partition.iterations,
                        }
                    )
                    rows.append(row)
                continue

            for artifact in sorted(artifacts, key=lambda value: int(value["dimension"])):
                dimension = int(artifact["dimension"])
                comparison_k = int(dimension_to_k.get(dimension, dimension + 1))
                representation = pd.read_csv(
                    artifact["embedding_path"], sep="\t", index_col="sample"
                )
                partition = fuzzy_c_means(
                    representation.to_numpy(dtype=float),
                    comparison_k,
                    fuzzifier=fuzzifier,
                    max_iter=fuzzy_max_iter,
                    tol=fuzzy_tol,
                    random_state=random_state,
                )
                row = evaluate_representation(
                    method=method,
                    dimension=dimension,
                    representation=representation,
                    memberships=partition.memberships,
                    original_input=fit_input,
                    metadata=metadata_target,
                    task_cfg=ctx.config["task"],
                    evaluation_cfg=ctx.config["evaluation"],
                    metric_cfg=metric_cfg,
                    native_reconstruction_rss=np.nan,
                    native_reconstruction_method="not_available",
                )
                row.update(
                    {
                        "family": "benchmark",
                        "k": np.nan,
                        "comparison_k": comparison_k,
                        "benchmark_fit_dimension": dimension,
                        "fuzzy_iterations": partition.iterations,
                    }
                )
                rows.append(row)

        metrics = pd.DataFrame(rows)
        if not metrics.empty:
            family_order = pd.Categorical(metrics["family"], categories=["aa", "benchmark"], ordered=True)
            metrics = metrics.assign(_family_order=family_order).sort_values(
                ["_family_order", "method", "dimension"]
            ).drop(columns="_family_order").reset_index(drop=True)
        return metrics

    def run(self, ctx: StageContext, overwrite: bool = False) -> dict[str, Any]:
        fp = self.fingerprint(ctx)
        outputs = self.required_outputs(ctx)
        if not overwrite and ctx.store.is_reusable(self.name, fp, outputs):
            return {"status": "reused", "outputs": outputs}

        selection = json.loads((ctx.store.stage_dir("select") / "selection.json").read_text())
        selected = [int(k) for k in selection["selected_k"]]
        fit_dir = ctx.store.stage_dir("fit")
        fit_input = pd.read_csv(fit_dir / "fit_input.tsv", sep="\t", index_col="sample")
        metadata = pd.read_csv(
            ctx.store.stage_dir("qc") / "metadata.tsv",
            sep="\t",
            index_col="sample",
            low_memory=False,
        )
        modules = ctx.config["analysis"].get("modules", []) or []

        stage_dir = ctx.store.stage_dir(self.name)
        metrics = self._quantitative_metrics(ctx, fit_input, metadata)
        metrics_tsv = stage_dir / "representation_metrics.tsv"
        metrics_json = stage_dir / "representation_metrics.json"
        metrics.to_csv(metrics_tsv, sep="\t", index=False)
        metrics_json.write_text(metrics.to_json(orient="records", indent=2))

        prediction_dir = ctx.store.stage_dir("prediction")
        stability_dir = ctx.store.stage_dir("stability")
        summary_copies = {
            "prediction_scores": (prediction_dir / "prediction_scores.tsv", stage_dir / "prediction_scores.tsv"),
            "initialization_stability": (stability_dir / "initialization_archetype_summary.tsv", stage_dir / "initialization_stability_summary.tsv"),
            "sampling_stability": (stability_dir / "sampling_archetype_summary.tsv", stage_dir / "sampling_stability_summary.tsv"),
            "heldout_projection_metrics": (stability_dir / "heldout_projection_metrics.tsv", stage_dir / "heldout_projection_metrics.tsv"),
        }
        for _name, (source, destination) in summary_copies.items():
            if source.exists():
                destination.write_bytes(source.read_bytes())
            else:
                pd.DataFrame().to_csv(destination, sep="\t", index=False)

        runtime_cfg = dict(ctx.config)
        runtime_cfg["_runtime"] = {
            "fit_dir": str(fit_dir),
            "benchmark_dir": str(ctx.store.stage_dir("benchmark")),
        }
        manifest: dict[str, Any] = {
            "selected_k": selected,
            "modules": modules,
            "metric_policy": METRIC_POLICY,
            "metric_rows": int(len(metrics)),
            "metrics": [str(metrics_tsv), str(metrics_json)],
            "consolidated_analysis_tables": {
                name: str(destination) for name, (_source, destination) in summary_copies.items()
            },
            "prediction": {
                "scores": str(prediction_dir / "prediction_scores.tsv"),
                "manifest": str(prediction_dir / "prediction_manifest.json"),
            },
            "stability": {
                "manifest": str(stability_dir / "stability_manifest.json"),
                "initialization_archetype_metrics": str(stability_dir / "initialization_archetype_metrics.tsv"),
                "sampling_archetype_metrics": str(stability_dir / "sampling_archetype_metrics.tsv"),
                "subsample_fit_metrics": str(stability_dir / "subsample_fit_metrics.tsv"),
                "heldout_projection_metrics": str(stability_dir / "heldout_projection_metrics.tsv"),
            },
            "outputs": {},
        }
        all_outputs: list[Path] = [
            metrics_tsv,
            metrics_json,
            *[destination for _source, destination in summary_copies.values()],
        ]
        for name in modules:
            if name not in ANALYSIS_REGISTRY:
                raise KeyError(
                    f"Unknown analysis module '{name}'. Registered modules: {sorted(ANALYSIS_REGISTRY)}"
                )
        for k in selected:
            k_dir = stage_dir / f"K_{k:03d}"
            k_dir.mkdir(parents=True, exist_ok=True)
            result = load_result(fit_dir / f"K_{k:03d}" / "result.json")
            manifest["outputs"][str(k)] = {}
            for name in modules:
                produced = ANALYSIS_REGISTRY[name](
                    result, metadata, fit_input, runtime_cfg, k_dir
                )
                manifest["outputs"][str(k)][name] = [str(p) for p in produced]
                all_outputs.extend(produced)

        manifest_path = stage_dir / "analysis_manifest.json"
        atomic_write_json(manifest_path, manifest)
        ctx.store.mark_success(
            self.name,
            fp,
            [manifest_path, *all_outputs],
            {
                "selected_k": selected,
                "modules": modules,
                "metric_rows": int(len(metrics)),
            },
        )
        return {
            "status": "ran",
            "selected_k": selected,
            "modules": modules,
            "metric_rows": int(len(metrics)),
            "outputs": [manifest_path, *all_outputs],
        }

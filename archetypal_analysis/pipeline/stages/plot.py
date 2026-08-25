"""Create all configured AA and dataset visualizations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ...analysis.experiments import load_result
from ...visualization.archetype_ordering import resolve_archetype_order
from ...visualization.embeddings import compute_projection, project_archetypes_from_similarity
from ...visualization.pipeline_plots import (
    build_supplemental_archetypes,
    plot_k_summary,
    plot_metrics,
    plot_multi_k_simplex,
    plot_projected_embedding,
    plot_projected_pcoa,
    plot_quality_selection,
    plot_similarity_stacked,
    plot_single_simplex,
    plot_taxa_stacked,
    relabel_result_for_visual_order,
    save_figure,
    visual_top_taxa_frame,
)
from .base import PipelineStage, StageContext


class PlotStage(PipelineStage):
    name = "plot"
    dependencies = ("select",)

    def config_fragment(self, cfg):
        return {"visualization": cfg["visualization"], "task": cfg["task"]}

    def required_outputs(self, ctx: StageContext) -> list[Path]:
        return [ctx.store.stage_dir(self.name) / "plot_manifest.json"]

    @staticmethod
    def _embedding_cfg(vis: dict[str, Any], name: str) -> dict[str, Any]:
        cfg = dict(vis.get(name, {}) or {})
        cfg.setdefault("archetype_colors", vis.get("archetype_colors"))
        global_projection = dict(vis.get("archetype_projection", {}) or {})
        local_projection = dict(cfg.get("archetype_projection", {}) or {})
        cfg["archetype_projection"] = {**global_projection, **local_projection}
        return cfg

    def run(self, ctx: StageContext, overwrite: bool = False) -> dict[str, Any]:
        fp = self.fingerprint(ctx)
        outputs = self.required_outputs(ctx)
        if not overwrite and ctx.store.is_reusable(self.name, fp, outputs):
            return {"status": "reused", "outputs": outputs}

        vis = dict(ctx.config["visualization"])
        if not vis.get("enabled", True):
            manifest = {"enabled": False, "outputs": []}
            outputs[0].write_text(json.dumps(manifest, indent=2))
            ctx.store.mark_success(self.name, fp, outputs, manifest)
            return {"status": "ran", **manifest}

        stage_dir = ctx.store.stage_dir(self.name)
        fit_dir = ctx.store.stage_dir("fit")
        selection = json.loads((ctx.store.stage_dir("select") / "selection.json").read_text())
        selected = [int(k) for k in selection["selected_k"]]
        metadata = pd.read_csv(
            ctx.store.stage_dir("filter") / "metadata.tsv",
            sep="\t",
            index_col="sample",
            low_memory=False,
        )
        fit_input = pd.read_csv(fit_dir / "fit_input.tsv", sep="\t", index_col="sample")
        filtered_otu = pd.read_csv(
            ctx.store.stage_dir("filter") / "otu.tsv",
            sep="\t",
            index_col="sample",
            low_memory=False,
        )
        raw_for_embeddings = filtered_otu.loc[fit_input.index]
        metadata_fit = metadata.loc[fit_input.index]
        transform = json.loads((fit_dir / "transform.json").read_text())
        metrics = pd.read_csv(ctx.store.stage_dir("evaluate") / "k_metrics.tsv", sep="\t")
        task = ctx.config["task"]
        ext = str(vis.get("format", "png")).lstrip(".")
        produced: list[Path] = []

        def emit(fig, path: Path, local_cfg):
            produced.append(
                save_figure(fig, path, {"dpi": vis.get("dpi", 180), **dict(local_cfg)})
            )

        common = stage_dir / "common"
        common.mkdir(parents=True, exist_ok=True)

        metrics_cfg = vis.get("metrics_over_k", {}) or {}
        if metrics_cfg.get("enabled", True):
            emit(
                plot_metrics(metrics, metrics_cfg),
                common / f"silhouette_davies_bouldin_over_k.{ext}",
                metrics_cfg,
            )
            if "quality_score" in metrics.columns:
                emit(
                    plot_quality_selection(metrics, metrics_cfg),
                    common / f"normalized_quality_over_k.{ext}",
                    metrics_cfg,
                )

        multi_cfg = dict(vis.get("multi_k_simplex", {}) or {})
        multi_cfg.setdefault("archetype_order", vis.get("archetype_order"))
        multi_cfg.setdefault("archetype_ordering", vis.get("archetype_ordering", {}))
        if multi_cfg.get("enabled", True):
            desired = multi_cfg.get("k_values") or ctx.config["fit"]["k_values"]
            multi_results = {
                int(k): load_result(fit_dir / f"K_{int(k):03d}" / "result.json")
                for k in desired
                if (fit_dir / f"K_{int(k):03d}" / "result.json").exists()
            }
            if multi_results:
                emit(
                    plot_multi_k_simplex(multi_results, metadata, task, multi_cfg),
                    common / f"simplex_across_k.{ext}",
                    multi_cfg,
                )

        # Resolve selected-K result/order once. Explicit archetype_order takes precedence;
        # otherwise the configured metadata-aware ordering is used.
        results = {
            k: load_result(fit_dir / f"K_{k:03d}" / "result.json") for k in selected
        }
        order_cfg = {
            "archetype_order": vis.get("archetype_order"),
            "archetype_ordering": vis.get("archetype_ordering", {}),
        }
        resolved_orders = {
            k: resolve_archetype_order(result, metadata, task, order_cfg)
            for k, result in results.items()
        }
        visual_results: dict[int, dict[str, Any]] = {}
        display_orders: dict[int, list[str]] = {}
        display_name_maps: dict[int, dict[str, str]] = {}
        for k, result in results.items():
            visual_result, display_order, name_map = relabel_result_for_visual_order(
                result,
                resolved_orders[k],
            )
            visual_results[k] = visual_result
            display_orders[k] = display_order
            display_name_maps[k] = name_map

        # Fit each embedding once on the samples. All selected-K archetypes are treated as
        # supplementary points, so the common sample coordinates and each K-specific plot
        # genuinely share the same embedding space.
        supplemental_otu, archetype_ids = build_supplemental_archetypes(
            results,
            transform,
            raw_for_embeddings,
        )
        projection_cache = {}
        summary_enabled = bool((vis.get("k_summary", {}) or {}).get("enabled", True))
        for name in ("pcoa", "umap", "tsne"):
            emb_cfg = self._embedding_cfg(vis, name)
            should_compute = bool(emb_cfg.get("enabled", True)) or (name == "pcoa" and summary_enabled)
            if not should_compute:
                continue
            projection = compute_projection(name, raw_for_embeddings, supplemental_otu, emb_cfg)
            if name in {"umap", "tsne"}:
                projection = project_archetypes_from_similarity(
                    projection,
                    results,
                    archetype_ids,
                    emb_cfg,
                )
            projection_cache[name] = projection

            # Common plots remain sample-only and use the same fitted coordinates as the
            # per-K projected plots below.
            if emb_cfg.get("enabled", True):
                if name == "pcoa":
                    fig = plot_projected_pcoa(
                        projection,
                        metadata_fit,
                        task,
                        emb_cfg,
                        {},
                        [],
                        emb_cfg.get("title", "PCoA of fitted dataset"),
                    )
                else:
                    fig = plot_projected_embedding(
                        projection,
                        metadata_fit,
                        task,
                        emb_cfg,
                        {},
                        [],
                        emb_cfg.get("title", f"{name.upper()} of fitted dataset"),
                    )
                emit(fig, common / f"{name}.{ext}", emb_cfg)

        for k in selected:
            result = visual_results[k]
            order = display_orders[k]
            id_map = {
                display_name: archetype_ids[k][original_name]
                for original_name, display_name in display_name_maps[k].items()
                if original_name in archetype_ids[k]
            }
            k_dir = stage_dir / f"K_{k:03d}"
            k_dir.mkdir(parents=True, exist_ok=True)

            simplex_cfg = dict(vis.get("simplex", {}) or {})
            if simplex_cfg.get("enabled", True):
                emit(
                    plot_single_simplex(result, metadata, task, simplex_cfg, order=order),
                    k_dir / f"simplex.{ext}",
                    simplex_cfg,
                )

            stacked_cfg = dict(vis.get("stacked_bar", {}) or {})
            stacked_cfg.setdefault("archetype_colors", vis.get("archetype_colors"))
            if stacked_cfg.get("enabled", True):
                emit(
                    plot_similarity_stacked(result, metadata, task, stacked_cfg, order=order),
                    k_dir / f"similarity_stacked_bar.{ext}",
                    stacked_cfg,
                )

            # Per-K taxa output intentionally keeps only the stacked contribution plot.
            taxa_cfg = dict(vis.get("taxa", {}) or {})
            if taxa_cfg.get("enabled", True) and taxa_cfg.get("stacked_enabled", True):
                visual_top_taxa_path = k_dir / "visual_top_taxa.tsv"
                visual_top_taxa_frame(result, transform, taxa_cfg, order=order).to_csv(
                    visual_top_taxa_path,
                    sep="\t",
                    index=False,
                )
                produced.append(visual_top_taxa_path)
                emit(
                    plot_taxa_stacked(result, transform, taxa_cfg, order=order),
                    k_dir / f"taxa_stacked_bar.{ext}",
                    taxa_cfg,
                )

            if "pcoa" in projection_cache and (vis.get("pcoa", {}) or {}).get("enabled", True):
                pcoa_cfg = self._embedding_cfg(vis, "pcoa")
                emit(
                    plot_projected_pcoa(
                        projection_cache["pcoa"],
                        metadata_fit,
                        task,
                        pcoa_cfg,
                        id_map,
                        order,
                        pcoa_cfg.get("projected_title", f"PCoA with archetypes (K={k})"),
                    ),
                    k_dir / f"pcoa.{ext}",
                    pcoa_cfg,
                )

            for name in ("umap", "tsne"):
                emb_cfg = self._embedding_cfg(vis, name)
                if name not in projection_cache or not emb_cfg.get("enabled", True):
                    continue
                emit(
                    plot_projected_embedding(
                        projection_cache[name],
                        metadata_fit,
                        task,
                        emb_cfg,
                        id_map,
                        order,
                        emb_cfg.get("projected_title", f"{name.upper()} with archetypes (K={k})"),
                    ),
                    k_dir / f"{name}.{ext}",
                    emb_cfg,
                )

            summary_cfg = vis.get("k_summary", {}) or {}
            if summary_cfg.get("enabled", True):
                if "pcoa" not in projection_cache:
                    raise RuntimeError("k_summary requires a PCoA projection")
                emit(
                    plot_k_summary(
                        result,
                        metadata,
                        task,
                        vis,
                        transform,
                        projection_cache["pcoa"],
                        id_map,
                        order,
                    ),
                    k_dir / f"summary.{ext}",
                    summary_cfg,
                )

        manifest = {
            "selected_k": selected,
            "resolved_archetype_orders": {str(k): order for k, order in resolved_orders.items()},
            "visual_archetype_labels": {
                str(k): display_name_maps[k] for k in selected
            },
            "embedding_projection_methods": {
                name: projection.projection_method for name, projection in projection_cache.items()
            },
            "outputs": [str(path) for path in produced],
        }
        outputs[0].write_text(json.dumps(manifest, indent=2))
        ctx.store.mark_success(
            self.name,
            fp,
            [outputs[0], *produced],
            {
                "selected_k": selected,
                "resolved_archetype_orders": manifest["resolved_archetype_orders"],
                "visual_archetype_labels": manifest["visual_archetype_labels"],
                "embedding_projection_methods": manifest["embedding_projection_methods"],
            },
        )
        return {
            "status": "ran",
            "selected_k": selected,
            "outputs": [outputs[0], *produced],
            "resolved_archetype_orders": manifest["resolved_archetype_orders"],
            "visual_archetype_labels": manifest["visual_archetype_labels"],
        }

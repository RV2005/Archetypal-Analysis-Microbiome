"""Metadata filtering, optional balanced subsampling, and metadata pruning."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ...data.dataset import DatasetBundle
from ...data.filtering import DatasetFilter
from .base import PipelineStage, StageContext


def required_metadata_columns(cfg: Mapping[str, Any]) -> list[str]:
    """Return metadata columns persisted with the filtered AA dataset.

    Only metadata used by the AA/evaluation/plot pipeline is retained here. Repeated-
    measures grouping variables used by prediction/stability are resolved on demand from
    the canonical standardized metadata file. Keeping those concerns separate means a
    prediction-only grouping change never invalidates or refits AA/benchmark artifacts,
    and heterogeneous studies can use different subject-ID columns.
    """
    task_cfg = cfg["task"]
    configured = task_cfg.get("required_metadata_columns", []) or []
    if not isinstance(configured, list) or not all(isinstance(column, str) for column in configured):
        raise ValueError("task.required_metadata_columns must be a list of column names")
    target = str(task_cfg["target"])
    ordered: list[str] = []
    for column in [*configured, target]:
        if column and column not in ordered:
            ordered.append(str(column))
    return ordered


class FilterStage(PipelineStage):
    name = "filter"
    dependencies = ("load",)

    def required_outputs(self, ctx: StageContext) -> list[Path]:
        d = ctx.store.stage_dir(self.name)
        return [d / "otu.tsv", d / "metadata.tsv"]

    def config_fragment(self, cfg):
        return {"filter": cfg["filter"], "task": cfg["task"]}

    def run(self, ctx: StageContext, overwrite: bool = False) -> dict[str, Any]:
        fp = self.fingerprint(ctx)
        outputs = self.required_outputs(ctx)
        if not overwrite and ctx.store.is_reusable(self.name, fp, outputs):
            return {"status": "reused", "outputs": outputs}

        bundle = ctx.transient.get("loaded_bundle")
        if bundle is None:
            raise RuntimeError("Filter requires the transient loaded dataset; run through PipelineRunner so load executes first")
        if ctx.config["filter"].get("enabled", True):
            bundle = DatasetFilter().apply(bundle, ctx.config["filter"], ctx.config["task"])
        else:
            bundle = DatasetBundle(bundle.otu.copy(), bundle.metadata.copy())

        keep_columns = required_metadata_columns(ctx.config)
        missing = [column for column in keep_columns if column not in bundle.metadata.columns]
        if missing:
            raise KeyError(
                f"Required downstream metadata columns are missing after filtering: {missing}. "
                f"Available columns include: {list(bundle.metadata.columns)[:30]}"
            )
        bundle = DatasetBundle(bundle.otu, bundle.metadata.loc[:, keep_columns].copy())

        d = ctx.store.stage_dir(self.name)
        bundle.otu.to_csv(d / "otu.tsv", sep="\t", index_label="sample")
        bundle.metadata.to_csv(d / "metadata.tsv", sep="\t", index_label="sample")
        details = {
            "n_samples": int(bundle.otu.shape[0]),
            "n_features": int(bundle.otu.shape[1]),
            "n_metadata_columns": int(bundle.metadata.shape[1]),
            "metadata_columns": keep_columns,
        }
        ctx.store.mark_success(self.name, fp, outputs, details)
        return {"status": "ran", **details, "outputs": outputs}
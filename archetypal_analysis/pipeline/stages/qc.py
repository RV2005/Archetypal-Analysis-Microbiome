"""OTU quality-control stage run after metadata filtering."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ...data.quality_control import QualityController
from ..artifacts import atomic_write_json
from .base import PipelineStage, StageContext


class QCStage(PipelineStage):
    name = "qc"
    dependencies = ("filter",)

    def required_outputs(self, ctx: StageContext) -> list[Path]:
        d = ctx.store.stage_dir(self.name)
        return [d / "otu.tsv", d / "metadata.tsv", d / "qc_summary.json"]

    def run(self, ctx: StageContext, overwrite: bool = False) -> dict[str, Any]:
        fp = self.fingerprint(ctx)
        outputs = self.required_outputs(ctx)
        if not overwrite and ctx.store.is_reusable(self.name, fp, outputs):
            return {"status": "reused", "outputs": outputs}

        filter_dir = ctx.store.stage_dir("filter")
        otu = pd.read_csv(filter_dir / "otu.tsv", sep="\t", index_col="sample", low_memory=False)
        metadata = pd.read_csv(filter_dir / "metadata.tsv", sep="\t", index_col="sample", low_memory=False)
        input_shape = {"n_samples": int(otu.shape[0]), "n_features": int(otu.shape[1])}
        missing_otu_values = int(otu.isna().sum().sum())
        if missing_otu_values:
            # Standardized sparse OTU tables can encode absent taxa as blank cells.
            # Downstream count/abundance preprocessing expects explicit zeros.
            otu = otu.fillna(0)

        qc_cfg = dict(ctx.config["qc"])
        enabled = bool(qc_cfg.pop("enabled", True))
        if enabled:
            allowed = {"min_taxa_level", "min_nonzero_samples", "min_total_counts", "std_filter", "min_counts"}
            kwargs = {k: v for k, v in qc_cfg.items() if k in allowed}
            otu = QualityController().quality_control(otu, verbose=ctx.verbose, **kwargs)
        else:
            otu = otu.copy()

        metadata = metadata.loc[metadata.index.intersection(otu.index, sort=False)].copy()
        otu = otu.loc[metadata.index].copy()
        output_shape = {"n_samples": int(otu.shape[0]), "n_features": int(otu.shape[1])}
        summary = {
            "enabled": enabled,
            "input": input_shape,
            "output": output_shape,
            "removed_samples": input_shape["n_samples"] - output_shape["n_samples"],
            "removed_features": input_shape["n_features"] - output_shape["n_features"],
            "missing_otu_values_filled": missing_otu_values,
            "n_metadata_columns": int(metadata.shape[1]),
            "metadata_columns": metadata.columns.tolist(),
            "config": {"enabled": enabled, **qc_cfg},
        }

        d = ctx.store.stage_dir(self.name)
        otu.to_csv(d / "otu.tsv", sep="\t", index_label="sample")
        metadata.to_csv(d / "metadata.tsv", sep="\t", index_label="sample")
        atomic_write_json(d / "qc_summary.json", summary)

        details = {
            "n_samples": int(otu.shape[0]),
            "n_features": int(otu.shape[1]),
            "n_metadata_columns": int(metadata.shape[1]),
            "removed_samples": summary["removed_samples"],
            "removed_features": summary["removed_features"],
            "missing_otu_values_filled": missing_otu_values,
            "persisted_dataset": True,
        }
        ctx.store.mark_success(self.name, fp, outputs, details)
        return {"status": "ran", **details, "outputs": outputs}
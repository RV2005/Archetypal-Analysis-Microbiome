"""Transient OTU quality-control stage run before metadata filtering."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...data.dataset import DatasetBundle
from ...data.quality_control import QualityController
from .base import PipelineStage, StageContext


class QCStage(PipelineStage):
    name = "qc"
    dependencies = ("load",)
    transient_stage = True

    def required_outputs(self, ctx: StageContext) -> list[Path]:
        # QC is intentionally transient: filter is the first stage that persists a dataset.
        return []

    def run(self, ctx: StageContext, overwrite: bool = False) -> dict[str, Any]:
        fp = self.fingerprint(ctx)
        bundle = ctx.transient.get("loaded_bundle")
        if bundle is None:
            raise RuntimeError("QC requires the transient loaded dataset; run through PipelineRunner so load executes first")

        qc_cfg = dict(ctx.config["qc"])
        enabled = bool(qc_cfg.pop("enabled", True))
        if enabled:
            allowed = {"min_taxa_level", "min_nonzero_samples", "min_total_counts", "std_filter", "min_counts"}
            kwargs = {k: v for k, v in qc_cfg.items() if k in allowed}
            otu = QualityController().quality_control(bundle.otu, verbose=ctx.verbose, **kwargs)
        else:
            otu = bundle.otu.copy()

        metadata = bundle.metadata.loc[bundle.metadata.index.intersection(otu.index, sort=False)].copy()
        otu = otu.loc[metadata.index].copy()
        qc_bundle = DatasetBundle(otu=otu, metadata=metadata)
        ctx.transient["qc_bundle"] = qc_bundle

        details = {
            "n_samples": int(otu.shape[0]),
            "n_features": int(otu.shape[1]),
            "n_metadata_columns": int(metadata.shape[1]),
            "persisted_dataset": False,
        }
        ctx.store.mark_success(self.name, fp, [], details)
        return {"status": "ran", **details, "outputs": []}
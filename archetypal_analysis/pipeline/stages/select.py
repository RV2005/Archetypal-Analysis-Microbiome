"""Choose downstream K values or deliberately keep all K values."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ...analysis.selection import select_k
from .base import PipelineStage, StageContext


class SelectStage(PipelineStage):
    name = "select"
    dependencies = ("evaluate",)

    def required_outputs(self, ctx: StageContext) -> list[Path]:
        return [ctx.store.stage_dir(self.name) / "selection.json"]

    def config_fragment(self, cfg):
        return {"selection": cfg["selection"], "fit_k_values": cfg["fit"]["k_values"]}

    def run(self, ctx: StageContext, overwrite: bool = False) -> dict[str, Any]:
        fp = self.fingerprint(ctx)
        outputs = self.required_outputs(ctx)
        if not overwrite and ctx.store.is_reusable(self.name, fp, outputs):
            return {"status": "reused", "outputs": outputs}

        metrics = pd.read_csv(ctx.store.stage_dir("evaluate") / "k_metrics.tsv", sep="\t")
        selection = select_k(metrics, ctx.config["selection"], [int(k) for k in ctx.config["fit"]["k_values"]])
        path = outputs[0]
        path.write_text(json.dumps(selection, indent=2))
        ctx.store.mark_success(self.name, fp, outputs, selection)
        return {"status": "ran", **selection, "outputs": outputs}
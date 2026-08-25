"""Run registered downstream analyses for selected K value(s)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ...analysis.experiments import ANALYSIS_REGISTRY, load_result
from .base import PipelineStage, StageContext


class AnalyzeStage(PipelineStage):
    name = "analyze"
    dependencies = ("select",)

    def config_fragment(self, cfg):
        return {"analysis": cfg["analysis"], "task": cfg["task"]}

    def required_outputs(self, ctx: StageContext) -> list[Path]:
        return [ctx.store.stage_dir(self.name) / "analysis_manifest.json"]

    def run(self, ctx: StageContext, overwrite: bool = False) -> dict[str, Any]:
        fp = self.fingerprint(ctx)
        outputs = self.required_outputs(ctx)
        if not overwrite and ctx.store.is_reusable(self.name, fp, outputs):
            return {"status": "reused", "outputs": outputs}

        selection = json.loads((ctx.store.stage_dir("select") / "selection.json").read_text())
        selected = [int(k) for k in selection["selected_k"]]
        fit_dir = ctx.store.stage_dir("fit")
        fit_input = pd.read_csv(fit_dir / "fit_input.tsv", sep="\t", index_col="sample")
        metadata = pd.read_csv(ctx.store.stage_dir("filter") / "metadata.tsv", sep="\t", index_col="sample", low_memory=False)
        modules = ctx.config["analysis"].get("modules", []) or []

        runtime_cfg = dict(ctx.config)
        runtime_cfg["_runtime"] = {"fit_dir": str(fit_dir)}
        manifest: dict[str, Any] = {"selected_k": selected, "modules": {}, "outputs": {}}
        all_outputs: list[Path] = []
        for name in modules:
            if name not in ANALYSIS_REGISTRY:
                raise KeyError(f"Unknown analysis module '{name}'. Registered modules: {sorted(ANALYSIS_REGISTRY)}")
        for k in selected:
            k_dir = ctx.store.stage_dir(self.name) / f"K_{k:03d}"
            k_dir.mkdir(parents=True, exist_ok=True)
            result = load_result(fit_dir / f"K_{k:03d}" / "result.json")
            manifest["outputs"][str(k)] = {}
            for name in modules:
                produced = ANALYSIS_REGISTRY[name](result, metadata, fit_input, runtime_cfg, k_dir)
                manifest["outputs"][str(k)][name] = [str(p) for p in produced]
                all_outputs.extend(produced)

        manifest_path = outputs[0]
        manifest_path.write_text(json.dumps(manifest, indent=2))
        ctx.store.mark_success(self.name, fp, [manifest_path, *all_outputs], {"selected_k": selected, "modules": modules})
        return {"status": "ran", "selected_k": selected, "modules": modules, "outputs": [manifest_path, *all_outputs]}
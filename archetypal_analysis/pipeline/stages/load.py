"""Load and validate the standardized dataset without persisting a copy."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...data.dataset import UnifiedDatasetLoader
from ..artifacts import file_signature, stable_hash
from ..config import dataset_paths
from .base import PipelineStage, StageContext


class LoadStage(PipelineStage):
    name = "load"
    transient_stage = True

    def required_outputs(self, ctx: StageContext) -> list[Path]:
        # The load stage intentionally does not persist OTU/metadata copies.
        return []

    def fingerprint(self, ctx: StageContext) -> str:
        otu_path, metadata_path = dataset_paths(ctx.config)
        return stable_hash({
            "stage": self.name,
            "dataset_config": ctx.config["dataset"],
            "task_target": ctx.config["task"]["target"],
            "otu": file_signature(otu_path),
            "metadata": file_signature(metadata_path),
        })

    def run(self, ctx: StageContext, overwrite: bool = False) -> dict[str, Any]:
        # Load is replayed when an invalid downstream persistent stage needs the data.
        # The only persisted record is the lightweight success marker.
        fp = self.fingerprint(ctx)
        otu_path, metadata_path = dataset_paths(ctx.config)
        loader = UnifiedDatasetLoader()
        bundle = loader.load(otu_path, metadata_path, ctx.config["dataset"])
        target = ctx.config["task"]["target"]
        validation_cfg = ctx.config.get("dataset", {}).get("validation", {}) or {}
        if isinstance(validation_cfg, bool):
            validation_cfg = {"enabled": validation_cfg}
        validation_enabled = bool(validation_cfg.get("enabled", True))
        validate_target = bool(validation_cfg.get("target_column", validation_enabled))
        if validate_target and target not in bundle.metadata.columns:
            raise KeyError(
                f"Target metadata column '{target}' not found. "
                f"Available columns: {list(bundle.metadata.columns)[:30]}"
            )

        ctx.transient["loaded_bundle"] = bundle
        details = {
            "n_samples": int(bundle.otu.shape[0]),
            "n_features": int(bundle.otu.shape[1]),
            "n_metadata_columns": int(bundle.metadata.shape[1]),
            "target": target,
            "target_non_null": (
                int(bundle.metadata[target].notna().sum())
                if validate_target and target in bundle.metadata.columns
                else None
            ),
            "source_otu": str(otu_path),
            "source_metadata": str(metadata_path),
            "persisted_dataset": False,
            "loader_timings": loader.last_timings,
            "loader_reader": loader.last_reader_info,
            "loader_validation": loader.last_validation_info,
        }
        ctx.store.mark_success(self.name, fp, [], details)
        return {"status": "ran", **details, "outputs": []}
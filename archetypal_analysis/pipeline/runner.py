"""Dependency-aware stage orchestration."""

from __future__ import annotations

import importlib
from time import perf_counter
from typing import Any, Iterable, Mapping

import yaml

from .artifacts import ArtifactStore
from .config import output_root
from .stages.base import PipelineStage, StageContext


DEFAULT_STAGE_ORDER = ["load", "qc", "filter", "fit", "evaluate", "select", "analyze", "plot"]
DEFAULT_STAGE_SPECS = {
    "load": ("archetypal_analysis.pipeline.stages.load", "LoadStage", ()),
    "qc": ("archetypal_analysis.pipeline.stages.qc", "QCStage", ("load",)),
    "filter": ("archetypal_analysis.pipeline.stages.filter", "FilterStage", ("qc",)),
    "fit": ("archetypal_analysis.pipeline.stages.fit", "FitStage", ("filter",)),
    "evaluate": ("archetypal_analysis.pipeline.stages.evaluate", "EvaluateStage", ("fit",)),
    "select": ("archetypal_analysis.pipeline.stages.select", "SelectStage", ("evaluate",)),
    "analyze": ("archetypal_analysis.pipeline.stages.analyze", "AnalyzeStage", ("select",)),
    "plot": ("archetypal_analysis.pipeline.stages.plot", "PlotStage", ("select",)),
}


class PipelineRunner:
    """Run any stage or the full pipeline while reusing valid artifacts."""

    def __init__(self, config: Mapping[str, Any], verbose: bool = True):
        self.config = dict(config)
        self.verbose = verbose
        self.store = ArtifactStore(output_root(config))
        self.order = list(DEFAULT_STAGE_ORDER)
        self._stage_specs = dict(DEFAULT_STAGE_SPECS)
        self._stage_instances: dict[str, PipelineStage] = {}
        self._custom_dependencies: dict[str, tuple[str, ...]] = {}
        self._write_resolved_config()

    def _get_stage(self, name: str) -> PipelineStage:
        if name in self._stage_instances:
            return self._stage_instances[name]
        if name not in self._stage_specs:
            raise KeyError(f"Unknown stage: {name}")
        module_name, class_name, _ = self._stage_specs[name]
        module = importlib.import_module(module_name)
        stage = getattr(module, class_name)()
        self._stage_instances[name] = stage
        return stage

    def _dependencies(self, name: str) -> tuple[str, ...]:
        if name in self._custom_dependencies:
            return self._custom_dependencies[name]
        if name in self._stage_specs:
            return tuple(self._stage_specs[name][2])
        return tuple(self._get_stage(name).dependencies)

    def register_stage(self, stage: PipelineStage, after: str | None = None) -> None:
        """Register a custom stage without modifying the runner implementation."""
        self._stage_instances[stage.name] = stage
        self._custom_dependencies[stage.name] = tuple(stage.dependencies)
        if stage.name not in self.order:
            if after and after in self.order:
                self.order.insert(self.order.index(after) + 1, stage.name)
            else:
                self.order.append(stage.name)

    def _write_resolved_config(self) -> None:
        payload = {k: v for k, v in self.config.items() if k != "_meta"}
        (self.store.root / "resolved_config.yaml").write_text(yaml.safe_dump(payload, sort_keys=False))

    def _normalize_targets(self, stages: Iterable[str]) -> list[str]:
        tokens: list[str] = []
        for value in stages:
            tokens.extend(part.strip() for part in str(value).split(",") if part.strip())
        if not tokens or "all" in tokens:
            return list(self.order)
        known = set(self.order)
        unknown = [x for x in tokens if x not in known]
        if unknown:
            raise KeyError(f"Unknown stages {unknown}; available: {self.order}")
        return [name for name in self.order if name in set(tokens)]

    def run(self, stages: Iterable[str] = ("all",), overwrite: bool = False) -> dict[str, Any]:
        requested = self._normalize_targets(stages)
        explicit = set(requested)
        completed: dict[str, Any] = {}
        visiting: set[str] = set()
        ctx = StageContext(config=self.config, store=self.store, verbose=self.verbose)

        reuse_memo: dict[str, bool] = {}

        def dependency_chain_reusable(name: str, checking: set[str] | None = None) -> bool:
            if name in reuse_memo:
                return reuse_memo[name]
            checking = set(checking or ())
            if name in checking:
                raise RuntimeError(f"Stage dependency cycle involving {name}")
            checking.add(name)
            stage = self._get_stage(name)
            reusable = stage.reusable(ctx) and all(
                dependency_chain_reusable(dep, checking) for dep in self._dependencies(name)
            )
            reuse_memo[name] = reusable
            return reusable

        def execute(name: str) -> None:
            if name in completed:
                return
            if name in visiting:
                raise RuntimeError(f"Stage dependency cycle involving {name}")

            stage = self._get_stage(name)
            force = bool(overwrite and name in explicit)

            # A persistent stage can short-circuit its dependency graph only when its
            # own artifact AND every upstream fingerprint are current. This is what
            # lets filter reuse skip transient load/QC without missing source-file or
            # QC-config changes.
            if not force and not stage.transient_stage and dependency_chain_reusable(name):
                result = {"status": "reused", "outputs": stage.required_outputs(ctx)}
                completed[name] = result
                if self.verbose:
                    print(f"\n[{name}] reuse")
                    print(f"[{name}] reused")
                return

            visiting.add(name)
            for dep in self._dependencies(name):
                if dep not in self.order:
                    raise KeyError(f"Stage '{name}' depends on unregistered stage '{dep}'")
                execute(dep)
            visiting.remove(name)

            if self.verbose:
                action = "overwrite" if force else "run/reuse"
                print(f"\n[{name}] {action}")
            started = perf_counter()
            result = stage.run(ctx, overwrite=force)
            elapsed = perf_counter() - started
            if result.get("status") == "ran":
                self.store.record_stage_duration(name, elapsed)
                result["duration_seconds"] = round(float(elapsed), 6)
            completed[name] = result
            if self.verbose:
                print(f"[{name}] {result.get('status', 'done')}")

        for target in requested:
            execute(target)
        return completed

    def status(self) -> list[dict[str, Any]]:
        rows = []
        for name in self.order:
            marker = self.store.stage_marker(name)
            rows.append({
                "stage": name,
                "complete": marker is not None,
                "fingerprint": marker.get("fingerprint") if marker else None,
                "completed_at": marker.get("completed_at") if marker else None,
                "duration_seconds": marker.get("duration_seconds") if marker else None,
            })
        return rows
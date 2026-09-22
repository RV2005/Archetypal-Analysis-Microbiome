"""Stage interface used by the pipeline runner."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ..artifacts import ArtifactStore, stable_hash


@dataclass
class StageContext:
    config: Mapping[str, Any]
    store: ArtifactStore
    verbose: bool = True
    # Transient stage-to-stage handoff. Load deliberately does not persist large
    # datasets; its DatasetBundle object lives here only for the current invocation.
    transient: dict[str, Any] = field(default_factory=dict)


class PipelineStage(ABC):
    name: str
    dependencies: tuple[str, ...] = ()
    # Transient stages have no reusable data artifact to materialize for downstream
    # consumers. When a downstream persistent stage must rerun, these stages replay in
    # memory even if their bookkeeping fingerprint is unchanged.
    transient_stage: bool = False

    def fingerprint(self, ctx: StageContext) -> str:
        upstream = {dep: ctx.store.upstream_fingerprint(dep) for dep in self.dependencies}
        return stable_hash({"stage": self.name, "config": self.config_fragment(ctx.config), "upstream": upstream})

    def config_fragment(self, cfg: Mapping[str, Any]) -> Any:
        return cfg.get(self.name, {})

    @abstractmethod
    def required_outputs(self, ctx: StageContext) -> list[Path]:
        raise NotImplementedError

    @abstractmethod
    def run(self, ctx: StageContext, overwrite: bool = False) -> dict[str, Any]:
        raise NotImplementedError

    def reusable(self, ctx: StageContext) -> bool:
        return ctx.store.is_reusable(self.name, self.fingerprint(ctx), self.required_outputs(ctx))
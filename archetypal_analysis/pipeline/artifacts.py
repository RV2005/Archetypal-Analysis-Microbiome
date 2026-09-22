"""Artifact layout, hashing, resumability, and stage bookkeeping helpers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


# Only stages that intentionally persist pipeline data/results get output directories.
# Load is transient and is tracked only by a success marker under _pipeline/.
STAGE_DIRS = {
    "filter": "01_filter",
    "qc": "02_qc",
    "fit": "03_fit",
    "evaluate": "04_evaluate",
    "select": "05_select",
    "benchmark": "06_benchmark",
    "prediction": "07_prediction",
    "stability": "08_stability",
    "analyze": "09_analyze",
    "plot": "10_plot",
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda x: str(x[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def stable_hash(value: Any) -> str:
    payload = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, suffix=".tmp") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)
        tmp = Path(fh.name)
    os.replace(tmp, path)


@dataclass
class ArtifactStore:
    root: Path

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def bookkeeping_dir(self) -> Path:
        path = self.root / "_pipeline"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def stage_dir(self, stage: str) -> Path:
        if stage not in STAGE_DIRS:
            raise KeyError(
                f"Stage '{stage}' does not persist a data/result directory. "
                f"Persistent stages are: {sorted(STAGE_DIRS)}"
            )
        path = self.root / STAGE_DIRS[stage]
        path.mkdir(parents=True, exist_ok=True)
        return path

    def marker_path(self, stage: str) -> Path:
        # Success markers are control-plane artifacts, separate from data outputs. This
        # lets load be fully transient while still supporting fingerprints, status,
        # timings, and dependency invalidation.
        return self.bookkeeping_dir / f"{stage}_SUCCESS.json"

    def stage_marker(self, stage: str) -> dict[str, Any] | None:
        path = self.marker_path(stage)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    def is_reusable(self, stage: str, fingerprint: str, required: Sequence[Path] = ()) -> bool:
        marker = self.stage_marker(stage)
        if not marker or marker.get("fingerprint") != fingerprint:
            return False
        required_paths = list(required)
        marker_paths = [Path(value) for value in marker.get("outputs", []) if value]
        # A stage is reusable only when both its contract-level required artifacts and
        # every output recorded by its last successful run still exist. This matters for
        # prediction/stability stages whose per-run artifacts are intentionally granular.
        return all(path.exists() for path in [*required_paths, *marker_paths])

    def mark_success(
        self,
        stage: str,
        fingerprint: str,
        outputs: Sequence[Path],
        details: Mapping[str, Any] | None = None,
        duration_seconds: float | None = None,
    ) -> None:
        payload = {
            "stage": stage,
            "fingerprint": fingerprint,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": None if duration_seconds is None else round(float(duration_seconds), 6),
            "outputs": [str(p) for p in outputs],
            "details": dict(details or {}),
        }
        atomic_write_json(self.marker_path(stage), payload)

    def record_stage_duration(self, stage: str, duration_seconds: float) -> None:
        """Attach elapsed wall-clock time to an already-written stage success marker."""
        marker = self.stage_marker(stage)
        if marker is None:
            return
        marker["duration_seconds"] = round(float(duration_seconds), 6)
        atomic_write_json(self.marker_path(stage), marker)

    def upstream_fingerprint(self, stage: str) -> str | None:
        marker = self.stage_marker(stage)
        return marker.get("fingerprint") if marker else None
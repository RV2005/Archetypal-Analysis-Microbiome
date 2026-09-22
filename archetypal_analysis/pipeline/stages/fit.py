"""Parallel archetypal-analysis fitting over configured K values."""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ...data.indexed_io import read_sample_indexed_tsv
from ...data.preprocessing import apply_preprocessing, resolve_preprocessing_mode
from ..artifacts import atomic_write_json, stable_hash
from .base import PipelineStage, StageContext


def _read_tsv(path: Path) -> pd.DataFrame:
    return read_sample_indexed_tsv(path)


def _prepare_fit_input(otu: pd.DataFrame, cfg: Mapping[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    prep = dict(cfg.get("preprocessing", {}))
    data = otu.copy()

    frac = float(prep.get("sample_fraction", 1.0))
    if not 0 < frac <= 1:
        raise ValueError("fit.preprocessing.sample_fraction must be in (0, 1]")
    if frac < 1:
        data = data.sample(frac=frac, random_state=prep.get("sample_random_state", cfg.get("random_state")))

    data, transform = apply_preprocessing(data, prep)
    transform.update({
        "sample_fraction": frac,
        "sample_random_state": prep.get("sample_random_state", cfg.get("random_state")),
        "standard_scale": bool(prep.get("standard_scale", False)),
    })

    data = data.replace([np.inf, -np.inf], np.nan)
    if prep.get("drop_nan_rows", True):
        data = data.dropna(axis=0, how="any")
    elif data.isna().any().any():
        mode = resolve_preprocessing_mode(prep)
        raise ValueError(
            f"NaN values remain after preprocessing mode '{mode}'; "
            "set drop_nan_rows=true or fix zero-sum/invalid input samples"
        )

    if prep.get("drop_zero_variance_features", True):
        std = data.std(axis=0)
        keep = std[(std > 0) & np.isfinite(std)].index
        data = data.loc[:, keep]

    if transform["standard_scale"]:
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        if (std == 0).any():
            raise ValueError("Zero-variance features remain before standard scaling")
        data = (data - mean) / std
        transform["mean"] = mean.tolist()
        transform["std"] = std.tolist()
    else:
        transform["mean"] = None
        transform["std"] = None

    transform["features"] = data.columns.tolist()
    transform["samples"] = data.index.tolist()
    return data, transform


def _serialize_result(result, k: int, feature_names: list[str], fit_cfg: Mapping[str, Any]) -> dict[str, Any]:
    loss = result.loss
    if isinstance(loss, np.ndarray):
        loss = loss.tolist()
    elif not isinstance(loss, list):
        try:
            loss = list(loss)
        except TypeError:
            loss = [float(loss)]
    info = result.info or {}
    return {
        "schema_version": 2,
        "k": int(k),
        "archetypes": np.asarray(result.archetypes).tolist(),
        "similarity_degree": result.similarity_degree.to_numpy().tolist(),
        "labels": np.asarray(result.labels).tolist(),
        "index": result.similarity_degree.index.tolist(),
        "columns": result.similarity_degree.columns.tolist(),
        "feature_names": feature_names,
        "loss": loss,
        "info_perms": info.get("perms", []),
        "info_scores": info.get("scores", []),
        "info_alphas": info.get("alphas", []),
        "fit_config": dict(fit_cfg),
    }


def _fit_one_k(k: int, processed_path: str, fit_cfg: dict[str, Any], k_dir: str, fingerprint: str, overwrite: bool) -> dict[str, Any]:
    """Worker entrypoint. Kept top-level so ProcessPoolExecutor can pickle it."""
    started = perf_counter()
    k_path = Path(k_dir)
    marker_path = k_path / "_SUCCESS.json"
    result_path = k_path / "result.json"
    similarity_path = k_path / "similarity.tsv"
    archetypes_path = k_path / "archetypes.tsv"

    if not overwrite and marker_path.exists() and result_path.exists() and similarity_path.exists() and archetypes_path.exists():
        try:
            marker = json.loads(marker_path.read_text())
            if marker.get("fingerprint") == fingerprint:
                return {"k": k, "status": "reused", "result_path": str(result_path)}
        except (OSError, json.JSONDecodeError):
            pass

    # Lazy import keeps config/selection tests usable without the heavy AA dependency installed.
    from ...analysis.archetypal import ArchetypalAnalysis

    data = _read_tsv(Path(processed_path))
    aa = ArchetypalAnalysis(
        n_archetypes=k,
        method=str(fit_cfg.get("method", "pgd")),
        max_iter=int(fit_cfg.get("max_iter", 100)),
        random_state=int(fit_cfg.get("random_state", 3407)),
        method_kwargs=dict(fit_cfg.get("method_kwargs", {})),
        model_kwargs=dict(fit_cfg.get("model_kwargs", {})),
    )
    result = aa.fit_transform(
        data,
        apply_log_transform=False,
        sample_fraction=1.0,
        std_scale=False,
        verbose=bool(fit_cfg.get("verbose_fit", False)),
    )

    k_path.mkdir(parents=True, exist_ok=True)
    payload = _serialize_result(result, k, data.columns.tolist(), fit_cfg)
    atomic_write_json(result_path, payload)
    result.similarity_degree.to_csv(similarity_path, sep="\t", index_label="sample")
    pd.DataFrame(
        np.asarray(result.archetypes),
        index=[f"Archetype {i + 1}" for i in range(k)],
        columns=data.columns,
    ).to_csv(archetypes_path, sep="\t", index_label="archetype")
    duration_seconds = perf_counter() - started
    atomic_write_json(marker_path, {
        "fingerprint": fingerprint,
        "k": k,
        "result": str(result_path),
        "duration_seconds": round(float(duration_seconds), 6),
    })
    return {
        "k": k,
        "status": "ran",
        "result_path": str(result_path),
        "duration_seconds": round(float(duration_seconds), 6),
    }


class FitStage(PipelineStage):
    name = "fit"
    dependencies = ("qc",)

    def required_outputs(self, ctx: StageContext) -> list[Path]:
        d = ctx.store.stage_dir(self.name)
        outputs = [d / "fit_input.tsv", d / "transform.json"]
        for k in ctx.config["fit"]["k_values"]:
            outputs.append(d / f"K_{int(k):03d}" / "result.json")
        return outputs

    def run(self, ctx: StageContext, overwrite: bool = False) -> dict[str, Any]:
        fp = self.fingerprint(ctx)
        outputs = self.required_outputs(ctx)
        if not overwrite and ctx.store.is_reusable(self.name, fp, outputs):
            return {"status": "reused", "outputs": outputs}

        qc_dir = ctx.store.stage_dir("qc")
        otu = _read_tsv(qc_dir / "otu.tsv")
        processed, transform = _prepare_fit_input(otu, ctx.config["fit"])
        if processed.empty or processed.shape[1] == 0:
            raise ValueError("AA preprocessing produced an empty matrix")

        d = ctx.store.stage_dir(self.name)
        processed_path = d / "fit_input.tsv"
        transform_path = d / "transform.json"
        processed.to_csv(processed_path, sep="\t", index_label="sample")
        atomic_write_json(transform_path, transform)

        k_values = sorted(int(k) for k in ctx.config["fit"]["k_values"])
        parallel = ctx.config["fit"].get("parallel", {}) or {}
        configured_jobs = int(parallel.get("n_jobs", 0) or 0)
        if configured_jobs <= 0:
            configured_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
        n_jobs = max(1, min(configured_jobs, len(k_values)))

        per_k_model_cfg = dict(ctx.config["fit"])
        per_k_model_cfg.pop("k_values", None)
        per_k_model_cfg.pop("parallel", None)
        per_k_model_cfg.pop("verbose_fit", None)
        per_k_base = stable_hash({
            "qc": ctx.store.upstream_fingerprint("qc"),
            "fit_model_config": per_k_model_cfg,
            "transform": transform,
        })
        tasks = []
        for k in k_values:
            k_fp = stable_hash({"fit_model": per_k_base, "k": k})
            tasks.append((k, str(processed_path), dict(ctx.config["fit"]), str(d / f"K_{k:03d}"), k_fp, overwrite))

        results: list[dict[str, Any]] = []
        backend = str(parallel.get("backend", "process")).lower()
        if n_jobs == 1 or backend == "sequential":
            for task in tasks:
                results.append(_fit_one_k(*task))
        else:
            with ProcessPoolExecutor(max_workers=n_jobs) as pool:
                future_map = {pool.submit(_fit_one_k, *task): task[0] for task in tasks}
                for future in as_completed(future_map):
                    k = future_map[future]
                    try:
                        results.append(future.result())
                    except Exception as exc:
                        raise RuntimeError(f"AA fit failed for K={k}") from exc

        details = {
            "k_values": k_values,
            "preprocessing_mode": transform["mode"],
            "n_jobs": n_jobs,
            "n_samples": int(processed.shape[0]),
            "n_features": int(processed.shape[1]),
            "per_k_status": sorted(results, key=lambda x: x["k"]),
        }
        ctx.store.mark_success(self.name, fp, outputs, details)
        return {"status": "ran", **details, "outputs": outputs}
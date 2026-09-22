"""Parallel, resumable dimensionality-reduction benchmark stage."""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from ...analysis.benchmarks import benchmark_tasks, run_benchmark_task
from ..artifacts import atomic_write_json, stable_hash
from .base import PipelineStage, StageContext


class BenchmarkStage(PipelineStage):
    name = "benchmark"
    dependencies = ("fit",)

    def config_fragment(self, cfg):
        return {
            "benchmark_schema": 2,
            "benchmarks": cfg.get("benchmarks", {}),
            "fit_k_values": cfg["fit"]["k_values"],
            "dimension_rule": "aa_k_minus_one",
        }

    def _tasks(self, ctx: StageContext):
        k_values = sorted(int(k) for k in ctx.config["fit"]["k_values"])
        return benchmark_tasks(k_values, ctx.config.get("benchmarks", {}))

    def required_outputs(self, ctx: StageContext) -> list[Path]:
        d = ctx.store.stage_dir(self.name)
        outputs = [d / "benchmark_manifest.json"]
        for task in self._tasks(ctx):
            task_dir = d / task.method / task.directory_name
            outputs.extend([task_dir / "embedding.tsv", task_dir / "benchmark.json"])
            if task.method == "pca":
                outputs.extend(
                    [task_dir / "components.tsv", task_dir / "mean.tsv", task_dir / "explained_variance.tsv"]
                )
            elif task.method == "pcoa":
                outputs.append(task_dir / "eigenvalues.tsv")
        return outputs

    def run(self, ctx: StageContext, overwrite: bool = False) -> dict[str, Any]:
        fp = self.fingerprint(ctx)
        outputs = self.required_outputs(ctx)
        if not overwrite and ctx.store.is_reusable(self.name, fp, outputs):
            return {"status": "reused", "outputs": outputs}

        cfg = dict(ctx.config.get("benchmarks", {}) or {})
        stage_dir = ctx.store.stage_dir(self.name)
        tasks = self._tasks(ctx)
        aa_k_values = sorted(int(k) for k in ctx.config["fit"]["k_values"])
        benchmark_dimensions = sorted({k - 1 for k in aa_k_values})
        max_k = max(aa_k_values)
        max_dimension = max(benchmark_dimensions)
        aa_k_to_dimension = {str(k): int(k - 1) for k in aa_k_values}
        fit_input_path = ctx.store.stage_dir("fit") / "fit_input.tsv"

        if not cfg.get("enabled", True):
            manifest = {
                "enabled": False,
                "max_k": max_k,
                "max_dimension": max_dimension,
                "aa_k_values": aa_k_values,
                "aa_k_to_dimension": aa_k_to_dimension,
                "dimensions": benchmark_dimensions,
                "methods": {},
                "tasks": [],
            }
            atomic_write_json(outputs[0], manifest)
            ctx.store.mark_success(self.name, fp, [outputs[0]], manifest)
            return {"status": "ran", **manifest, "outputs": [outputs[0]]}

        parallel = dict(cfg.get("parallel", {}) or {})
        configured_jobs = int(parallel.get("n_jobs", 0) or 0)
        if configured_jobs <= 0:
            configured_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
        n_jobs = max(1, min(configured_jobs, len(tasks))) if tasks else 1
        backend = str(parallel.get("backend", "process")).lower()

        base_fp = stable_hash(
            {
                "fit": ctx.store.upstream_fingerprint("fit"),
                "benchmark_config": cfg,
                "dimensions": benchmark_dimensions,
            }
        )
        task_args = []
        for task in tasks:
            method_cfg = dict(cfg.get(task.method, {}) or {})
            method_cfg.pop("enabled", None)
            task_fp = stable_hash(
                {
                    "benchmark": base_fp,
                    "method": task.method,
                    "dimension": task.dimension,
                    "method_config": method_cfg,
                }
            )
            task_args.append(
                (
                    task.method,
                    task.dimension,
                    str(fit_input_path),
                    method_cfg,
                    str(stage_dir / task.method / task.directory_name),
                    task_fp,
                    overwrite,
                )
            )

        results: list[dict[str, Any]] = []
        if n_jobs == 1 or backend == "sequential":
            for args in task_args:
                results.append(run_benchmark_task(*args))
        elif backend == "process":
            with ProcessPoolExecutor(max_workers=n_jobs) as pool:
                future_map = {
                    pool.submit(run_benchmark_task, *args): (args[0], args[1])
                    for args in task_args
                }
                for future in as_completed(future_map):
                    method, dimension = future_map[future]
                    try:
                        results.append(future.result())
                    except Exception as exc:
                        raise RuntimeError(
                            f"Benchmark failed for method={method}, dimension={dimension}"
                        ) from exc
        else:
            raise ValueError("benchmarks.parallel.backend must be 'process' or 'sequential'")

        sorted_results = sorted(results, key=lambda row: (row["method"], int(row["dimension"])))
        methods: dict[str, Any] = {}
        for method in ("pca", "pcoa", "umap", "tsne"):
            method_rows = [row for row in sorted_results if row["method"] == method]
            if not method_rows:
                continue
            fitted_dimensions = [int(row["dimension"]) for row in method_rows]
            methods[method] = {
                "fitted_dimensions": fitted_dimensions,
                "available_dimensions": (
                    benchmark_dimensions if method in {"pca", "pcoa"} else fitted_dimensions
                ),
                "fit_once_at_max_k": method in {"pca", "pcoa"},
                "fit_once_at_max_dimension": method in {"pca", "pcoa"},
                "artifacts": method_rows,
            }

        manifest = {
            "enabled": True,
            "max_k": max_k,
            "max_dimension": max_dimension,
            "aa_k_values": aa_k_values,
            "aa_k_to_dimension": aa_k_to_dimension,
            "dimensions": benchmark_dimensions,
            "n_jobs": n_jobs,
            "parallel_backend": backend,
            "methods": methods,
            "tasks": sorted_results,
        }
        manifest_path = stage_dir / "benchmark_manifest.json"
        atomic_write_json(manifest_path, manifest)
        ctx.store.mark_success(
            self.name,
            fp,
            outputs,
            {
                "max_k": max_k,
                "max_dimension": max_dimension,
                "dimension_rule": "aa_k_minus_one",
                "n_jobs": n_jobs,
                "parallel_backend": backend,
                "methods": {name: value["fitted_dimensions"] for name, value in methods.items()},
            },
        )
        return {
            "status": "ran",
            "max_k": max_k,
            "max_dimension": max_dimension,
            "n_jobs": n_jobs,
            "tasks": sorted_results,
            "outputs": outputs,
        }

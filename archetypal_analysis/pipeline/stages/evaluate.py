"""Evaluate each fitted K using metadata scores and reconstruction quality."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import davies_bouldin_score, silhouette_score

from ...analysis.selection import add_quality_score
from .base import PipelineStage, StageContext


def _discretize(labels: pd.Series, buckets: list[list[float]]) -> pd.Series:
    out = pd.Series(index=labels.index, dtype=object)
    numeric = pd.to_numeric(labels, errors="coerce")
    for lo, hi in buckets:
        if hi is None or (isinstance(hi, float) and np.isinf(hi)):
            mask = numeric >= lo
            name = f"{lo}+"
        else:
            mask = (numeric >= lo) & (numeric < hi)
            name = f"{lo}-{hi}"
        out.loc[mask] = name
    return out


def _safe_cluster_scores(points: pd.DataFrame, labels: pd.Series) -> tuple[float, float]:
    valid = labels.notna()
    x = points.loc[valid]
    y = labels.loc[valid].astype(str)
    n_classes = y.nunique()
    if len(x) < 3 or n_classes < 2 or n_classes >= len(x):
        return np.nan, np.nan
    try:
        silhouette = float(silhouette_score(x.to_numpy(), y.to_numpy()))
    except Exception:
        silhouette = np.nan
    try:
        db = float(davies_bouldin_score(x.to_numpy(), y.to_numpy()))
    except Exception:
        db = np.nan
    return silhouette, db


def _result(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


class EvaluateStage(PipelineStage):
    name = "evaluate"
    dependencies = ("fit",)

    def config_fragment(self, cfg):
        return {"evaluation": cfg["evaluation"], "task": cfg["task"], "fit_k_values": cfg["fit"]["k_values"]}

    def required_outputs(self, ctx: StageContext) -> list[Path]:
        d = ctx.store.stage_dir(self.name)
        return [d / "k_metrics.tsv", d / "k_metrics.json"]

    def run(self, ctx: StageContext, overwrite: bool = False) -> dict[str, Any]:
        fp = self.fingerprint(ctx)
        outputs = self.required_outputs(ctx)
        if not overwrite and ctx.store.is_reusable(self.name, fp, outputs):
            return {"status": "reused", "outputs": outputs}

        fit_dir = ctx.store.stage_dir("fit")
        fit_input = pd.read_csv(fit_dir / "fit_input.tsv", sep="\t", index_col="sample")
        metadata = pd.read_csv(ctx.store.stage_dir("filter") / "metadata.tsv", sep="\t", index_col="sample", low_memory=False)
        metadata.index = metadata.index.map(str)
        target = ctx.config["task"]["target"]
        eval_cfg = ctx.config["evaluation"]
        labels_all = metadata[target]
        if eval_cfg.get("discretize_target", False):
            buckets = eval_cfg.get("buckets")
            if not buckets:
                raise ValueError("evaluation.buckets is required when discretize_target=true")
            labels_all = _discretize(labels_all, buckets)

        rows = []
        for k in sorted(int(x) for x in ctx.config["fit"]["k_values"]):
            payload = _result(fit_dir / f"K_{k:03d}" / "result.json")
            similarity = pd.DataFrame(payload["similarity_degree"], index=pd.Index(payload["index"], name="sample"), columns=payload["columns"])
            common = similarity.index.intersection(labels_all.index, sort=False)
            if eval_cfg.get("drop_missing_target", True):
                common = common[labels_all.loc[common].notna()]
            silhouette, db = _safe_cluster_scores(similarity.loc[common], labels_all.loc[common])

            x = fit_input.loc[similarity.index].to_numpy(dtype=float)
            alpha = similarity.to_numpy(dtype=float)
            archetypes = np.asarray(payload["archetypes"], dtype=float)
            xhat = alpha @ archetypes
            residual = x - xhat
            mse = float(np.mean(residual ** 2))
            rmse = float(np.sqrt(mse))
            fro = float(np.linalg.norm(residual))
            denom = float(np.linalg.norm(x))
            relative = fro / denom if denom > 0 else np.nan
            loss = payload.get("loss", [])
            final_loss = float(loss[-1]) if loss else np.nan

            rows.append({
                "k": k,
                "n_samples_scored": int(len(common)),
                "silhouette": silhouette,
                "davies_bouldin": db,
                "reconstruction_mse": mse,
                "reconstruction_rmse": rmse,
                "frobenius_error": fro,
                "relative_frobenius_error": relative,
                "final_loss": final_loss,
            })

        metrics = add_quality_score(pd.DataFrame(rows).sort_values("k"))
        d = ctx.store.stage_dir(self.name)
        metrics.to_csv(d / "k_metrics.tsv", sep="\t", index=False)
        (d / "k_metrics.json").write_text(metrics.to_json(orient="records", indent=2))
        details = {"k_values": metrics["k"].astype(int).tolist()}
        ctx.store.mark_success(self.name, fp, outputs, details)
        return {"status": "ran", **details, "outputs": outputs}
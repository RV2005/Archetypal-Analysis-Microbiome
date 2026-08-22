#!/usr/bin/env python3
"""Validate archetypal-analysis taxa with RF importance, direction, overlap, and ablations."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

from archetypal_analysis.data.loaders import ArchetypalResultLoader
from archetypal_analysis.utils.metadata import extract_metadata
from archetypal_analysis.visualization.composition import CompositionAnalyzer

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def progress_iter(iterable, total: Optional[int], desc: str):
    if tqdm is not None:
        return tqdm(iterable, total=total, desc=desc, leave=False)
    return iterable


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("validate_taxa")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.handlers.clear()
    logger.addHandler(handler)
    return logger


def infer_delimiter(path: Path) -> str:
    with open(path, "r") as f:
        line = f.readline()
    if "\t" in line:
        return "\t"
    if "," in line:
        return ","
    return "\t"


def read_table(path: Path, sep_mode: str = "auto") -> pd.DataFrame:
    sep = infer_delimiter(path) if sep_mode == "auto" else ("," if sep_mode == "comma" else "\t")
    return pd.read_csv(path, sep=sep, index_col=0, low_memory=False)


def json_dump(data: dict, out: Path) -> None:
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
    with open(out, "w") as f:
        json.dump(data, f, indent=2, cls=NpEncoder)


def bucketize_with_edges(values: pd.Series, buckets: List[List[float]]) -> pd.Series:
    out = pd.Series(index=values.index, dtype="object")
    num = pd.to_numeric(values, errors="coerce")
    for lo, hi in buckets:
        mask = (num >= float(lo)) & (num < float(hi))
        out.loc[mask] = f"{lo}-{hi}"
    if len(buckets) > 0:
        lo, hi = buckets[-1]
        out.loc[num == float(hi)] = f"{lo}-{hi}"
    return out


def prepare_clinical_labels(
    y_raw: pd.Series,
    clinical_col: str,
    cfg: Dict,
    logger: logging.Logger,
) -> Tuple[pd.Series, Dict[str, str]]:
    transform = {"label_mode": "categorical_as_is"}
    y = y_raw.copy()

    remove_values = cfg.get("remove_values", {}) if isinstance(cfg, dict) else {}
    if clinical_col in remove_values:
        y = y.mask(y.astype(str) == str(remove_values[clinical_col]), np.nan)

    y_num = pd.to_numeric(y, errors="coerce")
    numeric_fraction = float(y_num.notna().mean()) if len(y_num) else 0.0

    if numeric_fraction > 0.9:
        if cfg.get("score_buckets"):
            y_binned = bucketize_with_edges(y, cfg["score_buckets"])
            transform["label_mode"] = "numeric_bucketed_from_config"
            transform["score_buckets"] = cfg["score_buckets"]
            y = y_binned
        else:
            non_na = y_num.dropna()
            n_unique = int(non_na.nunique())
            if n_unique > 20:
                q = min(10, max(3, int(np.sqrt(len(non_na)))))
                y = pd.qcut(non_na, q=q, duplicates="drop").astype(str).reindex(y.index)
                transform["label_mode"] = "numeric_quantile_binned"
                transform["n_bins"] = int(pd.Series(y.dropna()).nunique())

    y = y.astype("string").replace({"<NA>": np.nan, "nan": np.nan, "None": np.nan})
    vc = y.value_counts(dropna=True)
    if len(vc) > 0 and vc.min() < 2:
        rare = vc[vc < 2].index
        y = y.where(~y.isin(rare), np.nan)
        transform["dropped_singleton_labels"] = [str(v) for v in rare.tolist()]

    logger.info("Label preparation mode: %s", transform["label_mode"])
    return y, transform


def load_config(args: argparse.Namespace) -> Dict:
    cfg = {}
    if args.dataset:
        config_path = Path(args.dataset_config_file)
        with open(config_path, "r") as f:
            all_cfg = json.load(f)
        if args.dataset not in all_cfg:
            raise ValueError(f"Dataset '{args.dataset}' not found in {config_path}")
        cfg = all_cfg[args.dataset]
        base = config_path.parent
        for key in ("otu_file", "meta_file", "result_dir", "projects_file", "tags_file"):
            if key in cfg and cfg[key] is not None:
                p = Path(cfg[key])
                cfg[key] = str((base / p).resolve()) if not p.is_absolute() else str(p)
    return cfg


def resolve_arg(args: argparse.Namespace, cfg: Dict, arg_name: str, cfg_name: str) -> Optional[str]:
    val = getattr(args, arg_name)
    if val is not None:
        return val
    return cfg.get(cfg_name)


def align_samples(
    otu: pd.DataFrame,
    sample_meta: pd.DataFrame,
    projects: Optional[pd.DataFrame],
    tags: Optional[pd.DataFrame],
    clinical_col: str,
    metadata_dataset: str,
    metadata_sample_col: Optional[str],
    otu_orientation: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if otu_orientation == "taxa_rows":
        otu = otu.T
    elif otu_orientation == "auto":
        idx_meta = extract_metadata(
            index=pd.Index(otu.index.astype(str)),
            tags=tags,
            sample_meta=sample_meta,
            projects=projects,
            extract_cols=[clinical_col],
            dataset=metadata_dataset,
            verbose=False,
        )
        col_meta = extract_metadata(
            index=pd.Index(otu.columns.astype(str)),
            tags=tags,
            sample_meta=sample_meta,
            projects=projects,
            extract_cols=[clinical_col],
            dataset=metadata_dataset,
            verbose=False,
        )
        idx_hits = int(idx_meta[clinical_col].notna().sum()) if clinical_col in idx_meta.columns else 0
        col_hits = int(col_meta[clinical_col].notna().sum()) if clinical_col in col_meta.columns else 0
        if col_hits > idx_hits:
            otu = otu.T

    otu.index = otu.index.astype(str)
    sample_meta = sample_meta.copy()
    if metadata_sample_col:
        sample_meta.index = sample_meta[metadata_sample_col].astype(str)
    else:
        sample_meta.index = sample_meta.index.astype(str)

    aligned_meta = extract_metadata(
        index=pd.Index(otu.index),
        tags=tags,
        sample_meta=sample_meta,
        projects=projects,
        extract_cols=[clinical_col],
        dataset=metadata_dataset,
        verbose=False,
    )
    if clinical_col not in aligned_meta.columns:
        raise ValueError(f"Clinical column '{clinical_col}' was not extracted from metadata sources.")
    keep = aligned_meta[clinical_col].notna()
    if int(keep.sum()) < 3:
        raise ValueError("Too few matched samples between OTU and extracted metadata.")
    otu = otu.loc[keep.index[keep]]
    aligned_meta = aligned_meta.loc[keep.index[keep]]
    return otu, aligned_meta


def result_path(result_dir: str, k: int) -> Path:
    return Path(result_dir) / f"otu_AA_K{k}_1.0_None_jensenshannon.json"


def extract_aa_top_taxa(
    otu: pd.DataFrame,
    result_dir: str,
    main_k: int,
    k_values: List[int],
) -> Tuple[pd.DataFrame, Dict[int, List[str]]]:
    loader = ArchetypalResultLoader()
    _, _, aa_archetypes, _ = loader.load_parsed_result(str(result_path(result_dir, main_k)))
    comp = CompositionAnalyzer()
    arch = np.array(aa_archetypes)

    all_rows = []
    unions = {}
    for topk in k_values:
        top_taxa = comp.analyze_top_taxa(archetypes=arch, feature_names=list(otu.columns), num_taxa=topk)
        union = set()
        for i, arch_data in enumerate(top_taxa):
            for rank, (abundance, taxon) in enumerate(arch_data, start=1):
                if taxon == "Others":
                    continue
                union.add(taxon)
                all_rows.append(
                    {
                        "k": topk,
                        "archetype": f"Archetype {i+1}",
                        "rank": rank,
                        "taxon": taxon,
                        "abundance": float(abundance),
                        "is_union_member": True,
                    }
                )
        unions[topk] = sorted([t for t in union if t in otu.columns])
    return pd.DataFrame(all_rows), unions


def make_splitter(args: argparse.Namespace):
    if args.splitter == "sss":
        return StratifiedShuffleSplit(n_splits=args.n_resamples, test_size=args.test_size, random_state=args.random_seed)
    return RepeatedStratifiedKFold(n_splits=args.cv_folds, n_repeats=args.n_resamples, random_state=args.random_seed)


def compute_roc_auc(
    y_test: np.ndarray,
    y_proba: np.ndarray,
    n_classes: int,
    positive_idx: Optional[int],
) -> float:
    try:
        if n_classes == 2:
            idx = 1 if positive_idx is None else positive_idx
            return roc_auc_score(y_test, y_proba[:, idx])
        return roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
    except Exception:
        return float("nan")


def run_rf_importance(
    X_log1p: pd.DataFrame,
    y: pd.Series,
    args: argparse.Namespace,
    positive_idx: Optional[int],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    splitter = make_splitter(args)
    metrics_rows = []
    imp_rows = []
    split_rows = []
    n_classes = len(np.unique(y))

    split_pairs = list(splitter.split(X_log1p, y))
    for split_i, (train_idx, test_idx) in enumerate(
        progress_iter(split_pairs, total=len(split_pairs), desc="RF resamples")
    ):
        X_train, X_test = X_log1p.iloc[train_idx], X_log1p.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        rf = RandomForestClassifier(
            n_estimators=args.n_estimators,
            random_state=args.random_seed + split_i,
            class_weight="balanced",
            n_jobs=args.n_jobs,
        )
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        y_proba = rf.predict_proba(X_test)
        metrics_rows.append(
            {
                "split": split_i,
                "accuracy": accuracy_score(y_test, y_pred),
                "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
                "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
                "roc_auc": compute_roc_auc(y_test.values, y_proba, n_classes, positive_idx),
            }
        )

        split_rows.append(
            {"split": split_i, "train_indices": ",".join(map(str, train_idx)), "test_indices": ",".join(map(str, test_idx))}
        )

        perm = permutation_importance(
            rf,
            X_test,
            y_test,
            n_repeats=args.n_permutation_repeats,
            random_state=args.random_seed + split_i,
            n_jobs=args.n_jobs,
            scoring=args.rf_scoring_metric,
        )
        for j, taxon in enumerate(X_log1p.columns):
            imp_rows.append(
                {
                    "split": split_i,
                    "taxon": taxon,
                    "importance_mean": perm.importances_mean[j],
                    "importance_std": perm.importances_std[j],
                }
            )

    imp_long = pd.DataFrame(imp_rows)
    summary = (
        imp_long.groupby("taxon", as_index=False)
        .agg(
            mean_importance=("importance_mean", "mean"),
            median_importance=("importance_mean", "median"),
            std_importance=("importance_mean", "std"),
            mean_split_std=("importance_std", "mean"),
        )
    )
    sel = imp_long.groupby("taxon")["importance_mean"].apply(lambda s: (s > args.importance_threshold).mean())
    summary["selection_frequency"] = summary["taxon"].map(sel)
    summary = summary.sort_values("mean_importance", ascending=False).reset_index(drop=True)
    summary["rank"] = np.arange(1, len(summary) + 1)
    return pd.DataFrame(metrics_rows), imp_long, summary


def marker_call(delta: float, threshold: float, pos_name: str, neg_name: str) -> Optional[str]:
    if delta > threshold:
        return pos_name
    if delta < -threshold:
        return neg_name
    return None


def annotate_direction(
    X_log1p: pd.DataFrame,
    X_counts: pd.DataFrame,
    y_encoded: pd.Series,
    y_label: pd.Series,
    rf_summary: pd.DataFrame,
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    categories = sorted(y_encoded.unique())
    label_lookup = {enc: str(y_label[y_encoded == enc].iloc[0]) for enc in categories}
    rf_rows = list(rf_summary.iterrows())
    for _, rf_row in progress_iter(rf_rows, total=len(rf_rows), desc="Direction annotation"):
        taxon = rf_row["taxon"]
        for cat in categories:
            in_mask = y_encoded == cat
            out_mask = ~in_mask
            mean_in = float(X_log1p.loc[in_mask, taxon].mean())
            mean_out = float(X_log1p.loc[out_mask, taxon].mean())
            abundance_delta = mean_in - mean_out
            prev_in = float((X_counts.loc[in_mask, taxon] > 0).mean())
            prev_out = float((X_counts.loc[out_mask, taxon] > 0).mean())
            prevalence_delta = prev_in - prev_out
            m1 = marker_call(
                abundance_delta,
                args.direction_abundance_delta_threshold,
                "positive_abundance_marker",
                "negative_abundance_marker",
            )
            m2 = marker_call(
                prevalence_delta,
                args.direction_prevalence_delta_threshold,
                "positive_prevalence_marker",
                "negative_prevalence_marker",
            )
            marker_set = [m for m in [m1, m2] if m is not None]
            if not marker_set:
                marker_set = ["ambiguous_marker"]
            rows.append(
                {
                    "taxon": taxon,
                    "category_encoded": int(cat),
                    "category_label": label_lookup[cat],
                    "mean_importance": rf_row["mean_importance"],
                    "selection_frequency": rf_row["selection_frequency"],
                    "mean_log1p_in_category": mean_in,
                    "mean_log1p_out_category": mean_out,
                    "abundance_delta": abundance_delta,
                    "prevalence_in_category": prev_in,
                    "prevalence_out_category": prev_out,
                    "prevalence_delta": prevalence_delta,
                    "direction_markers": "|".join(marker_set),
                    "is_positive_marker": ("positive_abundance_marker" in marker_set) or ("positive_prevalence_marker" in marker_set),
                }
            )
    direction_df = pd.DataFrame(rows)
    positive = direction_df[direction_df["is_positive_marker"]].copy()
    return direction_df, positive


def overlap_positive_vs_aa(
    aa_unions: Dict[int, List[str]],
    positive_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    rf_ranked = (
        positive_df[["taxon", "mean_importance", "selection_frequency"]]
        .drop_duplicates(subset=["taxon"])
        .sort_values(["mean_importance", "selection_frequency"], ascending=False)
    )
    for k, aa_list in aa_unions.items():
        aa_set = set(aa_list)
        target_n = len(aa_set)
        rf_top = rf_ranked.head(target_n)["taxon"].tolist()
        rf_set = set(rf_top)
        overlap = sorted(aa_set & rf_set)
        union = aa_set | rf_set
        rows.append(
            {
                "k": k,
                "aa_n_features": len(aa_set),
                "rf_positive_n_features": len(rf_set),
                "n_common_features": len(overlap),
                "jaccard_index": (len(overlap) / len(union)) if union else np.nan,
                "overlapping_taxa": "|".join(overlap),
            }
        )
    return pd.DataFrame(rows)


def evaluate_feature_set(
    X: pd.DataFrame,
    y: pd.Series,
    split_pairs: List[Tuple[np.ndarray, np.ndarray]],
    args: argparse.Namespace,
    positive_idx: Optional[int],
) -> pd.DataFrame:
    rows = []
    n_classes = len(np.unique(y))
    for split_i, (train_idx, test_idx) in enumerate(split_pairs):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        rf = RandomForestClassifier(
            n_estimators=args.n_estimators,
            random_state=args.random_seed + split_i,
            class_weight="balanced",
            n_jobs=args.n_jobs,
        )
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        y_proba = rf.predict_proba(X_test)
        rows.append(
            {
                "split": split_i,
                "accuracy": accuracy_score(y_test, y_pred),
                "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
                "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
                "roc_auc": compute_roc_auc(y_test.values, y_proba, n_classes, positive_idx),
            }
        )
    return pd.DataFrame(rows)


def run_ablations(
    X_log1p: pd.DataFrame,
    y: pd.Series,
    aa_unions: Dict[int, List[str]],
    args: argparse.Namespace,
    positive_idx: Optional[int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    splitter = make_splitter(args)
    split_pairs = list(splitter.split(X_log1p, y))
    all_rows = []

    for k in progress_iter(sorted(aa_unions), total=len(aa_unions), desc="Ablation feature sets"):
        aa_features = [f for f in aa_unions[k] if f in X_log1p.columns]
        if aa_features:
            df = evaluate_feature_set(X_log1p[aa_features], y, split_pairs, args, positive_idx)
            df["feature_set"] = f"aa_top{k}"
            df["n_features"] = len(aa_features)
            all_rows.append(df)

        for split_i, (train_idx, test_idx) in enumerate(
            progress_iter(split_pairs, total=len(split_pairs), desc=f"random_top{k} resamples")
        ):
            rng = np.random.default_rng(args.random_seed + split_i + k)
            size = len(aa_features) if aa_features else min(k, X_log1p.shape[1])
            size = max(1, min(size, X_log1p.shape[1]))
            chosen = rng.choice(X_log1p.columns.values, size=size, replace=False).tolist()
            X_train = X_log1p.iloc[train_idx][chosen]
            X_test = X_log1p.iloc[test_idx][chosen]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
            rf = RandomForestClassifier(
                n_estimators=args.n_estimators,
                random_state=args.random_seed + split_i,
                class_weight="balanced",
                n_jobs=args.n_jobs,
            )
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            y_proba = rf.predict_proba(X_test)
            all_rows.append(
                pd.DataFrame(
                    [
                        {
                            "split": split_i,
                            "accuracy": accuracy_score(y_test, y_pred),
                            "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
                            "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
                            "roc_auc": compute_roc_auc(y_test.values, y_proba, len(np.unique(y)), positive_idx),
                            "feature_set": f"random_top{k}",
                            "n_features": len(chosen),
                        }
                    ]
                )
            )

    all_df = evaluate_feature_set(X_log1p, y, split_pairs, args, positive_idx)
    all_df["feature_set"] = "all_taxa"
    all_df["n_features"] = X_log1p.shape[1]
    all_rows.append(all_df)
    ablation_long = pd.concat(all_rows, ignore_index=True)
    ablation_summary = (
        ablation_long.groupby("feature_set", as_index=False)
        .agg(
            n_features=("n_features", "first"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            accuracy_median=("accuracy", "median"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            balanced_accuracy_median=("balanced_accuracy", "median"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            macro_f1_median=("macro_f1", "median"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            roc_auc_median=("roc_auc", "median"),
        )
    )
    return ablation_long, ablation_summary


def make_plots(
    out_plots: Path,
    rf_summary: pd.DataFrame,
    rf_metrics: pd.DataFrame,
    ablation_long: pd.DataFrame,
    overlap_df: pd.DataFrame,
    direction_df: pd.DataFrame,
) -> None:
    out_plots.mkdir(parents=True, exist_ok=True)
    stages = 4 + (3 if not ablation_long.empty else 0) + (1 if not overlap_df.empty else 0)
    p = tqdm(total=stages, desc="Plot generation", leave=False) if tqdm is not None else None

    top_rf = rf_summary.head(min(20, len(rf_summary))).copy()
    plt.figure(figsize=(10, 7))
    sns.barplot(data=top_rf, y="taxon", x="mean_importance", color="#3A86FF")
    plt.title("Top RF Taxa by Mean Permutation Importance")
    plt.tight_layout()
    plt.savefig(out_plots / "rf_top_importance.png", dpi=220)
    plt.close()
    if p is not None:
        p.update(1)

    mlong = rf_metrics.melt(value_vars=["accuracy", "balanced_accuracy", "macro_f1", "roc_auc"], var_name="metric", value_name="value")
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=mlong, x="metric", y="value")
    plt.title("RF Metrics Across Resamples")
    plt.tight_layout()
    plt.savefig(out_plots / "rf_metrics_boxplot.png", dpi=220)
    plt.close()
    if p is not None:
        p.update(1)

    if not ablation_long.empty:
        for metric in ["balanced_accuracy", "macro_f1", "roc_auc"]:
            plt.figure(figsize=(11, 5))
            sns.boxplot(data=ablation_long, x="feature_set", y=metric)
            plt.xticks(rotation=30, ha="right")
            plt.title(f"Ablation Comparison: {metric}")
            plt.tight_layout()
            plt.savefig(out_plots / f"ablation_{metric}_boxplot.png", dpi=220)
            plt.close()
            if p is not None:
                p.update(1)

    if not overlap_df.empty:
        plt.figure(figsize=(8, 5))
        sns.barplot(data=overlap_df.sort_values("k"), x="k", y="jaccard_index", color="#FF8C42")
        plt.title("AA vs RF Positive Overlap (Jaccard)")
        plt.tight_layout()
        plt.savefig(out_plots / "overlap_vs_k.png", dpi=220)
        plt.close()
        if p is not None:
            p.update(1)

    top_pos = direction_df[direction_df["is_positive_marker"]].sort_values("mean_importance", ascending=False).head(20)
    if not top_pos.empty:
        heat = top_pos.pivot_table(index="taxon", columns="category_label", values="abundance_delta", aggfunc="mean").fillna(0.0)
        plt.figure(figsize=(10, max(4, len(heat) * 0.25)))
        sns.heatmap(heat, cmap="coolwarm", center=0)
        plt.title("Abundance Delta Heatmap (Top RF-positive taxa)")
        plt.tight_layout()
        plt.savefig(out_plots / "abundance_delta_heatmap.png", dpi=220)
        plt.close()
    if p is not None:
        p.update(1)
        p.close()


def write_report(out_dir: Path, overlap_df: pd.DataFrame, ablation_summary: pd.DataFrame) -> None:
    report_path = out_dir / "report.md"
    with open(report_path, "w") as f:
        f.write("# Validation Report\n\n")
        f.write("## Overlap Metrics\n\n")
        f.write(
            "Jaccard overlap between AA top-k union taxa and RF-important taxa subsetted to positive markers.\n\n"
        )
        overlap_cols = ["k", "aa_n_features", "rf_positive_n_features", "n_common_features", "jaccard_index"]
        if overlap_df.empty:
            f.write("_No overlap results generated._\n\n")
        else:
            f.write(overlap_df[overlap_cols].sort_values("k").to_markdown(index=False))
            f.write("\n\n")

        f.write("## Ablation Metrics\n\n")
        if ablation_summary.empty:
            f.write("_Ablation not run or no ablation results available._\n")
        else:
            show_cols = [
                "feature_set",
                "n_features",
                "accuracy_mean",
                "balanced_accuracy_mean",
                "macro_f1_mean",
                "roc_auc_mean",
                "accuracy_std",
                "balanced_accuracy_std",
                "macro_f1_std",
                "roc_auc_std",
            ]
            f.write(ablation_summary[show_cols].to_markdown(index=False))
            f.write("\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AA-vs-RF taxa validation")
    p.add_argument("--dataset", default=None)
    p.add_argument("--dataset_config_file", default="dataset_configs.json")
    p.add_argument("--experiment_name", default=None)
    p.add_argument("--output_dir", default="./results_validate_taxa")

    p.add_argument("--otu_file", default=None)
    p.add_argument("--metadata_file", default=None)
    p.add_argument("--result_dir", default=None)
    p.add_argument("--main_k", type=int, default=None)
    p.add_argument("--clinical_col", default=None)
    p.add_argument("--projects_file", default=None)
    p.add_argument("--tags_file", default=None)
    p.add_argument("--metadata_dataset", choices=["compendium", "hadza"], default=None)

    p.add_argument("--splitter", choices=["sss", "rskf"], default="sss")
    p.add_argument("--n_resamples", type=int, default=30)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--cv_folds", type=int, default=5)
    p.add_argument("--n_estimators", type=int, default=1000)
    p.add_argument("--n_jobs", type=int, default=-1)
    p.add_argument("--random_seed", type=int, default=123)
    p.add_argument("--rf_scoring_metric", default="balanced_accuracy")

    p.add_argument("--n_permutation_repeats", type=int, default=10)
    p.add_argument("--importance_threshold", type=float, default=0.0)
    p.add_argument("--direction_abundance_delta_threshold", type=float, default=0.0)
    p.add_argument("--direction_prevalence_delta_threshold", type=float, default=0.0)
    p.add_argument("--positive_marker_rule", choices=["abundance_or_prevalence", "abundance_and_prevalence"], default="abundance_or_prevalence")
    p.add_argument("--class_positive_label", default=None)
    p.add_argument("--aa_k_values", default="5,10,15")
    p.add_argument("--run_ablation", action="store_true")

    p.add_argument("--sep", choices=["auto", "comma", "tab"], default="auto")
    p.add_argument("--metadata_sample_col", default=None)
    p.add_argument("--otu_orientation", choices=["auto", "taxa_rows", "samples_rows"], default="auto")
    p.add_argument("--force_rerun", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logger = setup_logger()
    start_time = time.time()
    cfg = load_config(args)

    otu_file = resolve_arg(args, cfg, "otu_file", "otu_file")
    metadata_file = resolve_arg(args, cfg, "metadata_file", "meta_file")
    result_dir = resolve_arg(args, cfg, "result_dir", "result_dir")
    projects_file = resolve_arg(args, cfg, "projects_file", "projects_file")
    tags_file = resolve_arg(args, cfg, "tags_file", "tags_file")
    metadata_dataset = args.metadata_dataset if args.metadata_dataset is not None else cfg.get("metadata_dataset", "compendium")
    main_k = args.main_k if args.main_k is not None else cfg.get("main_k")
    clinical_col = args.clinical_col if args.clinical_col is not None else cfg.get("metadata_col")
    if not all([otu_file, metadata_file, result_dir, main_k, clinical_col]):
        raise ValueError("Missing required inputs. Provide via --dataset config or explicit flags.")

    exp_name = args.experiment_name or args.dataset or "custom_experiment"
    out_dir = Path(args.output_dir) / exp_name
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[1/7] Loading OTU and metadata")
    otu = read_table(Path(otu_file), args.sep)
    meta = read_table(Path(metadata_file), args.sep)
    projects = read_table(Path(projects_file), args.sep) if projects_file else None
    tags = read_table(Path(tags_file), args.sep) if tags_file else None
    otu, meta = align_samples(
        otu=otu,
        sample_meta=meta,
        projects=projects,
        tags=tags,
        clinical_col=clinical_col,
        metadata_dataset=metadata_dataset,
        metadata_sample_col=args.metadata_sample_col,
        otu_orientation=args.otu_orientation,
    )
    otu = otu.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    y_raw = meta[clinical_col]
    y_prepared, label_transform = prepare_clinical_labels(y_raw, clinical_col, cfg, logger)
    keep = y_prepared.notna()
    otu = otu.loc[keep]
    y_raw = y_prepared.loc[keep].astype(str)
    if y_raw.nunique() < 2:
        raise ValueError("After label preparation, fewer than 2 classes remain for classification.")

    encoder = LabelEncoder()
    y_encoded = pd.Series(encoder.fit_transform(y_raw), index=y_raw.index, name="label_encoded")
    mapping = pd.DataFrame({"label_raw": encoder.classes_, "label_encoded": np.arange(len(encoder.classes_))})
    mapping.to_csv(out_dir / "label_encoder_mapping.tsv", sep="\t", index=False)

    positive_idx = None
    if args.class_positive_label is not None and len(encoder.classes_) == 2:
        matches = np.where(encoder.classes_ == args.class_positive_label)[0]
        if len(matches) == 1:
            positive_idx = int(matches[0])
        else:
            logger.warning("class_positive_label not found; defaulting to encoded label 1.")

    X_counts = otu.copy()
    X_log1p = np.log1p(X_counts)
    k_values = [int(x.strip()) for x in args.aa_k_values.split(",") if x.strip()]

    logger.info("[2/7] Extracting AA top taxa from archetypal results")
    aa_top_df, aa_unions = extract_aa_top_taxa(X_counts, result_dir, int(main_k), k_values)
    aa_top_df.to_csv(out_dir / "aa_top_taxa.tsv", sep="\t", index=False)

    logger.info("[3/7] Running RF importance and metrics")
    rf_metrics, rf_imp_long, rf_summary = run_rf_importance(X_log1p, y_encoded, args, positive_idx)
    rf_metrics.to_csv(out_dir / "rf_resample_metrics.tsv", sep="\t", index=False)
    rf_imp_long.to_csv(out_dir / "rf_feature_importances_long.tsv", sep="\t", index=False)
    rf_summary.to_csv(out_dir / "rf_feature_importances.tsv", sep="\t", index=False)

    logger.info("[4/7] Annotating RF feature direction")
    direction_df, positive_df = annotate_direction(X_log1p, X_counts, y_encoded, y_raw, rf_summary, args)
    if args.positive_marker_rule == "abundance_and_prevalence":
        positive_df = direction_df[
            direction_df["direction_markers"].str.contains("positive_abundance_marker")
            & direction_df["direction_markers"].str.contains("positive_prevalence_marker")
        ].copy()
    direction_df.to_csv(out_dir / "rf_direction_annotations.tsv", sep="\t", index=False)
    positive_df.to_csv(out_dir / "rf_positive_markers.tsv", sep="\t", index=False)

    overlap_df = overlap_positive_vs_aa(aa_unions, positive_df)
    overlap_df.to_csv(out_dir / "aa_rf_positive_overlap.tsv", sep="\t", index=False)

    ablation_long = pd.DataFrame()
    ablation_summary = pd.DataFrame()
    if args.run_ablation:
        logger.info("[5/7] Running ablation study")
        ablation_long, ablation_summary = run_ablations(X_log1p, y_encoded, aa_unions, args, positive_idx)
        ablation_long.to_csv(out_dir / "rf_ablations.tsv", sep="\t", index=False)
        ablation_summary.to_csv(out_dir / "rf_ablations_summary.tsv", sep="\t", index=False)

    logger.info("[6/7] Saving splits and run config")
    split_pairs = list(make_splitter(args).split(X_log1p, y_encoded))
    splits_df = pd.DataFrame(
        [{"split": i, "train_indices": ",".join(map(str, tr)), "test_indices": ",".join(map(str, te))}
         for i, (tr, te) in enumerate(split_pairs)]
    )
    splits_df.to_csv(out_dir / "splits.tsv", sep="\t", index=False)

    resolved = vars(args).copy()
    resolved.update(
        {
            "resolved_otu_file": str(Path(otu_file).resolve()),
            "resolved_metadata_file": str(Path(metadata_file).resolve()),
            "resolved_result_dir": str(Path(result_dir).resolve()),
            "resolved_main_k": int(main_k),
            "resolved_clinical_col": clinical_col,
            "label_transform": label_transform,
            "n_samples": int(X_counts.shape[0]),
            "n_taxa": int(X_counts.shape[1]),
        }
    )
    json_dump(resolved, out_dir / "run_config.json")

    logger.info("[7/7] Generating plots and markdown report")
    make_plots(plots_dir, rf_summary, rf_metrics, ablation_long, overlap_df, direction_df)
    write_report(out_dir, overlap_df, ablation_summary)
    elapsed = time.time() - start_time
    logger.info("Done in %.1fs. Output: %s", elapsed, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

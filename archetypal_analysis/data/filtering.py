"""Metadata filtering/balancing using the semantics of the legacy DataNormalizer."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .dataset import DatasetBundle


class DatasetFilter:
    """Apply task exclusions, control filters, then optional balanced subsampling.

    ``control``, ``normalize``, ``continuous``, ``ignore_values``, and ``tolerance``
    intentionally mirror the old ``DataNormalizer.normalize_data`` arguments. The only
    structural change is that metadata is already a single aligned table.
    """

    def apply(self, bundle: DatasetBundle, cfg: Mapping[str, Any], task_cfg: Mapping[str, Any]) -> DatasetBundle:
        otu = bundle.otu
        metadata = bundle.metadata.copy()
        target = task_cfg["target"]

        if target not in metadata.columns:
            raise KeyError(f"Target metadata column '{target}' not found")

        if cfg.get("drop_missing_target", True):
            metadata = metadata.loc[metadata[target].notna()]

        exclude = cfg.get("exclude_target_values", []) or []
        if exclude:
            metadata = metadata.loc[~metadata[target].isin(exclude)]

        # Legacy control behavior: None means all non-null values; list means membership.
        control = cfg.get("control", {}) or {}
        for label, values in control.items():
            if label not in metadata.columns:
                raise KeyError(f"Filter control column '{label}' not found in metadata")
            if values is None:
                metadata = metadata.loc[metadata[label].notna()]
            else:
                metadata = metadata.loc[metadata[label].isin(values)]

        normalize = cfg.get("normalize", {}) or {}
        if normalize:
            metadata = self._balanced_sample(
                metadata=metadata,
                normalize=normalize,
                continuous=bool(cfg.get("continuous", False)),
                ignore_zero=bool(cfg.get("ignore_zero", True)),
                ignore_values=cfg.get("ignore_values", []) or [],
                tolerance=float(cfg.get("tolerance", 0.0)),
                random_state=cfg.get("random_state"),
            )

        # Preserve metadata selection order and align OTU to it, matching legacy output.
        metadata = metadata.loc[~metadata.index.duplicated(keep="first")]
        common = metadata.index[metadata.index.isin(otu.index)]
        if len(common) == 0:
            raise ValueError("Filtering removed all samples")
        return DatasetBundle(otu.loc[common].copy(), metadata.loc[common].copy())

    @staticmethod
    def _group(metadata: pd.DataFrame, label: str, value: Any, continuous: bool) -> pd.DataFrame:
        if continuous:
            if not isinstance(value, Sequence) or isinstance(value, str) or len(value) != 2:
                raise ValueError(f"Continuous normalize value for {label} must be [min, max]")
            lo, hi = value
            return metadata.loc[(metadata[label] >= lo) & (metadata[label] <= hi)]
        return metadata.loc[metadata[label] == value]

    def _balanced_sample(
        self,
        metadata: pd.DataFrame,
        normalize: Mapping[str, Any],
        continuous: bool,
        ignore_zero: bool,
        ignore_values: Sequence[Any],
        tolerance: float,
        random_state: int | None,
    ) -> pd.DataFrame:
        min_counts = np.inf
        found = False

        for label, values in normalize.items():
            if label not in metadata.columns:
                raise KeyError(f"Normalize column '{label}' not found in metadata")
            if continuous and values is None:
                raise ValueError("Continuous normalization requires explicit [min, max] buckets")
            values = metadata[label].dropna().unique().tolist() if values is None else values
            for value in values:
                group = self._group(metadata, label, value, continuous)
                if value in ignore_values:
                    continue
                if ignore_zero and len(group) == 0:
                    continue
                min_counts = min(min_counts, len(group))
                found = True

        if not found or not np.isfinite(min_counts):
            raise ValueError("No eligible groups found for normalization")

        sampled = []
        rng = np.random.RandomState(random_state) if random_state is not None else None
        for label, values in normalize.items():
            values = metadata[label].dropna().unique().tolist() if values is None else values
            for value in values:
                group = self._group(metadata, label, value, continuous)
                if len(group) == 0 or value in ignore_values:
                    continue
                n = int(min(len(group), int(min_counts) * (1.0 + tolerance)))
                if n <= 0:
                    continue
                seed = int(rng.randint(0, 2**31 - 1)) if rng is not None else None
                sampled.append(group.sample(n=n, random_state=seed))

        if not sampled:
            raise ValueError("Normalization produced no samples")
        return pd.concat(sampled, axis=0)
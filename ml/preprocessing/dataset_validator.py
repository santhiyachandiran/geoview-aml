"""
GeoChange AI — Dataset Validator

Validates a saved pixel dataset produced by gee_extractor.extract_training_data().
Can be used standalone or imported by tests and prepare_dataset.py.

This module does NOT require GEE access.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ml.preprocessing.feature_builder import FEATURE_NAMES, N_FEATURES
from ml.preprocessing.gee_extractor import CLASS_NAMES, _VALID_LABELS


REQUIRED_FILES: list[str] = ["X.npy", "y.npy", "rows.npy", "cols.npy", "metadata.json"]

REQUIRED_METADATA_KEYS: list[str] = [
    "bbox",
    "composite_year",
    "date_start",
    "date_end_exclusive",
    "date_end_inclusive_desc",
    "date_convention",
    "scene_count",
    "n_sampled",
    "n_removed_nan_bands",
    "n_removed_missing_labels",
    "n_pixels",
    "n_features",
    "feature_names",
    "class_names",
    "class_distribution",
    "scale_m",
    "max_cloud_pct",
    "labels_source",
    "missing_label_handling",
    "temporal_caveat",
]


def validate_dataset(data_dir: str | Path) -> dict[str, Any]:
    """
    Load and validate a saved pixel dataset.  Does NOT contact GEE.

    Checks:
    - All required files are present
    - X is (N, 9) float32 with no NaN/Inf
    - y is (N,) int32 with values only in {0, 1, 2, 3}
    - rows and cols are (N,) integer arrays
    - metadata contains all required keys
    - metadata.n_pixels matches actual len(X)
    - metadata.feature_names matches FEATURE_NAMES
    - metadata.class_names matches CLASS_NAMES
    - No class has zero samples
    - Class imbalance ratio warning at > 20:1

    Args:
        data_dir: Path to the directory containing the .npy and metadata files.

    Returns:
        dict with:
            valid (bool)
            n_pixels (int)
            n_features (int)
            dtype_X (str)
            dtype_y (str)
            nan_rows_in_X (int)
            unique_labels (list[int])
            class_distribution (dict)
            date_start (str)
            date_end_exclusive (str)
            date_end_inclusive_desc (str)
            scene_count (int | None)
            n_removed_nan_bands (int | None)
            n_removed_missing_labels (int | None)
            labels_source (str | None)
            missing_metadata_keys (list[str])
            issues (list[str])  — empty when valid
    """
    data_dir = Path(data_dir)
    issues: list[str] = []

    # ── 1. Required files ───────────────────────────────────────────────
    for fname in REQUIRED_FILES:
        if not (data_dir / fname).exists():
            issues.append(f"Missing file: {fname}")

    if issues:
        # Can't load anything; stop early
        return {
            "valid": False,
            "n_pixels": None,
            "n_features": None,
            "dtype_X": None,
            "dtype_y": None,
            "nan_rows_in_X": None,
            "unique_labels": [],
            "class_distribution": {},
            "date_start": None,
            "date_end_exclusive": None,
            "date_end_inclusive_desc": None,
            "scene_count": None,
            "n_removed_nan_bands": None,
            "n_removed_missing_labels": None,
            "labels_source": None,
            "missing_metadata_keys": [],
            "issues": issues,
        }

    # ── 2. Load arrays ──────────────────────────────────────────────────
    X    = np.load(data_dir / "X.npy")
    y    = np.load(data_dir / "y.npy")
    rows = np.load(data_dir / "rows.npy")
    cols = np.load(data_dir / "cols.npy")
    meta = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))

    n = len(X)

    # ── 3. Shape and dtype ──────────────────────────────────────────────
    if X.ndim != 2:
        issues.append(f"X.ndim={X.ndim}; expected 2")
    elif X.shape[1] != N_FEATURES:
        issues.append(
            f"X has {X.shape[1]} feature columns; expected {N_FEATURES} {FEATURE_NAMES}"
        )

    if X.dtype != np.float32:
        issues.append(f"X.dtype={X.dtype}; expected float32")

    if y.ndim != 1:
        issues.append(f"y.ndim={y.ndim}; expected 1")
    elif len(y) != n:
        issues.append(f"len(y)={len(y)} != len(X)={n}")

    if y.dtype != np.int32:
        issues.append(f"y.dtype={y.dtype}; expected int32")

    if rows.ndim != 1 or len(rows) != n:
        issues.append(
            f"rows shape {rows.shape} should be ({n},)"
        )
    if cols.ndim != 1 or len(cols) != n:
        issues.append(
            f"cols shape {cols.shape} should be ({n},)"
        )

    # ── 4. Finite values ────────────────────────────────────────────────
    nan_rows = 0
    if X.ndim == 2:
        nan_rows = int((~np.isfinite(X)).any(axis=1).sum())
        if nan_rows > 0:
            issues.append(
                f"X contains {nan_rows} rows with NaN or Inf values"
            )

    # ── 5. Label validity ────────────────────────────────────────────────
    unique_labels: list[int] = []
    if y.ndim == 1:
        unique_labels = sorted(set(np.unique(y).tolist()))
        invalid = set(unique_labels) - _VALID_LABELS
        if invalid:
            issues.append(
                f"y contains invalid label values {invalid}; "
                f"valid set is {_VALID_LABELS}"
            )

    warnings_list: list[str] = []

    # ── 6. Class distribution ────────────────────────────────────────────
    class_dist: dict[str, int] = {}
    if y.ndim == 1:
        for c in sorted(_VALID_LABELS):
            count = int((y == c).sum())
            class_dist[CLASS_NAMES[c]] = count

        zero_classes = [k for k, v in class_dist.items() if v == 0]
        if zero_classes:
            issues.append(
                f"Classes with 0 samples: {zero_classes}. "
                "The model cannot learn these classes."
            )

        nonzero = [v for v in class_dist.values() if v > 0]
        if len(nonzero) >= 2:
            ratio = max(nonzero) / min(nonzero)
            if ratio > 20:
                warnings_list.append(
                    f"Severe class imbalance: ratio {ratio:.1f}:1 "
                    f"(max={max(nonzero)}, min={min(nonzero)}). "
                    "Consider class_weight='balanced' in the model."
                )

    # ── 7. Metadata completeness ─────────────────────────────────────────
    missing_keys = [k for k in REQUIRED_METADATA_KEYS if k not in meta]
    if missing_keys:
        issues.append(f"metadata.json is missing keys: {missing_keys}")

    if meta.get("n_pixels") is not None and meta["n_pixels"] != n:
        issues.append(
            f"metadata.n_pixels={meta['n_pixels']} != actual len(X)={n}"
        )

    if meta.get("feature_names") and meta["feature_names"] != FEATURE_NAMES:
        issues.append(
            f"metadata.feature_names {meta['feature_names']} "
            f"!= expected {FEATURE_NAMES}"
        )

    if meta.get("class_names") and meta["class_names"] != CLASS_NAMES:
        issues.append(
            f"metadata.class_names {meta['class_names']} "
            f"!= expected {CLASS_NAMES}"
        )

    return {
        "valid": len(issues) == 0,
        "n_pixels": n,
        "n_features": X.shape[1] if X.ndim == 2 else None,
        "dtype_X": str(X.dtype),
        "dtype_y": str(y.dtype),
        "nan_rows_in_X": nan_rows,
        "unique_labels": unique_labels,
        "class_distribution": class_dist,
        "date_start": meta.get("date_start"),
        "date_end_exclusive": meta.get("date_end_exclusive"),
        "date_end_inclusive_desc": meta.get("date_end_inclusive_desc"),
        "scene_count": meta.get("scene_count"),
        "n_removed_nan_bands": meta.get("n_removed_nan_bands"),
        "n_removed_missing_labels": meta.get("n_removed_missing_labels"),
        "labels_source": meta.get("labels_source"),
        "missing_metadata_keys": missing_keys,
        "issues": issues,
        "warnings": warnings_list,
    }

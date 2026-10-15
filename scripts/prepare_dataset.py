"""
GEOVIEW AML — Dataset Preparation (extraction + validation only)

Runs only the GEE extraction and dataset-validation steps.
Does NOT run model training, Optuna, or Docker.

Use this to generate and inspect the real dataset before committing to a
full pipeline run.  It is safe to run and interrupt at any time; the saved
.npy files can be reloaded by run_pipeline.py on the next run.

Usage (from project root, venv activated, GEE authenticated):
    python scripts/prepare_dataset.py [options]

Options:
    --year   INT   WorldCover label year and Sentinel-2 composite year.
                   Must be 2021 (WorldCover v200 is 2021 only). Default: 2021.
    --bbox   W S E N  Bounding box as four floats. Default: Coimbatore.
    --cloud  INT   Max CLOUDY_PIXEL_PERCENTAGE per scene. Default: 20.
    --pixels INT   Maximum pixels to sample from GEE.  Default: 100000.
                   Use a small number (e.g. 10000) for a quick test run.
    --scale  INT   Pixel size in metres. Default: 10.
    --out    PATH  Output directory.
                   Default: ml/data/processed/coimbatore_<year>/
    --validate-only  Do not extract; only validate an existing dataset at --out.

Examples
--------
# Quick test extraction (~10 k pixels, verifies GEE access):
    python scripts/prepare_dataset.py --pixels 10000

# Full extraction (~500 k pixels, takes 5–20 minutes over GEE):
    python scripts/prepare_dataset.py --pixels 500000

# Validate previously saved data without re-extracting:
    python scripts/prepare_dataset.py --validate-only

Expected output paths (all under the project root)
---------------------------------------------------
    ml/data/processed/coimbatore_2021/X.npy          feature matrix  (N, 9) float32
    ml/data/processed/coimbatore_2021/y.npy           label vector   (N,) int32
    ml/data/processed/coimbatore_2021/rows.npy         pixel row coords
    ml/data/processed/coimbatore_2021/cols.npy         pixel col coords
    ml/data/processed/coimbatore_2021/metadata.json    provenance record

Approximate storage
-------------------
Each pixel occupies 9 × 4 = 36 bytes for X plus 4 bytes for y.
    10 000 pixels  ≈   0.4 MB
   100 000 pixels  ≈   4 MB
   500 000 pixels  ≈  20 MB
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import numpy as np


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GEOVIEW AML — dataset preparation (extraction + validation)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--year",   type=int,   default=2021,
                   help="Composite/label year (default 2021)")
    p.add_argument("--bbox",   type=float, nargs=4,
                   metavar=("W", "S", "E", "N"),
                   default=[76.8, 10.9, 77.2, 11.1],
                   help="Bounding box [west south east north]")
    p.add_argument("--cloud",  type=int,   default=20,
                   help="Max CLOUDY_PIXEL_PERCENTAGE (default 20)")
    p.add_argument("--pixels", type=int,   default=100_000,
                   help="Max pixels to sample (default 100000)")
    p.add_argument("--scale",  type=int,   default=10,
                   help="Pixel resolution in metres (default 10)")
    p.add_argument("--out",    type=str,   default=None,
                   help="Output directory (default: ml/data/processed/coimbatore_<year>/)")
    p.add_argument("--validate-only", action="store_true",
                   help="Only validate an existing dataset; do not extract")
    return p.parse_args()


# ── Validation ─────────────────────────────────────────────────────────────

def validate_dataset(data_dir: Path) -> dict:
    """
    Load and validate a saved dataset.  Reports issues and returns a
    summary dict.  Does NOT require GEE access.

    Returns:
        dict with keys: valid (bool), n_pixels, n_features, class_distribution,
        missing_values, label_issues, issues (list of str).
    """
    from ml.preprocessing.feature_builder import FEATURE_NAMES, N_FEATURES
    from ml.preprocessing.gee_extractor import CLASS_NAMES, _VALID_LABELS

    issues: list[str] = []

    # Required files
    required = ["X.npy", "y.npy", "rows.npy", "cols.npy", "metadata.json"]
    for f in required:
        if not (data_dir / f).exists():
            issues.append(f"Missing file: {f}")

    if issues:
        return {"valid": False, "issues": issues}

    X    = np.load(data_dir / "X.npy")
    y    = np.load(data_dir / "y.npy")
    rows = np.load(data_dir / "rows.npy")
    cols = np.load(data_dir / "cols.npy")
    meta = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))

    # ── Shape checks ──────────────────────────────────────────
    if X.ndim != 2:
        issues.append(f"X must be 2D, got ndim={X.ndim}")
    elif X.shape[1] != N_FEATURES:
        issues.append(f"X has {X.shape[1]} features; expected {N_FEATURES}")

    if y.ndim != 1:
        issues.append(f"y must be 1D, got ndim={y.ndim}")
    elif len(y) != len(X):
        issues.append(f"len(y)={len(y)} != len(X)={len(X)}")

    if len(rows) != len(X):
        issues.append(f"len(rows)={len(rows)} != len(X)={len(X)}")
    if len(cols) != len(X):
        issues.append(f"len(cols)={len(cols)} != len(X)={len(X)}")

    # ── Finite values ──────────────────────────────────────────
    nan_count = int((~np.isfinite(X)).any(axis=1).sum()) if X.ndim == 2 else -1
    if nan_count > 0:
        issues.append(f"X contains {nan_count} rows with NaN/Inf values")

    # ── Label validity ─────────────────────────────────────────
    unique_labels = set(np.unique(y).tolist())
    invalid_labels = unique_labels - _VALID_LABELS
    if invalid_labels:
        issues.append(
            f"y contains invalid label values: {invalid_labels} "
            f"(valid: {_VALID_LABELS})"
        )

    # ── Class distribution ─────────────────────────────────────
    counts = {CLASS_NAMES[int(c)]: int((y == c).sum())
              for c in sorted(unique_labels)
              if int(c) < len(CLASS_NAMES)}

    # Warn on severely imbalanced classes (ratio > 20:1)
    if counts:
        max_c = max(counts.values())
        min_c = min(counts.values())
        if min_c == 0:
            issues.append("One or more classes have 0 samples.")
        elif max_c / min_c > 20:
            issues.append(
                f"Class imbalance ratio {max_c/min_c:.1f}:1 "
                f"(max={max_c}, min={min_c}) — consider class weighting."
            )

    # ── Metadata consistency ───────────────────────────────────
    if meta.get("n_pixels") != len(X):
        issues.append(
            f"metadata.n_pixels={meta.get('n_pixels')} "
            f"does not match actual len(X)={len(X)}"
        )
    if meta.get("feature_names") != FEATURE_NAMES:
        issues.append(
            f"metadata.feature_names {meta.get('feature_names')} "
            f"!= expected {FEATURE_NAMES}"
        )

    return {
        "valid": len(issues) == 0,
        "n_pixels": len(X),
        "n_features": X.shape[1] if X.ndim == 2 else None,
        "dtype_X": str(X.dtype),
        "dtype_y": str(y.dtype),
        "class_distribution": counts,
        "nan_rows_in_X": nan_count,
        "unique_labels": sorted(unique_labels),
        "date_start": meta.get("date_start"),
        "date_end_exclusive": meta.get("date_end_exclusive"),
        "date_end_inclusive_desc": meta.get("date_end_inclusive_desc"),
        "scene_count": meta.get("scene_count"),
        "n_removed_nan_bands": meta.get("n_removed_nan_bands"),
        "n_removed_missing_labels": meta.get("n_removed_missing_labels"),
        "labels_source": meta.get("labels_source"),
        "issues": issues,
    }


def print_validation(result: dict) -> None:
    print()
    status = "PASS" if result["valid"] else "FAIL"
    print(f"  Validation: {status}")
    if result.get("n_pixels") is not None:
        print(f"  Pixels:           {result['n_pixels']:,}")
        print(f"  Features:         {result['n_features']}  ({result.get('dtype_X','')})")
        print(f"  Labels dtype:     {result.get('dtype_y','')}")
        print(f"  NaN rows in X:    {result['nan_rows_in_X']}")
        print(f"  Unique labels:    {result['unique_labels']}")
        print(f"  Class dist:       {result['class_distribution']}")
        print(f"  Date start:       {result.get('date_start','?')}")
        print(f"  Date end (incl.): {result.get('date_end_inclusive_desc','?')}")
        print(f"  GEE end date:     {result.get('date_end_exclusive','?')}  (exclusive)")
        print(f"  Scenes used:      {result.get('scene_count','?')}")
        print(f"  Removed (NaN):    {result.get('n_removed_nan_bands','?')}")
        print(f"  Removed (label):  {result.get('n_removed_missing_labels','?')}")
        print(f"  Labels source:    {result.get('labels_source','?')}")
    for issue in result.get("issues", []):
        print(f"  [!] {issue}")


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    out_dir = Path(args.out) if args.out else (
        ROOT / "ml" / "data" / "processed" / f"coimbatore_{args.year}"
    )

    print("=" * 62)
    print("  GEOVIEW AML — Dataset Preparation")
    print("=" * 62)
    print(f"  Output directory : {out_dir}")
    print(f"  Composite year   : {args.year}")
    print(f"  Bounding box     : {args.bbox}")
    print(f"  Max cloud pct    : {args.cloud}%")
    print(f"  Max pixels       : {args.pixels:,}")
    print(f"  Pixel scale      : {args.scale} m")

    # Storage estimate
    bytes_per_pixel = 9 * 4 + 4   # X (float32) + y (int32)
    estimated_mb = args.pixels * bytes_per_pixel / 1_048_576
    print(f"  Est. storage     : ~{estimated_mb:.1f} MB for {args.pixels:,} pixels")
    print()

    # ── Validate-only mode ─────────────────────────────────────
    if args.validate_only:
        if not out_dir.exists() or not (out_dir / "X.npy").exists():
            print(f"  ERROR: No dataset found at {out_dir}")
            print("  Run without --validate-only to extract first.")
            sys.exit(1)
        print(f"  Validating existing dataset at {out_dir} ...")
        result = validate_dataset(out_dir)
        print_validation(result)
        sys.exit(0 if result["valid"] else 1)

    # ── GEE authentication ─────────────────────────────────────
    print("  Step 1: Checking GEE authentication ...")
    sys.path.insert(0, str(ROOT / "backend"))
    from backend.app.core.config import settings
    from backend.app.services.gee_service import authenticate_gee

    ok = authenticate_gee(project_id=settings.gee_project_id)
    if not ok:
        print()
        print("  ERROR: GEE authentication failed.")
        print("  Run:   earthengine authenticate")
        print("  Then set GEE_PROJECT_ID in .env and retry.")
        sys.exit(1)
    print("  GEE: authenticated")

    # ── Extraction ─────────────────────────────────────────────
    print(f"\n  Step 2: Extracting pixels ...")
    print(f"  (This contacts the GEE API; time depends on quota and network.)")

    if (out_dir / "X.npy").exists():
        print(f"  Found cached data at {out_dir}; skipping extraction.")
        print("  Delete the directory to force a fresh extraction.")
    else:
        from ml.preprocessing.gee_extractor import extract_training_data
        data = extract_training_data(
            bbox=args.bbox,
            composite_year=args.year,
            out_dir=out_dir,
            max_cloud_pct=args.cloud,
            scale_m=args.scale,
            max_pixels=args.pixels,
        )
        print(f"  Pixels extracted: {data['metadata']['n_pixels']:,}")
        print(f"  Removed (NaN):    {data['metadata']['n_removed_nan_bands']:,}")
        print(f"  Removed (label):  {data['metadata']['n_removed_missing_labels']:,}")

    # ── Validation ─────────────────────────────────────────────
    print(f"\n  Step 3: Validating saved dataset ...")
    result = validate_dataset(out_dir)
    print_validation(result)

    if result["valid"]:
        print()
        print("  Dataset is valid. Next step:")
        print(f"    python scripts/run_pipeline.py --year {args.year}")
        print()
    else:
        print()
        print("  Dataset failed validation. Fix the issues above before training.")
        sys.exit(1)


if __name__ == "__main__":
    main()

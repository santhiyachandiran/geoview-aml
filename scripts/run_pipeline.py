"""
GEOVIEW AML — End-to-End Training Pipeline

Steps:
  1. Authenticate GEE (must be done once externally: earthengine authenticate)
  2. Extract Sentinel-2 + WorldCover pixel arrays for Coimbatore
  3. Spatial block split (train 70% / val 15% / test 15%)
  4. Normalize features
  5. Train Random Forest baseline
  6. Train XGBoost + Optuna tuning
  7. Evaluate both models on held-out test set
  8. Save models, metrics, figures, metadata
  9. Log everything to MLflow

Usage (from project root, venv activated):
    python scripts/run_pipeline.py [--demo] [--trials 30] [--year 2021]

Flags:
    --demo      Use synthetic fixtures (no GEE needed). Results are clearly
                labelled DEMO and must not be reported as real findings.
    --trials N  Number of Optuna trials (default 30).
    --year YYYY WorldCover label year / composite year (default 2021).
    --timeout S Optuna timeout in seconds (default 600).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import numpy as np

from ml.preprocessing.feature_builder import (
    spatial_block_split, normalize_features, stack_features_demo, FEATURE_NAMES
)
from ml.models.train_rf import train_random_forest, save_model as save_rf
from ml.models.train_xgboost import train_xgboost_optuna, save_model as save_xgb
from ml.evaluation.evaluator import (
    evaluate_model, compare_models,
    save_confusion_matrix_fig, save_feature_importance_fig, log_to_mlflow,
)

CLASS_NAMES = ["built_up", "vegetation", "water", "bare"]
BBOX        = [76.8, 10.9, 77.2, 11.1]


def parse_args():
    p = argparse.ArgumentParser(description="GEOVIEW AML training pipeline")
    p.add_argument("--demo",    action="store_true", help="Use synthetic data (no GEE)")
    p.add_argument("--trials",  type=int, default=30,  help="Optuna trials")
    p.add_argument("--timeout", type=int, default=600, help="Optuna timeout (s)")
    p.add_argument("--year",    type=int, default=2021, help="Composite/label year")
    return p.parse_args()


def banner(msg: str):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def main():
    args = parse_args()

    from backend.app.core.config import settings
    artifacts_dir = settings.resolved_artifacts_dir
    models_dir    = artifacts_dir
    models_dir.mkdir(parents=True, exist_ok=True)

    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    # ── 1. Load or generate training data ────────────────────
    banner("Step 1: Load training data")

    demo_mode = args.demo
    metadata  = {}

    if demo_mode:
        print("  [DEMO MODE] Using synthetic pixel fixtures.")
        print("  These results are NOT valid research findings.")
        rng = np.random.default_rng(42)
        n   = 8000
        # Synthetic composites (4 classes, roughly realistic proportions)
        X_raw = np.vstack([
            stack_features_demo({"ndvi_mean": 0.55, "ndwi_mean": -0.05, "ndbi_mean": -0.30,
                                  "B2":0.06,"B3":0.09,"B4":0.08,"B8":0.45,"B11":0.15,"B12":0.10},
                                 n_pixels=n//4, seed=0),  # vegetation
            stack_features_demo({"ndvi_mean": 0.10, "ndwi_mean": -0.20, "ndbi_mean":  0.15,
                                  "B2":0.12,"B3":0.14,"B4":0.16,"B8":0.20,"B11":0.28,"B12":0.22},
                                 n_pixels=n//4, seed=1),  # built_up
            stack_features_demo({"ndvi_mean":-0.10, "ndwi_mean":  0.35, "ndbi_mean": -0.40,
                                  "B2":0.07,"B3":0.10,"B4":0.06,"B8":0.08,"B11":0.04,"B12":0.03},
                                 n_pixels=n//4, seed=2),  # water
            stack_features_demo({"ndvi_mean": 0.08, "ndwi_mean": -0.25, "ndbi_mean":  0.05,
                                  "B2":0.14,"B3":0.16,"B4":0.18,"B8":0.22,"B11":0.25,"B12":0.20},
                                 n_pixels=n//4, seed=3),  # bare
        ])
        y_raw = np.array(
            [1]*(n//4) + [0]*(n//4) + [2]*(n//4) + [3]*(n//4), dtype=np.int32
        )
        rows = np.arange(n) // 100
        cols = np.arange(n) %  100

        metadata = {
            "source": "SYNTHETIC_DEMO",
            "n_pixels": n,
            "feature_names": FEATURE_NAMES,
            "class_names": CLASS_NAMES,
            "class_distribution": {"built_up": n//4, "vegetation": n//4,
                                    "water": n//4, "bare": n//4},
            "warning": "DEMO DATA — do not cite as real results",
        }
        print(f"  Synthetic pixels: {n}")

    else:
        # Real GEE extraction
        print("  Authenticating GEE…")
        from backend.app.services.gee_service import authenticate_gee
        ok = authenticate_gee(project_id=settings.gee_project_id)
        if not ok:
            print(
                "\n  ERROR: GEE authentication failed.\n"
                "  Run:  earthengine authenticate\n"
                "  Then set GEE_PROJECT_ID in .env and re-run this script.\n"
                "  To run with synthetic fixtures: python scripts/run_pipeline.py --demo"
            )
            sys.exit(1)

        data_dir = ROOT / "ml" / "data" / "processed" / f"coimbatore_{args.year}"

        if (data_dir / "X.npy").exists():
            print(f"  Loading cached data from {data_dir}")
            from ml.preprocessing.gee_extractor import load_extracted_data
            data = load_extracted_data(data_dir)
        else:
            print(f"  Extracting Sentinel-2 + WorldCover pixels for {args.year}…")
            from ml.preprocessing.gee_extractor import extract_training_data
            data = extract_training_data(
                bbox=BBOX,
                composite_year=args.year,
                out_dir=data_dir,
            )

        X_raw    = data["X"]
        y_raw    = data["y"]
        rows     = data["rows"]
        cols     = data["cols"]
        metadata = data["metadata"]
        print(f"  Pixels loaded: {X_raw.shape[0]}")
        print(f"  Class dist:    {metadata['class_distribution']}")

    # ── 2. Spatial split ─────────────────────────────────────
    banner("Step 2: Spatial block split (70/15/15)")

    X_train, X_val, X_test, y_train, y_val, y_test = spatial_block_split(
        X_raw, y_raw,
        rows=rows, cols=cols,
        n_blocks=4,
        train_frac=0.70,
        val_frac=0.15,
        seed=42,
    )
    print(f"  Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    # Normalize — fit on train only
    X_train_n, scaler = normalize_features(X_train)
    X_val_n,   _      = normalize_features(X_val,  scaler)
    X_test_n,  _      = normalize_features(X_test, scaler)

    import joblib
    joblib.dump(scaler, models_dir / "scaler.joblib")

    # ── 3. Random Forest ──────────────────────────────────────
    banner("Step 3: Train Random Forest baseline")

    rf_model, rf_train_metrics = train_random_forest(X_train_n, y_train)
    rf_test_metrics = evaluate_model(rf_model, X_test_n, y_test, CLASS_NAMES)
    rf_test_metrics["evaluated_on"] = "test_set"
    rf_test_metrics["demo_data"]    = demo_mode

    print(f"  Accuracy:  {rf_test_metrics['accuracy']:.4f}")
    print(f"  Macro-F1:  {rf_test_metrics['macro_f1']:.4f}")
    print(f"  Bal.Acc:   {rf_test_metrics['balanced_accuracy']:.4f}")
    for cls, f1 in rf_test_metrics["per_class_f1"].items():
        print(f"    {cls:<12}: F1 {f1:.4f}")

    rf_path = save_rf(rf_model, models_dir / f"rf_{run_ts}.joblib")
    print(f"  Saved: {rf_path}")

    # Figures
    save_confusion_matrix_fig(
        rf_test_metrics["confusion_matrix"], CLASS_NAMES,
        models_dir / f"rf_confusion_matrix_{run_ts}.png"
    )
    save_feature_importance_fig(
        rf_train_metrics["feature_importances"], FEATURE_NAMES,
        models_dir / f"rf_feature_importance_{run_ts}.png"
    )

    # MLflow
    rf_params = {
        "n_estimators": 200, "max_depth": 20,
        "min_samples_split": 5, "min_samples_leaf": 2,
        "class_weight": "balanced", "random_state": 42,
        "split": "spatial_block_4x4", "demo_data": str(demo_mode),
    }
    rf_run_id = log_to_mlflow(
        rf_model, rf_params, rf_test_metrics,
        artifacts_dict={
            "confusion_matrix": str(models_dir / f"rf_confusion_matrix_{run_ts}.png"),
            "feature_importance": str(models_dir / f"rf_feature_importance_{run_ts}.png"),
        },
        run_name=f"random_forest_{run_ts}",
    )
    print(f"  MLflow run: {rf_run_id}")

    # ── 4. XGBoost + Optuna ───────────────────────────────────
    banner(f"Step 4: XGBoost + Optuna ({args.trials} trials, timeout {args.timeout}s)")

    xgb_model, best_params, xgb_val_metrics = train_xgboost_optuna(
        X_train_n, y_train, X_val_n, y_val,
        n_trials=args.trials, timeout=args.timeout, seed=42,
    )
    # Final evaluation on held-out test set
    xgb_test_metrics = evaluate_model(xgb_model, X_test_n, y_test, CLASS_NAMES)
    xgb_test_metrics["evaluated_on"] = "test_set"
    xgb_test_metrics["demo_data"]    = demo_mode
    xgb_test_metrics["best_params"]  = best_params

    print(f"  Best params: {best_params}")
    print(f"  Accuracy:    {xgb_test_metrics['accuracy']:.4f}")
    print(f"  Macro-F1:    {xgb_test_metrics['macro_f1']:.4f}")
    print(f"  Bal.Acc:     {xgb_test_metrics['balanced_accuracy']:.4f}")
    for cls, f1 in xgb_test_metrics["per_class_f1"].items():
        print(f"    {cls:<12}: F1 {f1:.4f}")

    xgb_path = save_xgb(xgb_model, models_dir / f"xgb_{run_ts}.joblib")
    print(f"  Saved: {xgb_path}")

    save_confusion_matrix_fig(
        xgb_test_metrics["confusion_matrix"], CLASS_NAMES,
        models_dir / f"xgb_confusion_matrix_{run_ts}.png"
    )
    save_feature_importance_fig(
        list(xgb_model.feature_importances_), FEATURE_NAMES,
        models_dir / f"xgb_feature_importance_{run_ts}.png"
    )

    xgb_log_params = {**best_params, "split": "spatial_block_4x4",
                      "demo_data": str(demo_mode), "n_trials": args.trials}
    xgb_run_id = log_to_mlflow(
        xgb_model, xgb_log_params, xgb_test_metrics,
        artifacts_dict={
            "confusion_matrix": str(models_dir / f"xgb_confusion_matrix_{run_ts}.png"),
            "feature_importance": str(models_dir / f"xgb_feature_importance_{run_ts}.png"),
        },
        run_name=f"xgboost_{run_ts}",
    )
    print(f"  MLflow run: {xgb_run_id}")

    # ── 5. Model comparison ───────────────────────────────────
    banner("Step 5: Model comparison")

    comparison = compare_models(rf_test_metrics, xgb_test_metrics)
    print(f"\n  {'Metric':<22} {'Random Forest':>15} {'XGBoost':>12} {'Winner':>14}")
    print(f"  {'-'*65}")
    for metric, row in comparison.items():
        rf_v  = f"{row['random_forest']:.4f}" if row['random_forest'] is not None else "—"
        xg_v  = f"{row['xgboost']:.4f}"       if row['xgboost']       is not None else "—"
        print(f"  {metric:<22} {rf_v:>15} {xg_v:>12} {row['winner']:>14}")

    # ── 6. Save results JSON ──────────────────────────────────
    results = {
        "run_timestamp": run_ts,
        "demo_data": demo_mode,
        "metadata": metadata,
        "random_forest": {
            "params": rf_params,
            "test_metrics": rf_test_metrics,
            "mlflow_run_id": rf_run_id,
            "model_path": str(rf_path),
        },
        "xgboost": {
            "best_params": best_params,
            "test_metrics": xgb_test_metrics,
            "mlflow_run_id": xgb_run_id,
            "model_path": str(xgb_path),
        },
        "comparison": comparison,
    }

    results_path = models_dir / f"results_{run_ts}.json"
    results_path.write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )

    banner("Pipeline complete")
    print(f"  Results saved to: {results_path}")
    print(f"  Model artifacts:  {models_dir}")
    print()
    if demo_mode:
        print("  [DEMO RUN] Results are from synthetic data.")
        print("     Do not cite these metrics in your paper.")
    else:
        print("  [REAL DATA] Metrics are from held-out test pixels.")
    print()
    print("  Next: mlflow ui --backend-store-uri artifacts/mlruns")


if __name__ == "__main__":
    main()

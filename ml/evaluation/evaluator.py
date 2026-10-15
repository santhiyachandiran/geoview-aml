"""
GeoChange AI — Model Evaluator and Experiment Tracker
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    classification_report, confusion_matrix,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str],
) -> dict:
    """
    Evaluate a fitted model on a held-out test set.

    Returns accuracy, macro_f1, balanced_accuracy, per-class metrics,
    confusion_matrix, and roc_auc where applicable.
    NEVER call this on training data and report it as test-set performance.
    """
    y_pred = model.predict(X_test)
    labels = sorted(np.unique(y_test).tolist())
    names  = class_names[:len(labels)]

    report = classification_report(
        y_test, y_pred, labels=labels, target_names=names,
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred, labels=labels).tolist()

    roc_auc = None
    if len(labels) <= 10 and hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_test)
            if len(labels) == 2:
                roc_auc = round(roc_auc_score(y_test, proba[:, 1]), 4)
            else:
                roc_auc = round(roc_auc_score(
                    y_test, proba, multi_class="ovr", average="macro",
                ), 4)
        except Exception as exc:
            logger.warning(f"ROC-AUC failed: {exc}")

    return {
        "accuracy":           round(accuracy_score(y_test, y_pred), 4),
        "balanced_accuracy":  round(balanced_accuracy_score(y_test, y_pred), 4),
        "macro_f1":           round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "per_class_precision": {n: round(report[n]["precision"], 4) for n in names if n in report},
        "per_class_recall":    {n: round(report[n]["recall"],    4) for n in names if n in report},
        "per_class_f1":        {n: round(report[n]["f1-score"],  4) for n in names if n in report},
        "confusion_matrix":   cm,
        "roc_auc":            roc_auc,
        "n_test_samples":     len(y_test),
        "class_names":        names,
    }


def compare_models(rf_metrics: dict, xgb_metrics: dict) -> dict:
    """Side-by-side comparison table of RF and XGBoost test-set metrics."""
    keys = ["accuracy", "balanced_accuracy", "macro_f1", "roc_auc"]
    comparison = {}
    for k in keys:
        v_rf  = rf_metrics.get(k)
        v_xgb = xgb_metrics.get(k)
        if v_rf is not None and v_xgb is not None:
            winner = "xgboost" if v_xgb > v_rf else "random_forest" if v_rf > v_xgb else "tie"
        else:
            winner = "n/a"
        comparison[k] = {"random_forest": v_rf, "xgboost": v_xgb, "winner": winner}
    return comparison


def save_confusion_matrix_fig(
    cm: list[list[int]],
    class_names: list[str],
    filepath: str | Path,
) -> Path:
    """Save a seaborn confusion-matrix heatmap as PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        np.array(cm), annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    fig.savefig(filepath, dpi=120)
    plt.close(fig)
    return filepath


def save_feature_importance_fig(
    importances: list[float],
    feature_names: list[str],
    filepath: str | Path,
) -> Path:
    """Save a feature-importance bar chart as PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    idx          = np.argsort(importances)[::-1]
    sorted_names = [feature_names[i] for i in idx]
    sorted_vals  = [importances[i]   for i in idx]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(sorted_names, sorted_vals, color="#40916C")
    ax.set_title("Feature Importances")
    ax.set_ylabel("Importance")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(filepath, dpi=120)
    plt.close(fig)
    return filepath


def save_shap_summary(
    model,
    X_test: np.ndarray,
    feature_names: list[str],
    filepath: str | Path,
    max_samples: int = 500,
) -> Path | None:
    """
    Save a SHAP mean-|SHAP| summary bar chart.
    Returns the path, or None if SHAP is unavailable or fails.
    """
    try:
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        idx      = np.random.default_rng(42).choice(
            len(X_test), min(max_samples, len(X_test)), replace=False,
        )
        X_sample = X_test[idx]

        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_sample)

        if isinstance(shap_vals, list):
            mean_abs = np.mean([np.abs(sv) for sv in shap_vals], axis=0).mean(axis=0)
        else:
            mean_abs = np.abs(shap_vals).mean(axis=0)

        idx_sorted = np.argsort(mean_abs)[::-1]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(
            [feature_names[i] for i in idx_sorted],
            [mean_abs[i]       for i in idx_sorted],
            color="#40916C",
        )
        ax.set_title("Mean |SHAP| Feature Importance")
        ax.set_ylabel("Mean |SHAP value|")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        fig.savefig(filepath, dpi=120)
        plt.close(fig)
        logger.info(f"SHAP summary saved: {filepath}")
        return filepath
    except Exception as exc:
        logger.warning(f"SHAP summary skipped: {exc}")
        return None


def log_to_mlflow(
    model,
    params: dict,
    metrics: dict,
    artifacts_dict: dict | None = None,
    run_name: str = "geochange_run",
) -> str | None:
    """
    Log a model run to MLflow. Returns the run_id or None on failure.
    """
    try:
        os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
        import mlflow

        import sys
        from pathlib import Path as _P
        _root = _P(__file__).resolve().parents[2]
        if str(_root / "backend") not in sys.path:
            sys.path.insert(0, str(_root / "backend"))

        from app.core.config import settings

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment("geochange_ai")

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params(params)
            loggable = {
                k: v for k, v in metrics.items()
                if isinstance(v, (int, float)) and v is not None
            }
            mlflow.log_metrics(loggable)
            if artifacts_dict:
                for name, path in artifacts_dict.items():
                    if Path(path).exists():
                        mlflow.log_artifact(str(path), artifact_path=name)
            return run.info.run_id
    except Exception as exc:
        logger.warning(f"MLflow logging failed: {exc}")
        return None

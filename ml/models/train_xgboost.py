"""
GeoChange AI — XGBoost + Optuna Hyperparameter Tuning
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import joblib
import numpy as np
import optuna
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)

CLASS_NAMES = ["built_up", "vegetation", "water", "bare"]


def train_xgboost_optuna(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 30,
    timeout: int = 600,
    seed: int = 42,
) -> tuple[object, dict, dict]:
    """
    Train XGBoost with Optuna hyperparameter search.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data (used for objective)
        n_trials: Number of Optuna trials
        timeout: Max seconds for search
        seed: Random seed

    Returns:
        (best_model, best_params, metrics_dict)
    """
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError("xgboost is not installed") from exc

    n_classes = len(np.unique(y_train))

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 10.0),
            "use_label_encoder": False,
            "eval_metric": "mlogloss",
            "random_state": seed,
            "n_jobs": -1,
        }
        if n_classes > 2:
            params["objective"] = "multi:softmax"
            params["num_class"] = n_classes

        clf = XGBClassifier(**params)
        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = clf.predict(X_val)
        return f1_score(y_val, preds, average="macro", zero_division=0)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    best_params = study.best_params
    logger.info(f"Best val macro_f1={study.best_value:.4f} | params={best_params}")

    # Retrain on full train+val with best params
    final_params = {**best_params, "random_state": seed, "n_jobs": -1,
                    "use_label_encoder": False, "eval_metric": "mlogloss"}
    if n_classes > 2:
        final_params.update({"objective": "multi:softmax", "num_class": n_classes})

    t0 = time.time()
    best_model = XGBClassifier(**final_params)
    X_full = np.vstack([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])
    best_model.fit(X_full, y_full)
    train_time = round(time.time() - t0, 2)

    # Evaluate on val
    y_pred = best_model.predict(X_val)
    metrics = _compute_metrics(y_pred, y_val, best_model, train_time, best_params)
    return best_model, best_params, metrics


def _compute_metrics(y_pred, y_true, model, train_time, best_params) -> dict:
    labels = sorted(np.unique(y_true).tolist())
    names = [CLASS_NAMES[i] for i in labels if i < len(CLASS_NAMES)]
    report = classification_report(y_true, y_pred, labels=labels,
                                   target_names=names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    importances = (model.feature_importances_.tolist()
                   if hasattr(model, "feature_importances_") else [])
    return {
        "model": "xgboost",
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "per_class_precision": {n: round(report[n]["precision"], 4) for n in names if n in report},
        "per_class_recall": {n: round(report[n]["recall"], 4) for n in names if n in report},
        "per_class_f1": {n: round(report[n]["f1-score"], 4) for n in names if n in report},
        "confusion_matrix": cm,
        "feature_importances": importances,
        "n_classes": len(labels),
        "class_names": names,
        "best_params": best_params,
        "train_time_sec": train_time,
    }


def save_model(model, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path

"""
GeoChange AI — Random Forest Baseline Training

train_random_forest() fits the model only.
Evaluation on a held-out test set is done by evaluator.evaluate_model().
"""

from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier


CLASS_NAMES = ["built_up", "vegetation", "water", "bare"]

DEFAULT_PARAMS = {
    "n_estimators": 200,
    "max_depth": 20,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: dict | None = None,
) -> tuple[RandomForestClassifier, dict]:
    """
    Fit a Random Forest on training data.

    Returns:
        (fitted_model, train_info_dict)
        train_info_dict contains training time and params only.
        Call evaluator.evaluate_model(model, X_test, y_test, CLASS_NAMES)
        for valid test-set metrics.
    """
    params = {**DEFAULT_PARAMS, **(config or {})}
    model  = RandomForestClassifier(**params)

    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = round(time.time() - t0, 2)

    importances = model.feature_importances_.tolist()

    return model, {
        "model":              "random_forest",
        "params":             params,
        "train_time_sec":     train_time,
        "n_train_samples":    len(X_train),
        "feature_importances": importances,
        "evaluated_on":       "not_yet_evaluated",
        "note": (
            "Call evaluator.evaluate_model(model, X_test, y_test, CLASS_NAMES) "
            "for accuracy, F1, confusion matrix on the held-out test set."
        ),
    }


def save_model(model: RandomForestClassifier, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path

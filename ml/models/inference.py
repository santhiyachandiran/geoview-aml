"""
GeoChange AI — Model Inference

load_model()       — load a saved joblib model
predict_landcover() — classify pixels
predict_change()    — classify two time points and derive a change map
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np


CLASS_NAMES = ["built_up", "vegetation", "water", "bare"]


def load_model(model_path: str | Path):
    """Load a joblib-serialised model. Raises FileNotFoundError if missing."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


def predict_landcover(
    model,
    features_array: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Classify pixels.

    Returns:
        (predictions, probabilities or None)
    """
    predictions = model.predict(features_array)
    probabilities = None
    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(features_array)
        except Exception:
            pass
    return predictions, probabilities


def predict_change(
    model,
    composite_t1_features: np.ndarray,
    composite_t2_features: np.ndarray,
) -> dict:
    """
    Classify two time points and produce a change map.

    Returns dict with:
        predictions_t1, predictions_t2,
        change_mask (bool array), changed_pixel_count,
        total_pixel_count, change_pct,
        class_names, transitions (list of dicts)
    """
    preds_t1, _ = predict_landcover(model, composite_t1_features)
    preds_t2, _ = predict_landcover(model, composite_t2_features)

    change_mask = preds_t1 != preds_t2
    changed     = int(change_mask.sum())
    total       = len(change_mask)

    # Build transition counts
    transitions = []
    for from_cls in range(len(CLASS_NAMES)):
        for to_cls in range(len(CLASS_NAMES)):
            mask  = (preds_t1 == from_cls) & (preds_t2 == to_cls)
            count = int(mask.sum())
            if count > 0:
                area_ha = round(count * 100 / 10_000, 2)  # 10 m pixels
                transitions.append({
                    "from_class": CLASS_NAMES[from_cls],
                    "to_class":   CLASS_NAMES[to_cls],
                    "pixel_count": count,
                    "area_ha":     area_ha,
                    "area_pct":    round(count / total * 100, 2),
                })

    transitions.sort(key=lambda x: x["pixel_count"], reverse=True)

    return {
        "predictions_t1":   preds_t1.tolist(),
        "predictions_t2":   preds_t2.tolist(),
        "change_mask":      change_mask.tolist(),
        "changed_pixel_count": changed,
        "total_pixel_count":   total,
        "change_pct":       round(changed / total * 100, 2) if total > 0 else 0.0,
        "class_names":      CLASS_NAMES,
        "transitions":      transitions,
    }

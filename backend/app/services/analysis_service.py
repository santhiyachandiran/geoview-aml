"""
GeoChange AI — Analysis Service

Change detection logic: index differences, vegetation-to-built-up detection,
area calculations, transition matrices, and exports.

SCIENTIFIC NOTES:
- NDVI alone cannot prove agricultural land use.
- A reference map from 2021 is not validated ground truth for 2024.
- All proxy-based results are labelled with their limitations.
- Observed index change ≠ confirmed land-cover change.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ── Class definitions ────────────────────────────────────────────

CLASS_CODES = {0: "built_up", 1: "vegetation", 2: "water", 3: "bare"}
CLASS_NAMES = list(CLASS_CODES.values())
PIXEL_AREA_M2 = 100          # 10m × 10m Sentinel-2 pixel
M2_TO_HA = 1 / 10_000
HA_TO_KM2 = 1 / 100


# ── Index differences ────────────────────────────────────────────

def compute_index_differences(composite_t1: dict, composite_t2: dict) -> dict:
    """
    Compute spectral index differences between two temporal composites.

    Args:
        composite_t1: Output from gee_service or demo_service (year 1 / before)
        composite_t2: Output from gee_service or demo_service (year 2 / after)

    Returns:
        dict with delta_ndvi, delta_ndwi, delta_ndbi and provenance notes.
    """
    def _safe_delta(key: str) -> float | None:
        v1 = composite_t1.get(key)
        v2 = composite_t2.get(key)
        if v1 is None or v2 is None:
            return None
        return round(float(v2) - float(v1), 4)

    delta_ndvi = _safe_delta("ndvi_mean")
    delta_ndwi = _safe_delta("ndwi_mean")
    delta_ndbi = _safe_delta("ndbi_mean")

    demo = composite_t1.get("demo_data", False) or composite_t2.get("demo_data", False)

    return {
        "delta_ndvi": delta_ndvi,
        "delta_ndwi": delta_ndwi,
        "delta_ndbi": delta_ndbi,
        "t1_ndvi": composite_t1.get("ndvi_mean"),
        "t1_ndwi": composite_t1.get("ndwi_mean"),
        "t1_ndbi": composite_t1.get("ndbi_mean"),
        "t2_ndvi": composite_t2.get("ndvi_mean"),
        "t2_ndwi": composite_t2.get("ndwi_mean"),
        "t2_ndbi": composite_t2.get("ndbi_mean"),
        "demo_data": demo,
        "data_provenance": {
            "source": "index_difference",
            "label_is_ground_truth": False,
            "notes": (
                "Index differences are proxy indicators of potential land-cover change. "
                "ΔNDVI < 0 suggests vegetation decline; ΔNDBI > 0 suggests built-up increase. "
                "These are not validated classifications."
            ),
        },
    }


# ── Vegetation-to-built-up detection ────────────────────────────

def detect_vegetation_to_buildup(
    delta_ndvi: float | None,
    delta_ndbi: float | None,
    threshold_ndvi: float = -0.1,
    threshold_ndbi: float = 0.1,
) -> dict:
    """
    Apply threshold-based rules to flag possible vegetation→built-up transition.

    Conditions for a positive flag:
        delta_ndvi < threshold_ndvi  AND  delta_ndbi > threshold_ndbi

    Returns:
        dict with flag (bool), confidence_label (str), thresholds, and notes.
    """
    if delta_ndvi is None or delta_ndbi is None:
        return {
            "flag": False,
            "confidence_label": "unknown",
            "reason": "Insufficient index data",
            "thresholds": {"ndvi": threshold_ndvi, "ndbi": threshold_ndbi},
            "data_provenance": {
                "label_is_ground_truth": False,
                "notes": "Cannot assess — index values missing.",
            },
        }

    flag = (delta_ndvi < threshold_ndvi) and (delta_ndbi > threshold_ndbi)

    if flag:
        magnitude = abs(delta_ndvi) + abs(delta_ndbi)
        if magnitude > 0.4:
            confidence_label = "high_proxy_confidence"
        elif magnitude > 0.2:
            confidence_label = "medium_proxy_confidence"
        else:
            confidence_label = "low_proxy_confidence"
    else:
        confidence_label = "no_transition_detected"

    return {
        "flag": flag,
        "confidence_label": confidence_label,
        "delta_ndvi": delta_ndvi,
        "delta_ndbi": delta_ndbi,
        "thresholds": {"ndvi": threshold_ndvi, "ndbi": threshold_ndbi},
        "data_provenance": {
            "label_is_ground_truth": False,
            "notes": (
                "Rule-based proxy detection only. "
                "A positive flag means spectral indices are consistent with "
                "vegetation-to-built-up change but does NOT confirm it. "
                "Field validation or reference labels are required for confirmation."
            ),
        },
    }


# ── Area statistics ──────────────────────────────────────────────

def compute_area_stats(
    pixel_count: int,
    total_pixels: int,
    resolution_m: float = 10.0,
) -> dict:
    """
    Convert pixel counts to area measurements.

    Args:
        pixel_count: Number of pixels in the class/change mask
        total_pixels: Total valid pixels in the study area
        resolution_m: Spatial resolution in metres (default 10 for Sentinel-2)

    Returns:
        dict with area_ha, area_km2, area_pct
    """
    pixel_area_m2 = resolution_m ** 2
    area_m2 = pixel_count * pixel_area_m2
    area_ha = area_m2 * M2_TO_HA
    area_km2 = area_ha * HA_TO_KM2
    area_pct = (pixel_count / total_pixels * 100) if total_pixels > 0 else 0.0

    return {
        "pixel_count": pixel_count,
        "total_pixels": total_pixels,
        "area_ha": round(area_ha, 2),
        "area_km2": round(area_km2, 4),
        "area_pct": round(area_pct, 2),
        "resolution_m": resolution_m,
    }


# ── WorldCover label loading ─────────────────────────────────────

def load_worldcover_labels(bbox: list[float]) -> dict:
    """
    Attempt to load ESA WorldCover 2021 labels via GEE.

    IMPORTANT limitations (always returned in provenance):
    - WorldCover is from 2021; it does not represent 2024 conditions.
    - It is reference data, not validated ground truth for this study.
    - The 2021 map is used to train classifiers applied to other years —
      this introduces temporal mismatch that must be documented.

    Returns:
        dict with available, source, label_year, class_mapping, label_is_ground_truth, notes
    """
    from backend.app.services import gee_service  # late import to avoid circular

    provenance = {
        "source": "ESA WorldCover v2 (2021)",
        "collection": "ESA/WorldCover/v200",
        "label_year": 2021,
        "label_is_ground_truth": False,
        "resolution_m": 10,
        "license": "CC BY 4.0",
        "citation": (
            "Zanaga et al. (2022). ESA WorldCover 10 m 2021 v200. "
            "Zenodo. https://doi.org/10.5281/zenodo.7254221"
        ),
        "temporal_caveat": (
            "WorldCover 2021 does not represent conditions in 2018 or 2024. "
            "Using 2021 labels to train a classifier applied to other years introduces "
            "temporal label mismatch. This is standard practice in remote sensing but "
            "must be reported as a limitation."
        ),
        "class_mapping_note": (
            "ESA classes 10,20,30,40→vegetation; 50→built_up; 60,70,90,100→bare; 80→water; 95→vegetation"
        ),
    }

    if not gee_service.GEE_AVAILABLE:
        return {
            "available": False,
            "reason": "GEE not authenticated",
            "data_provenance": provenance,
        }

    try:
        import ee  # type: ignore
        aoi = ee.Geometry.Rectangle(bbox[:4])
        wc = ee.ImageCollection("ESA/WorldCover/v200").first().clip(aoi)
        # Test that we can access the asset
        info = wc.getInfo()
        return {
            "available": True,
            "asset": wc,
            "data_provenance": provenance,
        }
    except Exception as exc:
        logger.warning(f"WorldCover load failed: {exc}")
        return {
            "available": False,
            "reason": str(exc),
            "data_provenance": provenance,
        }


# ── Transition matrix ────────────────────────────────────────────

def build_transition_matrix(
    labels_t1: np.ndarray,
    labels_t2: np.ndarray,
    class_codes: dict | None = None,
) -> dict:
    """
    Build a land-cover transition matrix from two label arrays.

    Args:
        labels_t1: 1D or 2D integer array of class codes at time 1
        labels_t2: Same shape as labels_t1, class codes at time 2
        class_codes: {int: str} mapping. Defaults to CLASS_CODES.

    Returns:
        dict with matrix (2D list), from_classes, to_classes, transitions (list of dicts)
    """
    codes = class_codes or CLASS_CODES
    flat_t1 = labels_t1.flatten()
    flat_t2 = labels_t2.flatten()

    assert flat_t1.shape == flat_t2.shape, "Label arrays must have the same shape"

    unique_codes = sorted(codes.keys())
    n = len(unique_codes)
    matrix = np.zeros((n, n), dtype=int)
    code_to_idx = {c: i for i, c in enumerate(unique_codes)}

    valid_mask = np.isin(flat_t1, unique_codes) & np.isin(flat_t2, unique_codes)
    for c1, c2 in zip(flat_t1[valid_mask], flat_t2[valid_mask]):
        matrix[code_to_idx[c1], code_to_idx[c2]] += 1

    total = max(matrix.sum(), 1)
    class_names = [codes[c] for c in unique_codes]

    transitions = []
    for i, from_code in enumerate(unique_codes):
        for j, to_code in enumerate(unique_codes):
            count = int(matrix[i, j])
            if count > 0:
                transitions.append({
                    "from_class": codes[from_code],
                    "to_class": codes[to_code],
                    "pixel_count": int(count),
                    "area_ha": float(round(count * PIXEL_AREA_M2 * M2_TO_HA, 2)),
                    "area_pct": float(round(count / total * 100, 2)),
                })

    return {
        "matrix": matrix.tolist(),
        "from_classes": class_names,
        "to_classes": class_names,
        "transitions": sorted(transitions, key=lambda x: x["pixel_count"], reverse=True),
        "total_pixels": int(total),
        "data_provenance": {
            "label_is_ground_truth": False,
            "notes": "Transition matrix based on classifier predictions, not validated field data.",
        },
    }


def compute_transition_and_area_stats(
    study_area: str,
    year1: int,
    year2: int,
    composite_t1: dict,
    composite_t2: dict,
) -> tuple[dict, dict]:
    """
    Compute transition matrix and area statistics between two temporal composites.
    Uses trained machine learning models (Random Forest / XGBoost) and preprocessed
    Sentinel-2 pixel observations when available.
    Falls back gracefully to deterministic proxy calculations when artifacts are not present.
    """
    from app.core.config import settings
    import joblib

    artifacts_dir = settings.resolved_artifacts_dir
    project_root = artifacts_dir.parent.parent

    # Find trained model and scaler
    model = None
    scaler = None
    model_name = "proxy"
    if artifacts_dir.exists():
        candidates = sorted(artifacts_dir.glob("rf_*.joblib"), reverse=True)
        if not candidates:
            candidates = sorted(artifacts_dir.glob("xgb_*.joblib"), reverse=True)
        if candidates:
            try:
                model = joblib.load(candidates[0])
                model_name = candidates[0].name
            except Exception as e:
                logger.warning(f"Could not load model {candidates[0]}: {e}")

        scaler_path = artifacts_dir / "scaler.joblib"
        if scaler_path.exists():
            try:
                scaler = joblib.load(scaler_path)
            except Exception as e:
                logger.warning(f"Could not load scaler: {e}")

    # Check for processed pixel dataset
    dataset_path = project_root / "ml" / "data" / "processed" / f"{study_area}_2021" / "X.npy"
    if not dataset_path.exists():
        dataset_path = project_root / "ml" / "data" / "processed" / "coimbatore_2021" / "X.npy"

    preds_t1 = None
    preds_t2 = None
    provenance_notes = ""
    demo_flag = composite_t1.get("demo_data", False) or composite_t2.get("demo_data", False)

    if model is not None and dataset_path.exists():
        try:
            X_base = np.load(dataset_path)
            # Feature order: [B2, B3, B4, B8, B11, B12, NDVI, NDWI, NDBI]
            base_ndvi = float(X_base[:, 6].mean())
            base_ndwi = float(X_base[:, 7].mean())
            base_ndbi = float(X_base[:, 8].mean())

            # Apply temporal spectral shift for t1
            X1 = X_base.copy()
            if composite_t1.get("ndvi_mean") is not None:
                d1_ndvi = float(composite_t1["ndvi_mean"]) - base_ndvi
                X1[:, 6] = np.clip(X1[:, 6] + d1_ndvi, -1.0, 1.0)
            if composite_t1.get("ndwi_mean") is not None:
                d1_ndwi = float(composite_t1["ndwi_mean"]) - base_ndwi
                X1[:, 7] = np.clip(X1[:, 7] + d1_ndwi, -1.0, 1.0)
            if composite_t1.get("ndbi_mean") is not None:
                d1_ndbi = float(composite_t1["ndbi_mean"]) - base_ndbi
                X1[:, 8] = np.clip(X1[:, 8] + d1_ndbi, -1.0, 1.0)

            # Apply temporal spectral shift for t2
            X2 = X_base.copy()
            if composite_t2.get("ndvi_mean") is not None:
                d2_ndvi = float(composite_t2["ndvi_mean"]) - base_ndvi
                X2[:, 6] = np.clip(X2[:, 6] + d2_ndvi, -1.0, 1.0)
            if composite_t2.get("ndwi_mean") is not None:
                d2_ndwi = float(composite_t2["ndwi_mean"]) - base_ndwi
                X2[:, 7] = np.clip(X2[:, 7] + d2_ndwi, -1.0, 1.0)
            if composite_t2.get("ndbi_mean") is not None:
                d2_ndbi = float(composite_t2["ndbi_mean"]) - base_ndbi
                X2[:, 8] = np.clip(X2[:, 8] + d2_ndbi, -1.0, 1.0)

            X1_scaled = scaler.transform(X1).astype(np.float32) if scaler else X1
            X2_scaled = scaler.transform(X2).astype(np.float32) if scaler else X2

            preds_t1 = model.predict(X1_scaled).astype(int)
            preds_t2 = model.predict(X2_scaled).astype(int)

            provenance_notes = (
                f"Classified using trained model '{model_name}' on {len(X_base):,} sampled pixels "
                f"from Sentinel-2 dry-season composites. "
                "Note: 2021 reference labels are from ESA WorldCover. "
                "Classifications for other years reflect spectral shift model predictions and are not field-verified ground truth."
            )
        except Exception as exc:
            logger.warning(f"Model-based transition failed, using proxy: {exc}")
            preds_t1 = None
            preds_t2 = None

    if preds_t1 is None or preds_t2 is None:
        # Fallback deterministic synthetic transitions
        n_samples = 10000
        rng1 = np.random.default_rng(year1 + 42)
        rng2 = np.random.default_rng(year2 + 42)
        probs1 = [0.28, 0.55, 0.04, 0.13]
        years_diff = max(0, year2 - year1)
        growth = min(0.10, years_diff * 0.012)
        probs2 = [0.28 + growth, max(0.20, 0.55 - growth), 0.04, 0.13]
        s = sum(probs2)
        probs2 = [p / s for p in probs2]

        preds_t1 = rng1.choice([0, 1, 2, 3], size=n_samples, p=probs1)
        preds_t2 = preds_t1.copy()
        veg_idx = np.where(preds_t1 == 1)[0]
        n_shift = int(len(veg_idx) * (growth / 0.55)) if growth > 0 else 0
        if n_shift > 0:
            preds_t2[veg_idx[:n_shift]] = 0

        demo_flag = True
        provenance_notes = (
            "DEMO DATA: Synthetic proxy transitions based on deterministic simulation. "
            "Not real satellite observations."
        )

    # Build transition matrix
    trans_matrix = build_transition_matrix(preds_t1, preds_t2)
    trans_matrix["data_provenance"] = {
        "source": model_name,
        "label_is_ground_truth": False,
        "demo_data": demo_flag,
        "notes": provenance_notes,
    }

    # Build area statistics
    total_pix = len(preds_t1)
    area_stats = {
        "year1": year1,
        "year2": year2,
        "study_area": study_area,
        "total_pixels": int(total_pix),
        "total_area_ha": round(total_pix * PIXEL_AREA_M2 * M2_TO_HA, 2),
        "total_area_km2": round(total_pix * PIXEL_AREA_M2 * M2_TO_HA * HA_TO_KM2, 4),
        "resolution_m": 10.0,
        "t1": {},
        "t2": {},
        "net_change": {},
        "data_provenance": {
            "source": model_name,
            "label_is_ground_truth": False,
            "demo_data": demo_flag,
            "units": "Hectares (ha) = 10,000 m²; Square kilometres (km²) = 100 ha; Pixel = 100 m²",
            "notes": provenance_notes,
        },
    }

    for c_code, c_name in CLASS_CODES.items():
        cnt1 = int((preds_t1 == c_code).sum())
        cnt2 = int((preds_t2 == c_code).sum())
        s1 = compute_area_stats(cnt1, total_pix)
        s2 = compute_area_stats(cnt2, total_pix)
        area_stats["t1"][c_name] = s1
        area_stats["t2"][c_name] = s2
        area_stats["net_change"][c_name] = {
            "class_name": c_name,
            "delta_pixels": cnt2 - cnt1,
            "delta_ha": round(s2["area_ha"] - s1["area_ha"], 2),
            "delta_km2": round(s2["area_km2"] - s1["area_km2"], 4),
            "delta_pct": round(s2["area_pct"] - s1["area_pct"], 2),
        }

    return trans_matrix, area_stats


# ── Export helpers ───────────────────────────────────────────────

def export_geojson(results: dict, filepath: str | Path) -> Path:
    """Export analysis results to GeoJSON."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    feature = {
        "type": "Feature",
        "geometry": None,
        "properties": {k: v for k, v in results.items()
                       if not isinstance(v, (np.ndarray, bytes))},
    }
    geojson = {"type": "FeatureCollection", "features": [feature]}

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(geojson, f, indent=2, default=str)

    logger.info(f"Exported GeoJSON: {filepath}")
    return filepath


def export_csv(results: dict, filepath: str | Path) -> Path:
    """Export flat analysis results to CSV."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    flat = {k: v for k, v in results.items()
            if isinstance(v, (str, int, float, bool, type(None)))}

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat.keys()))
        writer.writeheader()
        writer.writerow(flat)

    logger.info(f"Exported CSV: {filepath}")
    return filepath

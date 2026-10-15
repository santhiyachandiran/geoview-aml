"""
GeoChange AI — Demo Service

Provides the same interface as gee_service but returns synthetic data derived
from a deterministic seed so the application and UI work without GEE credentials.

Every returned dict has demo_data=True and a clear data_provenance note.
These results must never be presented as real satellite observations.
"""

from __future__ import annotations

import hashlib
import numpy as np
from datetime import date, timedelta


_DEMO_CLASSES = ["built_up", "vegetation", "water", "bare"]

# Coimbatore 2018 baseline land cover distribution (approximate, illustrative)
_BASE_COVER_PCT = {
    "built_up": 0.28,
    "vegetation": 0.55,
    "water": 0.04,
    "bare": 0.13,
}

# Annual urbanisation rate used for demo trends (illustrative only)
_URBAN_GROWTH_RATE = 0.012   # +1.2 pct/yr built_up, corresponding vegetation decline


def _seed_from(bbox: list[float], year: int) -> int:
    """Deterministic seed from bbox + year for reproducible synthetic data."""
    key = f"{bbox[0]:.4f},{bbox[1]:.4f},{bbox[2]:.4f},{bbox[3]:.4f},{year}"
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16) % (2**31)


def get_sentinel2_composite(
    bbox: list[float],
    year: int,
    start_month: int = 11,
    end_month: int = 3,
    max_cloud_pct: int = 20,
) -> dict:
    """
    Return a synthetic composite that mimics gee_service output.
    Urbanisation trend: NDVI declines ~0.005/yr, NDBI rises ~0.004/yr from 2018.

    DEMO DATA — not real satellite observations.
    """
    rng = np.random.default_rng(_seed_from(bbox, year))

    years_since_2018 = max(0, year - 2018)
    base_ndvi = 0.38
    ndvi_mean = round(base_ndvi - 0.005 * years_since_2018 + rng.normal(0, 0.005), 4)
    ndwi_mean = round(-0.12 + rng.normal(0, 0.004), 4)
    ndbi_mean = round(-0.15 + 0.004 * years_since_2018 + rng.normal(0, 0.004), 4)

    # Illustrative cloud cover
    cloud_pct = round(abs(rng.normal(8, 3)), 1)

    if start_month <= end_month:
        date_start = date(year, start_month, 1).isoformat()
        date_end = date(year, end_month, 28).isoformat()
    else:
        date_start = date(year, start_month, 1).isoformat()
        date_end = date(year + 1, end_month, 28).isoformat()

    return {
        "scene_count": int(rng.integers(4, 12)),
        "date_range": {"start": date_start, "end": date_end},
        "cloud_cover_pct": cloud_pct,
        "bands_available": ["B2", "B3", "B4", "B8", "B11", "B12", "NDVI", "NDWI", "NDBI"],
        "ndvi_mean": ndvi_mean,
        "ndwi_mean": ndwi_mean,
        "ndbi_mean": ndbi_mean,
        # Per-band mean reflectance (scaled 0-1, illustrative)
        "B2": round(float(rng.normal(0.08, 0.01)), 4),
        "B3": round(float(rng.normal(0.10, 0.01)), 4),
        "B4": round(float(rng.normal(0.12, 0.015)), 4),
        "B8": round(float(rng.normal(0.35, 0.03)), 4),
        "B11": round(float(rng.normal(0.20, 0.02)), 4),
        "B12": round(float(rng.normal(0.15, 0.02)), 4),
        "gee_available": False,
        "demo_data": True,
        "data_provenance": {
            "source": "SYNTHETIC_DEMO",
            "label_is_ground_truth": False,
            "label_year": None,
            "notes": (
                "DEMO DATA: Synthetic values generated from a deterministic seed. "
                "Not real satellite observations. "
                "Urbanisation trend is illustrative only."
            ),
        },
    }


def get_imagery_availability(
    bbox: list[float],
    year_start: int,
    year_end: int,
    start_month: int = 11,
    end_month: int = 3,
) -> list[dict]:
    """Return synthetic availability metadata for the year range."""
    results = []
    for year in range(year_start, year_end + 1):
        rng = np.random.default_rng(_seed_from(bbox, year))
        results.append({
            "year": year,
            "available": year >= 2017,
            "scene_count": int(rng.integers(4, 14)) if year >= 2017 else 0,
            "cloud_cover_pct": round(abs(float(rng.normal(8, 3))), 1),
            "date_range": {
                "start": date(year, start_month, 1).isoformat(),
                "end": date(year if end_month >= start_month else year + 1, end_month, 28).isoformat(),
            },
            "gee_available": False,
            "demo_data": True,
        })
    return results

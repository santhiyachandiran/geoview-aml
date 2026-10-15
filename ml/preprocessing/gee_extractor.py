"""
GeoChange AI — GEE Pixel Extraction for ML Training

Downloads Sentinel-2 composites and ESA WorldCover labels as pixel arrays
suitable for supervised classification. Requires authenticated GEE.

Usage (after earthengine authenticate):
    from ml.preprocessing.gee_extractor import extract_training_data
    data = extract_training_data(
        bbox=[76.8, 10.9, 77.2, 11.1],
        composite_year=2021,
        out_dir="ml/data/processed/coimbatore_2021"
    )

Date convention
---------------
GEE filterDate(start, end) is EXCLUSIVE of end.  This module always sets
end_date to the first day of the month after end_month so that every day of
end_month is included.  Example: end_month=2 → end_date=YYYY-03-01.

Missing-label handling
----------------------
Pixels whose WorldCover label is missing (None / NaN) or whose raw code is
not in WC_CLASS_MAP are EXCLUDED, not assigned to a default class.
The GEE remap uses defaultValue=-1 as a sentinel so unmapped codes surface
as -1 rather than being silently forced into class 3 (bare).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# WorldCover 2021 v200 → 4-class mapping
# Only codes in this dict are valid; everything else is excluded.
WC_CLASS_MAP: dict[int, int] = {
    10: 1,   # Tree cover         → vegetation
    20: 1,   # Shrubland          → vegetation
    30: 1,   # Grassland          → vegetation
    40: 1,   # Cropland           → vegetation
    50: 0,   # Built-up           → built_up
    60: 3,   # Bare / sparse      → bare
    70: 3,   # Snow / ice         → bare
    80: 2,   # Permanent water    → water
    90: 3,   # Herbaceous wetland → bare
    95: 1,   # Mangroves          → vegetation
    100: 3,  # Moss / lichen      → bare
}

# Sentinel value returned for unmapped / no-data WorldCover pixels.
# Must be outside [0, 3] so it is caught by the _VALID_LABELS check below.
_WC_SENTINEL: int = -1

_VALID_LABELS: set[int] = {0, 1, 2, 3}

CLASS_NAMES: list[str] = ["built_up", "vegetation", "water", "bare"]
BANDS: list[str] = ["B2", "B3", "B4", "B8", "B11", "B12"]


# ── GEE helpers ────────────────────────────────────────────────────────────

def _require_gee():
    try:
        import ee
        if not ee.data._initialized:  # type: ignore[attr-defined]
            raise RuntimeError(
                "GEE not initialised. Call gee_service.authenticate_gee() first."
            )
        return ee
    except AttributeError:
        import ee
        return ee


def _scl_mask(image):
    """Mask cloud, cloud-shadow, and cirrus pixels using the SCL band."""
    import ee
    scl = image.select("SCL")
    mask = (
        scl.neq(3)    # cloud shadow
        .And(scl.neq(8))   # medium-probability cloud
        .And(scl.neq(9))   # high-probability cloud
        .And(scl.neq(10))  # thin cirrus
        .And(scl.neq(11))  # snow / ice
    )
    return image.updateMask(mask)


def _add_indices(image):
    """Append NDVI, NDWI, NDBI bands to a Sentinel-2 image."""
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndwi = image.normalizedDifference(["B3", "B11"]).rename("NDWI")
    ndbi = image.normalizedDifference(["B11", "B8"]).rename("NDBI")
    return image.addBands([ndvi, ndwi, ndbi])


def _exclusive_end_date(year: int, end_month: int) -> str:
    """
    Return the exclusive end date for GEE filterDate covering all of end_month.

    GEE filterDate is [start, end) — end is exclusive.
    To include every day of end_month we use the 1st of the following month.

    Examples:
        end_month=2, year=2022  →  "2022-03-01"
        end_month=12, year=2021 →  "2022-01-01"
    """
    if end_month == 12:
        return f"{year + 1}-01-01"
    return f"{year}-{end_month + 1:02d}-01"


# ── Main extraction function ────────────────────────────────────────────────

def extract_training_data(
    bbox: list[float],
    composite_year: int,
    out_dir: str | Path,
    start_month: int = 11,
    end_month: int = 3,
    max_cloud_pct: int = 20,
    scale_m: int = 10,
    max_pixels: int = 500_000,
    seed: int = 42,
) -> dict:
    """
    Extract pixel-level Sentinel-2 features and ESA WorldCover 2021 labels.

    Pixels with missing or unmapped WorldCover labels are EXCLUDED.
    The seasonal date window is applied with GEE's exclusive end-date
    convention so that all days in end_month are captured.

    Args:
        bbox: [west, south, east, north] in EPSG:4326
        composite_year: Year for the seasonal start (window may cross year boundary)
        out_dir: Directory to save extracted arrays
        start_month: First month of the seasonal window (default 11 = November)
        end_month:   Last month of the seasonal window  (default  3 = March)
                     The actual GEE end date is the 1st of end_month+1.
        max_cloud_pct: Maximum CLOUDY_PIXEL_PERCENTAGE per scene
        scale_m: Pixel resolution in metres
        max_pixels: Maximum pixels to sample from GEE
        seed: Random seed for pixel sampling

    Returns:
        {
          "X": (N, 9) float32 feature array,
          "y": (N,) int32 label array   (values in {0,1,2,3} only),
          "rows": (N,) int pixel row coordinates,
          "cols": (N,) int pixel column coordinates,
          "feature_names": list[str],
          "class_names": list[str],
          "metadata": dict,
        }
    """
    ee = _require_gee()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    west, south, east, north = bbox
    aoi = ee.Geometry.Rectangle([west, south, east, north])

    # ── Seasonal date window ──────────────────────────────────
    date_start = f"{composite_year}-{start_month:02d}-01"

    # If end_month < start_month the window crosses a year boundary
    end_year = composite_year + 1 if end_month < start_month else composite_year
    date_end_exclusive = _exclusive_end_date(end_year, end_month)

    # Human-readable description of the inclusive range (for metadata)
    import calendar
    last_day = calendar.monthrange(end_year, end_month)[1]
    date_end_inclusive_desc = f"{end_year}-{end_month:02d}-{last_day:02d}"

    logger.info(
        f"Seasonal window: {date_start} to {date_end_inclusive_desc} "
        f"(GEE filterDate end = {date_end_exclusive}, exclusive)"
    )

    # ── Build Sentinel-2 composite ────────────────────────────
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(date_start, date_end_exclusive)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud_pct))
        .map(_scl_mask)
        .map(_add_indices)
    )

    scene_count = int(collection.size().getInfo())
    logger.info(f"Scenes in composite: {scene_count}")

    if scene_count == 0:
        raise ValueError(
            f"No Sentinel-2 scenes found for {date_start} – {date_end_inclusive_desc} "
            f"with CLOUDY_PIXEL_PERCENTAGE < {max_cloud_pct}. "
            f"Try increasing max_cloud_pct or widening the date range."
        )

    composite = collection.median()
    all_bands = BANDS + ["NDVI", "NDWI", "NDBI"]
    s2_image = composite.select(all_bands)

    # ── WorldCover 2021 labels ────────────────────────────────
    wc_raw = ee.ImageCollection("ESA/WorldCover/v200").first().clip(aoi)

    # Remap to 4-class scheme.
    # defaultValue=_WC_SENTINEL means any WorldCover code NOT in WC_CLASS_MAP
    # (including masked / no-data pixels that GEE fills with 0 before remapping)
    # is flagged for exclusion rather than silently assigned to a class.
    from_vals = list(WC_CLASS_MAP.keys())
    to_vals   = list(WC_CLASS_MAP.values())
    wc_4class = wc_raw.remap(
        from_vals, to_vals, defaultValue=_WC_SENTINEL
    ).rename("label")

    stacked = s2_image.addBands(wc_4class)

    # ── Sample pixels ─────────────────────────────────────────
    # GEE's getInfo() aborts if a single call returns > 5 000 features.
    # We work around this by fetching in batches with .toList(size, offset)
    # and calling getInfo() once per batch.
    #
    # Batch size is kept well under the 5 000 limit.
    # The outer loop runs until we have collected max_pixels features or the
    # collection is exhausted, whichever comes first.
    GEE_BATCH_SIZE: int = 4_500   # safely under the 5 000 element limit

    logger.info(
        f"Sampling up to {max_pixels} pixels at {scale_m} m "
        f"(batch size = {GEE_BATCH_SIZE}) …"
    )
    sample = stacked.sample(
        region=aoi,
        scale=scale_m,
        numPixels=max_pixels,
        seed=seed,
        geometries=True,   # include lat/lon for spatial split
    )

    features_list: list[dict] = []
    offset = 0
    while len(features_list) < max_pixels:
        batch_target = min(GEE_BATCH_SIZE, max_pixels - len(features_list))
        try:
            batch = sample.toList(batch_target, offset).getInfo()
        except Exception as exc:
            # If GEE itself raises the 5 000-element error on a batch it
            # means this batch is unexpectedly large; halve and retry once.
            if "5000" in str(exc) and GEE_BATCH_SIZE > 500:
                logger.warning(
                    f"GEE element limit hit at offset {offset}; "
                    f"halving batch size to {GEE_BATCH_SIZE // 2} and retrying."
                )
                batch = sample.toList(GEE_BATCH_SIZE // 2, offset).getInfo()
            else:
                raise

        if not batch:
            break   # collection exhausted

        features_list.extend(batch)
        offset += len(batch)
        logger.debug(f"  Fetched {len(features_list)} / {max_pixels} pixels …")

        if len(batch) < batch_target:
            break   # fewer items returned than requested → end of collection

    n_sampled = len(features_list)
    logger.info(f"Raw pixels from GEE: {n_sampled}")

    if n_sampled == 0:
        raise ValueError(
            "GEE returned 0 pixels. Check bbox, date range, and API quota."
        )

    # ── Parse GEE response ────────────────────────────────────
    band_arrays: dict[str, list] = {b: [] for b in all_bands}
    raw_labels: list[int | None] = []
    lats: list[float] = []
    lons: list[float] = []

    for feat in features_list:
        props = feat["properties"]
        geom  = feat["geometry"]["coordinates"]   # [lon, lat]
        for b in all_bands:
            band_arrays[b].append(props.get(b, np.nan))

        raw = props.get("label")
        # raw is None when GEE masked the pixel; keep as None for explicit handling
        raw_labels.append(int(raw) if raw is not None else None)
        lons.append(float(geom[0]))
        lats.append(float(geom[1]))

    # ── Build arrays ──────────────────────────────────────────
    pixel_data = {
        b: np.array(band_arrays[b], dtype=np.float32) for b in all_bands
    }
    # Use np.nan for missing labels so we can detect them with isfinite
    y_raw = np.array(
        [float(v) if v is not None else np.nan for v in raw_labels],
        dtype=np.float64,
    )
    lat_arr = np.array(lats, dtype=np.float64)
    lon_arr = np.array(lons, dtype=np.float64)

    # ── Filter 1: remove pixels with any NaN band value ──────
    band_valid = np.all(
        np.column_stack([np.isfinite(pixel_data[b]) for b in all_bands]),
        axis=1,
    )
    n_band_invalid = int((~band_valid).sum())

    # ── Filter 2: remove pixels with missing or invalid labels ─
    # Missing  → y_raw is NaN  (GEE returned None)
    # Sentinel → y_raw == _WC_SENTINEL  (unmapped WorldCover code)
    # Out of range → not in {0,1,2,3}
    label_valid = (
        np.isfinite(y_raw)
        & (y_raw != _WC_SENTINEL)
        & np.isin(y_raw.astype(int), list(_VALID_LABELS))
    )
    n_label_invalid = int((~label_valid & band_valid).sum())

    valid = band_valid & label_valid
    n_kept = int(valid.sum())
    n_removed = n_sampled - n_kept

    logger.info(
        f"Pixel filtering: {n_sampled} sampled → "
        f"{n_band_invalid} removed (NaN bands) + "
        f"{n_label_invalid} removed (missing/unmapped labels) → "
        f"{n_kept} kept"
    )

    if n_kept == 0:
        raise ValueError(
            f"All {n_sampled} sampled pixels were invalid after filtering. "
            "Check cloud masking thresholds, WorldCover coverage for the bbox, "
            "and GEE asset availability."
        )

    for b in all_bands:
        pixel_data[b] = pixel_data[b][valid]
    y         = y_raw[valid].astype(np.int32)
    lat_arr   = lat_arr[valid]
    lon_arr   = lon_arr[valid]

    # ── Build feature matrix ──────────────────────────────────
    from ml.preprocessing.feature_builder import stack_features, FEATURE_NAMES
    X = stack_features(pixel_data)

    # ── Approximate row/col from lat/lon ──────────────────────
    # These are relative raster coordinates used only for spatial block
    # assignment — not absolute CRS coordinates.
    resolution_deg = scale_m / 111_000.0
    rows = np.floor((north - lat_arr) / resolution_deg).astype(int)
    cols = np.floor((lon_arr - west)  / resolution_deg).astype(int)

    # ── Class distribution ────────────────────────────────────
    unique, counts = np.unique(y, return_counts=True)
    class_dist = {
        CLASS_NAMES[int(c)]: int(n)
        for c, n in zip(unique, counts)
        if int(c) < len(CLASS_NAMES)
    }
    logger.info(f"Class distribution (kept pixels): {class_dist}")

    # ── Save to disk ──────────────────────────────────────────
    np.save(out_path / "X.npy",    X)
    np.save(out_path / "y.npy",    y)
    np.save(out_path / "rows.npy", rows)
    np.save(out_path / "cols.npy", cols)

    metadata = {
        "bbox": bbox,
        "composite_year": composite_year,
        # Exact dates used with GEE (end is exclusive per GEE convention)
        "date_start": date_start,
        "date_end_exclusive": date_end_exclusive,
        "date_end_inclusive_desc": date_end_inclusive_desc,
        "date_convention": "GEE filterDate end is exclusive; all days of end_month included",
        "scene_count": scene_count,
        # Pixel counts
        "n_sampled": n_sampled,
        "gee_batch_size": GEE_BATCH_SIZE,
        "n_removed_nan_bands": n_band_invalid,
        "n_removed_missing_labels": n_label_invalid,
        "n_pixels": n_kept,
        "n_features": int(X.shape[1]),
        "feature_names": FEATURE_NAMES,
        "class_names": CLASS_NAMES,
        "class_distribution": class_dist,
        "scale_m": scale_m,
        "max_cloud_pct": max_cloud_pct,
        # Label provenance
        "labels_source": "ESA WorldCover v200 (2021)",
        "labels_license": "CC BY 4.0",
        "labels_citation": (
            "Zanaga et al. (2022). ESA WorldCover 10 m 2021 v200. "
            "Zenodo. https://doi.org/10.5281/zenodo.7254221"
        ),
        "missing_label_handling": (
            "Pixels with None, NaN, or unmapped WorldCover codes are excluded. "
            f"GEE remap defaultValue={_WC_SENTINEL} (sentinel, not a real class)."
        ),
        "temporal_caveat": (
            "WorldCover labels are from 2021. Classifying composites from other "
            "years with these labels introduces temporal label mismatch — a known "
            "limitation that must be disclosed in any published results."
        ),
    }
    (out_path / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    logger.info(f"Extraction complete. Saved to {out_path}")

    return {
        "X":             X,
        "y":             y,
        "rows":          rows,
        "cols":          cols,
        "feature_names": FEATURE_NAMES,
        "class_names":   CLASS_NAMES,
        "metadata":      metadata,
    }


def load_extracted_data(data_dir: str | Path) -> dict:
    """
    Load previously extracted pixel arrays from disk.
    Returns the same structure as extract_training_data().
    """
    p = Path(data_dir)
    required = ["X.npy", "y.npy", "rows.npy", "cols.npy", "metadata.json"]
    for f in required:
        if not (p / f).exists():
            raise FileNotFoundError(
                f"Missing '{f}' in {p}. Run extract_training_data() first."
            )

    from ml.preprocessing.feature_builder import FEATURE_NAMES
    meta = json.loads((p / "metadata.json").read_text(encoding="utf-8"))

    return {
        "X":             np.load(p / "X.npy"),
        "y":             np.load(p / "y.npy"),
        "rows":          np.load(p / "rows.npy"),
        "cols":          np.load(p / "cols.npy"),
        "feature_names": FEATURE_NAMES,
        "class_names":   meta["class_names"],
        "metadata":      meta,
    }

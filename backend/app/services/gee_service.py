"""
GeoChange AI — Google Earth Engine Service

Wraps the Earth Engine Python API. If authentication fails the module sets
GEE_AVAILABLE=False and every function returns a graceful fallback dict so
the rest of the application continues working.

Authentication (run once, outside this module):
    earthengine authenticate
Then set in .env:
    GEE_PROJECT_ID=your-cloud-project-id
    GEE_AUTH_MODE=user
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

# Module-level flag — set by authenticate_gee()
GEE_AVAILABLE: bool = False
_GEE_ERROR: str = "Not yet authenticated"

try:
    import ee  # type: ignore
    _EE_IMPORTED = True
except ImportError:
    _EE_IMPORTED = False
    logger.warning("earthengine-api package not importable. GEE disabled.")


def authenticate_gee(project_id: str = "") -> bool:
    """
    Try to initialise Earth Engine. Sets GEE_AVAILABLE at module level.
    Never raises — failures are logged and returned as False.

    Args:
        project_id: Google Cloud project that has Earth Engine enabled.

    Returns:
        True if initialisation succeeded, False otherwise.
    """
    global GEE_AVAILABLE, _GEE_ERROR

    if not _EE_IMPORTED:
        _GEE_ERROR = "earthengine-api not installed"
        GEE_AVAILABLE = False
        return False

    if not project_id:
        _GEE_ERROR = "GEE_PROJECT_ID not set in environment"
        GEE_AVAILABLE = False
        return False

    try:
        ee.Initialize(project=project_id)
        GEE_AVAILABLE = True
        _GEE_ERROR = ""
        logger.info(f"GEE initialised — project={project_id}")
        return True
    except Exception as exc:
        _GEE_ERROR = str(exc)
        GEE_AVAILABLE = False
        logger.warning(f"GEE initialisation failed: {exc}")
        return False


def _scl_cloud_mask(image: Any) -> Any:
    """Mask cloud, cloud shadow, cirrus pixels using the SCL band."""
    scl = image.select("SCL")
    # SCL classes to mask: 3=cloud shadow, 8=medium cloud, 9=high cloud, 10=thin cirrus, 11=snow
    mask = (
        scl.neq(3)
        .And(scl.neq(8))
        .And(scl.neq(9))
        .And(scl.neq(10))
        .And(scl.neq(11))
    )
    return image.updateMask(mask)


def _add_indices(image: Any) -> Any:
    """Add NDVI, NDWI, NDBI bands to a Sentinel-2 image."""
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndwi = image.normalizedDifference(["B3", "B11"]).rename("NDWI")
    ndbi = image.normalizedDifference(["B11", "B8"]).rename("NDBI")
    return image.addBands([ndvi, ndwi, ndbi])


def _season_date_range(year: int, start_month: int, end_month: int) -> tuple[str, str]:
    """
    Build start/end date strings for a seasonal window.
    Handles cross-year windows, e.g. November–February (Nov year → Feb year+1).
    """
    start = date(year, start_month, 1)
    if end_month < start_month:
        end_year = year + 1
    else:
        end_year = year
    # Last day of end_month
    if end_month == 12:
        end = date(end_year, 12, 31)
    else:
        end = date(end_year, end_month + 1, 1) - timedelta(days=1)
    return start.isoformat(), end.isoformat()


@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=False,
)
def _safe_gee_call(fn, *args, **kwargs):
    """Wrap a GEE call with retry logic."""
    return fn(*args, **kwargs)


def get_sentinel2_composite(
    bbox: list[float],
    year: int,
    start_month: int = 11,
    end_month: int = 3,
    max_cloud_pct: int = 20,
) -> dict:
    """
    Build a cloud-masked seasonal median composite for the given bbox and year.

    Args:
        bbox: [west, south, east, north] in EPSG:4326
        year: Year for the start of the seasonal window
        start_month: Start month (1-12)
        end_month: End month (1-12). If < start_month, window crosses year boundary.
        max_cloud_pct: Maximum scene-level cloud cover percentage to include

    Returns:
        dict with keys: scene_count, date_range, cloud_cover_pct, bands_available,
        ndvi_mean, ndwi_mean, ndbi_mean, gee_available, demo_data, data_provenance

    Raises:
        ValueError: for invalid dates or bbox
    """
    # Input validation
    if len(bbox) != 4:
        raise ValueError(f"bbox must have 4 elements [W,S,E,N], got {len(bbox)}")
    west, south, east, north = bbox
    if west >= east or south >= north:
        raise ValueError(f"Invalid bbox: [{west},{south},{east},{north}]")
    if not (1 <= start_month <= 12 and 1 <= end_month <= 12):
        raise ValueError(f"Months must be 1-12, got start={start_month}, end={end_month}")
    if year < 2017:
        raise ValueError("Sentinel-2 archive starts 2017. Use Landsat for earlier dates.")

    if not GEE_AVAILABLE:
        return {
            "scene_count": 0,
            "date_range": None,
            "cloud_cover_pct": None,
            "bands_available": [],
            "ndvi_mean": None,
            "ndwi_mean": None,
            "ndbi_mean": None,
            "gee_available": False,
            "demo_data": False,
            "error": _GEE_ERROR,
            "data_provenance": {
                "source": "COPERNICUS/S2_SR_HARMONIZED",
                "gee_available": False,
                "notes": "GEE not authenticated. Run earthengine authenticate.",
            },
        }

    try:
        date_start, date_end = _season_date_range(year, start_month, end_month)
        aoi = ee.Geometry.Rectangle([west, south, east, north])

        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(aoi)
            .filterDate(date_start, date_end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud_pct))
            .map(_scl_cloud_mask)
            .map(_add_indices)
        )

        scene_count = int(collection.size().getInfo())

        if scene_count == 0:
            return {
                "scene_count": 0,
                "date_range": {"start": date_start, "end": date_end},
                "cloud_cover_pct": None,
                "bands_available": [],
                "ndvi_mean": None,
                "ndwi_mean": None,
                "ndbi_mean": None,
                "gee_available": True,
                "demo_data": False,
                "warning": "No scenes found for this date range and cloud threshold.",
                "data_provenance": {
                    "source": "COPERNICUS/S2_SR_HARMONIZED",
                    "gee_available": True,
                    "label_is_ground_truth": False,
                    "notes": "Empty collection returned.",
                },
            }

        composite = collection.median()
        bands = ["B2", "B3", "B4", "B8", "B11", "B12", "NDVI", "NDWI", "NDBI"]

        # Sample mean index values over AOI for metadata
        means = (
            composite.select(["NDVI", "NDWI", "NDBI"])
            .reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=aoi,
                scale=100,
                maxPixels=1e8,
            )
            .getInfo()
        )

        # Mean cloud cover across scenes
        cloud_mean = collection.aggregate_mean("CLOUDY_PIXEL_PERCENTAGE").getInfo()

        return {
            "scene_count": scene_count,
            "date_range": {"start": date_start, "end": date_end},
            "cloud_cover_pct": round(cloud_mean, 2) if cloud_mean is not None else None,
            "bands_available": bands,
            "ndvi_mean": round(means.get("NDVI", 0), 4),
            "ndwi_mean": round(means.get("NDWI", 0), 4),
            "ndbi_mean": round(means.get("NDBI", 0), 4),
            "gee_available": True,
            "demo_data": False,
            "composite_image": composite,  # ee.Image object for further processing
            "data_provenance": {
                "source": "COPERNICUS/S2_SR_HARMONIZED",
                "collection_version": "Harmonized",
                "date_range": f"{date_start} to {date_end}",
                "cloud_filter_pct": max_cloud_pct,
                "cloud_mask": "SCL bands 3,8,9,10,11 excluded",
                "resolution_m": 10,
                "gee_available": True,
                "label_is_ground_truth": False,
                "notes": (
                    "Median composite over seasonal window. "
                    "Seasonal selection (Nov-Feb) reduces monsoon cloud contamination. "
                    "NDVI/NDWI/NDBI are proxy indicators, not validated land-cover labels."
                ),
            },
        }

    except Exception as exc:
        logger.error(f"GEE composite failed: {exc}")
        return {
            "scene_count": 0,
            "date_range": None,
            "cloud_cover_pct": None,
            "bands_available": [],
            "ndvi_mean": None,
            "ndwi_mean": None,
            "ndbi_mean": None,
            "gee_available": False,
            "demo_data": False,
            "error": str(exc),
            "data_provenance": {
                "source": "COPERNICUS/S2_SR_HARMONIZED",
                "gee_available": False,
                "notes": f"GEE call failed: {exc}",
            },
        }


def get_imagery_availability(
    bbox: list[float],
    year_start: int,
    year_end: int,
    start_month: int = 11,
    end_month: int = 3,
) -> list[dict]:
    """
    List Sentinel-2 availability for each year in range without downloading data.

    Returns:
        List of dicts: {year, available, scene_count, cloud_cover_pct, date_range}
    """
    results = []
    for year in range(year_start, year_end + 1):
        if not GEE_AVAILABLE:
            results.append({
                "year": year,
                "available": False,
                "scene_count": None,
                "cloud_cover_pct": None,
                "date_range": None,
                "gee_available": False,
            })
            continue
        try:
            date_start, date_end = _season_date_range(year, start_month, end_month)
            if year < 2017:
                results.append({"year": year, "available": False,
                                 "scene_count": 0, "note": "Pre-Sentinel-2 era"})
                continue
            aoi = ee.Geometry.Rectangle(bbox[:4])
            count = (
                ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterBounds(aoi)
                .filterDate(date_start, date_end)
                .size()
                .getInfo()
            )
            results.append({
                "year": year,
                "available": count > 0,
                "scene_count": count,
                "date_range": {"start": date_start, "end": date_end},
                "gee_available": True,
            })
        except Exception as exc:
            results.append({
                "year": year,
                "available": False,
                "scene_count": None,
                "error": str(exc),
                "gee_available": False,
            })
    return results

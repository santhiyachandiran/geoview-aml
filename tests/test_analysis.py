"""
GeoChange AI — Tests: Analysis Service and Demo Service

All GEE calls are mocked. No real network calls.
"""

import os
import sys

_HERE = os.path.abspath(os.path.dirname(__file__))
_BACKEND = os.path.abspath(os.path.join(_HERE, "..", "backend"))
_PROJECT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _BACKEND)
sys.path.insert(0, _PROJECT)

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
os.environ["SQLITE_DB_PATH"] = os.path.join(_BACKEND, "geochange_test.db")

import numpy as np
import pytest

from app.services.analysis_service import (
    compute_index_differences,
    detect_vegetation_to_buildup,
    compute_area_stats,
    load_worldcover_labels,
    build_transition_matrix,
    export_geojson,
    export_csv,
)
from app.services import demo_service


# ── Fixtures ──────────────────────────────────────────────────

COMPOSITE_T1 = {
    "ndvi_mean": 0.42, "ndwi_mean": -0.10, "ndbi_mean": -0.18,
    "scene_count": 8, "demo_data": True,
    "B2": 0.08, "B3": 0.10, "B4": 0.12, "B8": 0.36, "B11": 0.20, "B12": 0.15,
}
COMPOSITE_T2 = {
    "ndvi_mean": 0.28, "ndwi_mean": -0.13, "ndbi_mean": -0.04,
    "scene_count": 7, "demo_data": True,
    "B2": 0.09, "B3": 0.11, "B4": 0.13, "B8": 0.30, "B11": 0.22, "B12": 0.17,
}
BBOX = [76.8, 10.9, 77.2, 11.1]


# ── Index differences ─────────────────────────────────────────

class TestIndexDifferences:
    def test_keys_present(self):
        result = compute_index_differences(COMPOSITE_T1, COMPOSITE_T2)
        for k in ("delta_ndvi", "delta_ndwi", "delta_ndbi", "data_provenance"):
            assert k in result

    def test_delta_ndvi_correct(self):
        result = compute_index_differences(COMPOSITE_T1, COMPOSITE_T2)
        expected = round(0.28 - 0.42, 4)
        assert abs(result["delta_ndvi"] - expected) < 0.001

    def test_delta_ndbi_positive_when_buildup(self):
        result = compute_index_differences(COMPOSITE_T1, COMPOSITE_T2)
        # NDBI increased → built-up proxy
        assert result["delta_ndbi"] > 0

    def test_none_values_handled(self):
        c1 = {**COMPOSITE_T1, "ndvi_mean": None}
        result = compute_index_differences(c1, COMPOSITE_T2)
        assert result["delta_ndvi"] is None

    def test_provenance_label_not_ground_truth(self):
        result = compute_index_differences(COMPOSITE_T1, COMPOSITE_T2)
        assert result["data_provenance"]["label_is_ground_truth"] is False


# ── Vegetation-to-built-up detection ─────────────────────────

class TestVegToBuildup:
    def test_flag_true_when_both_thresholds_met(self):
        result = detect_vegetation_to_buildup(-0.15, 0.15)
        assert result["flag"] is True

    def test_flag_false_when_ndvi_stable(self):
        result = detect_vegetation_to_buildup(0.05, -0.05)
        assert result["flag"] is False

    def test_flag_false_when_ndbi_stable(self):
        result = detect_vegetation_to_buildup(-0.15, 0.05)
        assert result["flag"] is False

    def test_none_inputs_return_unknown(self):
        result = detect_vegetation_to_buildup(None, None)
        assert result["flag"] is False
        assert result["confidence_label"] == "unknown"

    def test_confidence_label_present(self):
        result = detect_vegetation_to_buildup(-0.15, 0.15)
        assert "confidence_label" in result
        assert result["confidence_label"] != "unknown"


# ── Area statistics ───────────────────────────────────────────

class TestAreaStats:
    def test_area_ha_for_100_pixels_at_10m(self):
        result = compute_area_stats(100, 1000, resolution_m=10)
        # 100 pixels × 100 m² = 10,000 m² = 1 ha
        assert abs(result["area_ha"] - 1.0) < 0.01

    def test_area_km2_correct(self):
        result = compute_area_stats(100, 1000, resolution_m=10)
        assert abs(result["area_km2"] - 0.01) < 0.0001

    def test_area_pct_correct(self):
        result = compute_area_stats(250, 1000)
        assert abs(result["area_pct"] - 25.0) < 0.01

    def test_zero_total_pixels_no_error(self):
        result = compute_area_stats(0, 0)
        assert result["area_pct"] == 0.0


# ── WorldCover ────────────────────────────────────────────────

class TestWorldCoverLabels:
    def test_returns_dict(self):
        result = load_worldcover_labels(BBOX)
        assert isinstance(result, dict)

    def test_has_available_key(self):
        result = load_worldcover_labels(BBOX)
        assert "available" in result

    def test_label_is_not_ground_truth(self):
        result = load_worldcover_labels(BBOX)
        assert result["data_provenance"]["label_is_ground_truth"] is False

    def test_unavailable_without_gee(self):
        # GEE is not authenticated in tests, so available must be False
        result = load_worldcover_labels(BBOX)
        assert result["available"] is False


# ── Transition matrix ─────────────────────────────────────────

class TestTransitionMatrix:
    def _make_labels(self):
        rng = np.random.default_rng(42)
        t1 = rng.integers(0, 4, 200)
        t2 = t1.copy()
        # Force some transitions
        t2[t1 == 1] = rng.choice([0, 1], 200)[t1 == 1]
        return t1, t2

    def test_returns_dict(self):
        t1, t2 = self._make_labels()
        result = build_transition_matrix(t1, t2)
        assert isinstance(result, dict)

    def test_has_matrix_key(self):
        t1, t2 = self._make_labels()
        result = build_transition_matrix(t1, t2)
        assert "matrix" in result
        assert len(result["matrix"]) == 4

    def test_matrix_is_square(self):
        t1, t2 = self._make_labels()
        result = build_transition_matrix(t1, t2)
        m = result["matrix"]
        assert all(len(row) == 4 for row in m)

    def test_transitions_list_present(self):
        t1, t2 = self._make_labels()
        result = build_transition_matrix(t1, t2)
        assert "transitions" in result
        assert isinstance(result["transitions"], list)


# ── Demo service ──────────────────────────────────────────────

class TestDemoService:
    def test_composite_keys_present(self):
        result = demo_service.get_sentinel2_composite(BBOX, 2020)
        for k in ("scene_count", "ndvi_mean", "ndwi_mean", "ndbi_mean",
                  "gee_available", "demo_data", "bands_available", "data_provenance"):
            assert k in result, f"Missing key: {k}"

    def test_demo_data_flag_true(self):
        result = demo_service.get_sentinel2_composite(BBOX, 2020)
        assert result["demo_data"] is True

    def test_gee_available_false(self):
        result = demo_service.get_sentinel2_composite(BBOX, 2020)
        assert result["gee_available"] is False

    def test_availability_returns_list(self):
        result = demo_service.get_imagery_availability(BBOX, 2018, 2022)
        assert isinstance(result, list)
        assert len(result) == 5

    def test_availability_years_correct(self):
        result = demo_service.get_imagery_availability(BBOX, 2019, 2021)
        years = [r["year"] for r in result]
        assert years == [2019, 2020, 2021]

    def test_ndvi_declines_over_time(self):
        # Urbanisation trend: NDVI should be lower in 2024 than 2018 on average
        v2018 = demo_service.get_sentinel2_composite(BBOX, 2018)["ndvi_mean"]
        v2024 = demo_service.get_sentinel2_composite(BBOX, 2024)["ndvi_mean"]
        assert v2024 < v2018

    def test_reproducible_with_same_inputs(self):
        r1 = demo_service.get_sentinel2_composite(BBOX, 2020)
        r2 = demo_service.get_sentinel2_composite(BBOX, 2020)
        assert r1["ndvi_mean"] == r2["ndvi_mean"]

    def test_label_not_ground_truth(self):
        result = demo_service.get_sentinel2_composite(BBOX, 2020)
        assert result["data_provenance"]["label_is_ground_truth"] is False


# ── Export helpers ────────────────────────────────────────────

class TestExports:
    def test_geojson_export(self, tmp_path):
        data = {"delta_ndvi": -0.14, "study_area": "coimbatore", "demo_data": True}
        out = export_geojson(data, tmp_path / "test.geojson")
        assert out.exists()
        import json
        content = json.loads(out.read_text())
        assert content["type"] == "FeatureCollection"

    def test_csv_export(self, tmp_path):
        data = {"delta_ndvi": -0.14, "study_area": "coimbatore", "year1": 2018}
        out = export_csv(data, tmp_path / "test.csv")
        assert out.exists()
        assert "delta_ndvi" in out.read_text()

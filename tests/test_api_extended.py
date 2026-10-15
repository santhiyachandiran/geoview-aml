"""
GeoChange AI — Tests: Extended API Endpoints
"""

import os
import sys

_HERE = os.path.abspath(os.path.dirname(__file__))
_BACKEND = os.path.abspath(os.path.join(_HERE, "..", "backend"))
_PROJECT = os.path.abspath(os.path.join(_HERE, ".."))
_TEST_DB = os.path.join(_BACKEND, "geochange_test.db")   # same path as test_health_api

sys.path.insert(0, _BACKEND)
sys.path.insert(0, _PROJECT)

os.environ["SQLITE_DB_PATH"] = _TEST_DB
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
os.environ.setdefault("GEE_PROJECT_ID", "")

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module", autouse=True)
def _init_db():
    # Use the same DB as test_health_api — just ensure it's initialized
    if not os.path.exists(_TEST_DB):
        from app.db.init_db import init_database
        init_database(_TEST_DB)
    yield
    # Don't delete — test_health_api owns the lifecycle when run together


@pytest.fixture(scope="module")
def client(_init_db):
    from app.main import app
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# ── Study Areas ───────────────────────────────────────────────

class TestStudyAreasEndpoint:
    def test_returns_200(self, client):
        r = client.get("/api/v1/study-areas")
        assert r.status_code == 200

    def test_returns_three_areas(self, client):
        r = client.get("/api/v1/study-areas")
        assert len(r.json()) == 3

    def test_each_area_has_required_fields(self, client):
        r = client.get("/api/v1/study-areas")
        for area in r.json():
            assert "slug" in area
            assert "name" in area
            assert "bbox_west" in area

    def test_coimbatore_present(self, client):
        r = client.get("/api/v1/study-areas")
        slugs = [a["slug"] for a in r.json()]
        assert "coimbatore" in slugs


# ── Imagery Availability ──────────────────────────────────────

class TestImageryAvailability:
    def test_valid_request_returns_200(self, client):
        r = client.get("/api/v1/imagery/availability",
                       params={"study_area": "coimbatore", "year_start": 2020, "year_end": 2022})
        assert r.status_code == 200

    def test_response_has_years(self, client):
        r = client.get("/api/v1/imagery/availability",
                       params={"study_area": "coimbatore", "year_start": 2020, "year_end": 2022})
        assert "years" in r.json()
        assert len(r.json()["years"]) == 3

    def test_unknown_study_area_returns_404(self, client):
        r = client.get("/api/v1/imagery/availability",
                       params={"study_area": "nonexistent", "year_start": 2020, "year_end": 2022})
        assert r.status_code == 404

    def test_invalid_year_range_returns_422(self, client):
        r = client.get("/api/v1/imagery/availability",
                       params={"study_area": "coimbatore", "year_start": 2024, "year_end": 2020})
        assert r.status_code == 422

    def test_demo_mode_flag_present(self, client):
        r = client.get("/api/v1/imagery/availability",
                       params={"study_area": "coimbatore", "year_start": 2020, "year_end": 2021})
        data = r.json()
        assert "demo_mode" in data


# ── Analysis: Indices ─────────────────────────────────────────

class TestAnalysisIndices:
    def test_valid_request_returns_200(self, client):
        r = client.post("/api/v1/analysis/indices",
                        json={"study_area": "coimbatore", "year1": 2018, "year2": 2024})
        assert r.status_code == 200

    def test_response_has_delta_fields(self, client):
        r = client.post("/api/v1/analysis/indices",
                        json={"study_area": "coimbatore", "year1": 2018, "year2": 2024})
        data = r.json()
        assert "delta_ndvi" in data
        assert "delta_ndwi" in data
        assert "delta_ndbi" in data

    def test_missing_study_area_returns_422(self, client):
        r = client.post("/api/v1/analysis/indices",
                        json={"year1": 2018, "year2": 2024})
        assert r.status_code == 422

    def test_year1_must_be_less_than_year2(self, client):
        r = client.post("/api/v1/analysis/indices",
                        json={"study_area": "coimbatore", "year1": 2024, "year2": 2018})
        assert r.status_code == 422

    def test_unknown_study_area_returns_404(self, client):
        r = client.post("/api/v1/analysis/indices",
                        json={"study_area": "atlantis", "year1": 2018, "year2": 2024})
        assert r.status_code == 404

    def test_provenance_label_is_not_ground_truth(self, client):
        r = client.post("/api/v1/analysis/indices",
                        json={"study_area": "coimbatore", "year1": 2018, "year2": 2024})
        data = r.json()
        assert data["data_provenance"]["label_is_ground_truth"] is False


# ── Analysis: Change Detection ────────────────────────────────

class TestChangeDetection:
    def test_valid_request_returns_200(self, client):
        r = client.post("/api/v1/analysis/change-detection",
                        json={"study_area": "coimbatore", "year1": 2018, "year2": 2024})
        assert r.status_code == 200

    def test_response_has_analysis_id(self, client):
        r = client.post("/api/v1/analysis/change-detection",
                        json={"study_area": "coimbatore", "year1": 2018, "year2": 2024})
        assert "analysis_id" in r.json()

    def test_result_retrievable_by_id(self, client):
        r1 = client.post("/api/v1/analysis/change-detection",
                         json={"study_area": "coimbatore", "year1": 2019, "year2": 2023})
        analysis_id = r1.json()["analysis_id"]
        r2 = client.get(f"/api/v1/analysis/results/{analysis_id}")
        assert r2.status_code == 200
        assert r2.json()["analysis_id"] == analysis_id

    def test_nonexistent_result_returns_404(self, client):
        r = client.get("/api/v1/analysis/results/00000000-0000-0000-0000-000000000000")
        assert r.status_code == 404


# ── Models & Experiments ─────────────────────────────────────

class TestModelsEndpoint:
    def test_returns_200(self, client):
        r = client.get("/api/v1/models")
        assert r.status_code == 200

    def test_response_has_models_list(self, client):
        data = client.get("/api/v1/models").json()
        assert "models" in data
        assert isinstance(data["models"], list)

    def test_count_matches_list_length(self, client):
        data = client.get("/api/v1/models").json()
        assert data["count"] == len(data["models"])


class TestExperimentsEndpoint:
    def test_returns_200(self, client):
        r = client.get("/api/v1/experiments")
        assert r.status_code == 200

    def test_response_has_runs_list(self, client):
        data = client.get("/api/v1/experiments").json()
        assert "runs" in data
        assert isinstance(data["runs"], list)

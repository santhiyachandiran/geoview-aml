"""
GeoChange AI — Tests: Database path consistency and fix verification.

These tests verify the root cause of the 503/500 errors is resolved:
- settings.resolved_db_path always returns the same absolute path
  regardless of which working directory Python is started from.
- The resolved path points to the real populated database, not an empty one.
- All API endpoints that need the DB use settings.resolved_db_path directly.
- Study-area endpoint returns data in demo mode without GEE credentials.
- Analysis endpoints return correct responses, not unhandled sqlite3 errors.
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
os.environ.setdefault("GEE_PROJECT_ID", "")

import sqlite3
import pytest
from pathlib import Path
from fastapi.testclient import TestClient


@pytest.fixture(scope="module", autouse=True)
def _init_test_db():
    test_db = os.environ["SQLITE_DB_PATH"]
    if os.path.exists(test_db):
        os.remove(test_db)
    from app.db.init_db import init_database
    init_database(test_db)
    yield
    if os.path.exists(test_db):
        os.remove(test_db)


@pytest.fixture(scope="module")
def client(_init_test_db):
    from app.main import app
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# ── Path resolution tests ─────────────────────────────────────

class TestResolvedDbPath:
    def test_resolved_db_path_is_absolute(self):
        """resolved_db_path must always be absolute."""
        from app.core.config import settings
        assert settings.resolved_db_path.is_absolute()

    def test_resolved_db_path_consistent_from_any_cwd(self, tmp_path):
        """
        Changing cwd must not change resolved_db_path.
        This was the root bug: relative paths resolved differently from
        backend/ vs project root.
        """
        from app.core.config import settings
        original_cwd = os.getcwd()
        try:
            path_from_project = settings.resolved_db_path
            os.chdir(tmp_path)          # totally different directory
            path_from_other = settings.resolved_db_path
            assert path_from_project == path_from_other, (
                f"Path changed with cwd!\n"
                f"  from project root: {path_from_project}\n"
                f"  from {tmp_path}:   {path_from_other}"
            )
        finally:
            os.chdir(original_cwd)

    def test_resolved_db_path_points_to_real_db(self):
        """The resolved path must point to a populated database (not a phantom 0-byte file)."""
        from app.core.config import settings
        db = settings.resolved_db_path
        assert db.exists(), f"DB does not exist at {db}"
        assert db.stat().st_size > 0, (
            f"DB at {db} is 0 bytes — this is the phantom file bug."
        )

    def test_no_phantom_db_at_project_root(self):
        """There must not be a 0-byte geochange.db in the project root."""
        phantom = Path(_PROJECT) / "geochange.db"
        if phantom.exists():
            assert phantom.stat().st_size > 0, (
                "Found a 0-byte phantom geochange.db in the project root. "
                "This is caused by the old relative path resolution bug."
            )

    def test_resolved_db_has_study_areas_table(self):
        """The database at resolved_db_path must have the study_areas table."""
        from app.core.config import settings
        conn = sqlite3.connect(str(settings.resolved_db_path))
        try:
            tables = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()}
        finally:
            conn.close()
        assert "study_areas" in tables, (
            f"study_areas table missing from {settings.resolved_db_path}"
        )

    def test_resolved_artifacts_dir_is_absolute(self):
        from app.core.config import settings
        assert settings.resolved_artifacts_dir.is_absolute()


# ── Study areas endpoint — demo mode ─────────────────────────

class TestStudyAreasInDemoMode:
    def test_returns_200_without_gee(self, client):
        """Must return 200, not 503, even without GEE credentials."""
        r = client.get("/api/v1/study-areas")
        assert r.status_code == 200, (
            f"Expected 200 in demo mode, got {r.status_code}: {r.text}"
        )

    def test_returns_three_areas(self, client):
        r = client.get("/api/v1/study-areas")
        areas = r.json()
        assert len(areas) == 3, f"Expected 3 study areas, got {len(areas)}"

    def test_coimbatore_bbox_correct(self, client):
        r = client.get("/api/v1/study-areas")
        coimb = next((a for a in r.json() if a["slug"] == "coimbatore"), None)
        assert coimb is not None
        assert abs(coimb["bbox_west"] - 76.8) < 0.001
        assert abs(coimb["bbox_east"] - 77.2) < 0.001

    def test_no_503_when_db_exists(self, client):
        """503 should only happen if DB file is missing, not in normal demo mode."""
        r = client.get("/api/v1/study-areas")
        assert r.status_code != 503, "Got 503 — DB path is wrong or DB is missing"


# ── Analysis endpoints — demo mode ───────────────────────────

class TestAnalysisInDemoMode:
    def test_indices_returns_200_not_500(self, client):
        """POST /analysis/indices must not raise unhandled sqlite3 errors."""
        r = client.post("/api/v1/analysis/indices",
                        json={"study_area": "coimbatore", "year1": 2018, "year2": 2024})
        assert r.status_code == 200, (
            f"Expected 200, got {r.status_code}: {r.text[:200]}"
        )

    def test_indices_response_has_delta_ndvi(self, client):
        r = client.post("/api/v1/analysis/indices",
                        json={"study_area": "coimbatore", "year1": 2018, "year2": 2024})
        assert r.status_code == 200
        data = r.json()
        assert "delta_ndvi" in data
        assert data["delta_ndvi"] is not None

    def test_indices_demo_data_flag_true(self, client):
        """In demo mode, demo_data must be True — never silently return wrong flag."""
        r = client.post("/api/v1/analysis/indices",
                        json={"study_area": "coimbatore", "year1": 2018, "year2": 2024})
        assert r.json()["demo_data"] is True

    def test_change_detection_returns_200_not_500(self, client):
        r = client.post("/api/v1/analysis/change-detection",
                        json={"study_area": "coimbatore", "year1": 2018, "year2": 2024})
        assert r.status_code == 200, (
            f"Expected 200, got {r.status_code}: {r.text[:200]}"
        )

    def test_change_detection_stores_result_in_db(self, client):
        r = client.post("/api/v1/analysis/change-detection",
                        json={"study_area": "coimbatore", "year1": 2019, "year2": 2023})
        assert r.status_code == 200
        analysis_id = r.json()["analysis_id"]
        r2 = client.get(f"/api/v1/analysis/results/{analysis_id}")
        assert r2.status_code == 200
        assert r2.json()["status"] == "completed"

    def test_unknown_study_area_returns_404_not_500(self, client):
        """sqlite3 OperationalError must not leak as 500 — must be caught as 404."""
        r = client.post("/api/v1/analysis/indices",
                        json={"study_area": "nonexistent_place", "year1": 2018, "year2": 2024})
        assert r.status_code == 404, (
            f"Expected 404 for unknown study_area, got {r.status_code}: {r.text[:200]}"
        )

    def test_imagery_availability_returns_200(self, client):
        r = client.get("/api/v1/imagery/availability",
                       params={"study_area": "coimbatore", "year_start": 2019, "year_end": 2022})
        assert r.status_code == 200, f"Got {r.status_code}: {r.text[:200]}"

    def test_imagery_availability_demo_mode_flag(self, client):
        r = client.get("/api/v1/imagery/availability",
                       params={"study_area": "coimbatore", "year_start": 2019, "year_end": 2022})
        assert r.json()["demo_mode"] is True

    def test_status_endpoint_shows_db_ok(self, client):
        r = client.get("/api/v1/status")
        assert r.status_code == 200
        db_component = next(
            (c for c in r.json()["components"] if c["name"] == "database"), None
        )
        assert db_component is not None
        assert db_component["status"] == "ok", (
            f"Database component not OK: {db_component}"
        )

"""
GeoChange AI — Tests: Health/Status API and Database Initialization.

Run from the project root with:
    .venv\Scripts\pytest tests/ -v

These tests use FastAPI's TestClient (synchronous) — no async needed.
A fresh test database is created before the module runs and removed after.
"""

import os
import sqlite3
import sys

import pytest

# ── Path setup (must be before any app imports) ────────────────
# Resolve absolute paths so they work regardless of cwd
_HERE = os.path.abspath(os.path.dirname(__file__))
_BACKEND = os.path.abspath(os.path.join(_HERE, "..", "backend"))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_TEST_DB = os.path.join(_BACKEND, "geochange_test.db")

sys.path.insert(0, _BACKEND)

os.environ["SQLITE_DB_PATH"] = _TEST_DB
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
# Ensure GEE is not configured so we can test the "unconfigured" path
os.environ.setdefault("GEE_PROJECT_ID", "")

# ── App imports (after path + env setup) ──────────────────────
from fastapi.testclient import TestClient
from app.main import app
from app.db.init_db import init_database, verify_database


# ── Module-scoped fixtures ─────────────────────────────────────

@pytest.fixture(scope="module", autouse=True)
def _init_test_db():
    """
    Create a clean test database before any test in this module runs.
    Removes it afterwards so each test run starts fresh.
    """
    # Remove leftover from a previous run
    if os.path.exists(_TEST_DB):
        os.remove(_TEST_DB)
    init_database(_TEST_DB)
    yield
    if os.path.exists(_TEST_DB):
        os.remove(_TEST_DB)


@pytest.fixture(scope="module")
def client(_init_test_db):
    """TestClient wrapping the FastAPI app. Shared across all tests in the module."""
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# ── Health: simple liveness ────────────────────────────────────

class TestHealthSimple:
    def test_returns_200(self, client):
        r = client.get("/api/v1/health")
        assert r.status_code == 200, r.text

    def test_returns_ok_status(self, client):
        body = client.get("/api/v1/health").json()
        assert body["status"] == "ok"

    def test_has_version_field(self, client):
        body = client.get("/api/v1/health").json()
        assert "version" in body

    def test_has_iso_timestamp(self, client):
        body = client.get("/api/v1/health").json()
        ts = body.get("timestamp", "")
        # ISO 8601 timestamps contain 'T'
        assert "T" in ts, f"timestamp not ISO 8601: {ts!r}"


# ── Health: detailed status ────────────────────────────────────

class TestHealthDetailed:
    def test_returns_200(self, client):
        r = client.get("/api/v1/status")
        assert r.status_code == 200, r.text

    def test_body_has_status_field(self, client):
        body = client.get("/api/v1/status").json()
        assert "status" in body

    def test_body_has_components_list(self, client):
        body = client.get("/api/v1/status").json()
        assert "components" in body
        assert isinstance(body["components"], list)
        assert len(body["components"]) >= 2

    def test_database_component_is_ok(self, client):
        """Database should be OK because _init_test_db set it up."""
        body = client.get("/api/v1/status").json()
        db = next((c for c in body["components"] if c["name"] == "database"), None)
        assert db is not None, "database component missing from status response"
        assert db["status"] == "ok", f"Expected database ok, got: {db}"

    def test_gee_component_is_unconfigured(self, client):
        """GEE_PROJECT_ID is blank in this test environment — must report unconfigured."""
        body = client.get("/api/v1/status").json()
        gee = next(
            (c for c in body["components"] if c["name"] == "google_earth_engine"),
            None,
        )
        assert gee is not None, "google_earth_engine component missing"
        assert gee["status"] == "unconfigured", (
            f"Expected unconfigured (no GEE_PROJECT_ID set), got: {gee['status']!r}. "
            f"Detail: {gee.get('detail')}"
        )

    def test_model_artifacts_component_ok(self, client):
        """artifacts/models directory was created in project setup."""
        body = client.get("/api/v1/status").json()
        ma = next(
            (c for c in body["components"] if c["name"] == "model_artifacts"), None
        )
        assert ma is not None, "model_artifacts component missing"
        assert ma["status"] == "ok", (
            f"Expected model_artifacts ok, got: {ma['status']!r}. "
            f"Detail: {ma.get('detail')}"
        )

    def test_overall_degraded_without_gee(self, client):
        """
        Overall status must be 'degraded' when GEE is unconfigured.
        This is correct and honest — the system is partially ready.
        """
        body = client.get("/api/v1/status").json()
        assert body["status"] == "degraded", (
            f"Expected overall degraded (GEE unconfigured), got: {body['status']!r}"
        )

    def test_has_environment_field(self, client):
        body = client.get("/api/v1/status").json()
        assert "environment" in body

    def test_has_version_field(self, client):
        body = client.get("/api/v1/status").json()
        assert "version" in body


# ── Routing: non-existent routes ──────────────────────────────

class TestRouting:
    def test_nonexistent_path_returns_404(self, client):
        r = client.get("/api/v1/nonexistent")
        assert r.status_code == 404, r.text

    def test_root_path_returns_404(self, client):
        """No root route is registered — FastAPI returns 404."""
        r = client.get("/")
        assert r.status_code == 404, r.text

    def test_method_not_allowed_returns_405(self, client):
        """POST on a GET-only endpoint should return 405."""
        r = client.post("/api/v1/health")
        assert r.status_code == 405, r.text


# ── API documentation ──────────────────────────────────────────

class TestApiDocs:
    def test_swagger_ui_accessible(self, client):
        r = client.get("/docs")
        assert r.status_code == 200, "Swagger UI not accessible"

    def test_openapi_schema_valid(self, client):
        r = client.get("/openapi.json")
        assert r.status_code == 200
        schema = r.json()
        assert schema["info"]["title"] == "GeoChange AI API"
        assert schema["info"]["version"] in ("0.1.0", "0.2.0")  # accept either phase

    def test_openapi_has_health_path(self, client):
        schema = client.get("/openapi.json").json()
        paths = schema.get("paths", {})
        assert "/api/v1/health" in paths, (
            f"Expected /api/v1/health in OpenAPI paths. Got: {list(paths.keys())}"
        )


# ── Database initialization ────────────────────────────────────

class TestDatabaseInit:
    """Tests that verify the SQLite database was set up correctly."""

    def test_db_file_exists(self, _init_test_db):
        assert os.path.exists(_TEST_DB), f"Test DB not found at {_TEST_DB}"

    def test_all_expected_tables_present(self, _init_test_db):
        result = verify_database(_TEST_DB)
        assert result["schema_complete"], (
            f"Schema incomplete. Missing: {result.get('missing_tables')}"
        )

    def test_table_names(self, _init_test_db):
        result = verify_database(_TEST_DB)
        expected = {
            "analyses",
            "change_transitions",
            "classification_results",
            "imagery_metadata_cache",
            "model_metrics",
            "study_areas",
        }
        found = set(result["tables"]) - {"sqlite_sequence"}
        assert found == expected, f"Table mismatch.\nExpected: {expected}\nFound: {found}"

    def test_three_study_areas_seeded(self, _init_test_db):
        result = verify_database(_TEST_DB)
        assert result["study_area_count"] == 3, (
            f"Expected 3 seeded study areas, got {result['study_area_count']}"
        )

    def test_coimbatore_bbox_correct(self, _init_test_db):
        conn = sqlite3.connect(_TEST_DB)
        row = conn.execute(
            "SELECT slug, bbox_west, bbox_south, bbox_east, bbox_north "
            "FROM study_areas WHERE slug='coimbatore'"
        ).fetchone()
        conn.close()
        assert row is not None, "Coimbatore not found in study_areas"
        slug, west, south, east, north = row
        assert abs(west - 76.8) < 0.001
        assert abs(south - 10.9) < 0.001
        assert abs(east - 77.2) < 0.001
        assert abs(north - 11.1) < 0.001

    def test_idempotent_init(self, _init_test_db):
        """Running init_database a second time must not raise or duplicate data."""
        init_database(_TEST_DB)  # second call
        result = verify_database(_TEST_DB)
        # study_areas uses INSERT OR IGNORE so no duplication
        assert result["study_area_count"] == 3, (
            f"Re-init duplicated study areas: {result['study_area_count']}"
        )
        assert result["schema_complete"]

    def test_db_excluded_by_gitignore(self):
        """*.db files must be ignored by Git."""
        gitignore_path = os.path.join(_PROJECT_ROOT, ".gitignore")
        assert os.path.exists(gitignore_path), ".gitignore not found at project root"
        content = open(gitignore_path, encoding="utf-8").read()
        assert "*.db" in content, "*.db not in .gitignore"
        assert ".env" in content, ".env not in .gitignore"
        assert ".venv" in content or "venv/" in content, "venv not in .gitignore"

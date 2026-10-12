"""
GeoChange AI — Health & Status Endpoint

GET /api/v1/health  — basic liveness check
GET /api/v1/status  — detailed component status (GEE, DB, models)
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

from app.core.config import settings

router = APIRouter()


# ── Response models ────────────────────────────────────────────


class ComponentStatus(BaseModel):
    name: str
    status: str          # "ok" | "degraded" | "unavailable" | "unconfigured"
    detail: str | None = None


class HealthResponse(BaseModel):
    status: str          # "ok" | "degraded" | "error"
    version: str
    environment: str
    timestamp: str
    components: list[ComponentStatus]


# ── Helpers ────────────────────────────────────────────────────


def _check_database() -> ComponentStatus:
    """Try opening and querying the SQLite database."""
    db_path = Path(settings.sqlite_db_path)
    try:
        if not db_path.exists():
            return ComponentStatus(
                name="database",
                status="unavailable",
                detail=f"Database file not found at {db_path}. Run db init.",
            )
        conn = sqlite3.connect(str(db_path), timeout=3)
        conn.execute("SELECT COUNT(*) FROM study_areas")
        conn.close()
        return ComponentStatus(name="database", status="ok", detail=str(db_path))
    except Exception as e:
        return ComponentStatus(name="database", status="degraded", detail=str(e))


def _check_gee() -> ComponentStatus:
    """
    Check whether Google Earth Engine is configured and authenticated.

    IMPORTANT: This does NOT attempt a live GEE API call during health check
    (that would be slow and could fail at startup for unrelated reasons).
    It only checks whether credentials are configured.
    GEE access is verified separately in /api/v1/imagery/validate.
    """
    if not settings.gee_project_id:
        return ComponentStatus(
            name="google_earth_engine",
            status="unconfigured",
            detail=(
                "GEE_PROJECT_ID is not set. "
                "Apply for access at https://earthengine.google.com, "
                "then run: earthengine authenticate"
            ),
        )
    if settings.gee_auth_mode == "service_account":
        key_path = Path(settings.gee_service_account_key_path)
        if not key_path.exists():
            return ComponentStatus(
                name="google_earth_engine",
                status="unconfigured",
                detail=f"Service account key not found at {key_path}",
            )
    return ComponentStatus(
        name="google_earth_engine",
        status="ok",
        detail=(
            f"auth_mode={settings.gee_auth_mode}, "
            f"project={settings.gee_project_id} "
            f"(connectivity not verified — use /api/v1/imagery/validate)"
        ),
    )


def _check_model_artifacts() -> ComponentStatus:
    """
    Check whether the model artifacts directory exists.
    Resolves relative paths from the project root (parent of backend/).
    """
    raw = settings.model_artifacts_dir
    if Path(raw).is_absolute():
        artifacts_path = Path(raw)
    else:
        # __file__ = backend/app/api/v1/health.py  → parents[3] = backend/  → parents[4] = project root
        project_root = Path(__file__).resolve().parents[4]
        # Strip leading ./ or / so Path division works correctly
        clean = raw.lstrip("./").lstrip("/")
        artifacts_path = project_root / clean
    if not artifacts_path.exists():
        return ComponentStatus(
            name="model_artifacts",
            status="unavailable",
            detail=f"Directory not found: {artifacts_path}",
        )
    model_files = list(artifacts_path.glob("*.pkl")) + list(artifacts_path.glob("*.joblib"))
    return ComponentStatus(
        name="model_artifacts",
        status="ok",
        detail=f"{len(model_files)} model file(s) found in {artifacts_path}",
    )


# ── Routes ─────────────────────────────────────────────────────


@router.get("/health", response_model=dict, summary="Liveness check")
async def health_simple() -> dict:
    """
    Minimal liveness check. Returns 200 if the API process is running.
    Used by Docker health checks and load balancers.
    """
    return {
        "status": "ok",
        "version": "0.1.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get(
    "/status",
    response_model=HealthResponse,
    summary="Detailed component status",
)
async def health_detailed() -> HealthResponse:
    """
    Returns the health of each major component:
    - SQLite database
    - Google Earth Engine configuration
    - Model artifact storage

    Note: GEE connectivity is NOT tested here to keep response time fast.
    Use GET /api/v1/imagery/validate to test live GEE access.
    """
    components = [
        _check_database(),
        _check_gee(),
        _check_model_artifacts(),
    ]

    # Overall status: degraded if any component is not "ok"
    status_values = {c.status for c in components}
    if "unavailable" in status_values or "error" in status_values:
        overall = "degraded"
    elif "unconfigured" in status_values or "degraded" in status_values:
        overall = "degraded"
    else:
        overall = "ok"

    return HealthResponse(
        status=overall,
        version="0.1.0",
        environment=settings.app_env,
        timestamp=datetime.now(timezone.utc).isoformat(),
        components=components,
    )

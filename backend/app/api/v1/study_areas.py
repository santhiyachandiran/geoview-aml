"""GET /api/v1/study-areas"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from fastapi import APIRouter, HTTPException
from app.core.config import settings
from app.models.schemas import StudyAreaResponse

router = APIRouter()


@router.get("/study-areas", response_model=list[StudyAreaResponse], summary="List study areas")
async def list_study_areas() -> list[StudyAreaResponse]:
    """Returns all pre-defined study areas from the database."""
    db_path = settings.resolved_db_path
    if not db_path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Database not initialised. Expected at: {db_path}. "
                   "Run: python -m app.db.init_db",
        )
    conn = sqlite3.connect(str(db_path), timeout=5)
    try:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, name, slug, bbox_west, bbox_south, bbox_east, bbox_north, crs, description "
            "FROM study_areas ORDER BY id"
        ).fetchall()
    finally:
        conn.close()
    return [StudyAreaResponse(**dict(r)) for r in rows]

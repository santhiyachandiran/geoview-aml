"""GET /api/v1/imagery/availability"""

from __future__ import annotations

import sqlite3

from fastapi import APIRouter, HTTPException, Query
from app.core.config import settings
from app.models.schemas import ImageryAvailabilityResponse, ImageryYearAvailability
from app.services import gee_service

router = APIRouter()


def _lookup_bbox(study_area: str) -> list[float]:
    db_path = settings.resolved_db_path
    if not db_path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Database not initialised. Expected at: {db_path}. "
                   "Run: python -m app.db.init_db",
        )
    conn = sqlite3.connect(str(db_path), timeout=5)
    try:
        row = conn.execute(
            "SELECT bbox_west, bbox_south, bbox_east, bbox_north FROM study_areas WHERE slug=?",
            (study_area,),
        ).fetchone()
    finally:
        conn.close()
    if not row:
        raise HTTPException(status_code=404, detail=f"Study area '{study_area}' not found")
    return list(row)


@router.get(
    "/imagery/availability",
    response_model=ImageryAvailabilityResponse,
    summary="Check Sentinel-2 imagery availability",
)
async def imagery_availability(
    study_area: str = Query(..., description="Study area slug, e.g. 'coimbatore'"),
    year_start: int = Query(..., ge=2013, le=2030),
    year_end: int = Query(..., ge=2013, le=2030),
    season_start_month: int = Query(default=11, ge=1, le=12),
    season_end_month: int = Query(default=3, ge=1, le=12),
) -> ImageryAvailabilityResponse:
    if year_start > year_end:
        raise HTTPException(status_code=422, detail="year_start must be <= year_end")

    bbox = _lookup_bbox(study_area)

    if gee_service.GEE_AVAILABLE:
        raw = gee_service.get_imagery_availability(bbox, year_start, year_end,
                                                   season_start_month, season_end_month)
        demo_mode = False
    else:
        from app.services import demo_service
        raw = demo_service.get_imagery_availability(bbox, year_start, year_end,
                                                    season_start_month, season_end_month)
        demo_mode = True

    years = [ImageryYearAvailability(**y) for y in raw]
    return ImageryAvailabilityResponse(
        study_area=study_area,
        years=years,
        gee_available=gee_service.GEE_AVAILABLE,
        demo_mode=demo_mode,
    )

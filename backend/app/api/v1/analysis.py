"""
POST /api/v1/analysis/indices
POST /api/v1/analysis/change-detection
GET  /api/v1/analysis/results/{analysis_id}
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, HTTPException
from app.core.config import settings
from app.models.schemas import AnalysisRequest, AnalysisResult, IndexResult, ChangeDetectionResult
from app.services import gee_service, analysis_service

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
            (study_area,)
        ).fetchone()
    finally:
        conn.close()
    if not row:
        raise HTTPException(status_code=404, detail=f"Study area '{study_area}' not found")
    return list(row)


def _get_composite(bbox, year, season_start_month, season_end_month):
    if gee_service.GEE_AVAILABLE:
        return gee_service.get_sentinel2_composite(
            bbox, year, season_start_month, season_end_month
        )
    from app.services import demo_service
    return demo_service.get_sentinel2_composite(
        bbox, year, season_start_month, season_end_month
    )


def _insert_analysis(analysis_id: str, study_area: str, year1: int, year2: int, status: str = "running"):
    conn = sqlite3.connect(str(settings.resolved_db_path), timeout=5)
    try:
        conn.execute(
            "INSERT OR IGNORE INTO analyses (id, date_start, date_end, data_source, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (analysis_id, str(year1), str(year2), "sentinel2", status,
             datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


def _update_analysis(analysis_id: str, status: str, stats_json: str | None = None, error: str | None = None):
    conn = sqlite3.connect(str(settings.resolved_db_path), timeout=5)
    try:
        conn.execute(
            "UPDATE analyses SET status=?, output_stats_json=?, error_message=?, completed_at=? WHERE id=?",
            (status, stats_json, error, datetime.now(timezone.utc).isoformat(), analysis_id),
        )
        conn.commit()
    finally:
        conn.close()


# ── Routes ────────────────────────────────────────────────────────

@router.post("/analysis/indices", response_model=IndexResult, summary="Compute spectral index differences")
async def compute_indices(req: AnalysisRequest) -> IndexResult:
    """
    Compute NDVI/NDWI/NDBI mean values for two years and return their differences.
    Returns immediately (synchronous for index stats).
    """
    bbox = _lookup_bbox(req.study_area)
    c1 = _get_composite(bbox, req.year1, req.season_start_month, req.season_end_month)
    c2 = _get_composite(bbox, req.year2, req.season_start_month, req.season_end_month)

    diff = analysis_service.compute_index_differences(c1, c2)
    veg = analysis_service.detect_vegetation_to_buildup(
        diff.get("delta_ndvi"), diff.get("delta_ndbi")
    )

    demo = c1.get("demo_data", False) or c2.get("demo_data", False)
    return IndexResult(
        year1=req.year1, year2=req.year2, study_area=req.study_area,
        delta_ndvi=diff.get("delta_ndvi"),
        delta_ndwi=diff.get("delta_ndwi"),
        delta_ndbi=diff.get("delta_ndbi"),
        t1_ndvi=diff.get("t1_ndvi"), t2_ndvi=diff.get("t2_ndvi"),
        t1_ndwi=diff.get("t1_ndwi"), t2_ndwi=diff.get("t2_ndwi"),
        t1_ndbi=diff.get("t1_ndbi"), t2_ndbi=diff.get("t2_ndbi"),
        vegetation_to_buildup_flag=veg.get("flag", False),
        confidence_label=veg.get("confidence_label", "unknown"),
        demo_data=demo,
        data_provenance=diff.get("data_provenance", {}),
        status="completed",
    )


@router.post("/analysis/change-detection", response_model=ChangeDetectionResult,
             summary="Run change detection analysis")
async def change_detection(req: AnalysisRequest, background_tasks: BackgroundTasks) -> ChangeDetectionResult:
    """
    Run change detection. Stores result in SQLite and returns immediately.
    """
    import json
    analysis_id = str(uuid.uuid4())
    bbox = _lookup_bbox(req.study_area)
    _insert_analysis(analysis_id, req.study_area, req.year1, req.year2)

    c1 = _get_composite(bbox, req.year1, req.season_start_month, req.season_end_month)
    c2 = _get_composite(bbox, req.year2, req.season_start_month, req.season_end_month)
    diff = analysis_service.compute_index_differences(c1, c2)
    veg = analysis_service.detect_vegetation_to_buildup(
        diff.get("delta_ndvi"), diff.get("delta_ndbi")
    )
    demo = c1.get("demo_data", False) or c2.get("demo_data", False)

    tm, area_stats = analysis_service.compute_transition_and_area_stats(
        req.study_area, req.year1, req.year2, c1, c2
    )

    result = ChangeDetectionResult(
        analysis_id=analysis_id,
        study_area=req.study_area,
        year1=req.year1, year2=req.year2,
        delta_ndvi=diff.get("delta_ndvi"),
        delta_ndwi=diff.get("delta_ndwi"),
        delta_ndbi=diff.get("delta_ndbi"),
        vegetation_to_buildup_flag=veg.get("flag", False),
        confidence_label=veg.get("confidence_label", "unknown"),
        transition_matrix=tm,
        area_stats=area_stats,
        demo_data=demo,
        data_provenance=diff.get("data_provenance", {}),
        status="completed",
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    _update_analysis(analysis_id, "completed", json.dumps(result.model_dump()))
    return result


@router.get("/analysis/results/{analysis_id}", response_model=AnalysisResult,
            summary="Get stored analysis result")
async def get_analysis_result(analysis_id: str) -> AnalysisResult:
    conn = sqlite3.connect(str(settings.resolved_db_path), timeout=5)
    try:
        row = conn.execute(
            "SELECT id, status, output_stats_json, error_message FROM analyses WHERE id=?",
            (analysis_id,)
        ).fetchone()
    finally:
        conn.close()
    if not row:
        raise HTTPException(status_code=404, detail=f"Analysis '{analysis_id}' not found")
    import json
    result_data = json.loads(row[2]) if row[2] else None
    return AnalysisResult(
        analysis_id=row[0], status=row[1],
        result=result_data, error=row[3],
    )

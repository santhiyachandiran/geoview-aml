"""
GeoChange AI — Pydantic v2 Request / Response Schemas
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


# ── Study Areas ──────────────────────────────────────────────────

class StudyAreaResponse(BaseModel):
    id: int
    name: str
    slug: str
    bbox_west: float
    bbox_south: float
    bbox_east: float
    bbox_north: float
    crs: str
    description: str | None = None


# ── Imagery ──────────────────────────────────────────────────────

class ImageryYearAvailability(BaseModel):
    year: int
    available: bool
    scene_count: int | None = None
    cloud_cover_pct: float | None = None
    date_range: dict | None = None
    gee_available: bool = False
    demo_data: bool = False


class ImageryAvailabilityResponse(BaseModel):
    study_area: str
    years: list[ImageryYearAvailability]
    gee_available: bool
    demo_mode: bool


# ── Analysis ─────────────────────────────────────────────────────

class AnalysisRequest(BaseModel):
    study_area: str = Field(..., description="Slug from /study-areas, e.g. 'coimbatore'")
    year1: int = Field(..., ge=2013, le=2030, description="First year (before)")
    year2: int = Field(..., ge=2013, le=2030, description="Second year (after)")
    season_start_month: int = Field(default=11, ge=1, le=12)
    season_end_month: int = Field(default=3, ge=1, le=12)

    @model_validator(mode="after")
    def years_must_differ(self) -> "AnalysisRequest":
        if self.year1 >= self.year2:
            raise ValueError("year1 must be earlier than year2")
        return self


class IndexResult(BaseModel):
    year1: int
    year2: int
    study_area: str
    delta_ndvi: float | None = None
    delta_ndwi: float | None = None
    delta_ndbi: float | None = None
    t1_ndvi: float | None = None
    t2_ndvi: float | None = None
    t1_ndwi: float | None = None
    t2_ndwi: float | None = None
    t1_ndbi: float | None = None
    t2_ndbi: float | None = None
    vegetation_to_buildup_flag: bool = False
    confidence_label: str = "unknown"
    demo_data: bool
    data_provenance: dict = Field(default_factory=dict)
    job_id: str | None = None
    status: str = "completed"


class ChangeDetectionResult(BaseModel):
    analysis_id: str
    study_area: str
    year1: int
    year2: int
    delta_ndvi: float | None = None
    delta_ndwi: float | None = None
    delta_ndbi: float | None = None
    vegetation_to_buildup_flag: bool = False
    confidence_label: str = "unknown"
    transition_matrix: dict | None = None
    area_stats: dict | None = None
    demo_data: bool
    data_provenance: dict = Field(default_factory=dict)
    status: str = "completed"
    created_at: str | None = None


class AnalysisResult(BaseModel):
    analysis_id: str
    status: str
    result: dict | None = None
    error: str | None = None


# ── Models ────────────────────────────────────────────────────────

class ModelInfo(BaseModel):
    name: str
    path: str
    size_kb: float | None = None
    model_type: str | None = None


class ModelListResponse(BaseModel):
    models: list[ModelInfo]
    count: int


# ── Experiments ──────────────────────────────────────────────────

class ExperimentRun(BaseModel):
    run_id: str
    run_name: str | None = None
    status: str | None = None
    start_time: str | None = None
    metrics: dict = Field(default_factory=dict)
    params: dict = Field(default_factory=dict)


class ExperimentListResponse(BaseModel):
    runs: list[ExperimentRun]
    count: int
    tracking_uri: str | None = None


# ── Model inference ───────────────────────────────────────────────

class PredictRequest(BaseModel):
    features: list[list[float]] = Field(
        ...,
        description="(N, 9) pixel feature rows in order: B2,B3,B4,B8,B11,B12,NDVI,NDWI,NDBI",
    )
    model_name: str | None = Field(
        default=None,
        description="Filename of saved model in artifacts/models/. Uses latest if omitted.",
    )


class PredictResponse(BaseModel):
    predictions: list[int]
    class_names: list[str]
    probabilities: list[list[float]] | None = None
    model_used: str
    n_pixels: int
    demo_data: bool = False

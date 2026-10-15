"""
GeoChange AI — FastAPI Application Entry Point

Start the server with:
    cd backend
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager
from loguru import logger
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging import setup_logging
from app.api.v1 import health, study_areas, imagery, analysis, models_api


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    setup_logging(log_level=settings.log_level, log_file=settings.log_file)
    logger.info(f"GeoChange AI starting — env={settings.app_env}")

    # Attempt GEE authentication (non-blocking — app starts regardless)
    from app.services.gee_service import authenticate_gee
    gee_ok = authenticate_gee(project_id=settings.gee_project_id)
    if gee_ok:
        logger.info("GEE authenticated successfully")
    else:
        logger.warning("GEE unavailable — demo mode active. Run: earthengine authenticate")

    yield
    logger.info("GeoChange AI shutting down")


app = FastAPI(
    title="GeoChange AI API",
    description=(
        "AI-Powered Satellite Imagery Land-Cover Change Detection. "
        "Detects agricultural land loss due to urban expansion using "
        "Sentinel-2 imagery and machine learning."
    ),
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────
app.include_router(health.router,       prefix="/api/v1", tags=["Health"])
app.include_router(study_areas.router,  prefix="/api/v1", tags=["Study Areas"])
app.include_router(imagery.router,      prefix="/api/v1", tags=["Imagery"])
app.include_router(analysis.router,     prefix="/api/v1", tags=["Analysis"])
app.include_router(models_api.router,   prefix="/api/v1", tags=["Models"])

# ── Static Files (Model Plots & Visualizations) ────────────────
from fastapi.staticfiles import StaticFiles
artifacts_dir = settings.resolved_artifacts_dir
artifacts_dir.mkdir(parents=True, exist_ok=True)
app.mount("/artifacts/models", StaticFiles(directory=str(artifacts_dir)), name="artifacts_models")


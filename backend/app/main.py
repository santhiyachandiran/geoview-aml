"""
GeoChange AI — FastAPI Application Entry Point

This file creates the FastAPI app, configures middleware, and includes API routers.
It does NOT run the server — that is done by uvicorn in the Dockerfile or via:

    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from loguru import logger

from app.core.config import settings
from app.core.logging import setup_logging
from app.api.v1 import health


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    setup_logging(log_level=settings.log_level, log_file=settings.log_file)
    logger.info(f"GeoChange AI starting — env={settings.app_env}")
    logger.info(f"GEE auth mode: {settings.gee_auth_mode}")
    logger.info(f"GEE project: {settings.gee_project_id or 'NOT CONFIGURED'}")
    yield
    logger.info("GeoChange AI shutting down")


app = FastAPI(
    title="GeoChange AI API",
    description=(
        "AI-Powered Satellite Imagery Land-Cover Change Detection. "
        "Detects agricultural land loss due to urban expansion using "
        "Sentinel-2 imagery and machine learning."
    ),
    version="0.1.0",
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
app.include_router(health.router, prefix="/api/v1", tags=["Health"])

# More routers will be added in Phase 7 (Backend API phase):
# app.include_router(imagery.router, prefix="/api/v1", tags=["Imagery"])
# app.include_router(analysis.router, prefix="/api/v1", tags=["Analysis"])
# app.include_router(export.router, prefix="/api/v1", tags=["Export"])
# app.include_router(history.router, prefix="/api/v1", tags=["History"])
# app.include_router(models.router, prefix="/api/v1", tags=["Models"])

"""
GET  /api/v1/models
GET  /api/v1/experiments
GET  /api/v1/models/results/latest   — latest pipeline results JSON
POST /api/v1/models/predict           — run inference with a saved model
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.config import settings
from app.models.schemas import ModelInfo, ModelListResponse, ExperimentRun, ExperimentListResponse

router = APIRouter()
logger = logging.getLogger(__name__)


class PredictRequest(BaseModel):
    features: list[list[float]]   # (N, 9) rows
    model_name: str | None = None  # filename; uses latest if None


class PredictResponse(BaseModel):
    predictions: list[int]
    class_names: list[str]
    probabilities: list[list[float]] | None = None
    model_used: str
    n_pixels: int


# ── Helpers ────────────────────────────────────────────────────

def _artifacts_dir() -> Path:
    return settings.resolved_artifacts_dir


def _latest_model(prefix: str = "") -> Path | None:
    d = _artifacts_dir()
    if not d.exists():
        return None
    candidates = sorted(d.glob(f"{prefix}*.joblib"), reverse=True)
    return candidates[0] if candidates else None


def _latest_results_json() -> dict | None:
    d = _artifacts_dir()
    if not d.exists():
        return None
    jsons = sorted(d.glob("results_*.json"), reverse=True)
    if not jsons:
        return None
    return json.loads(jsons[0].read_text(encoding="utf-8"))


# ── Routes ─────────────────────────────────────────────────────

@router.get("/models", response_model=ModelListResponse, summary="List saved models")
async def list_models() -> ModelListResponse:
    artifacts = _artifacts_dir()
    models_list = []
    if artifacts.exists():
        for ext in ("*.joblib", "*.pkl"):
            for p in sorted(artifacts.glob(ext)):
                if p.name == "scaler.joblib":
                    continue
                try:
                    size_kb = round(p.stat().st_size / 1024, 1)
                except OSError:
                    size_kb = None
                models_list.append(ModelInfo(
                    name=p.name, path=str(p),
                    size_kb=size_kb,
                    model_type="xgboost" if "xgb" in p.name else "random_forest",
                ))
    return ModelListResponse(models=models_list, count=len(models_list))


@router.get("/models/results/latest", summary="Latest pipeline results")
async def latest_results() -> dict:
    """Returns the metrics JSON from the most recent pipeline run."""
    data = _latest_results_json()
    if data is None:
        raise HTTPException(
            status_code=404,
            detail="No pipeline results found. Run: python scripts/run_pipeline.py",
        )
    return data


@router.post("/models/predict", response_model=PredictResponse, summary="Run model inference")
async def predict(req: PredictRequest) -> PredictResponse:
    """
    Classify pixel feature vectors with a saved model.
    Features must be in FEATURE_NAMES order:
    [B2, B3, B4, B8, B11, B12, NDVI, NDWI, NDBI]
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

    if req.model_name:
        model_path = _artifacts_dir() / req.model_name
        if not model_path.exists():
            raise HTTPException(status_code=404,
                                detail=f"Model not found: {req.model_name}")
    else:
        model_path = _latest_model()
        if model_path is None:
            raise HTTPException(
                status_code=404,
                detail="No saved models found. Run: python scripts/run_pipeline.py",
            )

    try:
        from ml.models.inference import load_model, predict_landcover, CLASS_NAMES
        model = load_model(model_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model load failed: {exc}")

    X = np.array(req.features, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] != 9:
        raise HTTPException(
            status_code=422,
            detail=f"Expected (N, 9) feature array, got shape {X.shape}. "
                   "Columns must be [B2,B3,B4,B8,B11,B12,NDVI,NDWI,NDBI].",
        )

    # Apply saved scaler if available
    scaler_path = _artifacts_dir() / "scaler.joblib"
    if scaler_path.exists():
        import joblib
        scaler = joblib.load(scaler_path)
        X = scaler.transform(X).astype(np.float32)

    preds, proba = predict_landcover(model, X)

    return PredictResponse(
        predictions=preds.tolist(),
        class_names=CLASS_NAMES,
        probabilities=proba.tolist() if proba is not None else None,
        model_used=model_path.name,
        n_pixels=len(preds),
    )


@router.get("/experiments", response_model=ExperimentListResponse,
            summary="List MLflow experiment runs")
async def list_experiments() -> ExperimentListResponse:
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    try:
        import mlflow
        tracking_uri = settings.mlflow_tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        runs = mlflow.search_runs(
            experiment_names=["geochange_ai"],
            max_results=50,
            output_format="list",
        )
        exp_runs = [
            ExperimentRun(
                run_id=r.info.run_id,
                run_name=r.info.run_name,
                status=r.info.status,
                start_time=str(r.info.start_time),
                metrics={k: v for k, v in r.data.metrics.items()},
                params={k: v for k, v in r.data.params.items()},
            )
            for r in runs
        ]
        if not exp_runs:
            latest = _latest_results_json()
            if latest:
                ts = latest.get("run_timestamp", "")
                rf = latest.get("random_forest", {})
                xgb = latest.get("xgboost", {})
                fallback_runs = []
                if rf and rf.get("test_metrics"):
                    fallback_runs.append(ExperimentRun(
                        run_id=rf.get("mlflow_run_id") or f"rf_{ts}",
                        run_name="random_forest",
                        status="FINISHED",
                        start_time=ts,
                        metrics=rf["test_metrics"],
                        params={str(k): str(v) for k, v in rf.get("params", {}).items()},
                    ))
                if xgb and xgb.get("test_metrics"):
                    fallback_runs.append(ExperimentRun(
                        run_id=xgb.get("mlflow_run_id") or f"xgb_{ts}",
                        run_name="xgboost",
                        status="FINISHED",
                        start_time=ts,
                        metrics=xgb["test_metrics"],
                        params={str(k): str(v) for k, v in xgb.get("best_params", {}).items()},
                    ))
                return ExperimentListResponse(
                    runs=fallback_runs, count=len(fallback_runs), tracking_uri=settings.mlflow_tracking_uri
                )

        return ExperimentListResponse(
            runs=exp_runs, count=len(exp_runs), tracking_uri=tracking_uri
        )
    except Exception as exc:
        logger.warning(f"MLflow search failed: {exc}")
        latest = _latest_results_json()
        if latest:
            ts = latest.get("run_timestamp", "")
            rf = latest.get("random_forest", {})
            xgb = latest.get("xgboost", {})
            fallback_runs = []
            if rf and rf.get("test_metrics"):
                fallback_runs.append(ExperimentRun(
                    run_id=rf.get("mlflow_run_id") or f"rf_{ts}",
                    run_name="random_forest",
                    status="FINISHED",
                    start_time=ts,
                    metrics=rf["test_metrics"],
                    params={str(k): str(v) for k, v in rf.get("params", {}).items()},
                ))
            if xgb and xgb.get("test_metrics"):
                fallback_runs.append(ExperimentRun(
                    run_id=xgb.get("mlflow_run_id") or f"xgb_{ts}",
                    run_name="xgboost",
                    status="FINISHED",
                    start_time=ts,
                    metrics=xgb["test_metrics"],
                    params={str(k): str(v) for k, v in xgb.get("best_params", {}).items()},
                ))
            return ExperimentListResponse(
                runs=fallback_runs, count=len(fallback_runs), tracking_uri=settings.mlflow_tracking_uri
            )
        return ExperimentListResponse(
            runs=[], count=0, tracking_uri=settings.mlflow_tracking_uri
        )


@router.get("/models/plots", summary="List generated model performance plots")
async def list_plots() -> dict:
    """Lists all saved visualization figures in the model artifacts directory."""
    d = _artifacts_dir()
    plots = []
    if d.exists():
        for p in sorted(d.glob("*.png")):
            plots.append({
                "filename": p.name,
                "url": f"/artifacts/models/{p.name}",
                "size_kb": round(p.stat().st_size / 1024, 1),
            })
    return {"plots": plots, "count": len(plots)}

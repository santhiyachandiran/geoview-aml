# GEOVIEW AML — How to Collect Paper Data

This file explains the exact commands that produce the real numbers,
tables, and figures for the IEEE report. Run these after GEE is authenticated.

## Prerequisites

```powershell
cd "D:\AML CAT PRO\geochange-ai"
.venv\Scripts\Activate.ps1
python scripts/check_gee_auth.py   # must show all [OK]
```

## Table V — Component status

```powershell
cd backend
uvicorn app.main:app --port 8000 &
# Wait 5 seconds, then:
curl http://localhost:8000/api/v1/status
```
Copy the JSON response directly into Table V.

## Scene counts (imagery availability table)

```powershell
curl "http://localhost:8000/api/v1/imagery/availability?study_area=coimbatore&year_start=2018&year_end=2024"
```
The `scene_count` and `cloud_cover_pct` per year populate the imagery table.

## Transition statistics

```powershell
curl -X POST http://localhost:8000/api/v1/analysis/change-detection `
  -H "Content-Type: application/json" `
  -d '{"study_area":"coimbatore","year1":2018,"year2":2024}'
```
The `transitions` array in the response gives from→to pixel counts and area (ha).

## Table VI — Model metrics (Random Forest and XGBoost)

```powershell
# Train both models (takes 15-60 minutes with real GEE data)
python scripts/run_pipeline.py --year 2021 --trials 30

# The results JSON is saved to artifacts/models/results_<timestamp>.json
# It contains:
#   random_forest.test_metrics.accuracy
#   random_forest.test_metrics.macro_f1
#   random_forest.test_metrics.per_class_precision/recall/f1
#   random_forest.test_metrics.confusion_matrix
#   xgboost.test_metrics.*  (same structure)
#   xgboost.best_params     (Optuna best hyperparameters)
#   comparison              (winner per metric)
```

## XGBoost best hyperparameters

Printed to console during run_pipeline.py and in:
- `artifacts/models/results_<timestamp>.json` → `xgboost.best_params`
- MLflow UI → http://localhost:5000 → experiment geochange_ai → XGBoost run → Params tab

## Confusion matrix PNG (Fig. 6)

```
artifacts/models/rf_confusion_matrix_<timestamp>.png
artifacts/models/xgb_confusion_matrix_<timestamp>.png
```

## Feature importance PNG

```
artifacts/models/rf_feature_importance_<timestamp>.png
artifacts/models/xgb_feature_importance_<timestamp>.png
```

## Figures 2 and 3 — Composite maps and index maps

These must come from Google Earth Engine Code Editor:
1. Open https://code.earthengine.google.com
2. Load the composite for T1 (2018-11-01 to 2019-02-28) and T2 (2024-11-01 to 2025-02-28)
3. Visualise as RGB (B4, B3, B2) → Screenshot = Fig. 2
4. Visualise NDVI, NDWI, NDBI maps → Screenshot = Fig. 3
5. Visualise NDVI difference (T2 - T1) → Screenshot = Fig. 8 (change map)

## Fig. 5 — Dashboard screenshot

```powershell
cd frontend; npm run dev
# Open http://localhost:5173, run an analysis, take screenshot with Win+Shift+S
```

## Software versions

```powershell
python --version
python -c "import sklearn,xgboost,optuna,mlflow,rasterio,geopandas,fastapi,numpy,pandas; [print(m, getattr(__import__(m),'__version__','?')) for m in ['sklearn','xgboost','optuna','mlflow','rasterio','geopandas','fastapi','numpy','pandas']]"
node --version
docker --version
```

## Hardware spec

```powershell
Get-WmiObject Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors
[math]::Round((Get-WmiObject Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1)
Get-WmiObject Win32_VideoController | Select-Object Name
```

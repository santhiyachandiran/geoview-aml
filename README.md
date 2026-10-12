# GeoChange AI
### AI-Powered Satellite Imagery Land-Cover Change Detection

> **Academic project** — Integrated M.Sc. AI & ML  
> Study area: Coimbatore, Tamil Nadu, India

---

## What this project does

GeoChange AI compares satellite observations from different years and detects where and how much land-cover has changed. The primary use case is **detecting agricultural land loss due to urban expansion** — for example, cropland being converted into built-up area around Coimbatore's IT corridor between 2018 and 2024.

The system answers:
- How has a selected area changed between two years?
- Where has agricultural or vegetated land changed into built-up land?
- How much area changed (in hectares and km²)?
- Which locations experienced the largest changes?
- How confident is the model in its predictions?
- What is the actual acquisition date of the satellite imagery?

---

## Status

| Component | Status |
|---|---|
| Project structure | ✅ Complete |
| Python environment (3.14.2) | ✅ All 27 packages installed |
| FastAPI backend skeleton | ✅ Running — health endpoint tested |
| SQLite database schema | ✅ Initialized and tested |
| Automated tests (26 tests) | ✅ 26/26 passing |
| Google Earth Engine access | ⚠️ **Not yet configured** — see setup below |
| Satellite data ingestion | 🔲 Phase 3 |
| ML models (RF, XGBoost) | 🔲 Phase 4 & 5 |
| React frontend | 🔲 Phase 8 |
| Docker | 🔲 Phase 10 |
| Cloud deployment | 🔲 Phase 11 |

---

## Architecture

```
Browser (React + Vite + Leaflet)
        │
        │ HTTP REST
        ▼
FastAPI Backend  ──────►  Google Earth Engine  (Sentinel-2, Landsat)
        │                  ESA WorldCover       (land-cover labels)
        │
        ├── SQLite DB       (analysis history, metadata)
        ├── ML Inference    (Random Forest / XGBoost models)
        └── Artifact Store  (trained models, exported maps)
```

---

## Technology Stack

**Frontend:** React 18, Vite, JavaScript, Leaflet, Recharts  
**Backend:** Python 3.14, FastAPI, Uvicorn, Pydantic v2, SQLAlchemy  
**ML:** scikit-learn, XGBoost, Optuna, MLflow, SHAP  
**Geospatial:** Rasterio, GeoPandas, Shapely, PyProj  
**Data source:** Google Earth Engine (Sentinel-2, Landsat, ESA WorldCover)  
**Database:** SQLite  
**Container:** Docker + Docker Compose  

---

## Project Structure

```
geochange-ai/
├── frontend/               React + Vite app (Phase 8)
├── backend/
│   ├── app/
│   │   ├── api/v1/         FastAPI route handlers
│   │   ├── core/           Config, logging
│   │   ├── db/             SQLite schema and init
│   │   ├── models/         Pydantic schemas
│   │   └── services/       Business logic
│   ├── requirements.txt    Pinned Python dependencies
│   └── geochange.db        SQLite database (gitignored)
├── ml/
│   ├── data/               Downloaded satellite data (gitignored)
│   ├── preprocessing/      Cloud masking, compositing, indices
│   ├── features/           Feature engineering
│   ├── models/             RF, XGBoost training code
│   ├── evaluation/         Metrics, change detection
│   └── notebooks/          EDA notebooks
├── tests/                  pytest test suite
├── configs/                Study area and model configs
├── scripts/                Data download, training runners
├── artifacts/              Saved models, exports (gitignored)
├── docs/                   Architecture, API docs
├── .env.example            Environment variable template
├── .gitignore
└── README.md
```

---

## Local Setup

### Prerequisites
- Python 3.14+ ([python.org](https://python.org))
- Node.js 20+ ([nodejs.org](https://nodejs.org))
- Docker Desktop ([docker.com](https://docker.com))
- Git

### 1. Clone and create virtual environment

```powershell
git clone https://github.com/<your-username>/geochange-ai.git
cd geochange-ai

# Create virtual environment
python -m venv .venv

# Activate (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Activate (macOS / Linux)
source .venv/bin/activate
```

### 2. Install backend dependencies

```powershell
pip install -r backend/requirements.txt
```

### 3. Configure environment variables

```powershell
# Copy the template
Copy-Item .env.example .env

# Edit .env and fill in:
# - GEE_PROJECT_ID (after GEE account approval)
# - GEE_AUTH_MODE=user
```

### 4. Initialize the database

```powershell
cd backend
python -m app.db.init_db
```

### 5. Start the backend server

```powershell
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API docs available at: http://localhost:8000/docs

### 6. Run tests

```powershell
# From project root
.venv\Scripts\pytest tests/ -v
```

---

## Google Earth Engine Setup

> ⚠️ **GEE access is NOT yet verified in this project.**  
> The application will start and the health endpoint will report `unconfigured` for GEE.  
> Satellite data ingestion (Phase 3) requires a confirmed GEE account.

### Steps to get access:

1. Go to [https://earthengine.google.com](https://earthengine.google.com)
2. Click **Sign Up** and use your university email for faster academic approval
3. Wait for approval (typically 1–3 days for academic users)
4. After approval, run in your activated venv:
   ```powershell
   earthengine authenticate
   ```
5. Create or select a Google Cloud project at [https://console.cloud.google.com](https://console.cloud.google.com)
6. Enable the Earth Engine API for your project
7. Add to your `.env`:
   ```
   GEE_PROJECT_ID=your-cloud-project-id
   GEE_AUTH_MODE=user
   ```

**Credentials are never committed to Git** — `.env` is in `.gitignore`.

---

## Planned Docker Workflow (Phase 10)

```powershell
docker compose build
docker compose up -d
docker compose ps
docker compose logs -f
docker compose down
```

---

## Data Sources

| Source | Use | Notes |
|---|---|---|
| Sentinel-2 SR (`COPERNICUS/S2_SR_HARMONIZED`) | Primary imagery | 10 m, 2017–present |
| Landsat 8/9 (`LANDSAT/LC08/C02/T1_L2`) | Historical backup | 30 m, pre-2017 |
| ESA WorldCover v2 (2021) | Training labels | 10 m, 11 classes |
| Dynamic World | Per-pixel probabilities | 10 m, per-scene |

**Seasonal window:** November–February composites (Tamil Nadu post-monsoon dry season)  
**Study area bbox:** `[76.8°E, 10.9°N, 77.2°E, 11.1°N]` (Coimbatore)

---

## Land-Cover Classes

| Class | Colour | ESA WorldCover codes |
|---|---|---|
| Built-up / Urban | `#FF3366` | 50 |
| Vegetation / Agriculture | `#10B981` | 10, 20, 30, 40 |
| Water | `#00E5FF` | 80 |
| Bare soil / Other | `#F59E0B` | 60, 70, 90, 95, 100 |

---

## Limitations

- ESA WorldCover labels only available for 2020 and 2021. The classifier is trained on 2021 labels and applied to multi-year composites — this is documented and standard practice.
- Sentinel-2 archive starts from 2017. Pre-2017 comparisons use Landsat (30 m resolution, different spectral characteristics).
- Cloud cover over Tamil Nadu is high from June–October. We use November–February composites to mitigate this.
- GEE free tier has export quotas; very large areas may need to be tiled.
- This project does not claim real-time or live imagery — all data is historical satellite observations.

---

## Academic Rubric Coverage

| Criterion | Marks | Planned evidence |
|---|---|---|
| Data engineering & visualization | 8 | GEE pipeline, preprocessing, 5+ charts |
| Advanced modeling & pipeline | 10 | RF baseline, XGBoost + Optuna, MLflow tracking |
| Benchmarking | 10 | Fair RF vs XGBoost comparison, spatial CV, SHAP |
| Deployment & containerization | 8 | Docker Compose, cloud deployment |
| Documentation | 4 | README, API docs, methodology section |

---

## License

MIT — see `LICENSE` file (to be added).

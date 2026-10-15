# GEOVIEW AML — AI-Powered Satellite Imagery Land-Cover Change Detection

> **Course / Evaluation:** Advanced Machine Learning (AML)  
> **Study Area:** Coimbatore, Tamil Nadu, India (`[76.8°E, 10.9°N, 77.2°E, 11.1°N]`)  
> **Repository:** [https://github.com/santhiyachandiran/geoview-aml](https://github.com/santhiyachandiran/geoview-aml)

---

## 1. Project Objective & Architecture

GEOVIEW AML is an end-to-end geospatial machine learning platform designed to detect and quantify land-cover changes across multi-temporal satellite observations. The system focuses on **urban expansion and agricultural/vegetation dynamics** around the Coimbatore metropolitan and industrial corridor between 2018 and 2024.

### System Architecture

```
User Interface (React 18 + Vite + Leaflet + Recharts)
                       │  HTTP REST (/api/v1)
                       ▼
FastAPI Backend (Python 3.14 + Uvicorn)
  ├── Data Access Layer: Google Earth Engine (Sentinel-2 Harmonized L2A)
  ├── Ground Reference: ESA WorldCover 10m v200 (2021)
  ├── Feature Engineering: 6 Spectral Bands + NDVI, NDWI, NDBI (10m)
  ├── ML Inference Engine: Random Forest & Optuna-Tuned XGBoost
  ├── Relational Storage: SQLite (geochange.db — study areas, analysis logs)
  └── Artifact Store: Trained models, scalers, confusion matrices, GeoJSON exports
```

---

## 2. Dataset Sources & Preprocessing Pipeline

| Layer | Source / Collection | Resolution | Temporal Window | Role |
|---|---|---|---|---|
| **Satellite Imagery** | `COPERNICUS/S2_SR_HARMONIZED` | 10 m | 2017–2025 (Nov–Mar dry season) | Multispectral surface reflectance |
| **Cloud Masking** | Sentinel-2 Scene Classification Layer (SCL) | 10 m | Per-scene | Masks cloud shadows (3), clouds (8, 9), cirrus (10), snow (11) |
| **Reference Labels** | ESA WorldCover v200 | 10 m | 2021 | 4 aggregated classes: Built-up, Vegetation, Water, Bare soil |
| **Real Sampled Dataset** | 10,000 spatial points (`ml/data/processed/coimbatore_2021/`) | 10 m | Dry Season 2021–2022 | Modular feature vectors (`X.npy`, `y.npy`, `rows.npy`, `cols.npy`) |

### Feature Engineering & Preprocessing
- **Spectral Bands:** Blue (`B2`), Green (`B3`), Red (`B4`), Near-Infrared (`B8`), SWIR-1 (`B11`), SWIR-2 (`B12`).
- **Derived Spectral Indices:**
  - **NDVI** (Normalized Difference Vegetation Index): `(B8 - B4) / (B8 + B4)` — vegetation health and density.
  - **NDWI** (Normalized Difference Water Index): `(B3 - B11) / (B3 + B11)` — surface water and moisture.
  - **NDBI** (Normalized Difference Built-up Index): `(B11 - B8) / (B11 + B8)` — urban surfaces and impervious cover.
- **Leakage Prevention:**
  - **Spatial Block Partitioning:** 4×4 spatial block splitting (`spatial_block_split`) allocates whole geographic grid tiles to train (70%), validation (15%), and test (15%), preventing adjacent pixel autocorrelation leakage.
  - **Pre-fit Standardization:** `StandardScaler` fitted strictly on training partition and applied to validation/test partitions.
- **Scientific Integrity Caveat:** ESA WorldCover labels are from 2021. In multi-year comparisons, classifying composites for 2018 or 2024 introduces temporal label mismatch. All results explicitly communicate this distinction between real satellite reflectance, reference labels, and model predictions.

---

## 3. Actual Machine Learning Experiment Results

Models were trained and evaluated on 10,000 real Sentinel-2 satellite pixels for Coimbatore using a 4×4 spatial block split (Test set: $N=1,876$). Class imbalance ($72.3\%$ vegetation, $26.2\%$ built-up, $0.9\%$ bare, $0.6\%$ water) was accounted for via `class_weight="balanced"` and macro-averaged metrics.

| Metric | Random Forest (Balanced) | XGBoost (Optuna Tuned) | Evaluation Winner |
|---|---|---|---|
| **Overall Test Accuracy** | 86.14% | **87.26%** | XGBoost |
| **Balanced Accuracy** | **58.10%** | 54.63% | Random Forest |
| **Macro-Averaged F1** | **0.5536** | 0.5481 | Random Forest |
| **ROC-AUC (OVR)** | 0.8755 | **0.9509** | XGBoost |
| **Built-up F1-Score** | **0.7783** (Precision: 0.753, Recall: 0.805) | 0.7764 (Precision: 0.803, Recall: 0.751) | Random Forest |
| **Vegetation F1-Score** | 0.9071 (Precision: 0.920, Recall: 0.894) | **0.9161** (Precision: 0.899, Recall: 0.934) | XGBoost |
| **Water F1-Score** | 0.4000 (Precision: 0.333, Recall: 0.500) | **0.5000** (Precision: 0.500, Recall: 0.500) | XGBoost |
| **Bare Soil F1-Score** | **0.1290** (Precision: 0.133, Recall: 0.125) | 0.0000 (Precision: 0.000, Recall: 0.000) | Random Forest |

### Saved Artifacts (`artifacts/models/`)
- `rf_20261014T204814.joblib`: Random Forest model weights ($n=200, \text{depth}=20$)
- `xgb_20261014T204814.joblib`: XGBoost model weights ($n=337, \text{depth}=3, \text{lr}=0.079$)
- `scaler.joblib`: Fitted `StandardScaler`
- `results_20261014T204814.json`: Complete metrics, confusion matrices, and Optuna hyperparameters
- High-resolution evaluation plots: `rf_confusion_matrix_*.png`, `rf_feature_importance_*.png`, `xgb_confusion_matrix_*.png`, `xgb_feature_importance_*.png`, `spectral_signatures_*.png`

---

## 4. Interactive Web Application & Visualizations

The application provides **6 distinct, interactive visualizations** with verified units:
1. **Before & After Satellite Imagery Map:** Side-by-side synchronized Leaflet maps displaying satellite views for Year 1 vs Year 2, with study area boundaries and change hotspot indicators.
2. **Land-Cover Area by Class (Hectares / km²):** Recharts bar chart comparing total surface area per class across comparison years.
3. **Net Area Gained or Lost (Hectares):** Horizontal/vertical delta chart showing net hectares gained (positive) or lost (negative) per class.
4. **Land-Cover Transition Matrix:** Interactive heat-mapped matrix displaying exact pixel counts, hectares ($1\text{ pixel} = 100\text{ m}^2 = 0.01\text{ ha}$), and percentages from initial to final class.
5. **Spectral-Index Signal Comparison:** Side-by-side and delta comparison of NDVI, NDWI, and NDBI with simple layman explanations.
6. **Satellite Imagery Availability Timeline:** Bar and line timeline of cloud-filtered Sentinel-2 scenes per dry season (2017–2025).

---

## 5. How to Run Locally

### Prerequisites
- Python 3.11+ (Python 3.14 verified)
- Node.js 18+ (Node.js 20+ recommended)
- Git

### 1. Environment Setup
```powershell
# In project root: D:\AML CAT PRO\geochange-ai
python -m venv .venv
.venv\Scripts\activate
pip install -r backend/requirements.txt
```

### 2. Google Earth Engine Configuration
```powershell
# Authenticate your Google Cloud / GEE account:
earthengine authenticate

# Ensure .env contains:
GEE_PROJECT_ID=sincere-baton-478517-m2
GEE_AUTH_MODE=user
```

### 3. Initialize Database & Run Backend
```powershell
cd backend
python -m app.db.init_db
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
- API Health Check: `http://127.0.0.1:8000/api/v1/health`
- Swagger Interactive Documentation: `http://127.0.0.1:8000/docs`

### 4. Run Frontend Development Server
```powershell
cd frontend
npm install
npm run dev
```
- Frontend Web App: `http://127.0.0.1:5173/`

### 5. Run Automated Tests
```powershell
# From project root:
pytest tests/ -v
# Verified: 223 / 223 tests passing
```

### 6. Production Build
```powershell
cd frontend
npm run build
# Verified: dist/index.html + bundled assets created cleanly
```

---

## 6. Docker & Containerized Execution

The application is fully containerized using Docker Compose v2:

```powershell
# Validate compose configuration:
docker compose config

# Build containers:
docker compose build

# Run multi-container stack:
docker compose up -d

# Verify services:
docker compose ps
# Frontend: http://localhost:3000
# Backend:  http://localhost:8000/api/v1/health
```

### Headless GEE Authentication in Containers
For Docker containers or headless cloud servers (where interactive browser OAuth cannot run), use a Google Cloud Service Account:
1. Create a service account with Earth Engine Viewer/Resource permissions.
2. Download the JSON key to `configs/gee_service_account_key.json` (gitignored).
3. Set `GEE_SERVICE_ACCOUNT_EMAIL` and `GEE_SERVICE_ACCOUNT_KEY_PATH` in `.env`.
4. Alternatively, mount host credentials: `-v ~/.config/earthengine:/root/.config/earthengine:ro`.

---

## 7. Deployment Status & Platform Readiness

- **Source Control:** GitHub remote verified at `https://github.com/santhiyachandiran/geoview-aml.git`.
- **Live Hosted URL:** *Not yet publicly hosted.* In accordance with assignment restrictions, cloud deployment will not be triggered without explicit approval.
- **Deployment Preparation:**
  - Supported Platforms: Render, Hugging Face Spaces (Docker SDK), Google Cloud Run.
  - Required Environment Variables: `GEE_PROJECT_ID`, `APP_ENV=production`, `CORS_ALLOWED_ORIGINS`, `SQLITE_DB_PATH`.
  - Health check endpoint configured: `GET /api/v1/health`.

---

## 8. Limitations & Known Boundaries

1. **Model Performance Page:** Temporarily postponed per user direction; standalone evaluation charts and results are served via `/api/v1/models/results/latest` and `/api/v1/models/plots`.
2. **Temporal Label Mismatch:** ESA WorldCover reference labels exist only for 2020/2021. Classification of other temporal composites represents model prediction, not verified in-situ ground truth.
3. **Cloud Contamination:** Monsoon months (June–October) in Tamil Nadu have high cloud cover; analysis is restricted to post-monsoon dry season (November–March).
4. **Spatial Resolution:** Sentinel-2 pixels are 10 m × 10 m ($100\text{ m}^2$). Sub-pixel structures (narrow roads, small buildings) are mixed pixels.

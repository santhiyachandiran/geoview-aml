# GEOVIEW AML — Deployment Guide

## Local development (no Docker)

```powershell
# From project root, venv activated
cd "D:\AML CAT PRO\geochange-ai"
.venv\Scripts\Activate.ps1

# Initialize database
cd backend
python -m app.db.init_db

# Start backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# In a second terminal — start frontend
cd frontend
npm run dev   # → http://localhost:5173
```

## ML pipeline (requires GEE authentication)

```powershell
# 1. Authenticate once
earthengine authenticate

# 2. Set project ID in .env
#    GEE_PROJECT_ID=your-cloud-project-id

# 3. Verify access
python scripts/check_gee_auth.py

# 4. Run full pipeline (real data)
python scripts/run_pipeline.py --year 2021 --trials 30

# 5. View experiments
mlflow ui --backend-store-uri artifacts/mlruns
# → http://localhost:5000

# Demo mode (no GEE, synthetic data — do not cite as real results)
python scripts/run_pipeline.py --demo --trials 5
```

## Docker (local)

```powershell
docker compose build
docker compose up -d
docker compose ps
docker compose logs -f backend
docker compose down
```

API available at http://localhost:8000/docs
Frontend at http://localhost:3000

**Note:** Trained model artifacts in `artifacts/models/` are not copied into
the Docker image at build time. Mount the volume or copy them in after training.

## Cloud deployment — Hugging Face Spaces (recommended for free tier)

HuggingFace Spaces supports Docker-based deployments. Steps:

1. Create a free account at https://huggingface.co
2. Create a new Space → Docker SDK
3. Push the project to a HuggingFace repo:
   ```bash
   git remote add hf https://huggingface.co/spaces/<username>/geoview-aml
   git push hf main
   ```
4. Add secrets in Space settings:
   - `GEE_PROJECT_ID`
   - `GEE_AUTH_MODE=service_account`
   - `GEE_SERVICE_ACCOUNT_KEY_PATH=/app/configs/gee_key.json`
   (Upload the service account JSON as a Space secret file)

**Limitations on HuggingFace free tier:**
- 16 GB RAM, 2 vCPUs — sufficient for inference, not model training
- No persistent storage between restarts (use external DB for production)
- Cold starts after inactivity

## Cloud deployment — AWS/GCP

For AWS: deploy backend as ECS Fargate + ALB; serve frontend from S3/CloudFront.
For GCP: deploy backend as Cloud Run; serve frontend from Firebase Hosting.

Both require:
- Container registry (ECR or GCR) to push the Docker image
- Secret Manager for GEE service account key
- Persistent storage (EFS or GCS bucket) for model artifacts

These steps require manual account setup and are not automated here.
**Do not claim deployment is live until the URL is tested.**

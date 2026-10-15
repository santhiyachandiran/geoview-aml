# GEOVIEW AML — Backend Dockerfile
# Build context: project root (see compose.yaml)
# Uses only minimal runtime libs — geospatial packages install from binary wheels.
# No compilation required; no libgdal-dev needed.

FROM python:3.12-slim

RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Minimal runtime C libraries required by binary geospatial wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
 && rm -rf /var/lib/apt/lists/*

# Python dependencies from pinned requirements
COPY backend/requirements-docker.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Application source (build context = project root)
COPY backend/app/ ./app/
COPY ml/ ./ml/

# Runtime directories
RUN mkdir -p /app/artifacts/models /app/artifacts/exports \
             /app/artifacts/mlruns /app/logs /app/data \
 && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=25s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

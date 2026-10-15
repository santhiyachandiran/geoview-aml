"""
GeoChange AI — Application Configuration
Uses pydantic-settings to load from environment variables and .env file.

The .env file is located at the project root (two levels above this file:
  config.py → core/ → app/ → backend/ → geochange-ai/.env)
Using an absolute path here means the app loads .env correctly regardless of
which directory uvicorn is started from.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, computed_field

# Absolute path to project root: config.py is at backend/app/core/config.py
# parents: [0]=core  [1]=app  [2]=backend  [3]=geochange-ai (project root)
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_ENV_FILE = _PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),          # absolute — works from any cwd
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────
    app_env: str = Field(default="development")
    app_debug: bool = Field(default=True)
    app_host: str = Field(default="0.0.0.0")
    app_port: int = Field(default=8000)

    # ── Database ─────────────────────────────────────────────
    sqlite_db_path: str = Field(default="./backend/geochange.db")

    # ── Google Earth Engine ───────────────────────────────────
    gee_auth_mode: str = Field(default="user")         # "user" | "service_account"
    gee_project_id: str = Field(default="")
    gee_service_account_email: str = Field(default="")
    gee_service_account_key_path: str = Field(default="")

    # ── CORS ─────────────────────────────────────────────────
    cors_allowed_origins: str = Field(
        default="http://localhost:5173,http://localhost:3000"
    )

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_allowed_origins.split(",") if o.strip()]

    # ── Paths ─────────────────────────────────────────────────
    model_artifacts_dir: str = Field(default="./artifacts/models")
    export_dir: str = Field(default="./artifacts/exports")
    mlflow_tracking_uri: str = Field(default="./artifacts/mlruns")

    # ── Logging ──────────────────────────────────────────────
    log_level: str = Field(default="INFO")
    log_file: str = Field(default="./logs/geochange.log")

    # ── Study Area Defaults ───────────────────────────────────
    default_study_area: str = Field(default="coimbatore")
    default_start_year: int = Field(default=2018)
    default_end_year: int = Field(default=2024)

    @property
    def resolved_db_path(self) -> Path:
        """
        Always returns an absolute Path to the SQLite database, resolved
        relative to the project root if the configured value is relative.

        Use this everywhere instead of reimplementing path resolution.
        """
        p = Path(self.sqlite_db_path)
        if p.is_absolute():
            return p
        return (_PROJECT_ROOT / self.sqlite_db_path.lstrip("./")).resolve()

    @property
    def resolved_artifacts_dir(self) -> Path:
        p = Path(self.model_artifacts_dir)
        if p.is_absolute():
            return p
        return (_PROJECT_ROOT / self.model_artifacts_dir.lstrip("./")).resolve()


# Singleton — import this wherever config is needed
settings = Settings()

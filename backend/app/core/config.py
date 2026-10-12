"""
GeoChange AI — Application Configuration
Uses pydantic-settings to load from environment variables and .env file.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
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
    sqlite_db_path: str = Field(default="./geochange.db")

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


# Singleton — import this wherever config is needed
settings = Settings()

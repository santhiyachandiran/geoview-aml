-- =============================================================
-- GeoChange AI — SQLite Database Schema
-- Stores analysis metadata and history only.
-- Large rasters and model files are stored on disk (artifacts/).
-- =============================================================

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- ─── Study Areas ──────────────────────────────────────────────
-- Pre-defined locations. Users can also supply a custom bbox.
CREATE TABLE IF NOT EXISTS study_areas (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,              -- e.g. "Coimbatore, India"
    slug        TEXT NOT NULL UNIQUE,       -- e.g. "coimbatore"
    bbox_west   REAL NOT NULL,              -- longitude min
    bbox_south  REAL NOT NULL,              -- latitude min
    bbox_east   REAL NOT NULL,              -- longitude max
    bbox_north  REAL NOT NULL,              -- latitude max
    crs         TEXT NOT NULL DEFAULT 'EPSG:4326',
    description TEXT,
    created_at  DATETIME DEFAULT (datetime('now'))
);

-- ─── Analyses ─────────────────────────────────────────────────
-- One record per change-detection run requested by the user.
CREATE TABLE IF NOT EXISTS analyses (
    id                  TEXT PRIMARY KEY,   -- UUID
    study_area_id       INTEGER REFERENCES study_areas(id),
    custom_bbox         TEXT,               -- JSON "{"west":x,"south":y,...}" if custom
    date_start          TEXT NOT NULL,      -- ISO date string for Period 1 composite
    date_end            TEXT NOT NULL,      -- ISO date string for Period 2 composite
    season_start_month  INTEGER,            -- e.g. 11 for November
    season_end_month    INTEGER,            -- e.g. 2 for February
    data_source         TEXT NOT NULL,      -- e.g. "sentinel2", "landsat8"
    model_name          TEXT,               -- e.g. "random_forest_v1"
    model_version       TEXT,
    status              TEXT NOT NULL DEFAULT 'pending',
    -- status: pending | running | completed | failed | cancelled
    error_message       TEXT,
    created_at          DATETIME DEFAULT (datetime('now')),
    started_at          DATETIME,
    completed_at        DATETIME,

    -- User-facing data quality info
    cloud_cover_pct_t1  REAL,              -- cloud cover % for Period 1 composite
    cloud_cover_pct_t2  REAL,              -- cloud cover % for Period 2 composite
    scene_count_t1      INTEGER,           -- number of scenes composited for Period 1
    scene_count_t2      INTEGER,           -- number of scenes composited for Period 2
    actual_date_t1      TEXT,              -- actual earliest/latest scene date in composite
    actual_date_t2      TEXT,

    -- Output file references (relative paths under artifacts/)
    output_change_map   TEXT,              -- e.g. "exports/{id}/change_map.tif"
    output_stats_json   TEXT,              -- e.g. "exports/{id}/statistics.json"
    output_geojson      TEXT,
    output_csv          TEXT
);

-- ─── Classification Results ────────────────────────────────────
-- Per-analysis summary of land-cover area for each class and date.
CREATE TABLE IF NOT EXISTS classification_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id     TEXT NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
    period          TEXT NOT NULL,         -- "t1" or "t2"
    class_code      INTEGER NOT NULL,      -- 0=built-up, 1=vegetation, 2=water, 3=bare
    class_name      TEXT NOT NULL,
    pixel_count     INTEGER NOT NULL,
    area_ha         REAL NOT NULL,         -- area in hectares
    area_km2        REAL NOT NULL,
    area_pct        REAL NOT NULL          -- percentage of total area
);

-- ─── Change Transitions ───────────────────────────────────────
-- Each row is one from→to transition with its area.
CREATE TABLE IF NOT EXISTS change_transitions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id     TEXT NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
    from_class_code INTEGER NOT NULL,
    from_class_name TEXT NOT NULL,
    to_class_code   INTEGER NOT NULL,
    to_class_name   TEXT NOT NULL,
    pixel_count     INTEGER NOT NULL,
    area_ha         REAL NOT NULL,
    area_km2        REAL NOT NULL,
    area_pct        REAL NOT NULL          -- percentage of total study area
);

-- ─── Model Metrics ────────────────────────────────────────────
-- Per-model evaluation results, linked to an analysis or a standalone training run.
CREATE TABLE IF NOT EXISTS model_metrics (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id     TEXT REFERENCES analyses(id) ON DELETE SET NULL,
    model_name      TEXT NOT NULL,
    model_version   TEXT,
    mlflow_run_id   TEXT,                  -- MLflow run ID for full tracking
    dataset_info    TEXT,                  -- JSON: training dates, split strategy
    macro_f1        REAL,
    accuracy        REAL,
    mean_iou        REAL,
    precision_json  TEXT,                  -- JSON: per-class precision
    recall_json     TEXT,                  -- JSON: per-class recall
    f1_json         TEXT,                  -- JSON: per-class F1
    confusion_matrix TEXT,                 -- JSON 2D array
    notes           TEXT,
    created_at      DATETIME DEFAULT (datetime('now'))
);

-- ─── Imagery Metadata Cache ────────────────────────────────────
-- Cache GEE metadata so we don't re-query on every page load.
CREATE TABLE IF NOT EXISTS imagery_metadata_cache (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    cache_key       TEXT NOT NULL UNIQUE,  -- hash of (bbox, date_range, collection)
    collection      TEXT NOT NULL,         -- e.g. "COPERNICUS/S2_SR_HARMONIZED"
    bbox_json       TEXT NOT NULL,
    date_from       TEXT NOT NULL,
    date_to         TEXT NOT NULL,
    scene_count     INTEGER,
    cloud_cover_pct REAL,
    earliest_date   TEXT,
    latest_date     TEXT,
    metadata_json   TEXT,                  -- full GEE response JSON
    fetched_at      DATETIME DEFAULT (datetime('now')),
    expires_at      DATETIME               -- cache expiry
);

-- ─── Seed data: pre-defined study areas ───────────────────────
INSERT OR IGNORE INTO study_areas (name, slug, bbox_west, bbox_south, bbox_east, bbox_north, description)
VALUES
  ('Coimbatore, Tamil Nadu, India',
   'coimbatore',
   76.8, 10.9, 77.2, 11.1,
   'Primary study area. Rapid urban expansion from IT corridor growth visible between 2018 and 2024. Surrounded by agricultural land, Nilgiri foothills, and Siruvani watershed.'),

  ('Chennai, Tamil Nadu, India',
   'chennai',
   80.1, 12.9, 80.4, 13.2,
   'Coastal metropolitan region. Useful for peri-urban growth and wetland change analysis.'),

  ('Bengaluru Urban, Karnataka, India',
   'bengaluru',
   77.4, 12.8, 77.8, 13.1,
   'One of the fastest urbanising areas in South Asia. Extensive agricultural-to-built-up transition visible in Sentinel-2 archive.');

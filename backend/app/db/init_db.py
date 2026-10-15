"""
GeoChange AI — Database Initialization

Run this script once to create the SQLite database from schema.sql:

    python -m app.db.init_db

Or call init_database() from application startup code.
"""

import sqlite3
from pathlib import Path

from loguru import logger
from app.core.config import settings


SCHEMA_FILE = Path(__file__).parent / "schema.sql"


def init_database(db_path: str | None = None) -> Path:
    """
    Create the SQLite database and run the schema if it doesn't exist.

    Args:
        db_path: Explicit path to the .db file.
                 Defaults to settings.resolved_db_path (always absolute).

    Returns:
        The resolved path to the database file.
    """
    if db_path is not None:
        resolved = Path(db_path).resolve()
    else:
        resolved = settings.resolved_db_path
    resolved.parent.mkdir(parents=True, exist_ok=True)

    if not SCHEMA_FILE.exists():
        raise FileNotFoundError(f"Schema file not found: {SCHEMA_FILE}")

    schema_sql = SCHEMA_FILE.read_text(encoding="utf-8")

    logger.info(f"Initialising database at: {resolved}")

    conn = sqlite3.connect(str(resolved))
    try:
        conn.executescript(schema_sql)
        conn.commit()
    finally:
        conn.close()

    logger.success(f"Database ready: {resolved}")
    return resolved


def verify_database(db_path: str | None = None) -> dict:
    """
    Check the database exists and all expected tables are present.

    Args:
        db_path: Explicit path. Defaults to settings.resolved_db_path.
    """
    if db_path is not None:
        resolved = Path(db_path).resolve()
    else:
        resolved = settings.resolved_db_path
    expected_tables = {
        "study_areas",
        "analyses",
        "classification_results",
        "change_transitions",
        "model_metrics",
        "imagery_metadata_cache",
    }

    if not resolved.exists():
        return {"path": str(resolved), "exists": False, "tables": [], "study_area_count": 0}

    conn = sqlite3.connect(str(resolved))
    try:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        found_tables = {row[0] for row in cursor.fetchall()}

        study_area_count = 0
        if "study_areas" in found_tables:
            row = conn.execute("SELECT COUNT(*) FROM study_areas").fetchone()
            study_area_count = row[0] if row else 0

        missing = expected_tables - found_tables
    finally:
        conn.close()

    return {
        "path": str(resolved),
        "exists": True,
        "tables": sorted(found_tables),
        "study_area_count": study_area_count,
        "missing_tables": sorted(missing),
        "schema_complete": len(missing) == 0,
    }


if __name__ == "__main__":
    import sys

    # Allow overriding the db path from the command line
    db_path = sys.argv[1] if len(sys.argv) > 1 else None

    result_path = init_database(db_path)
    verification = verify_database(str(result_path))

    print("\n── Database Verification ─────────────────────")
    print(f"  Path            : {verification['path']}")
    print(f"  Exists          : {verification['exists']}")
    print(f"  Tables found    : {verification['tables']}")
    print(f"  Missing tables  : {verification.get('missing_tables', [])}")
    print(f"  Schema complete : {verification['schema_complete']}")
    print(f"  Study areas     : {verification['study_area_count']}")

    if not verification["schema_complete"]:
        print("\nERROR: Schema is incomplete. Check schema.sql.")
        sys.exit(1)

    print("\nDatabase initialised successfully.")

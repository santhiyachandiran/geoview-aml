"""
GEOVIEW AML — GEE Authentication Check

Run this BEFORE run_pipeline.py to confirm GEE access works end-to-end.

Usage (from project root, venv activated):
    python scripts/check_gee_auth.py

What it checks:
  1. earthengine credentials file exists
  2. ee.Initialize() succeeds with GEE_PROJECT_ID from .env
  3. A minimal Sentinel-2 collection query over Coimbatore returns > 0 scenes
  4. ESA WorldCover 2021 asset is accessible

Exit codes:
  0 = all checks passed
  1 = one or more checks failed (details printed)
"""

from __future__ import annotations
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

CRED_PATH = Path.home() / ".config" / "earthengine" / "credentials"
BBOX      = [76.8, 10.9, 77.2, 11.1]


def check(label: str, ok: bool, detail: str = ""):
    mark = "[OK]" if ok else "[FAIL]"
    print(f"  {mark}  {label}")
    if detail:
        print(f"         {detail}")
    return ok


def main():
    print("\nGEEVIEW AML — GEE Authentication Check")
    print("=" * 50)
    all_ok = True

    # 1. Credentials file
    ok = check(
        "Credentials file",
        CRED_PATH.exists(),
        f"Expected at: {CRED_PATH}"
        if not CRED_PATH.exists()
        else str(CRED_PATH),
    )
    all_ok = all_ok and ok
    if not ok:
        print("\n  Run:  earthengine authenticate")
        print("  Then re-run this script.\n")
        sys.exit(1)

    # 2. Load GEE_PROJECT_ID from .env
    from app.core.config import settings
    project_id = settings.gee_project_id
    ok = check(
        "GEE_PROJECT_ID configured",
        bool(project_id) and project_id != "your-google-cloud-project-id",
        project_id or "NOT SET — edit .env",
    )
    all_ok = all_ok and ok
    if not ok:
        print("\n  Edit .env and set GEE_PROJECT_ID=your-cloud-project-id\n")
        sys.exit(1)

    # 3. ee.Initialize
    try:
        import ee
        ee.Initialize(project=project_id)
        ok = check("ee.Initialize()", True, f"project={project_id}")
    except Exception as exc:
        ok = check("ee.Initialize()", False, str(exc))
    all_ok = all_ok and ok
    if not ok:
        sys.exit(1)

    # 4. Sentinel-2 query
    try:
        aoi = ee.Geometry.Rectangle(BBOX)
        n = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(aoi)
            .filterDate("2021-11-01", "2022-02-28")
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
            .size()
            .getInfo()
        )
        ok = check(
            "Sentinel-2 query (Coimbatore Nov-2021–Feb-2022)",
            n > 0,
            f"{n} scene(s) found",
        )
    except Exception as exc:
        ok = check("Sentinel-2 query", False, str(exc))
    all_ok = all_ok and ok

    # 5. WorldCover access
    try:
        aoi = ee.Geometry.Rectangle(BBOX)
        wc  = ee.ImageCollection("ESA/WorldCover/v200").first().clip(aoi)
        wc.getInfo()   # triggers actual server call
        ok = check("ESA WorldCover v200 accessible", True)
    except Exception as exc:
        ok = check("ESA WorldCover v200 accessible", False, str(exc))
    all_ok = all_ok and ok

    print()
    if all_ok:
        print("  All checks passed. Run: python scripts/run_pipeline.py")
    else:
        print("  Some checks failed. Fix the issues above before running the pipeline.")
    print()
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

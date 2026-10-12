"""
pytest configuration for GeoChange AI tests.
Sets up paths and environment variables before any test module is imported.
"""
import os
import sys

# Ensure backend is importable from any test file
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

# Suppress MLflow startup messages in test output
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

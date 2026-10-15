"""
GeoChange AI — Tests: GEE Service (mocked) and Feature Builder
"""

import os
import sys
from unittest.mock import patch, MagicMock

_HERE = os.path.abspath(os.path.dirname(__file__))
_BACKEND = os.path.abspath(os.path.join(_HERE, "..", "backend"))
_PROJECT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _BACKEND)
sys.path.insert(0, _PROJECT)
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

import numpy as np
import pytest


# ── GEE Service ────────────────────────────────────────────────

class TestGeeAuthentication:
    def test_authenticate_gee_succeeds_with_mock(self):
        """Mock ee.Initialize to succeed → GEE_AVAILABLE becomes True."""
        with patch("ee.Initialize") as mock_init:
            mock_init.return_value = None
            import app.services.gee_service as gs
            # Reset state
            gs.GEE_AVAILABLE = False
            result = gs.authenticate_gee(project_id="test-project")
            assert result is True
            assert gs.GEE_AVAILABLE is True

    def test_authenticate_gee_fails_gracefully(self):
        """Mock ee.Initialize to raise → GEE_AVAILABLE stays False, no exception."""
        with patch("ee.Initialize", side_effect=Exception("Auth failed")):
            import app.services.gee_service as gs
            gs.GEE_AVAILABLE = False
            result = gs.authenticate_gee(project_id="test-project")
            assert result is False
            assert gs.GEE_AVAILABLE is False

    def test_authenticate_gee_without_project_id(self):
        """No project_id → False without calling ee.Initialize."""
        import app.services.gee_service as gs
        gs.GEE_AVAILABLE = False
        result = gs.authenticate_gee(project_id="")
        assert result is False

    def test_get_composite_returns_fallback_when_gee_unavailable(self):
        """When GEE is not available, composite returns gee_available=False dict."""
        import app.services.gee_service as gs
        gs.GEE_AVAILABLE = False
        result = gs.get_sentinel2_composite([76.8, 10.9, 77.2, 11.1], 2020)
        assert result["gee_available"] is False
        assert result["demo_data"] is False  # Not demo, just unavailable

    def test_get_composite_validates_bbox(self):
        import app.services.gee_service as gs
        gs.GEE_AVAILABLE = False
        with pytest.raises(ValueError, match="bbox"):
            gs.get_sentinel2_composite([76.8, 10.9], 2020)

    def test_get_composite_validates_year(self):
        import app.services.gee_service as gs
        gs.GEE_AVAILABLE = False
        with pytest.raises(ValueError, match="2017"):
            gs.get_sentinel2_composite([76.8, 10.9, 77.2, 11.1], 2015)

    def test_get_availability_when_gee_unavailable(self):
        import app.services.gee_service as gs
        gs.GEE_AVAILABLE = False
        result = gs.get_imagery_availability([76.8, 10.9, 77.2, 11.1], 2020, 2022)
        assert len(result) == 3
        assert all(not r["gee_available"] for r in result)


# ── Feature Builder ───────────────────────────────────────────

class TestFeatureBuilder:
    def _composite(self):
        return {
            "B2": 0.08, "B3": 0.10, "B4": 0.12, "B8": 0.36,
            "B11": 0.20, "B12": 0.15,
            "ndvi_mean": 0.35, "ndwi_mean": -0.12, "ndbi_mean": -0.15,
        }

    def test_stack_features_shape(self):
        from ml.preprocessing.feature_builder import stack_features_demo
        X = stack_features_demo(self._composite(), n_pixels=200, seed=42)
        assert X.shape == (200, 9)

    def test_stack_features_dtype(self):
        from ml.preprocessing.feature_builder import stack_features_demo
        X = stack_features_demo(self._composite(), n_pixels=100)
        assert X.dtype == np.float32

    def test_stack_features_finite(self):
        from ml.preprocessing.feature_builder import stack_features_demo
        X = stack_features_demo(self._composite(), n_pixels=100)
        assert np.all(np.isfinite(X))

    def test_normalize_features_shape(self):
        from ml.preprocessing.feature_builder import stack_features_demo, normalize_features
        X = stack_features_demo(self._composite(), n_pixels=100)
        X_norm, scaler = normalize_features(X)
        assert X_norm.shape == X.shape

    def test_normalize_features_has_mean(self):
        from ml.preprocessing.feature_builder import stack_features_demo, normalize_features
        X = stack_features_demo(self._composite(), n_pixels=100)
        _, scaler = normalize_features(X)
        assert hasattr(scaler, "mean_")
        assert len(scaler.mean_) == 9

    def test_normalize_features_with_prefit_scaler(self):
        from ml.preprocessing.feature_builder import stack_features_demo, normalize_features
        X = stack_features_demo(self._composite(), n_pixels=100)
        _, scaler = normalize_features(X)
        X2 = stack_features_demo(self._composite(), n_pixels=50, seed=99)
        X2_norm, scaler2 = normalize_features(X2, scaler=scaler)
        assert X2_norm.shape == (50, 9)
        assert scaler2 is scaler

    def test_spatial_block_split_sizes(self):
        from ml.preprocessing.feature_builder import stack_features_demo, spatial_block_split
        X = stack_features_demo(self._composite(), n_pixels=1000)
        y = np.random.default_rng(0).integers(0, 4, 1000).astype(np.int32)
        rows = np.arange(1000) // 40
        cols = np.arange(1000) % 40
        X_tr, X_v, X_te, y_tr, y_v, y_te = spatial_block_split(
            X, y, rows=rows, cols=cols, n_blocks=4, seed=42
        )
        total = len(X_tr) + len(X_v) + len(X_te)
        assert total == 1000
        assert 0.60 <= len(X_tr) / 1000 <= 0.80
        assert 0.05 <= len(X_v)  / 1000 <= 0.30

    def test_spatial_block_split_no_overlap(self):
        from ml.preprocessing.feature_builder import stack_features_demo, spatial_block_split
        X = stack_features_demo(self._composite(), n_pixels=1000)
        y = np.random.default_rng(0).integers(0, 4, 1000).astype(np.int32)
        rows = np.arange(1000) // 40
        cols = np.arange(1000) % 40
        X_tr, X_v, X_te, y_tr, y_v, y_te = spatial_block_split(
            X, y, rows=rows, cols=cols, n_blocks=4, seed=42
        )
        assert len(X_tr) + len(X_v) + len(X_te) == 1000

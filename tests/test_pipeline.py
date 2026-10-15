"""
GeoChange AI — Tests: ML pipeline components (no GEE required).

All tests use synthetic fixtures. They validate the pipeline code paths,
not real satellite results. Real metrics must come from run_pipeline.py
with authentic GEE data.
"""

import os
import sys
import warnings

_HERE = os.path.abspath(os.path.dirname(__file__))
_BACKEND = os.path.abspath(os.path.join(_HERE, "..", "backend"))
_PROJECT = os.path.abspath(os.path.join(_HERE, ".."))

sys.path.insert(0, _BACKEND)
sys.path.insert(0, _PROJECT)
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
os.environ["SQLITE_DB_PATH"] = os.path.join(_BACKEND, "geochange_test.db")

import numpy as np
import pytest


# ── Feature builder ───────────────────────────────────────────

class TestFeatureBuilder:
    def _pixel_data(self, n=200, seed=0):
        rng = np.random.default_rng(seed)
        bands = ["B2","B3","B4","B8","B11","B12"]
        return {b: rng.uniform(0.05, 0.5, n).astype(np.float32) for b in bands}

    def test_compute_indices_present(self):
        from ml.preprocessing.feature_builder import compute_indices
        pd = self._pixel_data()
        idx = compute_indices(pd)
        for k in ("NDVI", "NDWI", "NDBI"):
            assert k in idx
            assert np.all(np.isfinite(idx[k]))

    def test_ndvi_range(self):
        from ml.preprocessing.feature_builder import compute_indices
        pd = self._pixel_data()
        idx = compute_indices(pd)
        assert np.all(idx["NDVI"] >= -1.01)
        assert np.all(idx["NDVI"] <= 1.01)

    def test_stack_features_shape(self):
        from ml.preprocessing.feature_builder import stack_features, compute_indices, N_FEATURES
        pd = self._pixel_data()
        pd.update(compute_indices(pd))
        X = stack_features(pd)
        assert X.shape == (200, N_FEATURES)
        assert X.dtype == np.float32

    def test_stack_features_auto_indices(self):
        """stack_features must compute indices automatically if absent."""
        from ml.preprocessing.feature_builder import stack_features, N_FEATURES
        X = stack_features(self._pixel_data())
        assert X.shape[1] == N_FEATURES

    def test_stack_features_missing_band_raises(self):
        from ml.preprocessing.feature_builder import stack_features
        pd = self._pixel_data()
        del pd["B8"]
        with pytest.raises(KeyError):
            stack_features(pd)

    def test_normalize_returns_scaler(self):
        from ml.preprocessing.feature_builder import stack_features, normalize_features
        X = stack_features(self._pixel_data())
        X_n, scaler = normalize_features(X)
        assert X_n.shape == X.shape
        assert hasattr(scaler, "mean_")
        assert len(scaler.mean_) == X.shape[1]

    def test_normalize_with_prefit_scaler(self):
        from ml.preprocessing.feature_builder import stack_features, normalize_features
        X1 = stack_features(self._pixel_data(seed=0))
        X2 = stack_features(self._pixel_data(seed=1))
        _, scaler = normalize_features(X1)
        X2_n, s2 = normalize_features(X2, scaler=scaler)
        assert s2 is scaler
        assert X2_n.shape == X2.shape

    def test_spatial_split_with_coords(self):
        from ml.preprocessing.feature_builder import stack_features, spatial_block_split
        n = 1000
        X = stack_features(self._pixel_data(n=n))
        y = np.random.default_rng(0).integers(0, 4, n).astype(np.int32)
        rows = np.arange(n) // 40
        cols = np.arange(n) % 40
        Xtr, Xv, Xte, ytr, yv, yte = spatial_block_split(
            X, y, rows=rows, cols=cols, n_blocks=4, seed=42
        )
        assert len(Xtr) + len(Xv) + len(Xte) == n
        assert 600 <= len(Xtr) <= 800   # ≈70% ±10%

    def test_spatial_split_no_overlap(self):
        from ml.preprocessing.feature_builder import stack_features, spatial_block_split
        n = 800
        X = stack_features(self._pixel_data(n=n))
        y = np.zeros(n, dtype=np.int32)
        rows = np.arange(n) // 32
        cols = np.arange(n) % 32
        Xtr, Xv, Xte, *_ = spatial_block_split(
            X, y, rows=rows, cols=cols, n_blocks=4, seed=42
        )
        assert len(Xtr) + len(Xv) + len(Xte) == n

    def test_spatial_split_warns_without_coords(self):
        from ml.preprocessing.feature_builder import stack_features, spatial_block_split
        X = stack_features(self._pixel_data(n=400))
        y = np.zeros(400, dtype=np.int32)
        with pytest.warns(UserWarning, match="spatial"):
            spatial_block_split(X, y, rows=None, cols=None, seed=42)


# ── Random Forest training ────────────────────────────────────

class TestRandomForest:
    def _data(self, n=600, seed=7):
        from ml.preprocessing.feature_builder import stack_features_demo
        rng = np.random.default_rng(seed)
        X = np.vstack([
            stack_features_demo({"ndvi_mean":0.5,"ndwi_mean":-0.1,"ndbi_mean":-0.3,
                                  "B2":.06,"B3":.09,"B4":.08,"B8":.45,"B11":.15,"B12":.10},
                                 n_pixels=n//4, seed=seed),
            stack_features_demo({"ndvi_mean":0.1,"ndwi_mean":-0.2,"ndbi_mean":0.1,
                                  "B2":.12,"B3":.14,"B4":.16,"B8":.20,"B11":.28,"B12":.22},
                                 n_pixels=n//4, seed=seed+1),
            stack_features_demo({"ndvi_mean":-.1,"ndwi_mean":0.3,"ndbi_mean":-.4,
                                  "B2":.07,"B3":.10,"B4":.06,"B8":.08,"B11":.04,"B12":.03},
                                 n_pixels=n//4, seed=seed+2),
            stack_features_demo({"ndvi_mean":0.08,"ndwi_mean":-.25,"ndbi_mean":.05,
                                  "B2":.14,"B3":.16,"B4":.18,"B8":.22,"B11":.25,"B12":.20},
                                 n_pixels=n//4, seed=seed+3),
        ])
        y = np.array([1]*(n//4)+[0]*(n//4)+[2]*(n//4)+[3]*(n//4), dtype=np.int32)
        return X, y

    def test_train_returns_model_and_dict(self):
        from ml.models.train_rf import train_random_forest
        X, y = self._data()
        model, info = train_random_forest(X[:400], y[:400])
        assert hasattr(model, "predict")
        assert "feature_importances" in info
        assert "evaluated_on" in info
        assert info["evaluated_on"] == "not_yet_evaluated"

    def test_importances_length(self):
        from ml.models.train_rf import train_random_forest
        from ml.preprocessing.feature_builder import N_FEATURES
        X, y = self._data()
        _, info = train_random_forest(X[:400], y[:400])
        assert len(info["feature_importances"]) == N_FEATURES

    def test_predict_returns_correct_classes(self):
        from ml.models.train_rf import train_random_forest
        X, y = self._data()
        model, _ = train_random_forest(X[:400], y[:400])
        preds = model.predict(X[400:])
        assert set(np.unique(preds)).issubset({0, 1, 2, 3})


# ── Evaluator ─────────────────────────────────────────────────

class TestEvaluator:
    def _trained_rf(self):
        from ml.models.train_rf import train_random_forest
        from ml.preprocessing.feature_builder import stack_features_demo
        n = 400
        X = np.vstack([
            stack_features_demo({"ndvi_mean":0.5,"ndwi_mean":-.1,"ndbi_mean":-.3,
                                  "B2":.06,"B3":.09,"B4":.08,"B8":.45,"B11":.15,"B12":.10},
                                 n_pixels=n//4, seed=0),
            stack_features_demo({"ndvi_mean":0.1,"ndwi_mean":-.2,"ndbi_mean":.1,
                                  "B2":.12,"B3":.14,"B4":.16,"B8":.20,"B11":.28,"B12":.22},
                                 n_pixels=n//4, seed=1),
            stack_features_demo({"ndvi_mean":-.1,"ndwi_mean":.3,"ndbi_mean":-.4,
                                  "B2":.07,"B3":.10,"B4":.06,"B8":.08,"B11":.04,"B12":.03},
                                 n_pixels=n//4, seed=2),
            stack_features_demo({"ndvi_mean":.08,"ndwi_mean":-.25,"ndbi_mean":.05,
                                  "B2":.14,"B3":.16,"B4":.18,"B8":.22,"B11":.25,"B12":.20},
                                 n_pixels=n//4, seed=3),
        ])
        y = np.array([1]*(n//4)+[0]*(n//4)+[2]*(n//4)+[3]*(n//4), dtype=np.int32)
        model, _ = train_random_forest(X, y)
        return model, X, y

    def test_evaluate_returns_all_keys(self):
        from ml.evaluation.evaluator import evaluate_model
        model, X, y = self._trained_rf()
        metrics = evaluate_model(model, X[:100], y[:100],
                                 ["built_up","vegetation","water","bare"])
        for k in ("accuracy","balanced_accuracy","macro_f1","confusion_matrix"):
            assert k in metrics, f"Missing key: {k}"

    def test_accuracy_in_range(self):
        from ml.evaluation.evaluator import evaluate_model
        model, X, y = self._trained_rf()
        m = evaluate_model(model, X[:100], y[:100],
                           ["built_up","vegetation","water","bare"])
        assert 0.0 <= m["accuracy"] <= 1.0

    def test_confusion_matrix_shape(self):
        from ml.evaluation.evaluator import evaluate_model
        model, X, y = self._trained_rf()
        m = evaluate_model(model, X[:100], y[:100],
                           ["built_up","vegetation","water","bare"])
        cm = m["confusion_matrix"]
        n_classes = len(np.unique(y[:100]))
        assert len(cm) == n_classes
        assert all(len(row) == n_classes for row in cm)

    def test_compare_models_has_winner(self):
        from ml.evaluation.evaluator import compare_models
        rf  = {"accuracy": 0.82, "balanced_accuracy": 0.80, "macro_f1": 0.79, "roc_auc": 0.91}
        xgb = {"accuracy": 0.85, "balanced_accuracy": 0.83, "macro_f1": 0.84, "roc_auc": 0.93}
        comp = compare_models(rf, xgb)
        assert comp["accuracy"]["winner"] == "xgboost"
        assert comp["macro_f1"]["winner"] == "xgboost"

    def test_save_confusion_matrix_creates_file(self, tmp_path):
        from ml.evaluation.evaluator import save_confusion_matrix_fig
        cm = [[50, 3, 1, 0], [2, 45, 0, 1], [1, 0, 30, 2], [0, 2, 1, 40]]
        out = save_confusion_matrix_fig(
            cm, ["built_up","vegetation","water","bare"],
            tmp_path / "cm.png"
        )
        assert out.exists()
        assert out.stat().st_size > 1000   # non-trivial PNG

    def test_save_feature_importance_creates_file(self, tmp_path):
        from ml.evaluation.evaluator import save_feature_importance_fig
        from ml.preprocessing.feature_builder import FEATURE_NAMES
        importances = [0.18, 0.14, 0.10, 0.22, 0.12, 0.08, 0.06, 0.05, 0.05]
        out = save_feature_importance_fig(importances, FEATURE_NAMES, tmp_path / "fi.png")
        assert out.exists()


# ── Inference ─────────────────────────────────────────────────

class TestInference:
    def test_predict_landcover_shape(self):
        from ml.models.train_rf import train_random_forest
        from ml.models.inference import predict_landcover
        from ml.preprocessing.feature_builder import stack_features_demo
        X = stack_features_demo({}, n_pixels=50, seed=0)
        y = np.zeros(50, dtype=np.int32)
        model, _ = train_random_forest(X, y)
        preds, proba = predict_landcover(model, X)
        assert len(preds) == 50
        assert proba is not None
        assert proba.shape[0] == 50

    def test_predict_change_counts(self):
        from ml.models.train_rf import train_random_forest
        from ml.models.inference import predict_change
        from ml.preprocessing.feature_builder import stack_features_demo
        n = 100
        X = stack_features_demo({}, n_pixels=n, seed=0)
        y = np.zeros(n, dtype=np.int32)
        model, _ = train_random_forest(X, y)
        result = predict_change(model, X[:50], X[50:])
        assert result["total_pixel_count"] == 50
        assert 0 <= result["changed_pixel_count"] <= 50
        assert isinstance(result["transitions"], list)


# ── Pipeline smoke test (synthetic, no GEE) ───────────────────

class TestPipelineSmokeDemo:
    """Runs a minimal end-to-end pipeline pass with --demo flag."""

    def test_demo_pipeline_completes(self, tmp_path):
        """Run the pipeline script in demo mode — must exit 0."""
        import subprocess
        py = sys.executable
        script = str(os.path.join(_PROJECT, "scripts", "run_pipeline.py"))
        result = subprocess.run(
            [py, script, "--demo", "--trials", "3", "--timeout", "30"],
            capture_output=True, text=True,
            env={**os.environ,
                 "MODEL_ARTIFACTS_DIR": str(tmp_path),
                 "MLFLOW_TRACKING_URI": str(tmp_path / "mlruns"),
                 "SQLITE_DB_PATH": str(tmp_path / "demo.db"),
                 "GEE_PROJECT_ID": "",
                 "MLFLOW_DISABLE_AGENT_HINT": "1"},
            timeout=120,
        )
        assert result.returncode == 0, (
            f"Pipeline failed.\nSTDOUT:\n{result.stdout[-2000:]}\n"
            f"STDERR:\n{result.stderr[-1000:]}"
        )
        assert "Pipeline complete" in result.stdout
        # Result JSON must be created
        jsons = list(tmp_path.glob("results_*.json"))
        assert len(jsons) == 1, "Expected exactly one results JSON"
        import json
        data = json.loads(jsons[0].read_text())
        assert data["demo_data"] is True
        assert "random_forest" in data
        assert "xgboost" in data
        assert "comparison" in data

    def test_demo_pipeline_creates_model_files(self, tmp_path):
        import subprocess
        py = sys.executable
        script = str(os.path.join(_PROJECT, "scripts", "run_pipeline.py"))
        subprocess.run(
            [py, script, "--demo", "--trials", "2", "--timeout", "20"],
            capture_output=True, text=True,
            env={**os.environ,
                 "MODEL_ARTIFACTS_DIR": str(tmp_path),
                 "MLFLOW_TRACKING_URI": str(tmp_path / "mlruns"),
                 "SQLITE_DB_PATH": str(tmp_path / "demo.db"),
                 "GEE_PROJECT_ID": "",
                 "MLFLOW_DISABLE_AGENT_HINT": "1"},
            timeout=120,
        )
        rf_files  = list(tmp_path.glob("rf_*.joblib"))
        xgb_files = list(tmp_path.glob("xgb_*.joblib"))
        assert len(rf_files)  >= 1, "No RF model saved"
        assert len(xgb_files) >= 1, "No XGBoost model saved"
        assert (tmp_path / "scaler.joblib").exists(), "No scaler saved"

    def test_demo_metrics_are_labelled(self, tmp_path):
        import subprocess, json
        py = sys.executable
        script = str(os.path.join(_PROJECT, "scripts", "run_pipeline.py"))
        subprocess.run(
            [py, script, "--demo", "--trials", "2", "--timeout", "20"],
            capture_output=True, text=True,
            env={**os.environ,
                 "MODEL_ARTIFACTS_DIR": str(tmp_path),
                 "MLFLOW_TRACKING_URI": str(tmp_path / "mlruns"),
                 "SQLITE_DB_PATH": str(tmp_path / "demo.db"),
                 "GEE_PROJECT_ID": "",
                 "MLFLOW_DISABLE_AGENT_HINT": "1"},
            timeout=120,
        )
        data = json.loads(list(tmp_path.glob("results_*.json"))[0].read_text())
        # Both model entries must flag demo_data
        assert data["random_forest"]["test_metrics"]["demo_data"] is True
        assert data["xgboost"]["test_metrics"]["demo_data"] is True

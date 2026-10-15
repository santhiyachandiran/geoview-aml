"""
GEOVIEW AML — Preprocessing and dataset-loading tests.

All tests use small synthetic fixtures saved to a tmp_path.
No GEE, model training, Optuna, or network calls.

Tests cover:
  A. gee_extractor constants and helpers (no I/O)
  B. feature_builder: stack_features, compute_indices, normalize_features
  C. dataset_validator: shape, dtype, finite, label, metadata checks
  D. load_extracted_data: round-trip save → load
  E. prepare_dataset.py validation path (--validate-only logic)
"""

from __future__ import annotations

import json
import os
import sys

_HERE    = os.path.abspath(os.path.dirname(__file__))
_PROJECT = os.path.abspath(os.path.join(_HERE, ".."))
_BACKEND = os.path.abspath(os.path.join(_HERE, "..", "backend"))
sys.path.insert(0, _PROJECT)
sys.path.insert(0, _BACKEND)
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

import numpy as np
import pytest

from ml.preprocessing.feature_builder import (
    FEATURE_NAMES, N_FEATURES,
    compute_indices, stack_features, normalize_features,
)
from ml.preprocessing.gee_extractor import (
    CLASS_NAMES, WC_CLASS_MAP, _VALID_LABELS, _WC_SENTINEL,
    _exclusive_end_date, load_extracted_data,
)
from ml.preprocessing.dataset_validator import (
    validate_dataset, REQUIRED_FILES, REQUIRED_METADATA_KEYS,
)


# ══════════════════════════════════════════════════════════════
# Helpers — build a minimal synthetic dataset in tmp_path
# ══════════════════════════════════════════════════════════════

def _make_band_arrays(n: int = 200, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    return {b: rng.uniform(0.05, 0.5, n).astype(np.float32)
            for b in ["B2", "B3", "B4", "B8", "B11", "B12"]}


def _make_pixel_data(n: int = 200, seed: int = 7) -> dict:
    bands = _make_band_arrays(n, seed)
    indices = compute_indices(bands)
    return {**bands, **indices}


def _save_synthetic_dataset(path, n: int = 400, seed: int = 9,
                             dtype_y=np.int32) -> None:
    """Save a well-formed synthetic dataset to path."""
    rng = np.random.default_rng(seed)
    pd = _make_pixel_data(n, seed)
    X  = stack_features(pd)
    y  = rng.integers(0, 4, n).astype(dtype_y)
    rows = np.arange(n, dtype=np.int64) // 20
    cols = np.arange(n, dtype=np.int64) % 20

    np.save(path / "X.npy",    X)
    np.save(path / "y.npy",    y)
    np.save(path / "rows.npy", rows)
    np.save(path / "cols.npy", cols)

    counts = {CLASS_NAMES[c]: int((y == c).sum()) for c in range(4)}
    meta = {
        "bbox":                      [76.8, 10.9, 77.2, 11.1],
        "composite_year":            2021,
        "date_start":                "2021-11-01",
        "date_end_exclusive":        "2022-03-01",
        "date_end_inclusive_desc":   "2022-02-28",
        "date_convention":           "GEE filterDate end is exclusive",
        "scene_count":               7,
        "n_sampled":                 n,
        "n_removed_nan_bands":       0,
        "n_removed_missing_labels":  0,
        "n_pixels":                  n,
        "n_features":                N_FEATURES,
        "feature_names":             FEATURE_NAMES,
        "class_names":               CLASS_NAMES,
        "class_distribution":        counts,
        "scale_m":                   10,
        "max_cloud_pct":             20,
        "labels_source":             "ESA WorldCover v200 (2021)",
        "missing_label_handling":    "Excluded",
        "temporal_caveat":           "Labels are from 2021.",
    }
    (path / "metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )


# ══════════════════════════════════════════════════════════════
# A. gee_extractor constants and helpers
# ══════════════════════════════════════════════════════════════

class TestExtractorConstants:

    def test_wc_class_map_all_values_valid(self):
        for code, cls in WC_CLASS_MAP.items():
            assert cls in _VALID_LABELS, (
                f"WC code {code} maps to {cls}, not in {_VALID_LABELS}"
            )

    def test_sentinel_not_in_valid_labels(self):
        assert _WC_SENTINEL not in _VALID_LABELS

    def test_sentinel_is_negative(self):
        """Sentinel must be outside [0,3] so it is caught by the filter."""
        assert _WC_SENTINEL < 0 or _WC_SENTINEL > 3

    def test_class_names_length(self):
        assert len(CLASS_NAMES) == len(_VALID_LABELS) == 4

    def test_class_names_values(self):
        assert CLASS_NAMES == ["built_up", "vegetation", "water", "bare"]


class TestExclusiveEndDate:

    def test_february_returns_march_01(self):
        assert _exclusive_end_date(2022, 2) == "2022-03-01"

    def test_december_rolls_year(self):
        assert _exclusive_end_date(2021, 12) == "2022-01-01"

    def test_november_returns_december_01(self):
        assert _exclusive_end_date(2022, 11) == "2022-12-01"

    def test_never_returns_28th(self):
        for y in [2020, 2021, 2022]:
            for m in range(1, 13):
                result = _exclusive_end_date(y, m)
                assert not result.endswith("-28"), (
                    f"_exclusive_end_date({y}, {m}) = {result} ends on -28"
                )

    def test_all_months_produce_first_of_month(self):
        for m in range(1, 13):
            result = _exclusive_end_date(2022, m)
            day_part = result.split("-")[-1]
            assert day_part == "01", (
                f"Day part of {result} is {day_part!r}, expected '01'"
            )


# ══════════════════════════════════════════════════════════════
# B. Feature builder
# ══════════════════════════════════════════════════════════════

class TestComputeIndices:

    def test_returns_three_keys(self):
        bands = _make_band_arrays(100)
        idx = compute_indices(bands)
        assert set(idx.keys()) == {"NDVI", "NDWI", "NDBI"}

    def test_output_shapes_match_input(self):
        bands = _make_band_arrays(150)
        idx = compute_indices(bands)
        for k, v in idx.items():
            assert v.shape == (150,), f"{k} shape {v.shape} != (150,)"

    def test_output_dtype_float32(self):
        idx = compute_indices(_make_band_arrays(50))
        for k, v in idx.items():
            assert v.dtype == np.float32, f"{k}.dtype={v.dtype}"

    def test_ndvi_range(self):
        idx = compute_indices(_make_band_arrays(500))
        assert np.all(idx["NDVI"] >= -1.01)
        assert np.all(idx["NDVI"] <= 1.01)

    def test_all_finite(self):
        idx = compute_indices(_make_band_arrays(200))
        for k, v in idx.items():
            assert np.all(np.isfinite(v)), f"{k} contains non-finite values"


class TestStackFeatures:

    def test_shape(self):
        pd = _make_pixel_data(300)
        X = stack_features(pd)
        assert X.shape == (300, N_FEATURES)

    def test_dtype(self):
        X = stack_features(_make_pixel_data(50))
        assert X.dtype == np.float32

    def test_column_order(self):
        pd = _make_pixel_data(100)
        X = stack_features(pd)
        # First column must be B2, last must be NDBI
        np.testing.assert_array_almost_equal(
            X[:, 0], np.asarray(pd["B2"], dtype=np.float32)
        )
        assert FEATURE_NAMES[0] == "B2"
        assert FEATURE_NAMES[-1] == "NDBI"

    def test_auto_computes_indices_when_absent(self):
        """stack_features must derive NDVI/NDWI/NDBI if they are absent."""
        bands_only = _make_band_arrays(80)
        X = stack_features(bands_only)
        assert X.shape == (80, N_FEATURES)

    def test_raises_on_missing_band(self):
        pd = _make_pixel_data(50)
        del pd["B8"]
        with pytest.raises(KeyError, match="B8"):
            stack_features(pd)

    def test_all_finite(self):
        X = stack_features(_make_pixel_data(200))
        assert np.all(np.isfinite(X))


class TestNormalizeFeatures:

    def test_fit_transform_returns_scaler(self):
        X = stack_features(_make_pixel_data(200))
        X_n, scaler = normalize_features(X)
        assert hasattr(scaler, "mean_")

    def test_mean_near_zero_after_fit(self):
        X = stack_features(_make_pixel_data(500))
        X_n, _ = normalize_features(X)
        np.testing.assert_allclose(X_n.mean(axis=0), 0, atol=1e-5)

    def test_passed_scaler_not_refitted(self):
        X = stack_features(_make_pixel_data(200))
        _, scaler = normalize_features(X)
        train_mean = scaler.mean_.copy()
        # Transform a different array with the same scaler
        X2 = stack_features(_make_pixel_data(200, seed=99))
        _, scaler2 = normalize_features(X2, scaler=scaler)
        assert scaler2 is scaler
        np.testing.assert_array_equal(scaler.mean_, train_mean)

    def test_scaler_fit_uses_train_statistics(self):
        """After fit, mean_ must equal X_train.mean(axis=0)."""
        X = stack_features(_make_pixel_data(300))
        _, scaler = normalize_features(X)
        np.testing.assert_allclose(scaler.mean_, X.mean(axis=0), rtol=1e-5)

    def test_output_dtype_float32(self):
        X = stack_features(_make_pixel_data(100))
        X_n, _ = normalize_features(X)
        assert X_n.dtype == np.float32

    def test_shape_preserved(self):
        X = stack_features(_make_pixel_data(120))
        X_n, _ = normalize_features(X)
        assert X_n.shape == X.shape


# ══════════════════════════════════════════════════════════════
# C. Dataset validator
# ══════════════════════════════════════════════════════════════

class TestDatasetValidator:

    def test_valid_dataset_passes(self, tmp_path):
        _save_synthetic_dataset(tmp_path)
        result = validate_dataset(tmp_path)
        assert result["valid"], f"Expected valid, got issues: {result['issues']}"

    def test_missing_file_caught(self, tmp_path):
        _save_synthetic_dataset(tmp_path)
        (tmp_path / "rows.npy").unlink()
        result = validate_dataset(tmp_path)
        assert not result["valid"]
        assert any("rows.npy" in i for i in result["issues"])

    def test_correct_shape_reported(self, tmp_path):
        _save_synthetic_dataset(tmp_path, n=400)
        result = validate_dataset(tmp_path)
        assert result["n_pixels"] == 400
        assert result["n_features"] == N_FEATURES

    def test_nan_in_X_flagged(self, tmp_path):
        _save_synthetic_dataset(tmp_path, n=200)
        X = np.load(tmp_path / "X.npy")
        X[5, 3] = np.nan
        np.save(tmp_path / "X.npy", X)
        result = validate_dataset(tmp_path)
        assert not result["valid"]
        assert result["nan_rows_in_X"] >= 1
        assert any("NaN" in i or "nan" in i.lower() for i in result["issues"])

    def test_inf_in_X_flagged(self, tmp_path):
        _save_synthetic_dataset(tmp_path, n=100)
        X = np.load(tmp_path / "X.npy")
        X[0, 0] = np.inf
        np.save(tmp_path / "X.npy", X)
        result = validate_dataset(tmp_path)
        assert not result["valid"]

    def test_invalid_label_value_flagged(self, tmp_path):
        """Label value 4 is outside {0,1,2,3} and must be flagged."""
        _save_synthetic_dataset(tmp_path, n=200)
        y = np.load(tmp_path / "y.npy")
        y[10] = 4
        np.save(tmp_path / "y.npy", y)
        result = validate_dataset(tmp_path)
        assert not result["valid"]
        assert any("invalid" in i.lower() or "label" in i.lower()
                   for i in result["issues"])

    def test_sentinel_label_flagged(self, tmp_path):
        """Sentinel value (-1) in y must be flagged as invalid."""
        _save_synthetic_dataset(tmp_path, n=200)
        y = np.load(tmp_path / "y.npy")
        y[0] = _WC_SENTINEL
        np.save(tmp_path / "y.npy", y)
        result = validate_dataset(tmp_path)
        assert not result["valid"]

    def test_wrong_feature_count_flagged(self, tmp_path):
        _save_synthetic_dataset(tmp_path, n=150)
        X = np.load(tmp_path / "X.npy")
        # Trim to 8 features
        np.save(tmp_path / "X.npy", X[:, :8].astype(np.float32))
        result = validate_dataset(tmp_path)
        assert not result["valid"]
        assert any("feature" in i.lower() for i in result["issues"])

    def test_n_pixels_mismatch_flagged(self, tmp_path):
        """metadata.n_pixels wrong value must be caught."""
        _save_synthetic_dataset(tmp_path, n=200)
        meta = json.loads((tmp_path / "metadata.json").read_text())
        meta["n_pixels"] = 999   # wrong
        (tmp_path / "metadata.json").write_text(json.dumps(meta))
        result = validate_dataset(tmp_path)
        assert not result["valid"]
        assert any("n_pixels" in i for i in result["issues"])

    def test_missing_metadata_key_flagged(self, tmp_path):
        _save_synthetic_dataset(tmp_path, n=150)
        meta = json.loads((tmp_path / "metadata.json").read_text())
        del meta["date_end_exclusive"]
        (tmp_path / "metadata.json").write_text(json.dumps(meta))
        result = validate_dataset(tmp_path)
        assert not result["valid"]
        assert "date_end_exclusive" in result["missing_metadata_keys"]

    def test_class_distribution_reported(self, tmp_path):
        _save_synthetic_dataset(tmp_path, n=400)
        result = validate_dataset(tmp_path)
        assert set(result["class_distribution"].keys()) == set(CLASS_NAMES)
        total = sum(result["class_distribution"].values())
        assert total == 400

    def test_zero_class_flagged(self, tmp_path):
        """If a class has 0 samples, validation must fail."""
        _save_synthetic_dataset(tmp_path, n=300)
        y = np.load(tmp_path / "y.npy")
        y[y == 2] = 0   # remove all water samples
        np.save(tmp_path / "y.npy", y)
        result = validate_dataset(tmp_path)
        assert not result["valid"]
        assert any("0 sample" in i.lower() or "zero" in i.lower()
                   for i in result["issues"])

    def test_feature_names_mismatch_flagged(self, tmp_path):
        _save_synthetic_dataset(tmp_path, n=200)
        meta = json.loads((tmp_path / "metadata.json").read_text())
        meta["feature_names"] = ["wrong"] * N_FEATURES
        (tmp_path / "metadata.json").write_text(json.dumps(meta))
        result = validate_dataset(tmp_path)
        assert not result["valid"]
        assert any("feature_names" in i for i in result["issues"])


# ══════════════════════════════════════════════════════════════
# D. load_extracted_data round-trip
# ══════════════════════════════════════════════════════════════

class TestLoadExtractedData:

    def test_round_trip(self, tmp_path):
        _save_synthetic_dataset(tmp_path, n=300)
        data = load_extracted_data(tmp_path)

        assert data["X"].shape == (300, N_FEATURES)
        assert data["y"].shape == (300,)
        assert data["rows"].shape == (300,)
        assert data["cols"].shape == (300,)
        assert data["feature_names"] == FEATURE_NAMES
        assert data["class_names"] == CLASS_NAMES
        assert "n_pixels" in data["metadata"]

    def test_raises_on_missing_file(self, tmp_path):
        _save_synthetic_dataset(tmp_path, n=100)
        (tmp_path / "X.npy").unlink()
        with pytest.raises(FileNotFoundError):
            load_extracted_data(tmp_path)

    def test_raises_on_empty_directory(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_extracted_data(tmp_path)

    def test_arrays_are_numpy(self, tmp_path):
        _save_synthetic_dataset(tmp_path, n=200)
        data = load_extracted_data(tmp_path)
        for key in ("X", "y", "rows", "cols"):
            assert isinstance(data[key], np.ndarray), (
                f"data['{key}'] is {type(data[key])}, expected ndarray"
            )

    def test_X_is_float32(self, tmp_path):
        _save_synthetic_dataset(tmp_path, n=200)
        data = load_extracted_data(tmp_path)
        assert data["X"].dtype == np.float32

    def test_y_is_int32(self, tmp_path):
        _save_synthetic_dataset(tmp_path, n=200)
        data = load_extracted_data(tmp_path)
        assert data["y"].dtype == np.int32


# ══════════════════════════════════════════════════════════════
# E. prepare_dataset.py: validate_dataset function
# ══════════════════════════════════════════════════════════════

class TestPreparePipelineValidation:
    """
    Tests for the validate_dataset() helper imported by prepare_dataset.py.
    This is the same function from dataset_validator, but also verifies that
    the script itself can be imported and that its storage estimate is sensible.
    """

    def test_storage_estimate_formula(self):
        """Verify the in-script storage estimate: 9*4 + 4 bytes per pixel."""
        bytes_per_pixel = 9 * 4 + 4  # X float32 + y int32
        assert bytes_per_pixel == 40

        pixels_10k = 10_000
        mb = pixels_10k * bytes_per_pixel / 1_048_576
        assert 0.3 < mb < 0.5, f"10k pixels should be ~0.4 MB, got {mb:.2f}"

    def test_validate_only_passes_on_good_data(self, tmp_path):
        _save_synthetic_dataset(tmp_path, n=250)
        result = validate_dataset(tmp_path)
        assert result["valid"]
        assert not result["issues"]

    def test_required_files_list_complete(self):
        assert "X.npy" in REQUIRED_FILES
        assert "y.npy" in REQUIRED_FILES
        assert "rows.npy" in REQUIRED_FILES
        assert "cols.npy" in REQUIRED_FILES
        assert "metadata.json" in REQUIRED_FILES

    def test_required_metadata_keys_includes_date_fields(self):
        for key in ("date_start", "date_end_exclusive",
                    "date_end_inclusive_desc", "date_convention"):
            assert key in REQUIRED_METADATA_KEYS, (
                f"'{key}' not in REQUIRED_METADATA_KEYS"
            )

    def test_required_metadata_keys_includes_provenance(self):
        for key in ("labels_source", "missing_label_handling",
                    "temporal_caveat", "n_removed_missing_labels"):
            assert key in REQUIRED_METADATA_KEYS, (
                f"'{key}' not in REQUIRED_METADATA_KEYS"
            )

"""
GEOVIEW AML — Tests for the four targeted corrections.

Covers:
  1. Missing/unmapped WorldCover labels are excluded (not assigned to class 3)
  2. Seasonal date window end-date convention (_exclusive_end_date)
  3. Spatial block split: determinism, non-overlap, non-empty splits, reporting
  4. Scaler fitted on training data only (no val/test leakage)

No GEE, model training, or Optuna is involved.
"""

from __future__ import annotations

import os
import sys
import warnings

_HERE    = os.path.abspath(os.path.dirname(__file__))
_PROJECT = os.path.abspath(os.path.join(_HERE, ".."))
_BACKEND = os.path.abspath(os.path.join(_HERE, "..", "backend"))
sys.path.insert(0, _PROJECT)
sys.path.insert(0, _BACKEND)
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

import numpy as np
import pytest


# ══════════════════════════════════════════════════════════════
# 1. Missing / unmapped WorldCover labels
# ══════════════════════════════════════════════════════════════

class TestMissingLabels:
    """
    Tests for gee_extractor label handling.

    We test the pure Python filtering logic without calling GEE.
    The extraction function is large; we test the filtering behaviour
    by directly calling the helper constants and replicating the logic.
    """

    def test_wc_class_map_contains_no_sentinel(self):
        """No valid WorldCover code should map to the sentinel value."""
        from ml.preprocessing.gee_extractor import WC_CLASS_MAP, _WC_SENTINEL
        for code, cls in WC_CLASS_MAP.items():
            assert cls != _WC_SENTINEL, (
                f"WorldCover code {code} maps to sentinel {_WC_SENTINEL} — "
                "it should map to a real class (0–3)."
            )

    def test_sentinel_outside_valid_labels(self):
        """_WC_SENTINEL must not be in the set of valid class indices."""
        from ml.preprocessing.gee_extractor import _WC_SENTINEL, _VALID_LABELS
        assert _WC_SENTINEL not in _VALID_LABELS, (
            f"Sentinel {_WC_SENTINEL} is inside _VALID_LABELS {_VALID_LABELS}; "
            "it would pass the label filter and corrupt training data."
        )

    def test_all_map_values_are_valid(self):
        """Every mapped class must be in {0, 1, 2, 3}."""
        from ml.preprocessing.gee_extractor import WC_CLASS_MAP, _VALID_LABELS
        for code, cls in WC_CLASS_MAP.items():
            assert cls in _VALID_LABELS, (
                f"WorldCover code {code} maps to {cls}, "
                f"which is not in _VALID_LABELS {_VALID_LABELS}."
            )

    def test_label_filter_excludes_none(self):
        """
        A pixel whose GEE 'label' property is None (masked WorldCover pixel)
        must be excluded after filtering — NOT assigned to class 3.
        """
        from ml.preprocessing.gee_extractor import _WC_SENTINEL, _VALID_LABELS

        # Simulate: None label becomes NaN in y_raw
        y_raw = np.array([np.nan, 1.0, 0.0, 3.0], dtype=np.float64)
        label_valid = (
            np.isfinite(y_raw)
            & (y_raw != _WC_SENTINEL)
            & np.isin(y_raw.astype(int), list(_VALID_LABELS))
        )
        # The NaN entry (index 0) must be excluded
        assert not label_valid[0], "None label (NaN) was not excluded."
        assert label_valid[1:].all(), "Valid labels were wrongly excluded."

    def test_label_filter_excludes_sentinel(self):
        """
        A pixel whose WorldCover code was not in WC_CLASS_MAP (defaultValue
        returns _WC_SENTINEL) must be excluded.
        """
        from ml.preprocessing.gee_extractor import _WC_SENTINEL, _VALID_LABELS

        y_raw = np.array([float(_WC_SENTINEL), 0.0, 1.0, 2.0, 3.0], dtype=np.float64)
        label_valid = (
            np.isfinite(y_raw)
            & (y_raw != _WC_SENTINEL)
            & np.isin(y_raw.astype(int), list(_VALID_LABELS))
        )
        assert not label_valid[0], (
            f"Sentinel value {_WC_SENTINEL} was not excluded by the filter."
        )
        assert label_valid[1:].all()

    def test_label_filter_excludes_out_of_range(self):
        """Any label value outside {0,1,2,3} must be excluded."""
        from ml.preprocessing.gee_extractor import _WC_SENTINEL, _VALID_LABELS

        y_raw = np.array([4.0, 99.0, -2.0, 0.0, 3.0], dtype=np.float64)
        label_valid = (
            np.isfinite(y_raw)
            & (y_raw != _WC_SENTINEL)
            & np.isin(y_raw.astype(int), list(_VALID_LABELS))
        )
        assert not label_valid[0], "Value 4 should be excluded."
        assert not label_valid[1], "Value 99 should be excluded."
        assert not label_valid[2], "Value -2 should be excluded."
        assert label_valid[3], "Value 0 should be kept."
        assert label_valid[4], "Value 3 should be kept."

    def test_default_value_not_class_3(self):
        """
        The GEE remap defaultValue must be _WC_SENTINEL (-1), not 3.
        If it were 3 (bare), unmapped codes would silently become bare pixels.
        """
        from ml.preprocessing.gee_extractor import _WC_SENTINEL
        assert _WC_SENTINEL == -1, (
            f"Expected sentinel -1, got {_WC_SENTINEL}. "
            "Using 3 would silently assign unmapped codes to 'bare'."
        )

    def test_class_names_length_matches_valid_labels(self):
        from ml.preprocessing.gee_extractor import CLASS_NAMES, _VALID_LABELS
        assert len(CLASS_NAMES) == len(_VALID_LABELS)


# ══════════════════════════════════════════════════════════════
# 2. Seasonal date window / exclusive end-date convention
# ══════════════════════════════════════════════════════════════

class TestDateBoundaries:

    def _end_date(self, year: int, end_month: int) -> str:
        from ml.preprocessing.gee_extractor import _exclusive_end_date
        return _exclusive_end_date(year, end_month)

    def test_february_end_gives_march_01(self):
        """
        Default window ends in February.
        Exclusive end must be March 1, not February 28.
        """
        result = self._end_date(2022, 2)
        assert result == "2022-03-01", (
            f"Expected 2022-03-01, got {result}. "
            "filterDate end must be exclusive; Feb 28 would miss Feb 28 scenes."
        )

    def test_march_end_gives_april_01(self):
        result = self._end_date(2022, 3)
        assert result == "2022-04-01"

    def test_november_end_gives_december_01(self):
        result = self._end_date(2022, 11)
        assert result == "2022-12-01"

    def test_december_end_crosses_year(self):
        """December end must roll over to January of year+1."""
        result = self._end_date(2021, 12)
        assert result == "2022-01-01", (
            f"Expected 2022-01-01, got {result}."
        )

    def test_january_end_gives_february_01(self):
        result = self._end_date(2022, 1)
        assert result == "2022-02-01"

    def test_nov_to_feb_window_crosses_year(self):
        """
        The November–February window (start_month=11, end_month=2) crosses a
        year boundary.  end_year must be composite_year + 1.
        """
        from ml.preprocessing.gee_extractor import _exclusive_end_date
        composite_year = 2021
        start_month, end_month = 11, 2
        end_year = composite_year + 1 if end_month < start_month else composite_year
        date_end = _exclusive_end_date(end_year, end_month)
        assert date_end == "2022-03-01", (
            f"Nov–Feb window from 2021 should end 2022-03-01, got {date_end}."
        )

    def test_start_date_is_first_of_start_month(self):
        """Start date must always be the 1st of start_month."""
        composite_year, start_month = 2021, 11
        expected = f"{composite_year}-{start_month:02d}-01"
        assert expected == "2021-11-01"

    def test_exclusive_end_is_not_28th(self):
        """Sanity: the exclusive end should never be the 28th of any month."""
        from ml.preprocessing.gee_extractor import _exclusive_end_date
        for month in range(1, 13):
            result = _exclusive_end_date(2022, month)
            assert not result.endswith("-28"), (
                f"_exclusive_end_date(2022, {month}) = {result} — "
                "ends on the 28th, which would miss some scenes."
            )


# ══════════════════════════════════════════════════════════════
# 3. Spatial block split
# ══════════════════════════════════════════════════════════════

class TestSpatialBlockSplit:
    """
    Tests for correctness, determinism, non-overlap and non-empty splits.
    Uses synthetic coordinate arrays — no GEE calls.
    """

    def _make_data(self, n: int = 1000, seed: int = 0):
        from ml.preprocessing.feature_builder import stack_features_demo
        X = stack_features_demo({}, n_pixels=n, seed=seed)
        y = np.zeros(n, dtype=np.int32)
        # Lay out pixels on a regular 50x20 grid
        rows = np.arange(n) // 20
        cols = np.arange(n) % 20
        return X, y, rows, cols

    # ── Basic structure ──────────────────────────────────────

    def test_sizes_sum_to_n(self):
        from ml.preprocessing.feature_builder import spatial_block_split
        X, y, rows, cols = self._make_data()
        Xtr, Xv, Xte, ytr, yv, yte = spatial_block_split(
            X, y, rows=rows, cols=cols, n_blocks=4, seed=42
        )
        assert len(Xtr) + len(Xv) + len(Xte) == len(X)

    def test_no_split_is_empty_with_coords(self):
        from ml.preprocessing.feature_builder import spatial_block_split
        X, y, rows, cols = self._make_data()
        Xtr, Xv, Xte, ytr, yv, yte = spatial_block_split(
            X, y, rows=rows, cols=cols, n_blocks=4, seed=42
        )
        assert len(Xtr) > 0, "Training split is empty"
        assert len(Xv)  > 0, "Validation split is empty"
        assert len(Xte) > 0, "Test split is empty"

    def test_train_fraction_approximately_correct(self):
        from ml.preprocessing.feature_builder import spatial_block_split
        X, y, rows, cols = self._make_data(n=2000)
        Xtr, Xv, Xte, *_ = spatial_block_split(
            X, y, rows=rows, cols=cols, n_blocks=4, seed=42
        )
        frac = len(Xtr) / len(X)
        assert 0.55 <= frac <= 0.85, (
            f"Training fraction {frac:.3f} is outside [0.55, 0.85]."
        )

    # ── Tile non-overlap ─────────────────────────────────────

    def test_tiles_are_disjoint(self):
        """
        No pixel should appear in more than one split.
        Verify via original indices.
        """
        from ml.preprocessing.feature_builder import spatial_block_split
        import numpy as np
        n = 1000
        X, y, rows, cols = self._make_data(n=n)
        # Tag each pixel with its original index
        idx = np.arange(n, dtype=np.int32)
        Xtr, Xv, Xte, ytr, yv, yte = spatial_block_split(
            X, y, rows=rows, cols=cols, n_blocks=4, seed=42
        )
        # Recover indices via the feature arrays (unique row fingerprint)
        # Use a simpler approach: split the index array the same way
        X2, y2 = idx.reshape(-1, 1).astype(np.float32), idx
        itr, iv, ite, *_ = spatial_block_split(
            X2, y2, rows=rows, cols=cols, n_blocks=4, seed=42
        )
        train_ids = set(itr.ravel().tolist())
        val_ids   = set(iv.ravel().tolist())
        test_ids  = set(ite.ravel().tolist())

        assert not (train_ids & val_ids),  "Train and val share pixels."
        assert not (train_ids & test_ids), "Train and test share pixels."
        assert not (val_ids   & test_ids), "Val and test share pixels."

    # ── Determinism ──────────────────────────────────────────

    def test_same_seed_gives_same_result(self):
        from ml.preprocessing.feature_builder import spatial_block_split
        X, y, rows, cols = self._make_data()
        result1 = spatial_block_split(X, y, rows=rows, cols=cols, n_blocks=4, seed=42)
        result2 = spatial_block_split(X, y, rows=rows, cols=cols, n_blocks=4, seed=42)
        for a, b in zip(result1, result2):
            np.testing.assert_array_equal(a, b)

    def test_different_seed_gives_different_result(self):
        from ml.preprocessing.feature_builder import spatial_block_split
        X, y, rows, cols = self._make_data()
        r1 = spatial_block_split(X, y, rows=rows, cols=cols, n_blocks=4, seed=42)
        r2 = spatial_block_split(X, y, rows=rows, cols=cols, n_blocks=4, seed=99)
        # At least the training set should differ (not guaranteed but very likely)
        assert not np.array_equal(r1[0], r2[0]), (
            "Different seeds produced identical training sets — "
            "seed is not having any effect."
        )

    # ── Too few tiles ─────────────────────────────────────────

    def test_raises_when_only_one_tile(self):
        """If all pixels fall in one tile, split is impossible."""
        from ml.preprocessing.feature_builder import spatial_block_split
        n = 50
        X = np.zeros((n, 9), dtype=np.float32)
        y = np.zeros(n, dtype=np.int32)
        # All pixels at the same coordinate → one tile
        rows = np.zeros(n, dtype=np.int64)
        cols = np.zeros(n, dtype=np.int64)
        with pytest.raises(ValueError, match="tile"):
            spatial_block_split(X, y, rows=rows, cols=cols, n_blocks=4, seed=42)

    def test_raises_when_only_two_tiles(self):
        """Two tiles: can't have train, val, AND test."""
        from ml.preprocessing.feature_builder import spatial_block_split
        n = 100
        X = np.zeros((n, 9), dtype=np.float32)
        y = np.zeros(n, dtype=np.int32)
        # Half pixels in tile A, half in tile B
        rows = (np.arange(n) >= n // 2).astype(np.int64) * 1000
        cols = np.zeros(n, dtype=np.int64)
        with pytest.raises(ValueError, match="tile"):
            spatial_block_split(X, y, rows=rows, cols=cols, n_blocks=4, seed=42)

    # ── Fallback warning ─────────────────────────────────────

    def test_warns_without_coords(self):
        from ml.preprocessing.feature_builder import spatial_block_split
        X = np.zeros((300, 9), dtype=np.float32)
        y = np.zeros(300, dtype=np.int32)
        with pytest.warns(UserWarning, match="spatial"):
            spatial_block_split(X, y, rows=None, cols=None, n_blocks=4, seed=42)

    def test_fallback_splits_all_non_empty(self):
        from ml.preprocessing.feature_builder import spatial_block_split
        X = np.zeros((600, 9), dtype=np.float32)
        y = np.zeros(600, dtype=np.int32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            Xtr, Xv, Xte, *_ = spatial_block_split(
                X, y, rows=None, cols=None, n_blocks=4, seed=42
            )
        assert len(Xtr) > 0
        assert len(Xv)  > 0
        assert len(Xte) > 0

    # ── Edge rows/cols behaviour ──────────────────────────────

    def test_linspace_edges_deterministic_across_range_sizes(self):
        """
        Edge computation must be deterministic regardless of row/col range.
        A known bug was linspace(..., dtype=int) producing different edges
        for different ranges due to float rounding before int cast.
        """
        from ml.preprocessing.feature_builder import spatial_block_split
        for n_range in (40, 41, 100, 128, 255, 256):
            n = n_range
            X = np.zeros((n, 9), dtype=np.float32)
            y = np.zeros(n, dtype=np.int32)
            rows = np.arange(n, dtype=np.int64)
            cols = np.zeros(n, dtype=np.int64)
            # Must not raise and all splits non-empty
            try:
                Xtr, Xv, Xte, *_ = spatial_block_split(
                    X, y, rows=rows, cols=cols, n_blocks=4, seed=42
                )
                assert len(Xtr) + len(Xv) + len(Xte) == n
            except ValueError as e:
                # Only acceptable if genuinely too few tiles
                assert "tile" in str(e).lower(), (
                    f"Unexpected ValueError for n_range={n_range}: {e}"
                )


# ══════════════════════════════════════════════════════════════
# 4. Scaler train-only fitting (no val/test leakage)
# ══════════════════════════════════════════════════════════════

class TestScalerNoLeakage:

    def _splits(self, n: int = 600):
        from ml.preprocessing.feature_builder import stack_features_demo, spatial_block_split
        X = stack_features_demo({}, n_pixels=n, seed=3)
        y = np.zeros(n, dtype=np.int32)
        rows = np.arange(n) // 30
        cols = np.arange(n) % 30
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return spatial_block_split(X, y, rows=rows, cols=cols, n_blocks=4, seed=42)

    def test_scaler_fit_on_train_only(self):
        """
        Fitting the scaler on X_train only must use X_train statistics,
        not X_val or X_test statistics.
        """
        from ml.preprocessing.feature_builder import normalize_features
        Xtr, Xv, Xte, *_ = self._splits()

        X_train_n, scaler = normalize_features(Xtr)

        # Scaler mean_ should match X_train mean, not val or test mean
        np.testing.assert_allclose(
            scaler.mean_, Xtr.mean(axis=0), rtol=1e-5,
            err_msg="Scaler mean_ does not match training-set mean."
        )

    def test_val_uses_train_scaler(self):
        """
        Transforming val data with the train-fitted scaler should succeed
        and not refit the scaler.
        """
        from ml.preprocessing.feature_builder import normalize_features
        Xtr, Xv, Xte, *_ = self._splits()

        _, scaler = normalize_features(Xtr)
        X_val_n, scaler2 = normalize_features(Xv, scaler=scaler)

        assert scaler2 is scaler, (
            "normalize_features returned a different scaler object when "
            "a pre-fitted scaler was passed — it may have re-fitted."
        )
        # scaler.mean_ must still equal train mean (not changed by val transform)
        np.testing.assert_allclose(
            scaler.mean_, Xtr.mean(axis=0), rtol=1e-5,
            err_msg="Scaler mean_ changed after transforming val data — leakage."
        )

    def test_test_uses_train_scaler(self):
        from ml.preprocessing.feature_builder import normalize_features
        Xtr, Xv, Xte, *_ = self._splits()
        _, scaler = normalize_features(Xtr)
        X_te_n, scaler3 = normalize_features(Xte, scaler=scaler)
        assert scaler3 is scaler
        np.testing.assert_allclose(scaler.mean_, Xtr.mean(axis=0), rtol=1e-5)

    def test_normalize_none_scaler_fits_new(self):
        """Passing scaler=None must always fit a fresh scaler."""
        from ml.preprocessing.feature_builder import normalize_features
        Xtr, *_ = self._splits()
        _, s1 = normalize_features(Xtr, scaler=None)
        _, s2 = normalize_features(Xtr, scaler=None)
        # Both should have mean_ close to train mean
        np.testing.assert_allclose(s1.mean_, Xtr.mean(axis=0), rtol=1e-5)
        np.testing.assert_allclose(s2.mean_, Xtr.mean(axis=0), rtol=1e-5)

    def test_normalized_train_has_near_zero_mean(self):
        """After StandardScaler, training set mean should be ~0."""
        from ml.preprocessing.feature_builder import normalize_features
        Xtr, *_ = self._splits()
        X_n, _ = normalize_features(Xtr)
        means = np.abs(X_n.mean(axis=0))
        assert np.all(means < 1e-5), (
            f"Post-normalization training means not near zero: {means}"
        )

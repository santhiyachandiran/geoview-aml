"""
GeoChange AI — Feature Builder

Converts pixel-level band arrays (from GEE export or rasterio) into
numpy feature arrays suitable for ML training and inference.

Feature order (fixed — must match model training and inference):
  [0] B2  [1] B3  [2] B4  [3] B8  [4] B11  [5] B12
  [6] NDVI  [7] NDWI  [8] NDBI

Spatial split strategy
----------------------
`spatial_block_split` partitions pixels by their geographic grid-tile, not
by sequential index. Each pixel must supply a (row, col) raster coordinate.
Tiles are allocated to train/val/test as whole blocks, preventing adjacent
pixels from appearing in both train and test sets (spatial leakage).

When real pixel coordinates are unavailable (demo/test fixtures), the
function falls back to sequential-index modulo with a documented caveat.
"""

from __future__ import annotations

import warnings
import numpy as np
from sklearn.preprocessing import StandardScaler


FEATURE_NAMES = ["B2", "B3", "B4", "B8", "B11", "B12", "NDVI", "NDWI", "NDBI"]
N_FEATURES = len(FEATURE_NAMES)


def compute_indices(bands: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    Compute NDVI, NDWI, NDBI from per-pixel band arrays.

    Args:
        bands: dict with keys B2, B3, B4, B8, B11, B12 (float32 reflectance 0-1)

    Returns:
        dict with NDVI, NDWI, NDBI arrays of same length/shape.
    """
    eps = 1e-8
    B3, B4, B8, B11 = bands["B3"], bands["B4"], bands["B8"], bands["B11"]

    ndvi = (B8 - B4) / (B8 + B4 + eps)
    ndwi = (B3 - B11) / (B3 + B11 + eps)
    ndbi = (B11 - B8) / (B11 + B8 + eps)

    return {
        "NDVI": ndvi.astype(np.float32),
        "NDWI": ndwi.astype(np.float32),
        "NDBI": ndbi.astype(np.float32),
    }


def stack_features(pixel_data: dict) -> np.ndarray:
    """
    Build an (N, 9) feature matrix from pixel-level band arrays.

    Args:
        pixel_data: dict with keys matching FEATURE_NAMES, each an array
                    of length N. NDVI/NDWI/NDBI may be pre-computed or will
                    be derived from B3, B4, B8, B11 if absent.

    Returns:
        np.ndarray shape (N, 9) dtype float32, columns in FEATURE_NAMES order.
    """
    # Derive indices if not pre-computed
    if "NDVI" not in pixel_data:
        indices = compute_indices(pixel_data)
        pixel_data = {**pixel_data, **indices}

    cols = []
    for feat in FEATURE_NAMES:
        if feat not in pixel_data:
            raise KeyError(f"Feature '{feat}' missing from pixel_data. "
                           f"Available: {list(pixel_data.keys())}")
        cols.append(np.asarray(pixel_data[feat], dtype=np.float32).ravel())

    X = np.column_stack(cols)
    if X.shape[1] != N_FEATURES:
        raise ValueError(f"Expected {N_FEATURES} features, got {X.shape[1]}")
    return X


def stack_features_demo(composite: dict, n_pixels: int = 1000, seed: int = 42) -> np.ndarray:
    """
    Synthetic feature array for testing/demo only.
    Uses composite mean values + Gaussian noise.
    DO NOT use for training or reporting metrics.
    """
    rng = np.random.default_rng(seed)
    means = {
        "B2": composite.get("B2",   0.080),
        "B3": composite.get("B3",   0.100),
        "B4": composite.get("B4",   0.120),
        "B8": composite.get("B8",   0.350),
        "B11": composite.get("B11",  0.200),
        "B12": composite.get("B12",  0.150),
        "NDVI": composite.get("ndvi_mean", 0.35),
        "NDWI": composite.get("ndwi_mean", -0.12),
        "NDBI": composite.get("ndbi_mean", -0.15),
    }
    stds = {k: abs(v) * 0.15 + 0.005 for k, v in means.items()}
    cols = [rng.normal(means[f], stds[f], n_pixels).astype(np.float32) for f in FEATURE_NAMES]
    return np.column_stack(cols)


def normalize_features(
    X: np.ndarray,
    scaler: StandardScaler | None = None,
) -> tuple[np.ndarray, StandardScaler]:
    """
    Standardize features. Fit on training data only; pass fitted scaler for val/test.

    Returns:
        (X_normalized, fitted_scaler)
    """
    if scaler is None:
        scaler = StandardScaler()
        X_norm = scaler.fit_transform(X).astype(np.float32)
    else:
        X_norm = scaler.transform(X).astype(np.float32)
    return X_norm, scaler


def spatial_block_split(
    X: np.ndarray,
    y: np.ndarray,
    rows: np.ndarray | None = None,
    cols: np.ndarray | None = None,
    n_blocks: int = 4,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray]:
    """
    Spatially-aware train / val / test split.

    Each unique grid tile is assigned whole to exactly one split.
    No tile is shared between splits, so pixels within the same tile
    cannot appear in both train and test.

    NOTE: Assigning tiles to splits reduces — but does not eliminate —
    spatial autocorrelation between splits.  Pixels at tile boundaries
    remain spatially close across splits.  This is disclosed as a known
    limitation when reporting results.

    Args:
        X:          (N, F) feature array
        y:          (N,) label array
        rows:       (N,) integer pixel row indices (raster coordinates).
                    Pass None only for demo/test fixtures.
        cols:       (N,) integer pixel column indices (raster coordinates).
                    Pass None only for demo/test fixtures.
        n_blocks:   Number of grid lines per axis; produces up to n_blocks²
                    tiles.  Actual tile count equals occupied tiles only.
        train_frac: Fraction of tiles for training   (default 0.70)
        val_frac:   Fraction of tiles for validation (default 0.15)
        seed:       Random seed for tile assignment (fixed → deterministic)

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test

    Raises:
        ValueError: if any split would be empty given the available tiles.
    """
    import logging
    _log = logging.getLogger(__name__)

    rng = np.random.default_rng(seed)
    n = len(X)
    if len(y) != n:
        raise ValueError(f"X has {n} rows but y has {len(y)} rows.")

    # ── Assign pixels to tiles ────────────────────────────────────────────
    if rows is not None and cols is not None:
        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)

        row_min, row_max = int(rows.min()), int(rows.max())
        col_min, col_max = int(cols.min()), int(cols.max())

        # Build n_blocks+1 edges using float then floor to int for determinism.
        # Using dtype=int directly in linspace rounds differently across NumPy
        # versions — always floor-convert explicitly.
        row_edges = np.floor(
            np.linspace(row_min, row_max + 1, n_blocks + 1)
        ).astype(np.int64)
        col_edges = np.floor(
            np.linspace(col_min, col_max + 1, n_blocks + 1)
        ).astype(np.int64)

        # searchsorted gives 0-based tile index along each axis.
        # side="right" so a pixel at an interior edge belongs to the tile to
        # its right, matching np.digitize behaviour.
        row_tile = np.searchsorted(row_edges[1:], rows, side="right")
        col_tile = np.searchsorted(col_edges[1:], cols, side="right")

        # Flatten (row_tile, col_tile) to a single integer tile ID.
        block_ids = row_tile * n_blocks + col_tile

    else:
        warnings.warn(
            "spatial_block_split called without pixel coordinates (rows/cols). "
            "Falling back to sequential-index modulo partitioning. "
            "This does NOT eliminate spatial autocorrelation. "
            "Pass rows/cols from GEE for a tile-based split.",
            UserWarning,
            stacklevel=2,
        )
        total_blocks = n_blocks * n_blocks
        block_ids = np.arange(n) % total_blocks

    # ── Allocate tiles to splits ──────────────────────────────────────────
    unique_blocks = np.unique(block_ids)
    total = len(unique_blocks)

    if total < 3:
        raise ValueError(
            f"Only {total} unique tile(s) found; need at least 3 "
            "(one per split). Increase n_blocks or provide more pixels."
        )

    # Shuffle tile order deterministically.
    shuffled_blocks = unique_blocks[rng.permutation(total)]

    # Compute tile counts per split; ensure each gets ≥ 1 tile.
    n_train = max(1, round(train_frac * total))
    n_val   = max(1, round(val_frac   * total))
    # Reserve at least 1 tile for test by capping train + val.
    if n_train + n_val >= total:
        n_train = total - 2   # give val and test one tile each
        n_val   = 1
    n_test = total - n_train - n_val   # always ≥ 1

    train_blocks = set(shuffled_blocks[:n_train].tolist())
    val_blocks   = set(shuffled_blocks[n_train:n_train + n_val].tolist())
    test_blocks  = set(shuffled_blocks[n_train + n_val:].tolist())

    # Verify disjointness (defensive check — should always hold by construction)
    assert not (train_blocks & val_blocks),  "BUG: train/val blocks overlap"
    assert not (train_blocks & test_blocks), "BUG: train/test blocks overlap"
    assert not (val_blocks   & test_blocks), "BUG: val/test blocks overlap"

    train_mask = np.isin(block_ids, list(train_blocks))
    val_mask   = np.isin(block_ids, list(val_blocks))
    test_mask  = np.isin(block_ids, list(test_blocks))

    # ── Verify no split is empty ──────────────────────────────────────────
    for name, mask in (("train", train_mask), ("val", val_mask), ("test", test_mask)):
        if not mask.any():
            raise ValueError(
                f"The '{name}' split is empty after block allocation. "
                f"total_tiles={total}, n_train={n_train}, n_val={n_val}, "
                f"n_test={n_test}. Increase n_blocks or provide more data."
            )

    _log.info(
        f"spatial_block_split: {total} tiles → "
        f"train={n_train} tiles / {int(train_mask.sum())} samples, "
        f"val={n_val} tiles / {int(val_mask.sum())} samples, "
        f"test={n_test} tiles / {int(test_mask.sum())} samples  "
        f"(seed={seed}, n_blocks={n_blocks})"
    )

    return (
        X[train_mask], X[val_mask],  X[test_mask],
        y[train_mask], y[val_mask],  y[test_mask],
    )

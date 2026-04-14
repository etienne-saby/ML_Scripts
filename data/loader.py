"""
MetAIsAFe — data/loader.py
===========================
Multi-format data loading and categorical encoding.

Public API
----------
    load_data(filepath, fmt, use_polars)        -> pd.DataFrame
    encode_categoricals(df, features, ...)      -> (pd.DataFrame, dict)
    build_dataset(df, features, targets, ...)   -> (X, y, used_features, used_targets)

Supported formats
-----------------
- Parquet : preferred (fast, compressed, typed) — default
- CSV     : universal fallback (cast critical columns explicitly after loading)
- FST     : R binary format — requires pyreadr

Encoding
--------
encode_categoricals() is the single encoding entry point, called POST-SPLIT
to maintain architectural consistency. Two strategies are supported:

  ``"lightgbm"`` : cast to pandas ``category`` dtype — native LightGBM support,
                   no fitted transform, no leakage risk.
  ``"sklearn"``  : LabelEncoder fit on train only, applied to test.
                   Required for CART (sklearn DecisionTreeRegressor).
                   Never used for the main LightGBM model.

Module-level aliases
--------------------
ALL_FEATURES        = ALL_FEATURES_B1   (override for B2 in calling code)
CATEGORICAL_FEATURES = CATEGORICAL_FEATURES_B1

Author  : Étienne SABY
Updated : 2026-04
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from _version import __version__
from column_taxonomy import ALL_FEATURES_B1, ALL_TARGETS, CATEGORICAL_FEATURES_B1

ALL_FEATURES         = ALL_FEATURES_B1
CATEGORICAL_FEATURES = CATEGORICAL_FEATURES_B1

log = logging.getLogger(__name__)

# ── Optional dependencies ────────────────────────────────────────────────────
try:
    import polars as pl
    _POLARS_AVAILABLE = True
except ImportError:
    _POLARS_AVAILABLE = False

try:
    import pyreadr
    _PYREADR_AVAILABLE = True
except ImportError:
    _PYREADR_AVAILABLE = False


# ═════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════

def load_data(
    filepath: str | Path,
    fmt: str | None = None,
    use_polars: bool = False,
) -> pd.DataFrame:
    """
    Load the meta-table from a file, auto-detecting format if needed.

    Parameters
    ----------
    filepath : str or Path
        Path to the input file.
    fmt : str, optional
        Explicit format override: ``"parquet"``, ``"csv"``, or ``"fst"``.
        Auto-detected from file extension if ``None``.
    use_polars : bool, default False
        Load with Polars instead of pandas (faster for very large files).
        Returns a pandas DataFrame regardless.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the format cannot be inferred or is unsupported.
    ImportError
        If a required optional dependency (polars, pyreadr) is not installed.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    if fmt is None:
        fmt = filepath.suffix.lower().lstrip(".")
        if not fmt:
            raise ValueError(
                f"Cannot infer format from '{filepath.name}' (no extension). "
                "Specify fmt explicitly."
            )

    fmt = fmt.lower()

    if use_polars:
        if not _POLARS_AVAILABLE:
            raise ImportError("polars is not installed: pip install polars")
        return _load_polars(filepath, fmt)

    return _load_pandas(filepath, fmt)


def _load_pandas(filepath: Path, fmt: str) -> pd.DataFrame:
    """Internal: load with pandas."""
    log.info("📂 Loading %s  (format: %s, engine: pandas)", filepath.name, fmt)
    if fmt == "parquet":
        df = pd.read_parquet(filepath)
    elif fmt == "csv":
        df = pd.read_csv(filepath, na_values=["NA", "NaN", ""])
    elif fmt == "fst":
        if not _PYREADR_AVAILABLE:
            raise ImportError(
                "The 'fst' format requires pyreadr: pip install pyreadr\n"
                "Note: fst is a native R format, NOT Arrow Feather."
            )
        result = pyreadr.read_r(filepath)
        df = list(result.values())[0]
    else:
        raise ValueError(
            f"Unsupported format: '{fmt}'. Use: 'parquet', 'csv', or 'fst'."
        )
    log.info("✅ Loaded: %d rows × %d columns", len(df), len(df.columns))
    return df


def _load_polars(filepath: Path, fmt: str) -> pd.DataFrame:
    """Internal: load with Polars, return as pandas DataFrame."""
    log.info("📂 Loading %s  (format: %s, engine: polars)", filepath.name, fmt)
    if fmt == "parquet":
        df_pl = pl.read_parquet(filepath)
    elif fmt == "csv":
        df_pl = pl.read_csv(filepath)
    elif fmt == "fst":
        if not _PYREADR_AVAILABLE:
            raise ImportError("pyreadr required for fst: pip install pyreadr")
        result = pyreadr.read_r(filepath)
        df_pl = pl.from_pandas(list(result.values())[0])
    else:
        raise ValueError(f"Unsupported format: '{fmt}'.")
    df = df_pl.to_pandas()
    log.info("✅ Loaded: %d rows × %d columns", len(df), len(df.columns))
    return df


# ═════════════════════════════════════════════════════════════════════════════
# CATEGORICAL ENCODING  (single entry point — post-split)
# ═════════════════════════════════════════════════════════════════════════════

def encode_categoricals(
    df: pd.DataFrame,
    features: list[str],
    fit: bool = True,
    encoders: dict[str, LabelEncoder] | None = None,
    method: str = "lightgbm",
    inplace: bool = False,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """
    Encode categorical features — single entry point, called post-split.

    Protocol
    --------
    .. code-block:: python

        train_df, encoders = encode_categoricals(train_df, features, fit=True)
        test_df, _         = encode_categoricals(test_df,  features, fit=False,
                                                 encoders=encoders)

    Encoding strategies
    -------------------
    ``"lightgbm"``
        Cast to pandas ``category`` dtype. Native LightGBM support.
        No ``fit()`` involved — no leakage risk.
        Returns an empty encoders dict.

    ``"sklearn"``
        LabelEncoder fit on train (``fit=True``), applied to test
        (``fit=False``, ``encoders=encoders``).
        Unseen labels in test are replaced by the first known class.
        Required for CART; never used for the main LightGBM model.

    Only columns present in both ``features`` and ``CATEGORICAL_FEATURES``
    (module-level constant = ``CATEGORICAL_FEATURES_B1``) are encoded.

    Parameters
    ----------
    df : pd.DataFrame
    features : list of str
    fit : bool, default True
        Fit encoders on ``df`` (train). If ``False``, apply pre-fitted encoders.
    encoders : dict of {str: LabelEncoder}, optional
        Required when ``fit=False`` and ``method="sklearn"``.
    method : {"lightgbm", "sklearn"}
    inplace : bool, default False
    verbose : bool, default True

    Returns
    -------
    df : pd.DataFrame
    encoders : dict of {str: LabelEncoder}
        Fitted encoders for ``method="sklearn"``; empty dict for ``"lightgbm"``.

    Raises
    ------
    ValueError
        If ``method`` is not recognised, or if ``fit=False`` with
        ``method="sklearn"`` and no encoders are provided.
    """
    valid_methods = ("lightgbm", "sklearn")
    if method not in valid_methods:
        raise ValueError(
            f"Unknown encoding method: '{method}'. Valid: {valid_methods}"
        )
    if not fit and encoders is None and method == "sklearn":
        raise ValueError(
            "encode_categoricals: fit=False requires pre-fitted encoders dict."
        )

    df = df if inplace else df.copy()
    fitted_encoders: dict[str, LabelEncoder] = {}

    cat_cols = [
        c for c in CATEGORICAL_FEATURES
        if c in features and c in df.columns
    ]

    if not cat_cols:
        if verbose:
            log.info("ℹ️  No categorical features to encode.")
        return df, {}

    if verbose:
        log.info(
            "🏷️  Encoding %d categorical features (method='%s', fit=%s)",
            len(cat_cols), method, fit,
        )

    for col in cat_cols:
        if pd.api.types.is_bool_dtype(df[col]) or df[col].dtype == np.bool_:
            df[col] = df[col].astype(int)
        elif df[col].dtype == object:
            df[col] = df[col].astype(str)

        if method == "lightgbm":
            df[col] = df[col].astype("category")
            if verbose:
                log.info("   • %-25s → category (%d levels)", col, df[col].nunique())

        elif method == "sklearn":
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                fitted_encoders[col] = le
                if verbose:
                    log.info(
                        "   • %-25s → int (LabelEncoder fitted, %d classes)",
                        col, len(le.classes_),
                    )
            else:
                assert encoders is not None
                le = encoders[col]
                known = set(le.classes_)
                df[col] = df[col].astype(str).apply(
                    lambda x: x if x in known else le.classes_[0]
                )
                df[col] = le.transform(df[col])
                if verbose:
                    log.info("   • %-25s → int (LabelEncoder applied)", col)

    return df, fitted_encoders if method == "sklearn" else {}


# ═════════════════════════════════════════════════════════════════════════════
# DATASET BUILDER
# ═════════════════════════════════════════════════════════════════════════════

def build_dataset(
    df: pd.DataFrame,
    features: list[str] | None = None,
    targets: list[str] | None = None,
    exclude_features: list[str] | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """
    Extract (X, y) from a prepared meta-table.

    Defaults to ``ALL_FEATURES`` (module alias for ``ALL_FEATURES_B1``) and
    ``ALL_TARGETS`` from ``column_taxonomy``, intersected with available columns.

    Parameters
    ----------
    df : pd.DataFrame
        Prepared meta-table (post-encoding, post-winsorisation).
    features : list of str, optional
        Feature columns to use. Defaults to ``ALL_FEATURES``.
    targets : list of str, optional
        Target columns to extract. Defaults to ``ALL_TARGETS``.
    exclude_features : list of str, optional
        Features to drop from the final X (e.g. temporal features for Sobol).
    verbose : bool, default True

    Returns
    -------
    X : pd.DataFrame
    y : pd.DataFrame
    used_features : list of str
        Effective columns of X after intersection and exclusion.
    used_targets : list of str
        Effective columns of y after intersection.

    Raises
    ------
    ValueError
        If no features or no targets are found in ``df`` after intersection.
    """
    if features is None:
        features = ALL_FEATURES
    if targets is None:
        targets = ALL_TARGETS

    avail_features = [c for c in features if c in df.columns]
    avail_targets  = [c for c in targets  if c in df.columns]

    if exclude_features:
        avail_features = [c for c in avail_features if c not in exclude_features]

    if not avail_features:
        raise ValueError(
            f"No features found in DataFrame. "
            f"Requested: {features[:5]}... — "
            f"Available cols: {list(df.columns[:10])}..."
        )
    if not avail_targets:
        raise ValueError(
            f"No targets found in DataFrame. "
            f"Requested: {targets[:5]}... — "
            f"Available cols: {list(df.columns[:10])}..."
        )

    if verbose:
        zero_var = [
            c for c in avail_features
            if c in df.select_dtypes(include="number").columns
            and df[c].nunique() <= 1
        ]
        if zero_var:
            log.warning(
                "   ⚠ %d zero-variance features detected (will not contribute): %s",
                len(zero_var), zero_var,
            )

    X = df[avail_features].copy()
    y = df[avail_targets].copy()

    if verbose:
        log.info(
            "📐 Dataset built — X: %d rows × %d features | y: %d rows × %d targets",
            len(X), len(X.columns), len(y), len(y.columns),
        )
        missing_f = set(features) - set(df.columns)
        missing_t = set(targets)  - set(df.columns)
        if missing_f:
            log.warning("   ⚠ %d features not found: %s", len(missing_f), sorted(missing_f))
        if missing_t:
            log.warning("   ⚠ %d targets not found: %s", len(missing_t), sorted(missing_t))

    return X, y, avail_features, avail_targets

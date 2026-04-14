"""
MetAIsAFe — data/preprocessing.py
===================================
Post-split transformations: interpolation, winsorisation, and analytical
ratio computation.

All operations in this module are applied AFTER the train/test split
to prevent data leakage.

Public API
----------
    interpolate_dynamic_vars(df, ...)          -> pd.DataFrame
    apply_winsorization(df, fit, bounds, ...)  -> (pd.DataFrame, dict)
    compute_ratios_from_stocks(df, ...)        -> pd.DataFrame

Key design rules
----------------
- Winsorisation : quantile bounds fitted on TRAIN only, applied to both
  train and test splits. Bounds must be persisted for inference (predictor.py).
- Interpolation : restricted to INTERPOLABLE_STOCKS (tree carbon stocks only).
  Crop yields, deltas, _eff_, RR_, LER_ are never interpolated.
- Ratios (RR, LER) : computed analytically from _eff_ stocks after inference.
  Never predicted by the ML model.

Workflow position
-----------------
    [Step 3] splitter.py      → train_df, test_df
        ↓
    [Step 4] preprocessing.py → interpolate_dynamic_vars  (optional, B1 only)
                              → apply_winsorization        (fit on train, apply on test)
        ↓
    [Step 9] preprocessing.py → compute_ratios_from_stocks (post-inference only)

Author  : Étienne SABY
Updated : 2026-04
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from column_taxonomy import (
    INTERPOLABLE_STOCKS,
    NON_INTERPOLABLE_PATTERNS,
    WINSORIZE_STOCKS,
    WINSORIZE_EXCLUDE_PATTERNS,
)

log = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# STEP 4a — INTERPOLATION  (optional — B1 diagnostic only)
# ═════════════════════════════════════════════════════════════════════════════

def interpolate_dynamic_vars(
    df: pd.DataFrame,
    stock_cols: list[str] | None = None,
    na_pct_threshold: float = 0.20,
    method: str = "linear",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Fill NAs in interpolable stock time series using within-SimID interpolation.

    Only columns in ``INTERPOLABLE_STOCKS`` (tree carbon stocks) are eligible.
    Crop yields, deltas, ``_eff_``, ``RR_``, and ``LER_`` columns are never
    interpolated — their episodic nature makes linear interpolation physically
    meaningless.

    SimIDs with NA rate above ``na_pct_threshold`` in a given stock column are
    skipped for that column: too many missing values indicate a structural
    simulation issue that exclusion (Step 2) should have handled.

    Interpolation is performed per SimID via ``groupby().transform()``.
    ``limit_direction="both"`` ensures leading and trailing NaN values
    (start and end of a SimID time series) are also filled.

    Parameters
    ----------
    df : pd.DataFrame
        Meta-table after train/test split.
    stock_cols : list of str, optional
        Columns to interpolate. Defaults to ``INTERPOLABLE_STOCKS`` intersected
        with available columns, further filtered by ``NON_INTERPOLABLE_PATTERNS``.
    na_pct_threshold : float, default 0.20
        Skip interpolation for a SimID×column pair if NA rate exceeds this.
    method : str, default ``"linear"``
        Interpolation method passed to ``pd.Series.interpolate()``.
    verbose : bool, default True

    Returns
    -------
    pd.DataFrame
    """
    log.info("Interpolation of dynamic stock variables")
    df = df.copy()

    if stock_cols is None:
        stock_cols = [c for c in INTERPOLABLE_STOCKS if c in df.columns]

    # Filter out non-interpolable patterns
    stock_cols = [
        c for c in stock_cols
        if not any(pat in c for pat in NON_INTERPOLABLE_PATTERNS)
    ]

    if not stock_cols:
        log.info("   ℹ No interpolable columns found — skipped.")
        return df

    sort_keys = [c for c in ["SimID", "Harvest_Year_Absolute"] if c in df.columns]
    if sort_keys:
        df.sort_values(sort_keys, inplace=True)

    has_simid = "SimID" in df.columns
    total_filled = 0

    for col in stock_cols:
        n_na_before = df[col].isna().sum()
        if n_na_before == 0:
            continue

        if has_simid:
            def _interp_group(g: pd.Series) -> pd.Series:
                if g.isna().mean() > na_pct_threshold:
                    return g   # too many NAs — skip this SimID
                return g.interpolate(method=method, limit_direction="both")

            df[col] = df.groupby("SimID")[col].transform(_interp_group)
        else:
            df[col] = df[col].interpolate(method=method, limit_direction="both")

        n_filled = n_na_before - df[col].isna().sum()
        total_filled += n_filled
        if verbose and n_filled > 0:
            log.info("   • %-30s  filled %d NAs", col, n_filled)

    log.info("   ✓ Total NAs filled: %d across %d columns", total_filled, len(stock_cols))
    return df


# ═════════════════════════════════════════════════════════════════════════════
# STEP 4b — WINSORISATION (no data leakage)
# ═════════════════════════════════════════════════════════════════════════════

def apply_winsorization(
    df: pd.DataFrame,
    quantiles: tuple[float, float] = (0.01, 0.99),
    stock_cols: list[str] | None = None,
    fit: bool = True,
    bounds: dict[str, tuple[float, float]] | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    """
    Winsorise physical stock targets to cap extreme outliers.

    Quantile bounds are fitted on the train split only and applied to both
    train and test splits. Bounds should be persisted and reused at inference
    time (predictor.py).

    Usage (no leakage)
    ------------------
    .. code-block:: python

        train_df, bounds = apply_winsorization(train_df, fit=True)
        test_df,  _      = apply_winsorization(test_df,  fit=False, bounds=bounds)

    Only columns in ``WINSORIZE_STOCKS`` are affected. Columns matching any
    pattern in ``WINSORIZE_EXCLUDE_PATTERNS`` (RR_, LER_, _eff_, _delta_,
    Sobol parameters) are explicitly skipped.

    Parameters
    ----------
    df : pd.DataFrame
    quantiles : tuple of float, default (0.01, 0.99)
        Lower and upper quantile bounds.
    stock_cols : list of str, optional
        Explicit list of columns to winsorise. Defaults to
        ``WINSORIZE_STOCKS`` filtered by ``WINSORIZE_EXCLUDE_PATTERNS``.
    fit : bool, default True
        If ``True``, compute quantile bounds from ``df`` (train split).
        If ``False``, apply pre-computed ``bounds`` (test split or inference).
    bounds : dict, optional
        Pre-computed bounds ``{col: (lo, hi)}``. Required when ``fit=False``.
    verbose : bool, default True

    Returns
    -------
    df_winsorized : pd.DataFrame
    bounds : dict of {str: tuple[float, float]}
        When ``fit=True``  : bounds computed from ``df``.
        When ``fit=False`` : the ``bounds`` dict passed as input (unchanged).

    Raises
    ------
    ValueError
        If ``fit=False`` and ``bounds`` is None.
    """
    if not fit and bounds is None:
        raise ValueError(
            "apply_winsorization: fit=False requires pre-computed bounds dict."
        )

    log.info(
        "Winsorisation [%.3f, %.3f]  (fit=%s)",
        quantiles[0], quantiles[1], fit,
    )
    df = df.copy()

    if stock_cols is None:
        stock_cols = [
            c for c in WINSORIZE_STOCKS
            if c in df.columns
            and not any(pat in c for pat in WINSORIZE_EXCLUDE_PATTERNS)
        ]

    if not stock_cols:
        log.info("   ℹ No columns to winsorise — skipped.")
        return df, {}

    fitted_bounds: dict[str, tuple[float, float]] = {}

    for col in stock_cols:
        if fit:
            lo = float(df[col].quantile(quantiles[0]))
            hi = float(df[col].quantile(quantiles[1]))
            fitted_bounds[col] = (lo, hi)
        else:
            assert bounds is not None
            if col not in bounds:
                log.warning("   ⚠ No bounds for '%s' — skipping", col)
                continue
            lo, hi = bounds[col]

        n_clipped = int(((df[col] < lo) | (df[col] > hi)).sum())
        df[col] = df[col].clip(lower=lo, upper=hi)

        if verbose:
            log.info(
                "   • %-30s  [%.3f, %.3f]  — %d values clipped",
                col, lo, hi, n_clipped,
            )

    log.info("   ✓ Winsorisation complete — %d columns processed", len(stock_cols))
    return df, fitted_bounds if fit else bounds


# ═════════════════════════════════════════════════════════════════════════════
# STEP 9 — ANALYTICAL RATIO COMPUTATION (post-inference only)
# ═════════════════════════════════════════════════════════════════════════════

def compute_ratios_from_stocks(
    df: pd.DataFrame,
    zero_threshold: float = 0.001,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Compute RR and LER analytically from ``_eff_`` physical stocks.

    Called post-inference only — never during training. Requires that
    ``compute_effective_vars()`` (preparation.py) has already been applied
    to produce the ``_eff_`` columns.

    Ratios computed
    ---------------
    ``RR_crop_yield``
        yield_eff_AF / yield_eff_TA
    ``RR_crop_biomass``
        biomass_eff_AF / biomass_eff_TA
    ``RR_tree_carbonStem``
        carbonStem_eff_AF / carbonStem_eff_TF
    ``LER_yield_carbonStem``
        RR_crop_yield + RR_tree_carbonStem
    ``LER_biomass_carbonStem``
        RR_crop_biomass + RR_tree_carbonStem

    LER is computed as the **sum** of the two component RR values, following
    the standard Land Equivalent Ratio definition (Mead & Willey, 1980).

    Divisions are set to NA when the denominator < ``zero_threshold`` to avoid
    numerical instability in the juvenile phase (years 1–10).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain the relevant ``_eff_`` columns produced by
        ``compute_effective_vars()``.
    zero_threshold : float, default 0.001
        Minimum denominator value below which the ratio is set to NA.
    verbose : bool, default True

    Returns
    -------
    pd.DataFrame
        Input DataFrame with up to 5 ratio columns added in-place on a copy.
    """
    log.info("Analytical RR & LER from physical stocks")
    log.info("   Zero threshold: %.4f", zero_threshold)
    df = df.copy()
    n_ratios = 0

    # ── Crop RR ──────────────────────────────────────────────────────────
    if {"yield_eff_AF", "yield_eff_TA"}.issubset(df.columns):
        df["RR_crop_yield"] = np.where(
            df["yield_eff_TA"] > zero_threshold,
            df["yield_eff_AF"] / df["yield_eff_TA"],
            np.nan,
        )
        n_ratios += 1

    if {"biomass_eff_AF", "biomass_eff_TA"}.issubset(df.columns):
        df["RR_crop_biomass"] = np.where(
            df["biomass_eff_TA"] > zero_threshold,
            df["biomass_eff_AF"] / df["biomass_eff_TA"],
            np.nan,
        )
        n_ratios += 1

    # ── Tree RR ──────────────────────────────────────────────────────────
    if {"carbonStem_eff_AF", "carbonStem_eff_TF"}.issubset(df.columns):
        df["RR_tree_carbonStem"] = np.where(
            df["carbonStem_eff_TF"] > zero_threshold,
            df["carbonStem_eff_AF"] / df["carbonStem_eff_TF"],
            np.nan,
        )
        n_ratios += 1

    # ── LER (sum of RR components) ────────────────────────────────────────
    if {"RR_crop_yield", "RR_tree_carbonStem"}.issubset(df.columns):
        df["LER_yield_carbonStem"] = df["RR_crop_yield"] + df["RR_tree_carbonStem"]
        n_ratios += 1

    if {"RR_crop_biomass", "RR_tree_carbonStem"}.issubset(df.columns):
        df["LER_biomass_carbonStem"] = df["RR_crop_biomass"] + df["RR_tree_carbonStem"]
        n_ratios += 1

    if verbose:
        log.info("   ✓ %d ratio columns computed (RR + LER)", n_ratios)
        rr_cols  = [c for c in df.columns if c.startswith("RR_")]
        ler_cols = [c for c in df.columns if c.startswith("LER_")]
        if rr_cols:
            log.info(
                "   RR  stats:\n%s",
                df[rr_cols].describe().loc[["mean", "50%", "min", "max"]].to_string(),
            )
        if ler_cols:
            log.info(
                "   LER stats:\n%s",
                df[ler_cols].describe().loc[["mean", "50%", "min", "max"]].to_string(),
            )

    return df

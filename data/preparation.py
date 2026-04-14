"""
MetAIsAFe — data/preparation.py
=================================
Derived columns, crop filtering, population classification, conservative
cleaning, and analytical variable computation.

Public API
----------
    add_derived_columns(df, verbose)                    -> pd.DataFrame
    filter_crops(df, excluded_crops, ...)               -> (pd.DataFrame, dict)
    filter_population(df, population, ...)              -> (pd.DataFrame, pd.Series)
    clean(df, ...)                                      -> pd.DataFrame
    compute_effective_vars(df, verbose)                 -> pd.DataFrame
    compute_carbon_deltas(df, verbose)                  -> pd.DataFrame

Pipeline position
-----------------
    [Step 1] add_derived_columns → filter_crops → clean → compute_effective_vars
    [Step 2] filter_population  → routes to nominal population (yield_ok × tree_ok)

Notes
-----
- Winsorisation and interpolation are applied AFTER train/test split
  (see data/preprocessing.py) to prevent data leakage.
- compute_ratios_from_stocks() is post-inference only (see data/preprocessing.py).
- compute_carbon_deltas() is for diagnostic use only — not an ML target.

Author  : Étienne SABY
Updated : 2026-04
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from _version import __version__
from config import TREE_FAIL_THRESHOLD, YIELD_FAIL_THRESHOLD, YIELD_FAIL_RATE
from column_taxonomy import NOMINAL_POPULATION, POPULATION_LABELS

log = logging.getLogger(__name__)

_CROP_BASES = ["yield", "grainBiomass", "biomass"]
_TREE_BASES = ["carbonStem", "carbonBranches", "carbonCoarseRoots"]


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _match_patterns(columns: list[str], patterns: list[str]) -> list[str]:
    """Return column names containing any of the given substrings."""
    return [c for c in columns if any(p in c for p in patterns)]


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1a — DERIVED COLUMNS
# ═════════════════════════════════════════════════════════════════════════════

def add_derived_columns(
    df: pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Add engineered columns required by the ML pipeline.

    Columns created
    ---------------
    ``Harvest_Year_Absolute``
        Per-SimID rank of ``Harvest_Year_AF``: the first harvest year of each
        simulation becomes 1, the second 2, etc.
        This is the PRIMARY temporal feature used by the model.

    ``Rotation``
        Integer flag (0/1): 1 if ``rot_id`` is non-null and non-empty.

    ``rot_id`` NaN fill
        NaN values in ``rot_id`` are filled with ``"monoculture"``.

    ``main_crop`` NaN fill
        NaN values in ``main_crop`` are filled with ``"rotation"`` to avoid
        categorical encoding issues downstream.

    Parameters
    ----------
    df : pd.DataFrame
        Raw meta-table as loaded from disk.
    verbose : bool, default True

    Returns
    -------
    pd.DataFrame
    """
    log.info("Adding derived columns")
    df = df.copy()

    # ── Harvest_Year_Absolute ──────────────────────────────────────────────
    if "Harvest_Year_AF" in df.columns and "SimID" in df.columns:
        df["Harvest_Year_Absolute"] = (
            df.groupby("SimID")["Harvest_Year_AF"]
            .transform(lambda x: x - x.min() + 1)
        )
        if verbose:
            log.info("   ✓ Added 'Harvest_Year_Absolute' (normalised per SimID)")
    else:
        log.warning(
            "   ⚠ 'Harvest_Year_AF' or 'SimID' missing — "
            "'Harvest_Year_Absolute' not created."
        )

    # ── Rotation flag ──────────────────────────────────────────────────────
    if "rot_id" in df.columns:
        df["Rotation"] = (
            df["rot_id"].notna() &
            (df["rot_id"].astype(str).str.strip() != "")
        ).astype(int)
        if verbose:
            n_rot = df["Rotation"].sum()
            log.info(
                "   ✓ Added 'Rotation' flag (%d / %d cycles in rotation)",
                n_rot, len(df),
            )
        n_nan = df["rot_id"].isna().sum()
        if n_nan > 0:
            df["rot_id"] = df["rot_id"].fillna("monoculture")
            if verbose:
                log.info(
                    "   ✓ Filled %d NaN values in 'rot_id' with 'monoculture'",
                    n_nan,
                )
    else:
        log.warning("   ⚠ 'rot_id' missing — 'Rotation' not created.")

    # ── main_crop fallback ─────────────────────────────────────────────────
    if "main_crop" in df.columns:
        n_nan = df["main_crop"].isna().sum()
        if n_nan > 0:
            df["main_crop"] = df["main_crop"].fillna("rotation")
            if verbose:
                log.info(
                    "   ✓ Filled %d NaN values in 'main_crop' with 'rotation'",
                    n_nan,
                )

    return df


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1b — CROP FILTERING
# ═════════════════════════════════════════════════════════════════════════════

def filter_crops(
    df: pd.DataFrame,
    excluded_crops: list[str],
    crop_col: str = "Crop_Name",
    simid_col: str = "SimID",
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Remove rows belonging to excluded crop species.

    Reports the number of SimIDs completely lost (all their crops excluded)
    alongside per-crop row and SimID counts.

    Parameters
    ----------
    df : pd.DataFrame
    excluded_crops : list of str
        Crop names to exclude, e.g. ``["rape", "pea"]``.
    crop_col : str, default ``"Crop_Name"``
    simid_col : str, default ``"SimID"``
    verbose : bool, default True

    Returns
    -------
    df_filtered : pd.DataFrame
    report : dict
        Keys: ``crops``, ``n_rows_before``, ``n_rows_after``,
        ``n_rows_removed``, ``pct_rows_lost``,
        ``n_sims_before``, ``n_sims_after``, ``n_sims_lost``, ``detail``.
        Empty dict if no crops to exclude or ``crop_col`` absent.
    """
    log.info("Filtering crops: %s", excluded_crops)
    df = df.copy()

    if not excluded_crops or crop_col not in df.columns:
        log.info("   ℹ No crops to exclude — skipped.")
        return df, {}

    n_before    = len(df)
    sims_before = df[simid_col].nunique() if simid_col in df.columns else None

    detail = (
        df[df[crop_col].isin(excluded_crops)]
        .groupby(crop_col)
        .agg(n_rows=(crop_col, "size"), n_sims=(simid_col, "nunique"))
        .reset_index()
        .rename(columns={crop_col: "crop"})
    )

    if simid_col in df.columns:
        sims_crops = (
            df.groupby(simid_col)[crop_col]
            .apply(lambda x: list(x.unique()))
            .reset_index(name="all_crops")
        )
        sims_crops["n_excluded"] = sims_crops["all_crops"].apply(
            lambda crops: sum(c in excluded_crops for c in crops)
        )
        sims_crops["will_be_lost"] = (
            sims_crops["n_excluded"] == sims_crops["all_crops"].apply(len)
        )
        n_sims_lost = int(sims_crops["will_be_lost"].sum())
    else:
        n_sims_lost = 0

    df = df[~df[crop_col].isin(excluded_crops)].copy()

    n_after    = len(df)
    sims_after = df[simid_col].nunique() if simid_col in df.columns else None
    n_removed  = n_before - n_after
    pct_lost   = 100.0 * n_removed / max(n_before, 1)

    report: dict[str, Any] = {
        "crops":          excluded_crops,
        "n_rows_before":  n_before,
        "n_rows_after":   n_after,
        "n_rows_removed": n_removed,
        "pct_rows_lost":  pct_lost,
        "n_sims_before":  sims_before,
        "n_sims_after":   sims_after,
        "n_sims_lost":    n_sims_lost,
        "detail":         detail,
    }

    if verbose and n_removed > 0:
        log.info(
            "   ✗ Excluded crops [%s]: %d rows removed (%.1f%%)",
            ", ".join(excluded_crops), n_removed, pct_lost,
        )
        if n_sims_lost > 0:
            log.info(
                "   %d simulations completely lost (all crops excluded)",
                n_sims_lost,
            )

    return df, report


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — POPULATION CLASSIFICATION & FILTERING
# ═════════════════════════════════════════════════════════════════════════════

def filter_population(
    df: pd.DataFrame,
    population: str | None = "yield_ok × tree_ok",
    tree_fail_threshold: float = TREE_FAIL_THRESHOLD,
    yield_fail_threshold: float = YIELD_FAIL_THRESHOLD,
    yield_fail_rate: float = YIELD_FAIL_RATE,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Assign each SimID to one of four structural populations and optionally
    filter to a target population.

    Populations
    -----------
    yield_ok × tree_ok       : main meta-model training set
    yield_ok × tree_failed   : cultural-only model
    yield_fail × tree_ok     : geographic rejection rule
    yield_fail × tree_failed : full rejection (yield=0, carbon=0)

    Classification rules
    --------------------
    ``tree_failed``
        ``carbonStem_AF`` at the last cycle (highest ``Harvest_Year_Absolute``,
        or last row per SimID if absent) < ``tree_fail_threshold``.

    ``yield_failed``
        Fraction of cycles where ``yield_AF < yield_fail_threshold``
        OR ``yield_TA < yield_fail_threshold`` exceeds ``yield_fail_rate``.
        (fraction = number of failed *rows* / total rows for that SimID)

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: SimID, carbonStem_AF, yield_AF, yield_TA.
        Harvest_Year_Absolute is used for last-cycle detection if present.
    population : str or None, default ``"yield_ok × tree_ok"``
        Target population to keep. Pass ``None`` to return the full DataFrame
        with the ``'population'`` column added.
    tree_fail_threshold : float, default ``TREE_FAIL_THRESHOLD``
    yield_fail_threshold : float, default ``YIELD_FAIL_THRESHOLD``
    yield_fail_rate : float, default ``YIELD_FAIL_RATE``
    verbose : bool, default True

    Returns
    -------
    df_filtered : pd.DataFrame
        Rows of the target population (or full DataFrame if population=None),
        with a ``'population'`` column added.
    population_series : pd.Series
        Per-SimID population label (index = SimID). Always returned.

    Raises
    ------
    ValueError
        If required columns are missing, or if ``population`` is not in
        ``POPULATION_LABELS``.
    """
    required = {"SimID", "carbonStem_AF", "yield_AF", "yield_TA"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"filter_population: missing required columns: {sorted(missing)}"
        )

    if population is not None and population not in POPULATION_LABELS:
        raise ValueError(
            f"filter_population: unknown population '{population}'. "
            f"Valid: {POPULATION_LABELS}"
        )

    # ── 1. Tree failure : last cycle per SimID ────────────────────────────
    sort_col = "Harvest_Year_Absolute" if "Harvest_Year_Absolute" in df.columns else None

    if sort_col:
        last_cycle = (
            df.sort_values(["SimID", sort_col])
            .groupby("SimID", as_index=False)
            .last()
        )
    else:
        last_cycle = df.groupby("SimID", as_index=False).last()

    tree_failed = (
        last_cycle.set_index("SimID")["carbonStem_AF"]
        .lt(tree_fail_threshold)
        .rename("tree_failed")
    )

    # ── 2. Yield failure : fraction of failed rows per SimID ──────────────
    cycle_fail = (
        (df["yield_AF"] < yield_fail_threshold) |
        (df["yield_TA"] < yield_fail_threshold)
    )
    yield_fail_per_sim = (
        df.assign(_fail=cycle_fail)
        .groupby("SimID")["_fail"]
        .mean()
        .gt(yield_fail_rate)
        .rename("yield_failed")
    )

    # ── 3. Build population label per SimID ───────────────────────────────
    pop_df = pd.DataFrame({
        "tree_failed":  tree_failed,
        "yield_failed": yield_fail_per_sim,
    })

    def _label(row: pd.Series) -> str:
        y = "yield_ok"     if not row["yield_failed"] else "yield_fail"
        t = "tree_ok"      if not row["tree_failed"]  else "tree_failed"
        return f"{y} × {t}"

    population_series = pop_df.apply(_label, axis=1).rename("population")

    if verbose:
        log.info("Population distribution:")
        counts = population_series.value_counts()
        pcts   = (counts / len(population_series) * 100).round(1)
        for label in POPULATION_LABELS:
            n   = counts.get(label, 0)
            pct = pcts.get(label, 0.0)
            log.info("  %-35s : %4d  (%4.1f%%)", label, n, pct)

    # ── 4. Merge population label back onto df ────────────────────────────
    df = df.copy()
    df["population"] = df["SimID"].map(population_series)

    # ── 5. Filter to target population ───────────────────────────────────
    if population is not None:
        n_before = len(df)
        df = df[df["population"] == population].copy()
        n_after  = len(df)
        n_sims   = df["SimID"].nunique()
        if verbose:
            log.info(
                "Filtered to '%s' : %d rows | %d SimIDs  (%.1f%% of total rows)",
                population,
                n_after,
                n_sims,
                100.0 * n_after / max(n_before, 1),
            )

    return df, population_series


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1c — CONSERVATIVE CLEANING
# ═════════════════════════════════════════════════════════════════════════════

def clean(
    df: pd.DataFrame,
    drop_full_na_cols: bool = True,
    drop_duplicate_rows: bool = True,
    clip_fractions: bool = True,
    clip_physical: bool = True,
    high_na_warn_threshold: float = 0.30,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Conservative cleaning — no simulation exclusions.

    Steps applied
    -------------
    1. Drop fully-NA columns.
    2. Drop exact duplicate rows.
    3. Clip stress / fraction columns to [0, 1].
    4. Clip physical stock columns to ≥ 0.
    5. Warn about columns exceeding ``high_na_warn_threshold``.
    6. Warn about duplicate (SimID, Cycle_Nb, Crop_Name) row keys.

    Column matching for steps 3 and 4 is pattern-based (substring match).

    Parameters
    ----------
    df : pd.DataFrame
    drop_full_na_cols : bool, default True
    drop_duplicate_rows : bool, default True
    clip_fractions : bool, default True
        Clips columns matching stress/fraction patterns to [0, 1].
    clip_physical : bool, default True
        Clips columns matching physical stock patterns to ≥ 0.
    high_na_warn_threshold : float, default 0.30
        Emit a warning for columns with NA rate above this fraction.
    verbose : bool, default True

    Returns
    -------
    pd.DataFrame
    """
    log.info("Cleaning (conservative)")
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    stress_fraction_patterns = ["Stress", "stress", "relativeTotalParIncident"]
    positive_physical_patterns = [
        "yield_", "grainBiomass_", "biomass_",
        "carbonStem_", "carbonBranches_", "carbonCoarseRoots_",
        "carbonStump_", "carbonFineRoots_", "carbonFruit_",
        "totalLeafArea_", "tree_dbh_", "tree_height_",
        "density_", "plotWidth", "plotHeight", "strip_width",
        "waterUptake", "waterDemand", "precipitation", "irrigation",
        "ETP_cycle", "GDD_cycle", "globalRadiation",
        "soilDepth", "sand", "clay", "stone",
    ]

    # 1 — Fully-NA columns
    if drop_full_na_cols:
        full_na = df.columns[df.isna().all()].tolist()
        if full_na:
            df.drop(columns=full_na, inplace=True)
            num_cols = [c for c in num_cols if c not in full_na]
            log.info("   ✓ Dropped %d fully-NA columns: %s", len(full_na), full_na)

    # 2 — Exact duplicates
    if drop_duplicate_rows:
        n0 = len(df)
        df.drop_duplicates(inplace=True)
        n_dupes = n0 - len(df)
        if n_dupes:
            log.info("   ✓ Dropped %d duplicate rows", n_dupes)

    # 3 — Stress / fraction → [0, 1]
    if clip_fractions:
        frac_cols = _match_patterns(num_cols, stress_fraction_patterns)
        n_clip = 0
        for col in frac_cols:
            bad = (df[col] < 0) | (df[col] > 1)
            n_clip += bad.sum()
            if bad.any():
                df[col] = df[col].clip(0, 1)
        if n_clip:
            log.info(
                "   ✓ Clipped %d values to [0, 1] across %d stress columns",
                n_clip, len(frac_cols),
            )

    # 4 — Physical ≥ 0
    if clip_physical:
        phys_cols = _match_patterns(num_cols, positive_physical_patterns)
        n_neg = 0
        for col in phys_cols:
            bad = df[col] < 0
            n_neg += bad.sum()
            if bad.any():
                df[col] = df[col].clip(lower=0)
        if n_neg:
            log.info(
                "   ✓ Clipped %d negative values to 0 across %d physical columns",
                n_neg, len(phys_cols),
            )

    # 5 — High-NA warning
    na_pct  = df.isna().mean()
    high_na = na_pct[na_pct > high_na_warn_threshold].sort_values(ascending=False)
    if len(high_na):
        log.warning(
            "   ⚠ %d columns with >%.0f%% NA:\n%s",
            len(high_na), high_na_warn_threshold * 100,
            high_na.head(10).to_string(),
        )

    # 6 — Duplicate keys warning
    key_cols = ["SimID", "Cycle_Nb", "Crop_Name"]
    if all(c in df.columns for c in key_cols):
        dup_keys = df.duplicated(subset=key_cols, keep=False)
        if dup_keys.any():
            log.warning(
                "   ⚠ %d rows with duplicate (SimID, Cycle_Nb, Crop_Name)",
                dup_keys.sum(),
            )

    log.info("   ✓ Clean — %d rows × %d cols", *df.shape)
    return df


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1d — EFFECTIVE VARIABLES
# ═════════════════════════════════════════════════════════════════════════════

def compute_effective_vars(
    df: pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Compute per-hectare effective variables from physical stocks.

    Crop effective variables
    ------------------------
    ``{base}_eff_AF = {base}_AF × cult_frac``
        where ``cult_frac = (plotWidth − strip_width) / plotWidth``,
        clipped to [0, 1]. Corrects for tree strip reducing cultivated area.

    ``{base}_eff_TA = {base}_TA``
        No correction (100% cultivated in sole-crop reference).

    Tree effective variables
    ------------------------
    ``{base}_eff_AF = {base}_AF × density_AF``   [kgC/tree → kgC/ha]
    ``{base}_eff_TF = {base}_TF × density_TF``   [kgC/tree → kgC/ha]

    Bases covered: yield, grainBiomass, biomass (crop) and
    carbonStem, carbonBranches, carbonCoarseRoots (tree).

    These ``_eff_`` columns feed ``compute_ratios_from_stocks()``
    (data/preprocessing.py) and must NOT be used as ML features or targets.

    Parameters
    ----------
    df : pd.DataFrame
    verbose : bool, default True

    Returns
    -------
    pd.DataFrame
    """
    log.info("Effective variables (analytical)")
    df = df.copy()
    n_c_af = n_c_ta = n_t_af = n_t_tf = 0

    if {"plotWidth", "strip_width"}.issubset(df.columns):
        df["cult_frac"] = (
            (df["plotWidth"] - df["strip_width"]) / df["plotWidth"]
        ).clip(0, 1)
        for base in _CROP_BASES:
            af_col, ta_col = f"{base}_AF", f"{base}_TA"
            if af_col in df.columns:
                df[f"{base}_eff_AF"] = df[af_col] * df["cult_frac"]
                n_c_af += 1
            if ta_col in df.columns:
                df[f"{base}_eff_TA"] = df[ta_col]
                n_c_ta += 1
    else:
        log.warning("   ⚠ plotWidth/strip_width missing — crop effective vars skipped")

    for base in _TREE_BASES:
        af_col, tf_col = f"{base}_AF", f"{base}_TF"
        if "density_AF" in df.columns and af_col in df.columns:
            df[f"{base}_eff_AF"] = df[af_col] * df["density_AF"]
            n_t_af += 1
        if "density_TF" in df.columns and tf_col in df.columns:
            df[f"{base}_eff_TF"] = df[tf_col] * df["density_TF"]
            n_t_tf += 1

    total = n_c_af + n_c_ta + n_t_af + n_t_tf
    if verbose:
        log.info("   ✓ _eff variables computed:")
        log.info("     Crop : %d AF + %d TA", n_c_af, n_c_ta)
        log.info("     Tree : %d AF + %d TF", n_t_af, n_t_tf)
        log.info("     Total: %d columns added", total)
    return df


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1e — CARBON DELTAS (diagnostic only)
# ═════════════════════════════════════════════════════════════════════════════

def compute_carbon_deltas(
    df: pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Compute annual carbon increments (deltas) for tree components.

    For each tree base (carbonStem, carbonBranches, carbonCoarseRoots)
    and each suffix (_AF, _TF):

        ``{base}_delta{suffix} = {base}{suffix} − lag({base}{suffix})``

    Differences are computed within each SimID group. The first cycle
    of each SimID yields NaN (no previous value — expected behaviour).

    Sort order used: SimID → Harvest_Year_Absolute → Cycle_Nb
    (whichever columns are present).

    For diagnostic use only. Delta columns are NOT ML targets
    and must NOT be interpolated.

    Parameters
    ----------
    df : pd.DataFrame
    verbose : bool, default True

    Returns
    -------
    pd.DataFrame
    """
    log.info("Carbon deltas (diagnostic only — NOT ML targets)")
    df = df.copy()

    sort_keys = [
        c for c in ["SimID", "Harvest_Year_Absolute", "Cycle_Nb"]
        if c in df.columns
    ]
    if sort_keys:
        df.sort_values(sort_keys, inplace=True)

    has_group = "SimID" in df.columns
    n_deltas = 0

    for base in _TREE_BASES:
        for suffix in ("_AF", "_TF"):
            col       = f"{base}{suffix}"
            delta_col = f"{base}_delta{suffix}"
            if col not in df.columns:
                continue
            if has_group:
                df[delta_col] = df.groupby("SimID")[col].diff()
            else:
                df[delta_col] = df[col].diff()
            n_deltas += 1

    if verbose:
        log.info("   ✓ %d delta columns computed", n_deltas)
    return df

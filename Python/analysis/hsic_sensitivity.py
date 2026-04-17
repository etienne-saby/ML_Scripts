"""
hsic_sensitivity.py
====================
HSIC-based global sensitivity analysis for mixed (continuous + categorical)
inputs with correlated features.

Public API
----------
compute_hsic_indices(df, features_num, features_cat, targets, ...)
    Core computation on a given DataFrame slice.

compute_hsic_by_year(df, features_num, features_cat, targets, ...)
    Temporal loop: runs compute_hsic_indices for each Harvest_Year_Absolute,
    parallelised across years via ProcessPoolExecutor.

validate_hsic_vs_spearman(df, features_num, targets, ...)
    Sanity check: compares HSIC ranks vs Spearman² ranks.

plot_hsic_heatmap(results_by_year, target, ...)
    Heatmap of HSIC indices over time (feature × year).

plot_hsic_lines(results_by_year, target, top_n, ...)
    Line plot of top-N features over time for one target.

Design notes
------------
- All kernel computations are O(n²) via einsum — no O(n³) matrix products.
- Bootstrap subsampling keeps wall-clock time bounded regardless of N.
- RBF bandwidth (sigma) is estimated once per feature on the full slice via
  the median heuristic, then reused across all bootstrap iterations (Axe 3).
- Years are processed in parallel via ProcessPoolExecutor — each year is an
  independent computation unit (Axe 1).
- Crop_Name heterogeneity in rotations is handled via the `crop_mode`
  parameter in compute_hsic_by_year:
      "all"      → all crops pooled (recommended when main_crop is a Sobol
                   feature — avoids double-handling of crop identity)
      "dominant" → most frequent Crop_Name per year retained (legacy mode)
      "per_crop" → separate indices computed per Crop_Name (returns dict)
- Primary key of the input DataFrame is assumed to be
  (SimID, Harvest_Year_Absolute, Crop_Name). If multiple Crop_Name values
  exist for the same (SimID, year), an automatic aggregation guard is applied
  inside the year worker before HSIC computation.

Author  : Étienne SABY
Updated : 2026-04
"""

from __future__ import annotations

import logging
import os
import itertools
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Literal
from utils.plot_utils import save_figure
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

log = logging.getLogger(__name__)


# =============================================================================
# INTERNAL — Kernel primitives
# =============================================================================

def _estimate_sigma(x: np.ndarray) -> float:
    """
    Estimate the RBF bandwidth via the median heuristic on the full array.

    Formula
    -------
    sigma = sqrt(median(pairwise_squared_distances) / 2)

    This function is intended to be called **once per feature on the full
    DataFrame slice**, before the bootstrap loop, so that sigma is not
    recomputed at every iteration (Axe 3 optimisation).

    Parameters
    ----------
    x : 1-D numeric array of shape (n,).

    Returns
    -------
    sigma : Positive float. Returns 1.0 if the array has fewer than 2
            distinct values (degenerate case).
    """
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    D = cdist(x, x, metric="sqeuclidean")
    d_flat = D[np.triu_indices_from(D, k=1)]
    if len(d_flat) == 0 or np.median(d_flat) == 0:
        return 1.0
    return float(np.sqrt(np.median(d_flat) / 2))


def _rbf_kernel_centered(x: np.ndarray, sigma: float | None = None) -> np.ndarray:
    """
    Centered RBF (Gaussian) kernel matrix for a 1-D array.

    Parameters
    ----------
    x     : 1-D array of shape (n,).
    sigma : Bandwidth. If None, the median heuristic is applied on `x`
            directly (slower — prefer pre-computing sigma via _estimate_sigma
            and passing it explicitly for repeated bootstrap calls).

    Returns
    -------
    Kc : Double-centered kernel matrix of shape (n, n).

    Notes
    -----
    Double-centering is performed in O(n²) without forming the full
    centering matrix H explicitly:
        Kc = K - row_mean - col_mean + grand_mean
    """
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    D = cdist(x, x, metric="sqeuclidean")
    if sigma is None:
        d_flat = D[np.triu_indices_from(D, k=1)]
        sigma = float(np.sqrt(np.median(d_flat) / 2)) if len(d_flat) > 0 else 1.0
    sigma = max(sigma, 1e-10)
    K = np.exp(-D / (2.0 * sigma**2))
    row_mean   = K.mean(axis=1, keepdims=True)
    col_mean   = K.mean(axis=0, keepdims=True)
    grand_mean = K.mean()
    return K - row_mean - col_mean + grand_mean


def _hsic_normalized(Kxc: np.ndarray, Kyc: np.ndarray) -> float:
    """
    Normalized HSIC index in [0, 1].

    Formula
    -------
    HSIC_norm(X, Y) = HSIC(X, Y) / sqrt(HSIC(X, X) * HSIC(Y, Y))

    Interpretation
    --------------
    Analogous to a non-linear R²:
      - 0  → X and Y are statistically independent
      - 1  → X and Y are functionally dependent

    Both Kxc and Kyc must already be double-centered (output of
    _rbf_kernel_centered).

    Parameters
    ----------
    Kxc : Centered kernel matrix for X, shape (n, n).
    Kyc : Centered kernel matrix for Y, shape (n, n).

    Returns
    -------
    Normalized HSIC value clipped to [0, 1].
    """
    n = Kxc.shape[0]
    f = 1.0 / (n - 1) ** 2
    hsic_xy = f * float(np.einsum("ij,ij->", Kxc, Kyc))
    hsic_xx = f * float(np.einsum("ij,ij->", Kxc, Kxc))
    hsic_yy = f * float(np.einsum("ij,ij->", Kyc, Kyc))
    denom = np.sqrt(max(hsic_xx * hsic_yy, 1e-24))
    return float(np.clip(hsic_xy / denom, 0.0, 1.0))


# =============================================================================
# PUBLIC — Core computation
# =============================================================================

def compute_hsic_indices(
    df: pd.DataFrame,
    features_num: list[str],
    features_cat: list[str],
    targets: list[str],
    n_boot: int = 100,
    boot_size: int = 500,
    random_state: int = 42,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Compute bootstrap HSIC sensitivity indices for a given DataFrame slice.

    All features (numeric and categorical) are treated with an RBF kernel.
    Categorical features must be ordinally encoded as integers before calling
    this function.

    Optimisations applied
    ---------------------
    - Axe 3: RBF bandwidth (sigma) is estimated once per feature/target on
      the full slice via the median heuristic, then reused across all
      bootstrap iterations. This eliminates ~40% of cdist calls compared to
      re-estimating sigma at each bootstrap.

    Parameters
    ----------
    df           : DataFrame with one row per observation (SimID or cycle).
                   Must contain all features and targets as numeric columns.
    features_num : Continuous feature names.
    features_cat : Categorical feature names (ordinally encoded as int/float).
    targets      : Target variable names (must be numeric).
    n_boot       : Number of bootstrap repetitions. 100 is sufficient for
                   ranking; increase to 200 for publication-quality CIs.
    boot_size    : Subsample size per bootstrap. Complexity is O(boot_size²)
                   per feature per bootstrap — keep ≤ 600 for interactive use.
    random_state : Seed for reproducibility.
    verbose      : If True, log progress every 20 bootstraps.

    Returns
    -------
    DataFrame with columns:
        feature, target, hsic_mean, hsic_std, hsic_ci_low, hsic_ci_high
    Sorted by hsic_mean descending (best overall predictor first).
    """
    all_feat  = features_num + features_cat
    rng       = np.random.default_rng(random_state)
    n         = len(df)
    boot_size = min(boot_size, n)

    if n < 30:
        log.warning(
            "compute_hsic_indices: only %d observations — results unreliable.", n
        )

    # --- Pre-extract as numpy arrays to avoid pandas overhead in the loop ---
    X: dict[str, np.ndarray] = {
        f: df[f].values.astype(float) for f in all_feat
    }
    Y: dict[str, np.ndarray] = {
        t: df[t].values.astype(float) for t in targets
    }

    # --- Axe 3: pre-compute sigma for each feature and target once ----------
    # Sigma is estimated on the full slice (not a bootstrap subsample) for
    # stability, then passed explicitly to _rbf_kernel_centered at each
    # bootstrap iteration, skipping the costly median heuristic inside.
    sigmas_X: dict[str, float] = {
        f: _estimate_sigma(X[f]) if np.std(X[f]) > 1e-10 else 1.0
        for f in all_feat
    }
    sigmas_Y: dict[str, float] = {
        t: _estimate_sigma(Y[t]) if np.std(Y[t]) > 1e-10 else 1.0
        for t in targets
    }

    # Storage: feature → target → list of per-bootstrap HSIC values
    store: dict[str, dict[str, list[float]]] = {
        f: {t: [] for t in targets} for f in all_feat
    }

    for b in range(n_boot):
        idx = rng.choice(n, size=boot_size, replace=False)

        # --- Centered kernel for each target (computed once per bootstrap) ---
        Kyc: dict[str, np.ndarray | None] = {}
        for t in targets:
            y = Y[t][idx]
            if np.std(y) > 1e-10:
                Kyc[t] = _rbf_kernel_centered(y, sigma=sigmas_Y[t])
            else:
                Kyc[t] = None

        # --- Centered kernel for each feature ---
        for f in all_feat:
            x = X[f][idx]
            if np.std(x) < 1e-10:
                # Constant feature in this bootstrap → zero dependence
                for t in targets:
                    store[f][t].append(0.0)
                continue
            Kxc = _rbf_kernel_centered(x, sigma=sigmas_X[f])
            for t in targets:
                val = (
                    _hsic_normalized(Kxc, Kyc[t])
                    if Kyc[t] is not None
                    else 0.0
                )
                store[f][t].append(val)

        if verbose and (b + 1) % 20 == 0:
            log.info("  Bootstrap %d/%d ✓", b + 1, n_boot)

    # --- Aggregate bootstrap distributions ---
    rows = []
    for f in all_feat:
        for t in targets:
            v = np.array(store[f][t])
            rows.append(
                {
                    "feature"     : f,
                    "target"      : t,
                    "hsic_mean"   : float(np.mean(v)),
                    "hsic_std"    : float(np.std(v)),
                    "hsic_ci_low" : float(np.percentile(v, 2.5)),
                    "hsic_ci_high": float(np.percentile(v, 97.5)),
                }
            )

    out = pd.DataFrame(rows)
    # Sort features by max HSIC across targets (best overall predictor first)
    order = (
        out.groupby("feature")["hsic_mean"]
        .max()
        .sort_values(ascending=False)
        .index
    )
    out["feature"] = pd.Categorical(
        out["feature"], categories=order, ordered=True
    )
    return out.sort_values(["feature", "target"]).reset_index(drop=True)


# =============================================================================
# INTERNAL — Year worker (must be top-level for ProcessPoolExecutor pickling)
# =============================================================================

def _hsic_one_year(
    yr: int,
    yr_idx: int,
    df_yr: pd.DataFrame,
    features_num: list[str],
    features_cat: list[str],
    targets: list[str],
    simid_col: str,
    crop_mode: str,
    crop_col: str,
    min_obs: int,
    n_boot: int,
    boot_size: int,
    random_state: int,
) -> pd.DataFrame | None:
    """
    Compute HSIC indices for a single year slice.

    This function is the unit of work dispatched to each parallel worker.
    It must be a module-level function (not a closure or lambda) to be
    picklable by ProcessPoolExecutor.

    Aggregation guard
    -----------------
    The primary key of the input DataFrame is (SimID, Harvest_Year_Absolute,
    Crop_Name). If multiple Crop_Name values exist for the same (SimID, year)
    — possible in certain rotation configurations — the slice is aggregated
    to one row per SimID before HSIC computation:
        - Sobol plan features (fixed per SimID) : "first"
        - Targets (varying per cycle)           : "mean"

    Parameters
    ----------
    yr           : Absolute harvest year value.
    yr_idx       : Position of `yr` in the sorted year list (used to offset
                   random_state for reproducibility across parallel workers).
    df_yr        : Pre-filtered DataFrame for this year (already subset from
                   the main DataFrame in the parent process to avoid repeated
                   filtering inside each worker).
    features_num : Continuous feature names.
    features_cat : Categorical feature names (ordinally encoded).
    targets      : Target variable names.
    simid_col    : Column identifying simulation IDs.
    crop_mode    : "all" | "dominant" | "per_crop" (passed through from
                   compute_hsic_by_year).
    crop_col     : Column containing the observed crop species name.
    min_obs      : Minimum number of observations required to proceed.
    n_boot       : Bootstrap repetitions.
    boot_size    : Subsample size per bootstrap.
    random_state : Base seed (yr_idx is added to ensure year-level diversity).

    Returns
    -------
    DataFrame with HSIC results for this year, or None if skipped.
    """
    # --- Aggregation guard: ensure one row per SimID -----------------------
    # Triggered when multiple Crop_Name values exist for the same (SimID, yr).
    # This preserves correctness even if the "currently no" assumption breaks.
    if df_yr.groupby(simid_col).size().max() > 1:
        log.warning(
            "Year %d: multiple Crop_Name per SimID detected — "
            "aggregating to one row per SimID (targets: mean, features: first).",
            yr,
        )
        agg_dict: dict[str, str] = {
            f: "first" for f in features_num + features_cat
        }
        agg_dict.update({t: "mean" for t in targets})
        df_yr = df_yr.groupby(simid_col).agg(agg_dict).reset_index()

    # --- Crop filtering (legacy dominant mode) -----------------------------
    crop_label = "all"
    if crop_mode == "dominant" and crop_col in df_yr.columns:
        dominant_crop = df_yr[crop_col].mode()
        if len(dominant_crop) > 0:
            dominant_crop = dominant_crop.iloc[0]
            df_yr         = df_yr[df_yr[crop_col] == dominant_crop]
            crop_label    = str(dominant_crop)

    # --- Drop rows with NA in any required column --------------------------
    required_cols = features_num + features_cat + targets
    df_slice      = df_yr.dropna(subset=required_cols)
    n_obs         = len(df_slice)

    if n_obs < min_obs:
        log.warning(
            "Year %d: only %d obs after filtering (min_obs=%d) — skipped.",
            yr, n_obs, min_obs,
        )
        return None

    # --- Core HSIC computation ---------------------------------------------
    res = compute_hsic_indices(
        df_slice,
        features_num,
        features_cat,
        targets,
        n_boot       = n_boot,
        boot_size    = min(boot_size, n_obs),
        random_state = random_state + yr_idx,
        verbose      = False,
    )
    res.insert(0, "year",      yr)
    res.insert(1, "n_obs",     n_obs)
    res.insert(2, "crop_name", crop_label)
    return res

def _hsic_one_year_crop(
    yr: int,
    yr_idx: int,
    crop: str,
    df_slice: pd.DataFrame,
    features_num: list[str],
    features_cat: list[str],
    targets: list[str],
    simid_col: str,
    min_obs: int,
    n_boot: int,
    boot_size: int,
    random_state: int,
) -> tuple[str, pd.DataFrame | None]:
    """
    Worker for per_crop mode — one (year, crop) pair.
    Returns (crop, result_or_None).
    """
    # Agrégation défensive
    if df_slice.groupby(simid_col).size().max() > 1:
        agg_dict = {f: "first" for f in features_num + features_cat}
        agg_dict.update({t: "mean" for t in targets})
        df_slice = df_slice.groupby(simid_col).agg(agg_dict).reset_index()

    required_cols = features_num + features_cat + targets
    df_slice = df_slice.dropna(subset=required_cols)
    n_obs = len(df_slice)

    if n_obs < min_obs:
        log.warning(
            "  Year %d | crop=%-8s : only %d obs (min_obs=%d) — skipped.",
            yr, crop, n_obs, min_obs,
        )
        return crop, None

    res = compute_hsic_indices(
        df_slice, features_num, features_cat, targets,
        n_boot       = n_boot,
        boot_size    = min(boot_size, n_obs),
        random_state = random_state + yr_idx,
        verbose      = False,
    )
    res.insert(0, "year",  yr)
    res.insert(1, "n_obs", n_obs)
    return crop, res

# =============================================================================
# PUBLIC — Temporal loop (parallelised across years)
# =============================================================================

def compute_hsic_by_year(
    df: pd.DataFrame,
    features_num: list[str],
    features_cat: list[str],
    targets: list[str],
    year_col: str = "Harvest_Year_Absolute",
    simid_col: str = "SimID",
    nominal_simids: pd.Index | None = None,
    crop_mode: Literal["all", "dominant", "per_crop"] = "all",
    crop_col: str = "Crop_Name",
    min_obs: int = 100,
    n_boot: int = 100,
    boot_size: int = 400,
    random_state: int = 42,
    n_workers: int | None = None,
    verbose: bool = True,
) -> pd.DataFrame | dict[str, pd.DataFrame]:
    """
    Run HSIC sensitivity analysis for each value of Harvest_Year_Absolute.

    This captures how feature importance evolves as the tree grows and the
    agroforestry system transitions through juvenile → competition →
    complementarity phases.

    Parallelisation (Axe 1)
    -----------------------
    Years are independent computation units and are dispatched to a
    ProcessPoolExecutor. Each worker receives a pre-sliced DataFrame for its
    year (slicing is done in the parent process before forking to avoid
    redundant filtering in each worker). Results are collected as they
    complete and re-sorted by year before concatenation.

    The number of workers defaults to min(os.cpu_count(), n_years). Set
    n_workers=1 to disable parallelisation (useful for debugging or when
    running inside an existing parallel context).

    Crop_Name heterogeneity (rotations)
    ------------------------------------
    In rotation SimIDs, Crop_Name varies across cycles within the same SimID.
    The `crop_mode` parameter controls how this is handled:

    - "all"      : Pool all Crop_Name values for the given year.
                   Recommended when main_crop is already included as a
                   categorical Sobol feature — crop identity is then captured
                   directly by the HSIC index of main_crop, with no need to
                   filter by observed Crop_Name.
    - "dominant" : Keep only the most frequent Crop_Name for that year
                   (legacy mode — risks excluding monoculture SimIDs that are
                   not the dominant crop that year).
    - "per_crop" : Run HSIC separately for each Crop_Name present in the year.
                   Returns a dict {crop_name: DataFrame} instead of a single
                   DataFrame. Use when crop-specific sensitivity is needed.

    Parameters
    ----------
    df             : Full meta-table (all cycles, all SimIDs).
    features_num   : Continuous feature names (Sobol plan parameters).
    features_cat   : Categorical feature names (encoded as int).
    targets        : Target variable names.
    year_col       : Column identifying the absolute crop year (1 → 40).
    simid_col      : SimID column name.
    nominal_simids : If provided, restrict to these SimIDs (e.g. AF_ok×TA_ok).
    crop_mode      : How to handle Crop_Name heterogeneity. See above.
                     Defaults to "all" (recommended with main_crop as feature).
    crop_col       : Column containing the observed crop species name.
    min_obs        : Minimum observations required to run HSIC for a year.
                     Years with fewer observations are skipped with a warning.
    n_boot         : Bootstrap repetitions per year.
    boot_size      : Subsample size per bootstrap.
    random_state   : Base seed (incremented by year index for reproducibility
                     across parallel workers).
    n_workers      : Number of parallel workers. Defaults to
                     min(os.cpu_count(), n_years). Set to 1 to disable
                     parallelisation.
    verbose        : Log progress as years complete.

    Returns
    -------
    If crop_mode in ("all", "dominant"):
        DataFrame with columns:
            year, n_obs, crop_name, feature, target,
            hsic_mean, hsic_std, hsic_ci_low, hsic_ci_high
    If crop_mode == "per_crop":
        dict mapping crop_name → DataFrame (same schema without crop_name col).
    """
    df_work = df.copy()

    # --- Restrict to nominal population if provided ------------------------
    if nominal_simids is not None:
        df_work = df_work[df_work[simid_col].isin(nominal_simids)].copy()
        log.info(
            "compute_hsic_by_year: restricted to %d nominal SimIDs → %d rows",
            len(nominal_simids), len(df_work),
        )

    years = sorted(df_work[year_col].dropna().unique().astype(int))
    log.info(
        "compute_hsic_by_year: %d years × %d features × %d targets | "
        "crop_mode='%s' | n_boot=%d | boot_size=%d",
        len(years),
        len(features_num) + len(features_cat),
        len(targets),
        crop_mode,
        n_boot,
        boot_size,
    )

    # ── crop_mode == "per_crop" → separate dict per crop ──────────────────
    # Note: per_crop mode is not parallelised across years here to avoid
    # excessive worker spawning (n_years × n_crops processes). If needed,
    # wrap the outer loop with ProcessPoolExecutor at the crop level.
    if crop_mode == "per_crop":
        all_crops = sorted(df_work[crop_col].dropna().unique())
        results_per_crop: dict[str, list[pd.DataFrame]] = {c: [] for c in all_crops}

        # Pré-découpage dans le process parent (year, crop)
        slices: dict[tuple[int, str], pd.DataFrame] = {}
        for yr, crop in itertools.product(years, all_crops):
            df_yr = df_work[df_work[year_col] == yr]
            slices[(yr, crop)] = df_yr[df_yr[crop_col] == crop].copy()

        n_workers = min(os.cpu_count() or 1, len(slices))
        log.info(
            "per_crop: parallelising %d (year × crop) tasks on %d workers",
            len(slices), n_workers,
        )
        t0 = time.time()

        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(
                    _hsic_one_year_crop,
                    yr, yr_idx, crop,
                    slices[(yr, crop)],
                    features_num, features_cat, targets,
                    simid_col, min_obs, n_boot, boot_size, random_state,
                ): (yr, crop)
                for yr_idx, yr in enumerate(years)
                for crop in all_crops
            }

            for future in as_completed(futures):
                yr, crop = futures[future]
                crop_key, res = future.result()
                if res is not None:
                    results_per_crop[crop_key].append(res)
                if verbose:
                    log.info(
                        "  Year %d | crop=%-8s done | %.1fs elapsed",
                        yr, crop, time.time() - t0,
                    )

        return {
            crop: pd.concat(frames, ignore_index=True).sort_values("year")
            for crop, frames in results_per_crop.items()
            if frames
        }

    # ── crop_mode in ("all", "dominant") — parallelised across years ──────

    # Pre-slice the DataFrame by year in the parent process.
    # This avoids each worker filtering the full DataFrame independently,
    # and ensures each worker receives only the data it needs.
    df_by_year: dict[int, pd.DataFrame] = {
        yr: df_work[df_work[year_col] == yr].copy() for yr in years
    }

    effective_workers = min(
        n_workers if n_workers is not None else (os.cpu_count() or 1),
        len(years),
    )
    log.info(
        "Dispatching %d year jobs across %d workers...",
        len(years), effective_workers,
    )

    all_results: list[pd.DataFrame] = []
    t0 = time.time()

    if effective_workers == 1:
        # Sequential fallback — useful for debugging or profiling
        for yr_idx, yr in enumerate(years):
            res = _hsic_one_year(
                yr, yr_idx, df_by_year[yr],
                features_num, features_cat, targets,
                simid_col, crop_mode, crop_col,
                min_obs, n_boot, boot_size, random_state,
            )
            if res is not None:
                all_results.append(res)
            if verbose:
                elapsed = time.time() - t0
                eta = elapsed / (yr_idx + 1) * (len(years) - yr_idx - 1)
                log.info(
                    "  Year %2d/%d | n=%4d | %.1fs elapsed | ETA ~%.0fs",
                    yr, years[-1], len(df_by_year[yr]), elapsed, eta,
                )
    else:
        # Parallel execution — one future per year
        with ProcessPoolExecutor(max_workers=effective_workers) as pool:
            futures = {
                pool.submit(
                    _hsic_one_year,
                    yr, yr_idx, df_by_year[yr],
                    features_num, features_cat, targets,
                    simid_col, crop_mode, crop_col,
                    min_obs, n_boot, boot_size, random_state,
                ): yr
                for yr_idx, yr in enumerate(years)
            }

            completed = 0
            for future in as_completed(futures):
                yr  = futures[future]
                completed += 1
                try:
                    res = future.result()
                except Exception as exc:
                    log.error("Year %d raised an exception: %s", yr, exc)
                    res = None
                if res is not None:
                    all_results.append(res)
                if verbose:
                    elapsed = time.time() - t0
                    log.info(
                        "  Year %2d done (%d/%d) | %.1fs elapsed",
                        yr, completed, len(years), elapsed,
                    )

    if not all_results:
        log.error(
            "compute_hsic_by_year: no results produced — "
            "check min_obs and data coverage."
        )
        return pd.DataFrame()

    # Re-sort by year (as_completed does not guarantee order)
    all_results.sort(key=lambda d: int(d["year"].iloc[0]))
    return pd.concat(all_results, ignore_index=True)


# =============================================================================
# PUBLIC — Validation
# =============================================================================

def validate_hsic_vs_spearman(
    df: pd.DataFrame,
    features_num: list[str],
    targets: list[str],
    sample_size: int = 400,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Sanity check: compare HSIC indices vs Spearman² on a random subsample.

    HSIC should be of the same order of magnitude as Spearman² and can be
    larger when the relationship is non-linear. Rank concordance (Spearman
    of ranks) across features is also reported to the log.

    Parameters
    ----------
    df           : DataFrame containing features and targets.
    features_num : Numeric features to validate (categorical features are
                   excluded as Spearman is not meaningful for them).
    targets      : Target variables.
    sample_size  : Number of rows to subsample.
    random_state : Seed.

    Returns
    -------
    DataFrame with columns:
        feature, target, spearman_r, spearman_r2, hsic, ratio_hsic_over_r2
    """
    rng = np.random.default_rng(random_state)
    n   = len(df)
    idx = rng.choice(n, size=min(sample_size, n), replace=False)
    df_val = df.iloc[idx].reset_index(drop=True)

    rows = []
    for feat in features_num:
        for tgt in targets:
            x = df_val[feat].values.astype(float)
            y = df_val[tgt].values.astype(float)
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 10:
                continue
            rho, _ = spearmanr(x[mask], y[mask])
            Kxc    = _rbf_kernel_centered(x[mask])
            Kyc    = _rbf_kernel_centered(y[mask])
            h      = _hsic_normalized(Kxc, Kyc)
            r2     = rho**2
            rows.append(
                {
                    "feature"           : feat,
                    "target"            : tgt,
                    "spearman_r"        : round(rho, 4),
                    "spearman_r2"       : round(r2,  4),
                    "hsic"              : round(h,   4),
                    "ratio_hsic_over_r2": round(h / max(r2, 1e-6), 2),
                }
            )

    val_df = pd.DataFrame(rows)

    # --- Rank concordance summary (logged, not returned) -------------------
    log.info("=== Rank concordance HSIC vs Spearman² ===")
    for tgt in targets:
        sub = val_df[val_df["target"] == tgt].set_index("feature")
        if len(sub) < 3:
            continue
        rho_ranks, _ = spearmanr(
            sub["hsic"].rank(ascending=False),
            sub["spearman_r2"].rank(ascending=False),
        )
        log.info("  %-22s : rank concordance = %.3f", tgt, rho_ranks)

    return val_df


# =============================================================================
# PUBLIC — Visualisation
# =============================================================================

def plot_hsic_heatmap(
    results: pd.DataFrame,
    save_path: "Path",
    targets: list[str],
    cmap: str = "YlOrRd",
    title: str | None = None,
    figsize: tuple[int, int] = (18, 10),
    annotate: bool = False,
    show: bool | None = None,
) -> "Path":
    """
    Grid of HSIC heatmaps (one subplot per target) on a single figure.

    Parameters
    ----------
    results   : Output of compute_hsic_by_year (single DataFrame).
    save_path : Destination file path.
    targets   : List of target variables to plot (one subplot each).
    cmap      : Colormap name.
    title     : Figure suptitle. Auto-generated if None.
    figsize   : Overall figure size.
    annotate  : If True, write HSIC values in each cell.
    show      : None = auto, True = always, False = never.

    Returns
    -------
    Path : resolved save path.
    """

    n     = len(targets)
    ncols = 2
    nrows = (n + 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_flat = np.array(axes).flatten()

    for ax, target in zip(axes_flat, targets):
        sub = results[results["target"] == target]
        if sub.empty:
            log.warning("plot_hsic_heatmap: no data for target '%s'", target)
            ax.set_visible(False)
            continue

        pivot = sub.pivot_table(
            index="feature", columns="year", values="hsic_mean", aggfunc="mean"
        )
        pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

        im = ax.imshow(pivot.values, aspect="auto", cmap=cmap,
                       vmin=0, vmax=pivot.values.max())
        plt.colorbar(im, ax=ax, label="HSIC (normalized)")

        ax.set_xticks(range(pivot.shape[1]))
        ax.set_xticklabels(pivot.columns.astype(int), fontsize=7)
        ax.set_yticks(range(pivot.shape[0]))
        ax.set_yticklabels(pivot.index, fontsize=8)
        ax.set_xlabel("Harvest Year (absolute)")
        ax.set_ylabel("Feature")
        ax.set_title(target, fontweight="bold")

        if annotate:
            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    val = pivot.values[i, j]
                    if not np.isnan(val):
                        ax.text(
                            j, i, f"{val:.2f}",
                            ha="center", va="center", fontsize=5,
                            color="black" if val < 0.5 * pivot.values.max()
                                  else "white",
                        )

    # Hide unused subplots
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle(title or "HSIC sensitivity — all targets",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    return save_figure(fig, save_path, show=show)


def plot_hsic_lines(
    results: pd.DataFrame,
    save_path: "Path",
    targets: list[str],
    top_n: int = 8,
    title: str | None = None,
    figsize: tuple[int, int] = (18, 10),
    add_ci: bool = True,
    show: bool | None = None,
) -> "Path":
    """
    Grid of HSIC line plots (one subplot per target) on a single figure.

    Parameters
    ----------
    results   : Output of compute_hsic_by_year (single DataFrame).
    save_path : Destination file path.
    targets   : List of target variables to plot (one subplot each).
    top_n     : Number of top features per subplot (ranked by mean HSIC).
    title     : Figure suptitle. Auto-generated if None.
    figsize   : Overall figure size.
    add_ci    : If True, shade the bootstrap 95 % CI band.
    show      : None = auto, True = always, False = never.

    Returns
    -------
    Path : resolved save path.
    """

    n     = len(targets)
    ncols = 2
    nrows = (n + 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_flat = np.array(axes).flatten()

    for ax, target in zip(axes_flat, targets):
        sub = results[results["target"] == target]
        if sub.empty:
            log.warning("plot_hsic_lines: no data for target '%s'", target)
            ax.set_visible(False)
            continue

        top_features = (
            sub.groupby("feature")["hsic_mean"]
            .mean()
            .sort_values(ascending=False)
            .head(top_n)
            .index.tolist()
        )

        cmap_tab = plt.get_cmap("tab10")
        colors   = {f: cmap_tab(i % 10) for i, f in enumerate(top_features)}

        for feat in top_features:
            df_feat = sub[sub["feature"] == feat].sort_values("year")
            color   = colors[feat]
            ax.plot(
                df_feat["year"].values,
                df_feat["hsic_mean"].values,
                label=feat, color=color, linewidth=2, marker="o", markersize=3,
            )
            if add_ci:
                ax.fill_between(
                    df_feat["year"].values,
                    df_feat["hsic_ci_low"].values,
                    df_feat["hsic_ci_high"].values,
                    alpha=0.12, color=color,
                )

        ax.set_xlabel("Harvest Year (absolute)")
        ax.set_ylabel("HSIC (normalized)")
        ax.set_title(target, fontweight="bold")
        ax.legend(loc="upper left", fontsize=7, ncol=2)
        ax.set_xlim(left=1)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle(title or "HSIC sensitivity over time — all targets",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    return save_figure(fig, save_path, show=show)
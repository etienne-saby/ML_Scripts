"""
MetAIsAFe — Diagnostics v4.0
==============================
Step 2: Identification and analysis of problematic simulations.

Three independent criteria are applied per SimID:
    1. Carbon dead     — final carbonStem_AF below threshold
    2. Yield failures  — too many cycles with near-zero yield_TA
    3. High NA rate    — excessive missing values in key stock columns

Also runs Kolmogorov-Smirnov tests to detect whether flagged simulations
are systematically biased across Sobol design parameters.

WORKFLOW POSITION
-----------------
    [Step 1] preparation.py   → add_derived_columns
                              → filter_crops
                              → clean
        ↓
    [Step 2] diagnostics.py   → analyze_problematic_simulations (Optional)
                              → apply_exclusions (Optional)
        ↓
    [Step 3] preparation.py   → compute_effective_vars
                              → compute_carbon_deltas
             preprocessing.py → interpolate_dynamic_vars  (optional)
                              → compute_ratios_from_stocks  (optional)



Author  : Étienne SABY
Updated : 2026-05 (v4.0 — extracted from prepare_data.py)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from column_taxonomy import NA_KEY_COLUMNS, SOBOL_FEATURES

log = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    _PLOTTING_AVAILABLE = True
except ImportError:
    _PLOTTING_AVAILABLE = False
    log.warning("matplotlib/seaborn not available — diagnostic plots disabled.")

try:
    from scipy.stats import ks_2samp
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — DIAGNOSTIC ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════

def analyze_problematic_simulations(
    df: pd.DataFrame,
    carbon_stem_final_min: float = 1.0,
    yield_ta_min: float = 0.05,
    yield_failure_pct_max: float = 0.30,
    na_pct_max: float = 0.20,
    na_key_columns: list[str] | None = None,
    export_plots: bool = True,
    exclusion_dir: Path | None = None,
    show_plots: bool | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Identify and analyse problematic simulations across three criteria.

    Criteria
    --------
    1. **Carbon dead** — final ``carbonStem_AF`` < ``carbon_stem_final_min``.
       Indicates a tree mortality event or simulation crash.
    2. **Yield failures** — fraction of cycles with ``yield_TA`` <
       ``yield_ta_min`` exceeds ``yield_failure_pct_max``.
       Indicates systematic crop failure (calibration issue).
    3. **High NA rate** — NA% in ``na_key_columns`` exceeds ``na_pct_max``.
       Indicates incomplete simulation output.

    Parameters
    ----------
    df : pd.DataFrame
        Output of Step 1 (preparation).
    carbon_stem_final_min : float, default 1.0
        Minimum final ``carbonStem_AF`` [kgC/tree] to consider tree alive.
    yield_ta_min : float, default 0.05
        Minimum ``yield_TA`` [t/ha] to avoid flagging a cycle as failure.
    yield_failure_pct_max : float, default 0.30
        Maximum fraction of failed cycles before flagging a SimID.
    na_pct_max : float, default 0.20
        Maximum NA% in key columns before flagging a SimID.
    na_key_columns : list of str, optional
        Defaults to ``NA_KEY_COLUMNS`` from ``column_taxonomy``.
    export_plots : bool, default True
    exclusion_dir : Path, optional
        Directory to save diagnostic plots.
    show_plots : bool or None, default None (auto)
    verbose : bool, default True

    Returns
    -------
    dict with keys:
        ``exclusion_summary``  : pd.DataFrame — per-criterion counts
        ``flagged_sims``       : pd.DataFrame — SimIDs flagged and reasons
        ``sobol_analysis``     : dict — KS test results
        ``plots``              : dict[str, Path]
        ``thresholds``         : dict — parameter values used
    """
    if na_key_columns is None:
        na_key_columns = NA_KEY_COLUMNS

    required = ["SimID", "Harvest_Year_Absolute"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    log.info("Step 2 — Diagnostic analysis of problematic simulations")

    flagged: dict[str, set] = {
        "carbon_dead_AF": set(),
        "carbon_dead_TF": set(),
        "yield_failure": set(),
        "high_na": set(),
    }

    # ── 2.1 Carbon dead ───────────────────────────────────────────────────
    log.info("  Step 2.1 — Carbon-dead flagging (AF & TF)")
    last_year = (
        df.sort_values("Harvest_Year_Absolute")
        .groupby("SimID", as_index=False)
        .tail(1)
    )
    
    if "carbonStem_AF" in df.columns:
        dead_af = last_year.loc[
            last_year["carbonStem_AF"] < carbon_stem_final_min, "SimID"
        ]
        flagged["carbon_dead_AF"] = set(dead_af)
        log.info(
            "   ✓ %d carbon-dead AF simulations (final carbonStem_AF < %.2f)",
            len(flagged["carbon_dead_AF"]), carbon_stem_final_min,
        )
    else:
        log.warning("   ⚠ carbonStem_AF not found — AF flagging skipped")

    if "carbonStem_TF" in df.columns:
        dead_tf = last_year.loc[
            last_year["carbonStem_TF"] < carbon_stem_final_min, "SimID"
        ]
        flagged["carbon_dead_TF"] = set(dead_tf)
        log.info(
            "   ✓ %d carbon-dead TF simulations (final carbonStem_TF < %.2f)",
            len(flagged["carbon_dead_TF"]), carbon_stem_final_min,
        )
        
        # Overlap analysis
        overlap = flagged["carbon_dead_AF"] & flagged["carbon_dead_TF"]
        log.info("   ℹ %d simulations failed both in AF and TF", len(overlap))
    else:
        log.warning("   ⚠ carbonStem_TF not found — TF flagging skipped")

    # ── 2.2 Yield failures ─────────────────────────────────────────────────
    log.info("  Step 2.2 — Yield-failure flagging")
    if "yield_TA" in df.columns:
        df["_yield_fail_cycle"] = df["yield_TA"] < yield_ta_min
        fail_rate = df.groupby("SimID")["_yield_fail_cycle"].mean()
        flagged["yield_failure"] = set(fail_rate[fail_rate > yield_failure_pct_max].index)
        df.drop(columns=["_yield_fail_cycle"], inplace=True)
        log.info(
            "   ✓ %d yield-failure simulations (>%.0f%% cycles below %.2f t/ha)",
            len(flagged["yield_failure"]), yield_failure_pct_max * 100, yield_ta_min,
        )
    else:
        log.warning("   ⚠ yield_TA not found — yield-failure flagging skipped")

    # ── 2.3 High NA rate ───────────────────────────────────────────────────
    log.info("  Step 2.3 — High-NA flagging")
    avail_na_cols = [c for c in na_key_columns if c in df.columns]
    if avail_na_cols:
        na_per_sim = df.groupby("SimID")[avail_na_cols].apply(
            lambda g: g.isna().mean().mean()
        )
        flagged["high_na"] = set(na_per_sim[na_per_sim > na_pct_max].index)
        log.info(
            "   ✓ %d high-NA simulations (mean NA rate > %.0f%% in key columns)",
            len(flagged["high_na"]), na_pct_max * 100,
        )
    else:
        log.warning("   ⚠ No key NA columns found — high-NA flagging skipped")

    # ── Build flagged summary ──────────────────────────────────────────────
    all_flagged = (
        flagged["carbon_dead_AF"] |
        flagged["carbon_dead_TF"] |
        flagged["yield_failure"] |
        flagged["high_na"]
    )

    flagged_df = pd.DataFrame(
        {
            "SimID":          list(all_flagged),
            "carbon_dead_AF": [s in flagged["carbon_dead_AF"] for s in all_flagged],
            "carbon_dead_TF": [s in flagged["carbon_dead_TF"] for s in all_flagged],
            "yield_failure":  [s in flagged["yield_failure"]  for s in all_flagged],
            "high_na":        [s in flagged["high_na"]        for s in all_flagged],
        }
    ).sort_values("SimID").reset_index(drop=True)

    exclusion_summary = pd.DataFrame(
        [
            {"criterion": "carbon_dead_AF",    "n_flagged": len(flagged["carbon_dead_AF"])},
            {"criterion": "carbon_dead_TF",    "n_flagged": len(flagged["carbon_dead_TF"])},
            {"criterion": "carbon_dead_BOTH",  "n_flagged": len(flagged["carbon_dead_AF"] & flagged["carbon_dead_TF"])},
            {"criterion": "yield_failure", "n_flagged": len(flagged["yield_failure"])},
            {"criterion": "high_na",       "n_flagged": len(flagged["high_na"])},
            {"criterion": "TOTAL (union)", "n_flagged": len(all_flagged)},
        ]
    )

    if verbose:
        log.info("\n%s\n", exclusion_summary.to_string(index=False))

    # ── 2.4 Sobol KS tests ────────────────────────────────────────────────
    sobol_analysis: dict[str, Any] = {"ks_tests": pd.DataFrame()}
    if _SCIPY_AVAILABLE and len(all_flagged) > 0:
        sobol_numeric = [
            c for c in SOBOL_FEATURES
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
        ]
        ks_rows = []
        flagged_mask  = df["SimID"].isin(all_flagged)
        for col in sobol_numeric:
            a = df.loc[ flagged_mask, col].dropna().values
            b = df.loc[~flagged_mask, col].dropna().values
            if len(a) > 10 and len(b) > 10:
                stat, pval = ks_2samp(a, b)
                ks_rows.append({"parameter": col, "ks_stat": stat, "p_value": pval})
        sobol_analysis["ks_tests"] = (
            pd.DataFrame(ks_rows)
            .sort_values("ks_stat", ascending=False)
            .reset_index(drop=True)
        )

    # ── 2.5 Diagnostic plots ──────────────────────────────────────────────
    plots: dict[str, Path] = {}
    if export_plots and _PLOTTING_AVAILABLE and exclusion_dir is not None:
        plots = _plot_diagnostics(
            df, flagged, exclusion_dir, show_plots
        )

    return {
        "exclusion_summary": exclusion_summary,
        "flagged_sims":      flagged_df,
        "sobol_analysis":    sobol_analysis,
        "plots":             plots,
        "thresholds": {
            "carbon_stem_final_min": carbon_stem_final_min,
            "yield_ta_min":          yield_ta_min,
            "yield_failure_pct_max": yield_failure_pct_max,
            "na_pct_max":            na_pct_max,
            "na_key_columns":        avail_na_cols,
        },
    }


def apply_exclusions(
    df: pd.DataFrame,
    flagged_sims: pd.DataFrame,
    simid_col: str = "SimID",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Remove flagged simulations from the meta-table.

    Parameters
    ----------
    df : pd.DataFrame
        Full meta-table.
    flagged_sims : pd.DataFrame
        Output of ``analyze_problematic_simulations()`` — must contain
        a ``SimID`` column.
    simid_col : str, default ``"SimID"``
    verbose : bool, default True

    Returns
    -------
    pd.DataFrame
        Meta-table with flagged SimIDs removed.
    """
    if len(flagged_sims) == 0:
        log.info("   ℹ No simulations to exclude.")
        return df

    sims_to_exclude = set(flagged_sims[simid_col])
    n_before = df[simid_col].nunique()
    df = df[~df[simid_col].isin(sims_to_exclude)].copy()
    n_after = df[simid_col].nunique()

    if verbose:
        log.info(
            "   ✓ Exclusions applied: %d → %d simulations (%d removed)",
            n_before, n_after, n_before - n_after,
        )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Internal: diagnostic plots
# ─────────────────────────────────────────────────────────────────────────────

def _plot_diagnostics(
    df: pd.DataFrame,
    flagged: dict[str, set],
    exclusion_dir: Path,
    show_plots: bool | None,
) -> dict[str, Path]:
    """Generate and save diagnostic plots for problematic simulations."""
    from utils.plot_utils import save_figure, is_interactive

    plots: dict[str, Path] = {}
    exclusion_dir = Path(exclusion_dir)
    exclusion_dir.mkdir(parents=True, exist_ok=True)

    show = is_interactive() if show_plots is None else show_plots

    all_flagged = (
        flagged["carbon_dead_AF"]
        | flagged["carbon_dead_TF"]
        | flagged["yield_failure"]
        | flagged["high_na"]
    )

    # ── Plot 1 : Carbon stem distributions (AF & TF) ──────────────────────
    carbon_cols = [c for c in ("carbonStem_AF", "carbonStem_TF") if c in df.columns]
    dead_union = flagged["carbon_dead_AF"] | flagged["carbon_dead_TF"]

    if carbon_cols and len(dead_union) > 0:
        try:
            fig, axes = plt.subplots(1, len(carbon_cols), figsize=(7 * len(carbon_cols), 4), squeeze=False)
            criterion_map = {
                "carbonStem_AF": flagged["carbon_dead_AF"],
                "carbonStem_TF": flagged["carbon_dead_TF"],
            }
            for ax, col in zip(axes[0], carbon_cols):
                dead_sims = criterion_map[col]
                mask_dead = df["SimID"].isin(dead_sims)
                df.loc[~mask_dead, col].hist(
                    bins=50, ax=ax, alpha=0.7, label="Retained", color="#52B788"
                )
                df.loc[mask_dead, col].hist(
                    bins=50, ax=ax, alpha=0.7, label="Carbon dead", color="#E63946"
                )
                ax.set_xlabel(f"{col} (kgC/tree)")
                ax.set_ylabel("Count")
                ax.set_title(f"{col} — Flagged vs Retained")
                ax.legend()
            plt.tight_layout()
            p = save_figure(fig, exclusion_dir / "diag_carbon_dead.png", show=show)
            plots["carbon_dead_dist"] = p
        except Exception as e:
            log.warning("   ⚠ Carbon dead plot failed: %s", e)

    # ── Plot 2 : yield_TA distribution — flagged vs retained ─────────────
    if "yield_TA" in df.columns and len(all_flagged) > 0:
        try:
            fig, ax = plt.subplots(figsize=(8, 4))
            mask_flag = df["SimID"].isin(all_flagged)
            df.loc[~mask_flag, "yield_TA"].hist(
                bins=50, ax=ax, alpha=0.7, label="Retained", color="#52B788"
            )
            df.loc[mask_flag, "yield_TA"].hist(
                bins=50, ax=ax, alpha=0.7, label="Flagged", color="#E63946"
            )
            ax.set_xlabel("yield_TA (t/ha)")
            ax.set_ylabel("Count")
            ax.set_title("yield_TA Distribution — Flagged vs Retained")
            ax.legend()
            plt.tight_layout()
            p = save_figure(fig, exclusion_dir / "diag_yield_ta.png", show=show)
            plots["yield_ta_dist"] = p
        except Exception as e:
            log.warning("   ⚠ yield_TA plot failed: %s", e)

    # ── Plot 3 : NA rate per SimID — flagged vs retained ─────────────────
    na_key_columns = [c for c in df.columns if df[c].isna().any()]  # fallback
    # Prefer to re-derive from what's available rather than depend on a global
    try:
        from column_taxonomy import NA_KEY_COLUMNS
        avail_na_cols = [c for c in NA_KEY_COLUMNS if c in df.columns]
    except ImportError:
        avail_na_cols = na_key_columns

    if avail_na_cols and len(all_flagged) > 0:
        try:
            na_per_sim = (
                df.groupby("SimID")[avail_na_cols]
                .apply(lambda g: g.isna().mean().mean())
                .rename("na_rate")
                .reset_index()
            )
            na_per_sim["flagged"] = na_per_sim["SimID"].isin(all_flagged)

            fig, ax = plt.subplots(figsize=(8, 4))
            for flag_val, label, color in [
                (False, "Retained", "#52B788"),
                (True,  "Flagged",  "#E63946"),
            ]:
                subset = na_per_sim.loc[na_per_sim["flagged"] == flag_val, "na_rate"]
                subset.hist(bins=40, ax=ax, alpha=0.7, label=label, color=color)
            ax.set_xlabel("Mean NA rate across key columns")
            ax.set_ylabel("Number of SimIDs")
            ax.set_title("NA Rate Distribution per SimID — Flagged vs Retained")
            ax.legend()
            plt.tight_layout()
            p = save_figure(fig, exclusion_dir / "diag_na_rate.png", show=show)
            plots["na_rate_dist"] = p
        except Exception as e:
            log.warning("   ⚠ NA rate plot failed: %s", e)

    return plots
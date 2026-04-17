"""
run_hsic_analysis.py
============================
Driver script for the HSIC sensitivity analysis pipeline.

Workflow
--------
Step 1 : Load & prepare data
Step 2 : Build nominal population (AF_ok x TA_ok quadrant)
Step 3 : Global HSIC (SimID level) — sanity check + final indices
Step 4 : Temporal HSIC (year-by-year) — feature importance dynamics
Step 5 : Visualisation

Usage
-----
Run interactively cell-by-cell in a Jupyter notebook, or as a script:
    python run_hsic_analysis.py

Author  : Étienne SABY
Updated : 2026-05
"""

from __future__ import annotations

import logging
import time
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from column_taxonomy import (
    CATEGORICAL_FEATURES,
    SOBOL_FEATURES,
    STOCK_TARGETS_MINIMAL,
)
from config import get_campaign_paths, cleanup_empty_campaign_dirs
from data.loader import encode_categoricals, load_data
from data.preparation import add_derived_columns, filter_crops
from analysis.hsic_sensitivity import (
    compute_hsic_by_year,
    compute_hsic_indices,
    plot_hsic_heatmap,
    plot_hsic_lines,
    validate_hsic_vs_spearman,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("sensitivity.run")

# =============================================================================
# CLI
# =============================================================================

def _parse_args() -> argparse.Namespace:
    """
    Command-line interface for the HSIC sensitivity analysis pipeline.

    Examples
    --------
    python run_sensitivity_analysis.py
    python run_sensitivity_analysis.py --crop_mode per_crop
    python run_sensitivity_analysis.py --crop_mode all --n_boot 200 --boot_size 600
    python run_sensitivity_analysis.py --campaign sobol_S11111_n4096 --no_warm_start
    """
    parser = argparse.ArgumentParser(
        description="HSIC sensitivity analysis pipeline — MetAIsAFe",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--campaign",
        type=str,
        default="sobol_S11111_n2048",
        help="Campaign name (must match a folder in models_dir).",
    )
    parser.add_argument(
        "--crop_mode",
        type=str,
        choices=["all", "dominant", "per_crop"],
        default="all",
        help="How to handle Crop_Name heterogeneity across years.",
    )
    parser.add_argument(
        "--n_boot",
        type=int,
        default=100,
        help="Number of bootstrap repetitions per year.",
    )
    parser.add_argument(
        "--boot_size",
        type=int,
        default=500,
        help="Subsample size per bootstrap.",
    )
    parser.add_argument(
        "--min_obs",
        type=int,
        default=80,
        help="Minimum observations per year to run HSIC (else skipped).",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=None,
        help="Number of parallel workers. Defaults to os.cpu_count().",
    )
    parser.add_argument(
        "--no_warm_start",
        action="store_true",
        help="Disable warm start — run precise HSIC directly on all features.",
    )
    parser.add_argument(
        "--top_n_warm",
        type=int,
        default=8,
        help="Number of top features retained after warm start.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Global random seed.",
    )

    return parser.parse_args()


# =============================================================================
# CONFIGURATION (defaults — overridden by CLI args)
# =============================================================================

CAMPAIGN_NAME       = "sobol_S11111_n2048"
YIELD_FAIL_THRESH   = 0.5    # t/ha — yield below this = failure
FAIL_RATE_THRESH    = 0.5    # SimID flagged if > 50% cycles below threshold
TREE_FAIL_THRESH = 1.0    # kgC/tree — carbonStem_AF final below this = tree failure

# HSIC hyperparameters
N_BOOT_GLOBAL  = 100         # bootstraps for global (SimID-level) analysis
N_BOOT_YEAR    = 100         # bootstraps per year in temporal analysis
BOOT_SIZE      = 500         # subsample size per bootstrap
RANDOM_STATE   = 42

WARM_START     = True

# Crop mode for temporal analysis:
#   "dominant" for yield targets (avoids mixing wheat/maize distributions)
#   "all"      for carbon targets (crop species irrelevant for tree growth)
CROP_MODE : str = "all"

MIN_OBS_PER_YEAR : int = 80  # skip year if fewer observations after filtering

if __name__ == "__main__":

    args = _parse_args()

    # --- Surcharge des constantes par les args CLI ---
    CAMPAIGN_NAME  = args.campaign
    CROP_MODE      = args.crop_mode
    N_BOOT_YEAR    = args.n_boot
    BOOT_SIZE      = args.boot_size
    MIN_OBS_PER_YEAR = args.min_obs
    N_WORKERS      = args.n_workers
    WARM_START     = not args.no_warm_start
    TOP_N_WARM     = args.top_n_warm
    RANDOM_STATE   = args.random_state

    campaign = get_campaign_paths(CAMPAIGN_NAME)

    from utils.io_utils import setup_file_logging
    fh = setup_file_logging(campaign.reports_dir, CAMPAIGN_NAME)
    
    log.info("=" * 60)
    log.info("CLI args: %s", vars(args))
    log.info("=" * 60)

    # =============================================================================
    # STEP 1 — Load & prepare data
    # =============================================================================
    log.info("=" * 60)
    log.info("STEP 1 — Load & prepare data")
    log.info("=" * 60)

    df_raw   = load_data(campaign.raw_meta)
    df       = add_derived_columns(df_raw)
    df, _    = filter_crops(df, excluded_crops=["rape"])
    df, enc_report = encode_categoricals(df, features=SOBOL_FEATURES, method="sklearn")

    log.info("Shape after preparation: %d rows × %d cols", *df.shape)

    # Feature lists (after encoding)
    num_features = [f for f in SOBOL_FEATURES if f not in CATEGORICAL_FEATURES]
    cat_features = [f for f in SOBOL_FEATURES if f in CATEGORICAL_FEATURES]

    log.info("Numeric features  : %s", num_features)
    log.info("Categorical features: %s", cat_features)


    # =============================================================================
    # STEP 2 — Build nominal population (AF_ok × TA_ok)
    # =============================================================================
    log.info("=" * 60)
    log.info("STEP 2 — Build nominal population")
    log.info("=" * 60)

    # --- Per-SimID yield & tree failure flags ---
    # carbonStem_AF : value at last cycle (final cumulated stocks)
    last_cycle_idx = df.groupby("SimID")["Cycle_Nb"].idxmax()
    last_cycle_snap = df.loc[last_cycle_idx, ["SimID", "carbonStem_AF"]].set_index("SimID")

    sim_stats = (
        df.groupby("SimID")
        .agg(
            zero_rate_AF=("yield_AF", lambda x: (x < YIELD_FAIL_THRESH).mean()),
            zero_rate_TA=("yield_TA", lambda x: (x < YIELD_FAIL_THRESH).mean()),
        )
        .reset_index()
    )
    sim_stats["yield_AF_failed"] = sim_stats["zero_rate_AF"] > FAIL_RATE_THRESH
    sim_stats["yield_TA_failed"] = sim_stats["zero_rate_TA"] > FAIL_RATE_THRESH
    sim_stats["tree_failed"]     = (
        last_cycle_snap.reindex(sim_stats["SimID"].values)["carbonStem_AF"].values
        < TREE_FAIL_THRESH
    )

    def _population_label(row):
        y = "yield_ok"  if not row["yield_AF_failed"] and not row["yield_TA_failed"] else "yield_fail"
        t = "tree_ok"   if not row["tree_failed"] else "tree_failed"
        return f"{y} × {t}"

    sim_stats["population"] = sim_stats.apply(_population_label, axis=1)

    log.info(
        "Population distribution:\n%s",
        sim_stats["population"].value_counts().to_string()
    )
    log.info(
        "Population distribution (%%):\n%s",
        (sim_stats["population"].value_counts() / len(sim_stats) * 100).round(1).to_string()
    )

    # --- Population nominale : yield_ok × tree_ok ---
    # Composite criterion with 3 conditions:
    #   1. zero_rate_AF <= FAIL_RATE_THRESH  (viable yield AF)
    #   2. zero_rate_TA <= FAIL_RATE_THRESH  (viable yield TA)
    #   3. final carbonStem_AF >= TREE_FAIL_THRESH  (functional tree component)
    # The “yield_ok × tree_failed” SimIDs are explicitly excluded:
    nominal_simids = sim_stats.loc[
        sim_stats["population"] == "yield_ok × tree_ok", "SimID"
    ]
    df_nominal = df[df["SimID"].isin(nominal_simids)].copy()

    log.info(
        "Nominal population (yield_ok × tree_ok): %d SimIDs | %d rows",
        len(nominal_simids), len(df_nominal),
    )


    # =============================================================================
    # STEP 3 — Global HSIC (SimID level)
    # =============================================================================
    log.info("=" * 60)
    log.info("STEP 3 — Global HSIC (SimID level)")
    log.info("=" * 60)

    # Aggregate targets at SimID level
    # yield    : mean over cycles (site-level productivity)
    # carbonStem : last value (cumulative stock at end of simulation)
    targets_agg = (
        df_nominal.groupby("SimID")
        .agg(
            yield_AF      =("yield_AF",      "mean"),
            yield_TA      =("yield_TA",      "mean"),
            carbonStem_AF =("carbonStem_AF", "last"),
            carbonStem_TF =("carbonStem_TF", "last"),
        )
        .reset_index()
    )

    # Plan snapshot at SimID level
    plan_snap = (
        df_nominal[["SimID"] + num_features + cat_features]
        .drop_duplicates("SimID")
    )

    df_global = plan_snap.merge(targets_agg, on="SimID", how="inner").dropna()
    log.info("Global HSIC dataset: %d SimIDs", len(df_global))

    # --- Sanity check: HSIC vs Spearman ---
    log.info("Validation: HSIC vs Spearman² on random subsample...")
    val_df = validate_hsic_vs_spearman(
        df_global, num_features, STOCK_TARGETS_MINIMAL,
        sample_size=400, random_state=RANDOM_STATE,
    )
    log.info("\n=== Validation HSIC vs Spearman² ===")
    log.info(val_df.to_string(index=False))

    # --- Global HSIC indices ---
    log.info(
        "Computing global HSIC: N=%d, boot_size=%d, n_boot=%d...",
        len(df_global), BOOT_SIZE, N_BOOT_GLOBAL,
    )
    t0 = time.time()
    hsic_global = compute_hsic_indices(
        df          = df_global,
        features_num = num_features,
        features_cat = cat_features,
        targets      = STOCK_TARGETS_MINIMAL,
        n_boot       = N_BOOT_GLOBAL,
        boot_size    = BOOT_SIZE,
        random_state = RANDOM_STATE,
        verbose      = True,
    )
    log.info("Global HSIC done in %.1fs", time.time() - t0)

    # Pivot for readability
    pivot_global = hsic_global.pivot(
        index="feature", columns="target", values="hsic_mean"
    )
    pivot_global = pivot_global.loc[
        pivot_global.max(axis=1).sort_values(ascending=False).index
    ]
    log.info("\n=== Global HSIC indices — nominal population ===")
    log.info(pivot_global.round(4))

    # =============================================================================
    # STEP 4 — Temporal HSIC (year-by-year) — warm start strategy
    # =============================================================================

    if WARM_START:
        # ── Phase 1 : Warm start for quick ranking top-N ──
        log.info("STEP 4a — Warm start (quick ranking)...")
        hsic_warm = compute_hsic_by_year(
            df             = df_nominal,
            features_num   = num_features,
            features_cat   = cat_features,
            targets        = STOCK_TARGETS_MINIMAL,
            nominal_simids = nominal_simids,
            crop_mode      = CROP_MODE,
            min_obs        = MIN_OBS_PER_YEAR,
            n_boot         = 30,  # fast
            boot_size      = 300, # fast
            random_state   = RANDOM_STATE,
            verbose        = False,
        )
        hsic_warm_df = (
            pd.concat(hsic_warm.values(), ignore_index=True)
            if isinstance(hsic_warm, dict)
            else hsic_warm
        )

        TOP_N = 10
        top_num = [
            f for f in (
                hsic_warm_df.groupby("feature")["hsic_mean"]
                .mean()
                .sort_values(ascending=False)
                .head(TOP_N)
                .index.tolist()
            )
            if f in num_features
        ]
        top_cat = [f for f in cat_features]
    else:
        top_num = num_features
        top_cat = cat_features
        log.info("Warm start disabled — using all %d features.", len(top_num + top_cat))
    
    log.info("Top features for precise run : %s", top_num + top_cat)

    # ── Phase 2 : Precise run on top features ──────────────────
    log.info("STEP 4b — Precise run on top features...")
    hsic_temporal = compute_hsic_by_year(
        df             = df_nominal,
        features_num   = top_num,
        features_cat   = top_cat,
        targets        = STOCK_TARGETS_MINIMAL,
        nominal_simids = nominal_simids,
        crop_mode      = CROP_MODE,
        min_obs        = MIN_OBS_PER_YEAR,
        n_boot         = N_BOOT_YEAR,   # 100
        boot_size      = BOOT_SIZE,     # 500
        random_state   = RANDOM_STATE,
        verbose        = True,
    )

    # =============================================================================
    # STEP 5 — Visualisation
    # =============================================================================
    log.info("=" * 60)
    log.info("STEP 5 — Visualisation")
    log.info("=" * 60)

    ALL_TARGETS = ["yield_AF", "yield_TA", "carbonStem_AF", "carbonStem_TF"]

    # Normalise en dict pour boucle unique : mode standard → {"all": df}
    if isinstance(hsic_temporal, pd.DataFrame):
        results_by_crop = {"all": hsic_temporal}
    else:
        results_by_crop = hsic_temporal  # dict {crop_name: DataFrame}

    for crop, df_results in results_by_crop.items():
        suffix = f"_{crop}" if crop != "all" else ""
        crop_label = f" — {crop}" if crop != "all" else ""

        plot_hsic_heatmap(
            df_results,
            save_path = campaign.sobol_dir / f"hsic_heatmaps{suffix}",
            targets   = ALL_TARGETS,
            title     = f"HSIC sensitivity — all targets{crop_label}",
        )
        plot_hsic_lines(
            df_results,
            save_path = campaign.sobol_dir / f"hsic_lines{suffix}",
            targets   = ALL_TARGETS,
            title     = f"HSIC over time — all targets{crop_label}",
            top_n     = 8,
        )

    # =============================================================================
    # EXPORT
    # =============================================================================
    # hsic_global.to_parquet("hsic_global_results.parquet", index=False)
    # hsic_temporal.to_parquet("hsic_temporal_results.parquet", index=False)
    # log.info("Results saved to hsic_global_results.parquet and hsic_temporal_results.parquet")

    # =============================================================================
    # REMOVE EMPTY FOLDERS
    # =============================================================================
    removed = cleanup_empty_campaign_dirs(campaign, verbose=True)
    if removed:
        log.info("Cleaned up %d empty directories.", len(removed))
    else:
        log.info("No empty directories to clean up.")

    fh.flush()
    logging.getLogger().removeHandler(fh)
    fh.close()
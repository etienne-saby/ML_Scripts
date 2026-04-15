"""
MetAIsAFe — full_training.py
==============================
Full training pipeline B2 — v5.0 (Step 2 row-by-row).

Runs the complete workflow in a single pass:
    STEP 1  — Data loading & preparation
    STEP 2  — Population filtering
    STEP 3  — Train/test split + CV groups
    STEP 4  — Categorical encoding & winsorization
    STEP 5a — Classifiers: CLF1 (binary tree status) + CLF2 (binary yield fail)
               + stunted fallback model
    STEP 5b — Stage 1: carbonStem_AF_h{h} + carbonStem_TF_h{h}
               (log1p target space, SimID-level aggregation, × 5 horizons)
    STEP 5c — Stage 2: yield_AF (row-by-row, real carbonStem_AF as feature)
                       yield_TA (row-by-row, no tree features)
    STEP 6  — SHAP analysis (h=40 for Stage 1 targets + classifiers;
               full test set for Stage 2 row-by-row targets)
    STEP 7  — Final evaluation + quality summary table
    STEP 8  — End-to-end inference test on an example scenario

Architecture notes
------------------
    Stage 1 (carbonStem):
        Targets are aggregated at SimID level for each discrete horizon
        h ∈ {5, 10, 20, 30, 40}. Right-skewed cumulative biomass distributions
        are log1p-transformed before training; expm1 back-transform is applied
        at inference. Full 40-year trajectories are reconstructed via PCHIP
        interpolation in predictor.py.

    Stage 2 (yield — v5.0 row-by-row):
        yield_AF and yield_TA are predicted at the annual row level
        (SimID × Crop_Name × Harvest_Year_Absolute). No horizon aggregation,
        no PCHIP interpolation. carbonStem_AF from the meta-table (real values)
        is used as a feature for yield_AF; yield_TA is trained without any
        tree-related feature. At inference, the carbonStem_AF trajectory
        predicted by Stage 1 is fed as a feature into the yield_AF model.
        GroupKFold splits remain at SimID level to prevent temporal leakage.

Cascade routing
---------------
    CLF1 (binary, SimID-level) → tree_degraded (0) | tree_ok (1)
        Sub-routing via P(tree_ok) vs STUNTED_PROBA_THRESHOLD:
            P(tree_ok) > threshold → stunted fallback (conditional median)
            P(tree_ok) ≤ threshold → tree_failed (carbon = 0)
    CLF2 (binary, row-level)   → yield_failed (0=ok | 1=failed)

Saved model artifacts
---------------------
    clf1_tree_fail.joblib              CLF1 — binary tree status
    clf2_yield_fail.joblib             CLF2 — binary yield failure
    stunted_model.joblib               Conditional median fallback for stunted trees
    winsorization_bounds.joblib        Winsorization bounds (fit on train only)
    lgbm_carbonStem_AF_h{h}.joblib     Stage 1 — × 5 horizons
    lgbm_carbonStem_TF_h{h}.joblib     Stage 1 — × 5 horizons
    lgbm_yield_AF_rowwise.joblib       Stage 2 — single model (row-by-row)
    lgbm_yield_TA_rowwise.joblib       Stage 2 — single model (row-by-row)

CLI usage
---------
    python full_training.py --campaign sobol_training_1_n2048
    python full_training.py --campaign sobol_training_1_n2048 --no-shap --no-inference
    python full_training.py --help

Author  : Etienne SABY
Created : 2026-04
Version : 5.0 — Stage 2 row-by-row (yield_AF / yield_TA)
"""
from __future__ import annotations

from _version import __version__

import argparse
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for cluster execution
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# ── Suppress noisy third-party warnings ──────────────────────────────────────
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Ensure project root is on sys.path ───────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Project imports ───────────────────────────────────────────────────────────
from config import (
    get_campaign_paths,
    cleanup_empty_campaign_dirs,
    RANDOM_STATE,
    TREE_FAIL_THRESHOLD,
    TREE_STUNT_THRESHOLD,
    MIN_CARBON_HORIZON,
    MIN_ENRICH_HORIZON,       # used by Stage 1 build_horizon_dataset only
    YIELD_FAIL_THRESHOLD,
    YIELD_FAIL_RATE,
    PARAMS_YIELD_AF_ROW,      # Stage 2 row-by-row hyperparameters
    PARAMS_YIELD_TA_ROW,      # Stage 2 row-by-row hyperparameters
    CARBON_HORIZONS,
    LGB_PARAMS,
    CV_N_FOLDS,
    FIGURE_DPI,
)
from column_taxonomy import (
    ACTIVE_FEATURES_B2,
    CATEGORICAL_FEATURES_B2,
    CLIMATE_FEATURES,
    STOCK_TARGETS_MINIMAL,
    NOMINAL_POPULATION,
    STEP2_AF_FEATURES,        # row-by-row feature set for yield_AF
    STEP2_TA_FEATURES,        # row-by-row feature set for yield_TA
    STEP2_TARGETS,            # ["yield_AF", "yield_TA"]
    # CLIMATE_HORIZON_FEATURES is defined in column_taxonomy for Stage 1 use only
    # and is NOT imported here — Stage 2 uses raw annual CLIMATE_FEATURES directly.
)
from data.loader import load_data, encode_categoricals, build_dataset
from data.preparation import (
    add_derived_columns,
    filter_crops,
    clean,
    compute_effective_vars,
    filter_population,
    build_horizon_dataset,
)
from data.preprocessing import apply_winsorization
from data.splitter import (
    stratified_split_by_rotation,
    split_by_simid,
    build_cv_groups,
    make_group_kfold,
    summarise_rotations,
    get_rotation_signature,
)
from modeling.models import build_lgb, build_lgb_classifier, get_feature_importances
from modeling.trainer import train_final_model, train_classifier, cross_validate
from modeling.classifiers import (
    build_tree_fail_classifier,
    build_yield_fail_classifier,
    build_classifier_features,
    build_tree_fail_labels_multiclass,
    build_tree_degraded_labels,
    build_yield_fail_labels,
    evaluate_clf1_binary,
    evaluate_classifier,
    apply_geographic_rule,
    predict_routing,
    save_classifiers,
    CLF1_FEATURES,
    CLF2_FEATURES,
    TREE_BINARY_LABELS,
    TREE_BINARY_OK,
    TREE_BINARY_DEGRADED,
    STUNTED_PROBA_THRESHOLD,
)
from modeling.evaluator import compute_metrics, plot_pred_vs_obs, plot_residuals
from modeling.shap_analysis import (
    run_shap_classifiers,
    plot_shap_by_target,
    export_shap_for_shiny,
    compute_shap_values,
    summarise_shap,
)
from modeling.predictor import (
    build_inference_grid,
    predict_single_sim,
    format_output,
)
from utils.io_utils import (
    save_model,
    save_metrics,
    save_cv_results,
    save_predictions,
    save_feature_importances,
    print_campaign_summary,
    setup_file_logging,
)

log = logging.getLogger("metaisafe.training")


# =============================================================================
# DEFAULT HYPERPARAMETERS
# All dicts are overridable via run_full_training(config_overrides=...).
# =============================================================================

# CLF1 — binary tree-status classifier (tree_degraded vs tree_ok)
PARAMS_CLF1_DEFAULT: dict = {
    "objective":         "binary",
    "metric":            "binary_logloss",
    "n_estimators":      300,
    "learning_rate":     0.05,
    "num_leaves":        31,
    "min_child_samples": 20,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "reg_alpha":         0.3,
    "reg_lambda":        0.5,
    "random_state":      RANDOM_STATE,
    "n_jobs":            -1,
    "verbose":           -1,
    "importance_type":   "gain",
}

# Stage 1 — carbonStem horizon models (log1p target space)
PARAMS_CARBON_SHORT: dict = {   # h ∈ {5, 10}
    "n_estimators":      600,
    "learning_rate":     0.03,
    "num_leaves":        63,
    "min_child_samples": 10,
    "subsample":         0.9,
    "colsample_bytree":  0.9,
    "reg_alpha":         0.1,
    "reg_lambda":        0.3,
    "random_state":      RANDOM_STATE,
    "n_jobs":            -1,
    "verbose":           -1,
}

PARAMS_CARBON_LONG: dict = {    # h ∈ {20, 30, 40}
    "n_estimators":      300,
    "learning_rate":     0.05,
    "num_leaves":        31,
    "min_child_samples": 20,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "reg_alpha":         0.5,
    "reg_lambda":        1.0,
    "random_state":      RANDOM_STATE,
    "n_jobs":            -1,
    "verbose":           -1,
    "importance_type":   "gain",
}

# Example parameter set for STEP 8 inference test
EXAMPLE_PARAMS_DEFAULT: dict = {
    "latitude":                  46.0,
    "plotWidth":                 10.0,
    "plotHeight":                15.0,
    "soilDepth":                  1.2,
    "sand":                      35.0,
    "clay":                      25.0,
    "stone":                     10.0,
    "waterTable":                   0,
    "main_crop":              "wheat",
    "w_type":                 "CONST",
    "GDD_cycle_AF":          1800.0,
    "ETP_cycle_AF":           550.0,
    "precipitation_AF":       650.0,
    "frost_events_cycle_AF":    5.0,
    "globalRadiation_AF":    4200.0,
    "maxTemperature_extreme_AF": 35.0,
    "minTemperature_extreme_AF": -5.0,
}

# Quality thresholds — used in STEP 7 summary table
QUALITY_THRESHOLDS: dict = {
    "clf1_f1_macro":  0.70,
    "clf1_roc_auc":   0.85,
    "clf2_accuracy":  0.80,
    "carbon_r2":      0.60,
    "carbon_rho":     0.85,
    "yield_af_r2":    0.55,
    "yield_ta_r2":    0.55,
}


# =============================================================================
# STEP 1 — Data Loading & Preparation
# =============================================================================

def _step1_load_and_prepare(
    campaign: Any,
    excluded_crops: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load the raw meta-table and apply the full preparation pipeline.

    Operations performed (in order):
        1. load_data()            — raw parquet / CSV ingestion
        2. add_derived_columns()  — creates Harvest_Year_Absolute, Rotation
        3. filter_crops()         — drops excluded crop types (e.g. rape)
        4. clean()                — conservative NA / duplicate handling
        5. compute_effective_vars() — analytical _eff_ variables

    Parameters
    ----------
    campaign : CampaignPaths
    excluded_crops : list of str, optional
        Crop types to drop before training. Default: ["rape"].

    Returns
    -------
    df : pd.DataFrame
        Fully prepared meta-table — all populations, all rows.
    """
    if excluded_crops is None:
        excluded_crops = ["rape"]

    t0 = time.perf_counter()
    log.info("=" * 70)
    log.info("STEP 1 — Data Loading & Preparation")
    log.info("=" * 70)

    # 1a. Raw ingestion
    df_raw = load_data(campaign.raw_meta)
    log.info("Raw shape: %d rows × %d cols", *df_raw.shape)

    required_cols = (
        ["SimID", "Harvest_Year_AF",
         "carbonStem_AF", "carbonStem_TF", "yield_AF", "yield_TA"]
        + [f for f in ACTIVE_FEATURES_B2
           if f not in ("SimID", "Harvest_Year_Absolute")]
    )
    missing = [c for c in required_cols if c not in df_raw.columns]
    if missing:
        raise ValueError(f"STEP 1: missing columns in meta-table: {missing}")

    # 1b. Derived columns (Harvest_Year_Absolute, Rotation, …)
    df = add_derived_columns(df_raw, verbose=True)
    assert "Harvest_Year_Absolute" in df.columns, \
        "add_derived_columns did not create Harvest_Year_Absolute."
    assert df["Harvest_Year_Absolute"].isna().sum() == 0, \
        "Harvest_Year_Absolute contains NaN values."
    assert df["Harvest_Year_Absolute"].min() >= 1, \
        "Harvest_Year_Absolute minimum value < 1."
    log.info(
        "Harvest_Year_Absolute — min=%d  max=%d  unique=%d",
        int(df["Harvest_Year_Absolute"].min()),
        int(df["Harvest_Year_Absolute"].max()),
        df["Harvest_Year_Absolute"].nunique(),
    )

    # 1c. Crop filtering
    df, crop_report = filter_crops(df, excluded_crops=excluded_crops, verbose=True)
    if crop_report:
        log.info(
            "Crop filter — rows: %d → %d (−%.1f%%) | SimIDs dropped: %d",
            crop_report["n_rows_before"], crop_report["n_rows_after"],
            crop_report["pct_rows_lost"], crop_report["n_sims_lost"],
        )

    # 1d. Conservative cleaning
    df = clean(df, verbose=True)
    key_cols = [c for c in STOCK_TARGETS_MINIMAL + ACTIVE_FEATURES_B2
                if c in df.columns]
    na_check = df[key_cols].isna().mean() * 100
    na_high  = na_check[na_check > 5].sort_values(ascending=False)
    if len(na_high):
        log.warning("Columns with >5%% NA after cleaning:\n%s", na_high.to_string())
    else:
        log.info("No key column with >5%% NA after cleaning.")

    # 1e. Effective (analytical) variables
    df = compute_effective_vars(df, verbose=True)
    eff_cols = [c for c in df.columns if "_eff_" in c]
    assert eff_cols, "compute_effective_vars produced no _eff_ columns."
    for col in eff_cols:
        assert (df[col] < 0).sum() == 0, \
            f"Negative values detected in effective variable '{col}'."

    log.info(
        "STEP 1 complete — %.1fs | %d rows × %d cols",
        time.perf_counter() - t0, *df.shape,
    )
    return df


# =============================================================================
# STEP 2 — Population Filtering
# =============================================================================

def _step2_filter_populations(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Classify all SimIDs into populations and extract the nominal subset.

    Population taxonomy (defined by carbonStem_AF(t=40) and yield failure rate):
        yield_ok × tree_ok      → NOMINAL population (Stage 1 & Stage 2 training)
        yield_ok × tree_stunted → stunted fallback calibration
        yield_ok × tree_failed  → CLF1 training signal
        yield_fail × *          → CLF2 training signal

    Parameters
    ----------
    df : pd.DataFrame
        Fully prepared meta-table (all populations, all rows).

    Returns
    -------
    df_all_pops : pd.DataFrame
        Full DataFrame with a 'population' column attached (SimID-level label).
    df_nominal : pd.DataFrame
        Rows belonging to the nominal population only (yield_ok × tree_ok).
    """
    t0 = time.perf_counter()
    log.info("=" * 70)
    log.info("STEP 2 — Population Filtering")
    log.info("=" * 70)

    # Full population distribution across all SimIDs
    df_all_pops, population_series = filter_population(
        df,
        population=None,
        tree_fail_threshold=TREE_FAIL_THRESHOLD,
        yield_fail_threshold=YIELD_FAIL_THRESHOLD,
        yield_fail_rate=YIELD_FAIL_RATE,
        verbose=True,
    )
    pop_counts = population_series.value_counts()
    pop_pct    = (pop_counts / len(population_series) * 100).round(1)
    log.info(
        "Population distribution (SimID level):\n%s",
        pd.DataFrame({"n_sims": pop_counts, "pct": pop_pct}).to_string(),
    )

    # carbonStem_AF(t=40) — diagnostic threshold breakdown
    cs40 = (
        df.sort_values(["SimID", "Harvest_Year_Absolute"])
        .groupby("SimID")["carbonStem_AF"].last()
    )
    log.info("carbonStem_AF(t=40) — %d SimIDs", len(cs40))
    log.info(
        "  < %.1f kgC  (tree_failed)   : %d (%.1f%%)",
        TREE_FAIL_THRESHOLD,
        (cs40 < TREE_FAIL_THRESHOLD).sum(),
        100.0 * (cs40 < TREE_FAIL_THRESHOLD).mean(),
    )
    log.info(
        "  [%.1f, %.1f[ (tree_stunted) : %d (%.1f%%)",
        TREE_FAIL_THRESHOLD, TREE_STUNT_THRESHOLD,
        ((cs40 >= TREE_FAIL_THRESHOLD) & (cs40 < TREE_STUNT_THRESHOLD)).sum(),
        100.0 * ((cs40 >= TREE_FAIL_THRESHOLD) & (cs40 < TREE_STUNT_THRESHOLD)).mean(),
    )
    log.info(
        "  >= %.1f kgC  (tree_ok)      : %d (%.1f%%)",
        TREE_STUNT_THRESHOLD,
        (cs40 >= TREE_STUNT_THRESHOLD).sum(),
        100.0 * (cs40 >= TREE_STUNT_THRESHOLD).mean(),
    )

    # Nominal population (yield_ok × tree_ok)
    df_nominal, _ = filter_population(
        df,
        population=NOMINAL_POPULATION,
        tree_fail_threshold=TREE_FAIL_THRESHOLD,
        yield_fail_threshold=YIELD_FAIL_THRESHOLD,
        yield_fail_rate=YIELD_FAIL_RATE,
        verbose=True,
    )

    # Anti-contamination guard — no tree_failed SimIDs in the nominal set
    cs_last = (
        df_nominal.sort_values(["SimID", "Harvest_Year_Absolute"])
        .groupby("SimID")["carbonStem_AF"].last()
    )
    assert (cs_last >= TREE_FAIL_THRESHOLD).all(), (
        f"tree_failed SimIDs detected in nominal population! "
        f"min(carbonStem_AF) = {cs_last.min():.3f}"
    )

    log.info(
        "Nominal population: %d rows | %d SimIDs",
        len(df_nominal), df_nominal["SimID"].nunique(),
    )
    log.info("STEP 2 complete — %.1fs", time.perf_counter() - t0)
    return df_all_pops, df_nominal


# =============================================================================
# STEP 3 — Train/Test Split + CV Groups
# =============================================================================

def _step3_split(
    df_nominal: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, set, set, Any, Any]:
    """
    Split the nominal population into train/test sets and build CV groups.

    Stratified split by rotation type (maize / wheat) ensures balanced
    representation in both sets. GroupKFold on SimID prevents temporal
    leakage within cross-validation folds.

    Parameters
    ----------
    df_nominal : pd.DataFrame

    Returns
    -------
    train_df   : pd.DataFrame
    test_df    : pd.DataFrame
    train_sims : set  — unique SimIDs in training set
    test_sims  : set  — unique SimIDs in test set
    groups     : pd.Series  — SimID groups aligned to train_df index
    cv         : GroupKFold instance
    """
    t0 = time.perf_counter()
    log.info("=" * 70)
    log.info("STEP 3 — Train/Test Split + CV Groups")
    log.info("=" * 70)

    rot_summary = summarise_rotations(df_nominal)
    log.info("Rotation types:\n%s", rot_summary.to_string(index=False))

    train_df, test_df, _, _ = stratified_split_by_rotation(
        df_nominal, test_size=0.20, random_state=RANDOM_STATE, verbose=True,
    )
    train_sims = set(train_df["SimID"].unique())
    test_sims  = set(test_df["SimID"].unique())

    # SimID-level leakage guard
    overlap = train_sims & test_sims
    assert not overlap, (
        f"SimID leakage: {len(overlap)} SimIDs shared between train and test sets."
    )

    actual_test_ratio = len(test_sims) / df_nominal["SimID"].nunique()
    assert abs(actual_test_ratio - 0.20) < 0.05, (
        f"Test ratio {actual_test_ratio:.2f} outside acceptable range [0.15, 0.25]."
    )

    # Stratification balance check
    sig_train = get_rotation_signature(train_df).value_counts(normalize=True) * 100
    sig_test  = get_rotation_signature(test_df).value_counts(normalize=True) * 100
    strat_df  = pd.DataFrame(
        {"train_%": sig_train, "test_%": sig_test}
    ).fillna(0).round(1)
    strat_df["delta"] = (strat_df["train_%"] - strat_df["test_%"]).abs()
    if (strat_df["delta"] > 10.0).any():
        log.warning("Stratification imbalance > 10%%:\n%s", strat_df)
    else:
        log.info("Stratification balance check passed:\n%s", strat_df.to_string())

    # CV groups (SimID-level, aligned with train_df)
    groups = build_cv_groups(train_df)
    cv     = make_group_kfold(n_splits=CV_N_FOLDS)
    assert len(groups) == len(train_df), \
        "groups length mismatch with train_df."
    assert groups.nunique() == len(train_sims), \
        "groups unique count mismatch with training SimID count."

    # Per-fold leakage guard
    X_dummy = train_df[["SimID"]].reset_index(drop=True)
    y_dummy = pd.Series(0, index=range(len(train_df)))
    for fold_i, (tr_idx, val_idx) in enumerate(
        cv.split(X_dummy, y_dummy, groups=groups)
    ):
        sims_tr  = set(groups.iloc[tr_idx])
        sims_val = set(groups.iloc[val_idx])
        assert not (sims_tr & sims_val), \
            f"SimID leakage detected in CV fold {fold_i + 1}."

    log.info(
        "Train: %d SimIDs | Test: %d SimIDs | %d CV folds — no leakage detected.",
        len(train_sims), len(test_sims), CV_N_FOLDS,
    )
    log.info("STEP 3 complete — %.1fs", time.perf_counter() - t0)
    return train_df, test_df, train_sims, test_sims, groups, cv


# =============================================================================
# STEP 4 — Categorical Encoding & Winsorization
# =============================================================================

def _step4_encode_and_winsorize(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    campaign: Any,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame, pd.DataFrame]:
    """
    Apply categorical encoding and winsorization — fit on train only.

    Encoding  : LightGBM native category dtype (no statistical fit required).
    Winsorize : quantile [0.01, 0.99] clipping, fitted on train and applied
                to test with train bounds to prevent any data leakage.

    Parameters
    ----------
    train_df : pd.DataFrame
    test_df  : pd.DataFrame
    campaign : CampaignPaths

    Returns
    -------
    train_wins : pd.DataFrame  — winsorized train set
    test_wins  : pd.DataFrame  — winsorized test set (train bounds applied)
    win_bounds : dict          — {column: (lo, hi)} winsorization bounds
    X_train    : pd.DataFrame  — base feature matrix (ACTIVE_FEATURES_B2)
    X_test     : pd.DataFrame  — base feature matrix (ACTIVE_FEATURES_B2)
    """
    t0 = time.perf_counter()
    log.info("=" * 70)
    log.info("STEP 4 — Categorical Encoding & Winsorization")
    log.info("=" * 70)

    # Categorical encoding — LightGBM native dtype='category'
    train_df, encoders = encode_categoricals(
        train_df, features=ACTIVE_FEATURES_B2,
        fit=True, method="lightgbm", verbose=True,
    )
    test_df, _ = encode_categoricals(
        test_df, features=ACTIVE_FEATURES_B2,
        fit=False, encoders=encoders, method="lightgbm", verbose=True,
    )
    for col in CATEGORICAL_FEATURES_B2:
        if col in train_df.columns:
            assert train_df[col].dtype.name == "category", \
                f"'{col}' is not category dtype in train after encoding."
            assert test_df[col].dtype.name == "category", \
                f"'{col}' is not category dtype in test after encoding."
    log.info("Categorical encoding (LightGBM native) validated.")

    # Winsorization — fit on train, apply to test with fixed bounds
    train_wins, win_bounds = apply_winsorization(
        train_df, quantiles=(0.01, 0.99), fit=True, verbose=True,
    )
    test_wins, _ = apply_winsorization(
        test_df, quantiles=(0.01, 0.99), fit=False,
        bounds=win_bounds, verbose=True,
    )

    # Verify that clipping bounds are respected in train
    for col, (lo, hi) in win_bounds.items():
        if col in train_wins.columns:
            assert train_wins[col].min() >= lo - 1e-9, \
                f"'{col}': min value below lower winsorization bound."
            assert train_wins[col].max() <= hi + 1e-9, \
                f"'{col}': max value above upper winsorization bound."

    # Persist winsorization bounds — required at inference time
    win_path = campaign.metamodels_dir / "winsorization_bounds.joblib"
    joblib.dump(win_bounds, win_path)
    log.info(
        "Winsorization bounds saved (%d columns) → %s", len(win_bounds), win_path,
    )

    # Base feature matrices (all ACTIVE_FEATURES_B2, no stage-specific subsetting)
    X_train, _, _, _ = build_dataset(
        train_wins, features=ACTIVE_FEATURES_B2,
        targets=STOCK_TARGETS_MINIMAL, verbose=True,
    )
    X_test, _, _, _ = build_dataset(
        test_wins, features=ACTIVE_FEATURES_B2,
        targets=STOCK_TARGETS_MINIMAL, verbose=True,
    )
    assert X_train.shape[1] == X_test.shape[1], \
        "Feature count mismatch between X_train and X_test."
    assert list(X_train.columns) == list(X_test.columns), \
        "Feature column order mismatch between X_train and X_test."
    log.info(
        "X_train: %d × %d | X_test: %d × %d",
        *X_train.shape, *X_test.shape,
    )

    log.info("STEP 4 complete — %.1fs", time.perf_counter() - t0)
    return train_wins, test_wins, win_bounds, X_train, X_test


# =============================================================================
# STEP 5a — Classifiers (CLF1, CLF2) + Stunted Fallback Model
# =============================================================================

def _step5a_classifiers(
    df: pd.DataFrame,
    df_nominal: pd.DataFrame,
    campaign: Any,
    params_clf1: dict,
) -> tuple[Any, Any, dict, dict, dict, pd.DataFrame, pd.DataFrame]:
    """
    Train CLF1 (binary tree status) and CLF2 (binary yield failure).
    Build the conditional median fallback model for stunted-tree simulations.

    Both classifiers are trained on the FULL dataset (all populations) using
    a dedicated SimID-level random split that is independent of the nominal
    split from STEP 3. This ensures classifiers see the complete population
    distribution, including failed and stunted cases.

    Parameters
    ----------
    df         : pd.DataFrame  — full prepared meta-table (all populations)
    df_nominal : pd.DataFrame  — nominal population (routing sanity check only)
    campaign   : CampaignPaths
    params_clf1 : dict         — LightGBM hyperparameters for CLF1

    Returns
    -------
    clf1_fitted   : fitted LGBMClassifier (binary)
    clf2_fitted   : fitted LGBMClassifier (binary) or None
    clf1_eval     : dict  — evaluation metrics for CLF1
    clf2_eval     : dict  — evaluation metrics for CLF2
    stunted_model : dict  — conditional median fallback
    df_clf_train  : pd.DataFrame  — classifier training split
    df_clf_test   : pd.DataFrame  — classifier test split (used in STEP 6 SHAP)
    """
    t0 = time.perf_counter()
    log.info("=" * 70)
    log.info("STEP 5a — Classifiers + Stunted Fallback Model")
    log.info("=" * 70)

    # Dedicated SimID-level split on the full dataset (all populations)
    df_clf_train, df_clf_test, _, _ = split_by_simid(
        df, test_size=0.20, random_state=RANDOM_STATE, verbose=True,
    )
    log.info(
        "Classifier split — train: %d SimIDs | test: %d SimIDs",
        df_clf_train["SimID"].nunique(),
        df_clf_test["SimID"].nunique(),
    )

    # ── CLF1 — Binary tree status (tree_degraded vs tree_ok) ──────────────
    log.info("\n--- CLF1: Tree Status (binary — tree_degraded / tree_ok) ---")
    clf1_model, _, _ = build_tree_fail_classifier(
        df, params=params_clf1, multiclass=False, verbose=True,
    )

    X_clf1_train = build_classifier_features(df_clf_train, CLF1_FEATURES)
    X_clf1_test  = build_classifier_features(df_clf_test,  CLF1_FEATURES)
    y_clf1_train = build_tree_degraded_labels(df_clf_train)
    y_clf1_test  = build_tree_degraded_labels(df_clf_test)

    # Index alignment (SimID-level labels may differ in length)
    X_clf1_train = X_clf1_train.loc[X_clf1_train.index.intersection(y_clf1_train.index)]
    y_clf1_train = y_clf1_train.loc[X_clf1_train.index]
    X_clf1_test  = X_clf1_test.loc[X_clf1_test.index.intersection(y_clf1_test.index)]
    y_clf1_test  = y_clf1_test.loc[X_clf1_test.index]
    assert len(X_clf1_test) > 0, "X_clf1_test is empty after index alignment."

    for i, label in enumerate(TREE_BINARY_LABELS):
        n = (y_clf1_train == i).sum()
        log.info(
            "  CLF1 train — %-20s: %d (%.1f%%)",
            label, n, 100.0 * n / len(y_clf1_train),
        )

    clf1_fitted, _ = train_classifier(
        clf1_model, X_clf1_train, y_clf1_train,
        X_clf1_test, y_clf1_test, verbose=True,
    )
    y_clf1_pred  = clf1_fitted.predict(X_clf1_test)
    y_clf1_proba = clf1_fitted.predict_proba(X_clf1_test)[:, 1]   # P(tree_ok)
    clf1_eval    = evaluate_clf1_binary(
        y_clf1_test, y_clf1_pred, y_clf1_proba,
        classifier_name="CLF1 — Binary Tree Status",
        verbose=True,
    )

    if clf1_eval["f1_macro"] < QUALITY_THRESHOLDS["clf1_f1_macro"]:
        log.warning(
            "CLF1 F1 macro below threshold (%.3f < %.2f).",
            clf1_eval["f1_macro"], QUALITY_THRESHOLDS["clf1_f1_macro"],
        )
    if clf1_eval.get("roc_auc", 1.0) < QUALITY_THRESHOLDS["clf1_roc_auc"]:
        log.warning(
            "CLF1 ROC-AUC below threshold (%.3f < %.2f).",
            clf1_eval.get("roc_auc", 0.0), QUALITY_THRESHOLDS["clf1_roc_auc"],
        )
    else:
        log.info(
            "CLF1 — Accuracy: %.1f%%  F1 macro: %.3f  ROC-AUC: %.3f",
            clf1_eval["accuracy"] * 100,
            clf1_eval["f1_macro"],
            clf1_eval.get("roc_auc", float("nan")),
        )

    # ── CLF2 — Binary yield failure ────────────────────────────────────────
    log.info("\n--- CLF2: Yield Failure (binary) ---")
    clf2_model, _, _ = build_yield_fail_classifier(df, verbose=True)

    X_clf2_train = build_classifier_features(df_clf_train, CLF2_FEATURES)
    X_clf2_test  = build_classifier_features(df_clf_test,  CLF2_FEATURES)
    y_clf2_train = build_yield_fail_labels(df_clf_train)
    y_clf2_test  = build_yield_fail_labels(df_clf_test)

    X_clf2_train = X_clf2_train.loc[X_clf2_train.index.intersection(y_clf2_train.index)]
    y_clf2_train = y_clf2_train.loc[X_clf2_train.index]
    X_clf2_test  = X_clf2_test.loc[X_clf2_test.index.intersection(y_clf2_test.index)]
    y_clf2_test  = y_clf2_test.loc[X_clf2_test.index]
    assert len(X_clf2_test) > 0, "X_clf2_test is empty after index alignment."

    log.info(
        "CLF2 — train: %d SimIDs (%.1f%% failed) | test: %d SimIDs (%.1f%% failed)",
        len(y_clf2_train), 100.0 * y_clf2_train.mean(),
        len(y_clf2_test),  100.0 * y_clf2_test.mean(),
    )

    if clf2_model is None:
        # Geographic deterministic fallback (insufficient yield-fail data)
        y_clf2_pred = apply_geographic_rule(X_clf2_test)
        clf2_eval   = evaluate_classifier(
            y_clf2_test, y_clf2_pred,
            classifier_name="CLF2 (geographic rule fallback)", verbose=True,
        )
        clf2_fitted = None
        log.info("CLF2: using geographic rule fallback (insufficient yield-fail data).")
    else:
        clf2_fitted, _ = train_classifier(
            clf2_model, X_clf2_train, y_clf2_train,
            X_clf2_test, y_clf2_test, verbose=True,
        )
        y_clf2_pred  = clf2_fitted.predict(X_clf2_test)
        y_clf2_proba = clf2_fitted.predict_proba(X_clf2_test)[:, 1]
        clf2_eval    = evaluate_classifier(
            y_clf2_test, y_clf2_pred, y_clf2_proba,
            classifier_name="CLF2 — Yield Failure", verbose=True,
        )

    if clf2_eval.get("accuracy", 1.0) < QUALITY_THRESHOLDS["clf2_accuracy"]:
        log.warning(
            "CLF2 accuracy below threshold (%.1f%% < %.0f%%).",
            clf2_eval.get("accuracy", 0.0) * 100,
            QUALITY_THRESHOLDS["clf2_accuracy"] * 100,
        )
    else:
        log.info("CLF2 accuracy: %.1f%%", clf2_eval.get("accuracy", 1.0) * 100)

    # Serialize both classifiers
    save_classifiers(
        clf1_fitted, clf2_fitted, campaign,
        clf1_meta={
            "accuracy_test": clf1_eval["accuracy"],
            "f1_macro_test": clf1_eval["f1_macro"],
            "f1_weighted":   clf1_eval.get("f1_weighted"),
            "n_train":       len(y_clf1_train),
            "n_test":        len(y_clf1_test),
            "n_classes":     2,
        },
        clf2_meta={
            "accuracy_test": clf2_eval.get("accuracy"),
            "f1_test":       clf2_eval.get("f1"),
            "roc_auc_test":  clf2_eval.get("roc_auc"),
            "n_train":       len(y_clf2_train),
            "n_test":        len(y_clf2_test),
        } if clf2_fitted is not None else None,
    )
    log.info("CLF1 + CLF2 serialized.")

    # ── Routing sanity check on nominal population ─────────────────────────
    log.info("\n--- Routing sanity check (nominal population) ---")
    all_clf_feats = CLF1_FEATURES + [c for c in CLF2_FEATURES if c not in CLF1_FEATURES]
    X_nominal_clf = build_classifier_features(df_nominal, feature_cols=all_clf_feats)
    routing_check = predict_routing(X_nominal_clf, clf1_fitted, clf2_fitted)

    for i, label in enumerate(TREE_BINARY_LABELS):
        n   = (routing_check["tree_status"] == i).sum()
        pct = 100.0 * n / len(routing_check)
        log.info("  CLF1 predicts %-20s: %d (%.1f%%)", label, n, pct)

    n_misrouted   = (routing_check["tree_status"] == TREE_BINARY_DEGRADED).sum()
    pct_misrouted = 100.0 * n_misrouted / len(routing_check)
    if pct_misrouted > 10.0:
        log.warning(
            "High misrouting rate on nominal population: %d / %d (%.1f%%).",
            n_misrouted, len(routing_check), pct_misrouted,
        )
    else:
        log.info(
            "Misrouted nominal SimIDs (predicted tree_degraded): %d / %d (%.1f%%).",
            n_misrouted, len(routing_check), pct_misrouted,
        )

    if "tree_ok_proba" in routing_check.columns:
        log.info(
            "Mean P(tree_ok) on nominal population: %.3f  "
            "[stunted sub-routing threshold: %.2f]",
            routing_check["tree_ok_proba"].mean(),
            STUNTED_PROBA_THRESHOLD,
        )

    # ── Stunted fallback model ─────────────────────────────────────────────
    log.info("\n--- Stunted fallback model (conditional median) ---")
    cs_final_all = (
        df.sort_values(["SimID", "Harvest_Year_Absolute"])
        .groupby("SimID")["carbonStem_AF"].last()
    )
    mask_stunted   = (
        (cs_final_all >= TREE_FAIL_THRESHOLD)
        & (cs_final_all < TREE_STUNT_THRESHOLD)
    )
    simids_stunted = cs_final_all[mask_stunted].index
    df_stunted     = df[df["SimID"].isin(simids_stunted)].copy()
    log.info(
        "Stunted population: %d SimIDs  "
        "(carbonStem_AF(t=40) ∈ [%.1f, %.1f[ kgC/tree)",
        len(simids_stunted), TREE_FAIL_THRESHOLD, TREE_STUNT_THRESHOLD,
    )

    # Calibrate on the training split only (no test leakage)
    stunted_train_sims = set(df_clf_train["SimID"].unique()) & set(simids_stunted)
    df_stunted_train   = df_stunted[df_stunted["SimID"].isin(stunted_train_sims)]

    cs_final_stunted = (
        df_stunted_train.sort_values(["SimID", "Harvest_Year_Absolute"])
        .groupby("SimID").last()[["carbonStem_AF", "main_crop"]]
    )
    median_by_crop = cs_final_stunted.groupby("main_crop")["carbonStem_AF"].median()
    global_median  = float(cs_final_stunted["carbonStem_AF"].median())

    for crop, med in median_by_crop.items():
        log.info("  Stunted median  %-12s: %.3f kgC/tree", crop, med)
    log.info("  Stunted median  %-12s: %.3f kgC/tree", "global", global_median)

    # Normalized temporal profile: ratio = carbonStem_AF(t) / carbonStem_AF(t=40)
    df_sp        = df_stunted_train.copy().sort_values(["SimID", "Harvest_Year_Absolute"])
    cs_final_map = df_sp.groupby("SimID")["carbonStem_AF"].last().rename("cs_final")
    df_sp        = df_sp.join(cs_final_map, on="SimID")
    df_sp        = df_sp[df_sp["cs_final"] > 0].copy()
    df_sp["profile"] = (df_sp["carbonStem_AF"] / df_sp["cs_final"]).clip(0, 1)
    stunted_profile  = df_sp.groupby("Harvest_Year_Absolute")["profile"].median()

    stunted_model = {
        "type":           "conditional_median",
        "median_by_crop": median_by_crop.to_dict(),
        "global_median":  global_median,
        "profile_median": stunted_profile.to_dict(),
        "n_train_simids": len(stunted_train_sims),
    }
    stunted_path = campaign.metamodels_dir / "stunted_model.joblib"
    joblib.dump(stunted_model, stunted_path)
    log.info("Stunted model saved → %s", stunted_path)

    log.info("STEP 5a complete — %.1fs", time.perf_counter() - t0)
    return (
        clf1_fitted, clf2_fitted,
        clf1_eval, clf2_eval,
        stunted_model,
        df_clf_train, df_clf_test,
    )


# =============================================================================
# STEP 5b — Stage 1: carbonStem Horizon Models (log1p)
# =============================================================================

def _step5b_stage1_carbon(
    df_horizon_pop: pd.DataFrame,
    train_sims: set,
    test_sims: set,
    campaign: Any,
    params_carbon_short: dict,
    params_carbon_long: dict,
    static_feats: list[str],
) -> tuple[dict, dict, list]:
    """
    Train 10 SimID-level carbonStem horizon models (5 horizons × 2 targets).

    Targets  : carbonStem_AF, carbonStem_TF
    Horizons : CARBON_HORIZONS (default: 5, 10, 20, 30, 40)

    Carbon stock distributions are right-skewed (cumulative biomass over years),
    so targets are log1p-transformed before training. expm1 back-transform
    is applied during evaluation and at inference (predictor.py).

    MIN_ENRICH_HORIZON (from config.py) controls the threshold below which
    enriched climate aggregates (p10, p90, trend, recent_mean) are excluded
    from the feature set — Stage 1 only.

    Parameters
    ----------
    df_horizon_pop      : pd.DataFrame  — nominal population with MIN_CARBON_HORIZON filter
    train_sims          : set           — SimIDs for training
    test_sims           : set           — SimIDs for evaluation
    campaign            : CampaignPaths
    params_carbon_short : dict          — LightGBM params for h ≤ 10
    params_carbon_long  : dict          — LightGBM params for h > 10
    static_feats        : list[str]     — non-climate, non-temporal features

    Returns
    -------
    horizon_models   : dict  — {(target, h): fitted LGBMRegressor}
    horizon_datasets : dict  — {(target, h): (X_h, y_h)}
    horizon_results  : list  — list of per-model metric dicts
    """
    t0 = time.perf_counter()
    log.info("=" * 70)
    log.info("STEP 5b — Stage 1: carbonStem Horizon Models (log1p)")
    log.info("=" * 70)

    horizon_models:   dict = {}
    horizon_datasets: dict = {}
    horizon_results:  list = []

    for target in ["carbonStem_AF", "carbonStem_TF"]:
        for h in CARBON_HORIZONS:
            log.info("\n--- %s | h=%d ---", target, h)

            params_carbon = params_carbon_short if h <= 10 else params_carbon_long

            X_h, y_h = build_horizon_dataset(
                df_horizon_pop,
                horizon=h,
                target_col=target,
                feature_cols=static_feats,
                min_enrich_horizon=MIN_ENRICH_HORIZON,
            )
            # Categorical encoding — LightGBM native, no statistical fit
            X_h, _ = encode_categoricals(
                X_h, features=CATEGORICAL_FEATURES_B2,
                fit=True, method="lightgbm", verbose=False,
            )

            X_train_h = X_h[X_h.index.isin(train_sims)]
            y_train_h = y_h[y_h.index.isin(train_sims)]
            X_test_h  = X_h[X_h.index.isin(test_sims)]
            y_test_h  = y_h[y_h.index.isin(test_sims)]

            n_train, n_test = len(y_train_h), len(y_test_h)
            log.info("  train: %d SimIDs | test: %d SimIDs", n_train, n_test)

            if n_train < 30 or n_test < 10:
                log.warning("  Insufficient data — skipping horizon.")
                continue

            # log1p transform (right-skewed cumulative biomass)
            y_train_log = np.log1p(y_train_h)
            y_test_log  = np.log1p(y_test_h)

            cat_feats_h   = [c for c in CATEGORICAL_FEATURES_B2 if c in X_train_h.columns]
            model_h       = build_lgb(params=params_carbon, categorical_feature=cat_feats_h)
            model_h_fitted, _ = train_final_model(
                model_h, X_train_h, y_train_log,
                X_test_h, y_test_log, verbose=False,
            )

            # Evaluate in original space (expm1 back-transform)
            y_pred = np.clip(np.expm1(model_h_fitted.predict(X_test_h)), 0.0, None)
            y_true = y_test_h.values
            m      = compute_metrics(y_true, y_pred, prefix="test_")
            rho, p = spearmanr(y_true, y_pred)

            log.info(
                "  %s h=%d — R²=%.3f | RMSE=%.2f | rho=%.3f (p=%.1e)",
                target, h, m["test_r2"], m["test_rmse"], rho, p,
            )
            if m["test_r2"] < QUALITY_THRESHOLDS["carbon_r2"]:
                log.warning(
                    "  R²=%.3f below threshold (%.2f).",
                    m["test_r2"], QUALITY_THRESHOLDS["carbon_r2"],
                )
            if h == 40 and rho < QUALITY_THRESHOLDS["carbon_rho"]:
                log.warning(
                    "  rho=%.3f below threshold (%.2f) at h=40.",
                    rho, QUALITY_THRESHOLDS["carbon_rho"],
                )

            model_key  = f"{target}_h{h}"
            model_path = campaign.metamodels_dir / f"lgbm_{model_key}.joblib"
            save_model(
                model_h_fitted, model_path,
                metadata={
                    "target":         target,
                    "horizon":        h,
                    "model_key":      model_key,
                    "log_transform":  True,
                    "feature_names":  list(X_h.columns),
                    "n_features":     X_h.shape[1],
                    "r2_test":        m["test_r2"],
                    "spearman_rho":   rho,
                },
            )

            horizon_models[(target, h)]   = model_h_fitted
            horizon_datasets[(target, h)] = (X_h, y_h)
            horizon_results.append({
                "target":       target,
                "horizon":      h,
                "n_train":      n_train,
                "n_test":       n_test,
                "r2":           m["test_r2"],
                "rmse":         m["test_rmse"],
                "spearman_rho": rho,
                "spearman_p":   p,
            })

    df_hr = pd.DataFrame(horizon_results).sort_values(["target", "horizon"])
    log.info(
        "\nStage 1 — %d carbonStem models trained:\n%s",
        len(horizon_results),
        df_hr[["target", "horizon", "r2", "rmse", "spearman_rho"]].to_string(index=False),
    )

    log.info("STEP 5b complete — %.1fs", time.perf_counter() - t0)
    return horizon_models, horizon_datasets, horizon_results


# =============================================================================
# STEP 5c — Stage 2: yield_AF and yield_TA (row-by-row, v5.0)
# =============================================================================

def _step5c_stage2_yield(
    df_train_nominal: pd.DataFrame,
    df_test_nominal: pd.DataFrame,
    campaign: Any,
    params_yield_af: dict,
    params_yield_ta: dict,
) -> tuple[Any, Any, dict, dict]:
    """
    Train row-by-row regression models for yield_AF and yield_TA (Stage 2, v5.0).

    Design principles:
      - One model per target (vs 5 × horizons in v4.1).
      - Trained on annual rows (SimID × Crop_Name × Harvest_Year_Absolute).
        No horizon aggregation, no PCHIP interpolation.
      - yield_AF feature set includes carbonStem_AF (real value from the
        meta-table). At inference, the Stage 1 predicted trajectory is used
        instead — training always on real values, no injection leakage.
      - yield_TA has no tree-related features (pure crop, no AF interaction).
      - GroupKFold CV remains at SimID level to prevent temporal leakage.

    Parameters
    ----------
    df_train_nominal : pd.DataFrame
        Training rows — nominal population (tree_ok × yield_ok), winsorized,
        at annual resolution (SimID × Crop_Name × Harvest_Year_Absolute).
    df_test_nominal : pd.DataFrame
        Test rows — same format, same population filter.
    campaign : CampaignPaths
    params_yield_af : dict  — LightGBM hyperparameters for yield_AF model
    params_yield_ta : dict  — LightGBM hyperparameters for yield_TA model

    Returns
    -------
    model_yield_af   : fitted LGBMRegressor
    model_yield_ta   : fitted LGBMRegressor
    metrics_yield_af : dict  — {train_r2, train_rmse, train_mae, test_r2, test_rmse, test_mae}
    metrics_yield_ta : dict  — same structure
    """
    t0 = time.perf_counter()
    log.info("=" * 70)
    log.info("STEP 5c — Stage 2 Row-by-Row: yield_AF and yield_TA")
    log.info("=" * 70)
    log.info(
        "Train set: %d rows | %d SimIDs",
        len(df_train_nominal), df_train_nominal["SimID"].nunique(),
    )
    log.info(
        "Test set:  %d rows | %d SimIDs",
        len(df_test_nominal), df_test_nominal["SimID"].nunique(),
    )

    # GroupKFold CV groups — SimID level (prevents temporal leakage)
    groups_train = build_cv_groups(df_train_nominal)

    # ── yield_AF — with real carbonStem_AF as feature ──────────────────────
    log.info("\n--- yield_AF (carbonStem_AF real value as feature) ---")

    # Validate that carbonStem_AF is present and non-negative
    assert "carbonStem_AF" in df_train_nominal.columns, \
        "carbonStem_AF column missing from df_train_nominal."
    assert (df_train_nominal["carbonStem_AF"] >= 0).all(), \
        "Negative carbonStem_AF values detected in training set."

    # Validate feature availability
    missing_af = [f for f in STEP2_AF_FEATURES if f not in df_train_nominal.columns]
    if missing_af:
        raise ValueError(f"STEP 5c — Missing STEP2_AF_FEATURES columns: {missing_af}")

    X_train_af = df_train_nominal[STEP2_AF_FEATURES]
    y_train_af = df_train_nominal["yield_AF"]
    X_test_af  = df_test_nominal[STEP2_AF_FEATURES]
    y_test_af  = df_test_nominal["yield_AF"]

    model_yield_af = build_lgb(
        params=params_yield_af,
        categorical_feature=CATEGORICAL_FEATURES_B2,
    )
    cv_results_af = cross_validate(
        model_yield_af, X_train_af, y_train_af,
        groups=groups_train, verbose=True,
    )
    save_cv_results(cv_results_af, campaign.cv_dir / "cv_yield_AF_rowwise.json")

    model_yield_af, metrics_yield_af = train_final_model(
        model_yield_af,
        X_train_af, y_train_af,
        X_test_af,  y_test_af,
        verbose=True,
    )
    save_model(
        model_yield_af,
        campaign.metamodels_dir / "lgbm_yield_AF_rowwise.joblib",
        metadata={
            "target":        "yield_AF",
            "architecture":  "row-by-row",
            "log_transform": False,
            "feature_names": list(STEP2_AF_FEATURES),
            "n_features":    len(STEP2_AF_FEATURES),
            "r2_test":       metrics_yield_af.get("test_r2"),
        },
    )
    save_metrics(metrics_yield_af, campaign.metrics_dir / "metrics_yield_AF_rowwise.json")

    fi_af = get_feature_importances(model_yield_af, STEP2_AF_FEATURES)
    save_feature_importances(fi_af, campaign.featimps_dir / "fimp_yield_AF_rowwise.csv")

    log.info(
        "yield_AF — CV R² val: %.3f ± %.3f | Test R²: %.3f",
        cv_results_af["mean_r2_val"],
        cv_results_af["std_r2_val"],
        metrics_yield_af["test_r2"],
    )
    if metrics_yield_af["test_r2"] < QUALITY_THRESHOLDS["yield_af_r2"]:
        log.warning(
            "yield_AF Test R²=%.3f below threshold (%.2f). "
            "Consider tuning or switching to RR_crop target.",
            metrics_yield_af["test_r2"], QUALITY_THRESHOLDS["yield_af_r2"],
        )

    # ── yield_TA — no carbonStem, pure crop model ──────────────────────────
    log.info("\n--- yield_TA (no tree features — pure crop control) ---")

    missing_ta = [f for f in STEP2_TA_FEATURES if f not in df_train_nominal.columns]
    if missing_ta:
        raise ValueError(f"STEP 5c — Missing STEP2_TA_FEATURES columns: {missing_ta}")

    X_train_ta = df_train_nominal[STEP2_TA_FEATURES]
    y_train_ta = df_train_nominal["yield_TA"]
    X_test_ta  = df_test_nominal[STEP2_TA_FEATURES]
    y_test_ta  = df_test_nominal["yield_TA"]

    model_yield_ta = build_lgb(
        params=params_yield_ta,
        categorical_feature=CATEGORICAL_FEATURES_B2,
    )
    cv_results_ta = cross_validate(
        model_yield_ta, X_train_ta, y_train_ta,
        groups=groups_train, verbose=True,
    )
    save_cv_results(cv_results_ta, campaign.cv_dir / "cv_yield_TA_rowwise.json")

    model_yield_ta, metrics_yield_ta = train_final_model(
        model_yield_ta,
        X_train_ta, y_train_ta,
        X_test_ta,  y_test_ta,
        verbose=True,
    )
    save_model(
        model_yield_ta,
        campaign.metamodels_dir / "lgbm_yield_TA_rowwise.joblib",
        metadata={
            "target":        "yield_TA",
            "architecture":  "row-by-row",
            "log_transform": False,
            "feature_names": list(STEP2_TA_FEATURES),
            "n_features":    len(STEP2_TA_FEATURES),
            "r2_test":       metrics_yield_ta.get("test_r2"),
        },
    )
    save_metrics(metrics_yield_ta, campaign.metrics_dir / "metrics_yield_TA_rowwise.json")

    fi_ta = get_feature_importances(model_yield_ta, STEP2_TA_FEATURES)
    save_feature_importances(fi_ta, campaign.featimps_dir / "fimp_yield_TA_rowwise.csv")

    log.info(
        "yield_TA — CV R² val: %.3f ± %.3f | Test R²: %.3f",
        cv_results_ta["mean_r2_val"],
        cv_results_ta["std_r2_val"],
        metrics_yield_ta["test_r2"],
    )
    if metrics_yield_ta["test_r2"] < QUALITY_THRESHOLDS["yield_ta_r2"]:
        log.warning(
            "yield_TA Test R²=%.3f below threshold (%.2f).",
            metrics_yield_ta["test_r2"], QUALITY_THRESHOLDS["yield_ta_r2"],
        )

    log.info("STEP 5c complete — %.1fs", time.perf_counter() - t0)
    return model_yield_af, model_yield_ta, metrics_yield_af, metrics_yield_ta


# =============================================================================
# STEP 6 — SHAP Analysis
# =============================================================================

def _step6_shap(
    horizon_models: dict,
    horizon_datasets: dict,
    model_yield_af: Any,
    X_test_af: pd.DataFrame,
    model_yield_ta: Any,
    X_test_ta: pd.DataFrame,
    clf1_fitted: Any,
    clf2_fitted: Any,
    X_clf1_test: pd.DataFrame,
    X_clf2_test: pd.DataFrame,
    test_sims: set,
    campaign: Any,
) -> dict:
    """
    Compute and export SHAP values for all models.

    Targets analysed:
        Stage 1  : carbonStem_AF h=40, carbonStem_TF h=40
        Stage 2  : yield_AF (full test set, row-by-row)
                   yield_TA (full test set, row-by-row)
        Classifiers: CLF1, CLF2

    Parameters
    ----------
    horizon_models    : dict  — {(target, h): model}
    horizon_datasets  : dict  — {(target, h): (X_h, y_h)}
    model_yield_af    : fitted LGBMRegressor — Stage 2 yield_AF
    X_test_af         : pd.DataFrame  — test feature matrix for yield_AF
    model_yield_ta    : fitted LGBMRegressor — Stage 2 yield_TA
    X_test_ta         : pd.DataFrame  — test feature matrix for yield_TA
    clf1_fitted, clf2_fitted : classifiers
    X_clf1_test, X_clf2_test : pd.DataFrame
    test_sims         : set  — SimIDs in test split
    campaign          : CampaignPaths

    Returns
    -------
    all_shap : dict  — consolidated SHAP results keyed by model name
    """
    t0 = time.perf_counter()
    log.info("=" * 70)
    log.info("STEP 6 — SHAP Analysis")
    log.info("=" * 70)

    all_shap: dict = {}

    # ── Stage 1 — carbonStem h=40 ─────────────────────────────────────────
    log.info("\n--- SHAP: Stage 1 carbonStem (h=40) ---")
    shap_stage1: dict = {}
    for cs_target in ["carbonStem_AF", "carbonStem_TF"]:
        key = f"{cs_target}_h40"
        if (cs_target, 40) not in horizon_models:
            log.warning("Model (%s, h=40) not found — skipping SHAP.", cs_target)
            continue
        model_h40     = horizon_models[(cs_target, 40)]
        X_h40_full, _ = horizon_datasets[(cs_target, 40)]
        X_h40_test    = X_h40_full[X_h40_full.index.isin(test_sims)]
        if len(X_h40_test) == 0:
            log.warning("No test samples for %s h=40 — skipping SHAP.", cs_target)
            continue
        sv, Xs = compute_shap_values(
            model_h40, X_h40_test, max_samples=2000, random_state=RANDOM_STATE,
        )
        shap_stage1[key] = {
            "shap_values":   sv,
            "X_sample":      Xs,
            "summary_df":    summarise_shap(sv, list(Xs.columns)),
            "feature_names": list(Xs.columns),
        }
    if shap_stage1:
        plot_shap_by_target(
            shap_stage1, save_dir=campaign.shap_dir,
            plot_type="both", max_display=15,
        )
        all_shap.update(shap_stage1)

    # ── Stage 2 — yield_AF (row-by-row) ───────────────────────────────────
    log.info("\n--- SHAP: Stage 2 yield_AF (row-by-row) ---")
    if model_yield_af is not None and len(X_test_af) > 0:
        try:
            sv_af, Xs_af = compute_shap_values(
                model_yield_af, X_test_af,
                max_samples=2000, random_state=RANDOM_STATE,
            )
            shap_yaf = {
                "yield_AF_rowwise": {
                    "shap_values":   sv_af,
                    "X_sample":      Xs_af,
                    "summary_df":    summarise_shap(sv_af, list(Xs_af.columns)),
                    "feature_names": list(Xs_af.columns),
                }
            }
            plot_shap_by_target(
                shap_yaf, save_dir=campaign.shap_dir,
                plot_type="both", max_display=15,
            )
            all_shap.update(shap_yaf)
        except Exception as exc:
            log.warning("SHAP failed for yield_AF: %s", exc)
    else:
        log.warning("yield_AF model or test set unavailable — skipping SHAP.")

    # ── Stage 2 — yield_TA (row-by-row) ───────────────────────────────────
    log.info("\n--- SHAP: Stage 2 yield_TA (row-by-row) ---")
    if model_yield_ta is not None and len(X_test_ta) > 0:
        try:
            sv_ta, Xs_ta = compute_shap_values(
                model_yield_ta, X_test_ta,
                max_samples=2000, random_state=RANDOM_STATE,
            )
            shap_yta = {
                "yield_TA_rowwise": {
                    "shap_values":   sv_ta,
                    "X_sample":      Xs_ta,
                    "summary_df":    summarise_shap(sv_ta, list(Xs_ta.columns)),
                    "feature_names": list(Xs_ta.columns),
                }
            }
            plot_shap_by_target(
                shap_yta, save_dir=campaign.shap_dir,
                plot_type="both", max_display=15,
            )
            all_shap.update(shap_yta)
        except Exception as exc:
            log.warning("SHAP failed for yield_TA: %s", exc)
    else:
        log.warning("yield_TA model or test set unavailable — skipping SHAP.")

    # ── Classifiers ───────────────────────────────────────────────────────
    log.info("\n--- SHAP: Classifiers ---")
    try:
        shap_clf = run_shap_classifiers(
            clf1=clf1_fitted, clf2=clf2_fitted,
            X_clf1=X_clf1_test, X_clf2=X_clf2_test,
            random_state=RANDOM_STATE,
        )
        plot_shap_by_target(
            shap_clf, save_dir=campaign.shap_dir,
            plot_type="bar", max_display=10,
        )
        all_shap.update(shap_clf)
    except Exception as exc:
        log.warning("Classifier SHAP failed: %s", exc)

    # ── Export for R/Shiny ────────────────────────────────────────────────
    export_path = export_shap_for_shiny(
        all_shap, campaign.shap_data_dir, include_raw=False,
    )
    log.info("SHAP data exported → %s", export_path)
    log.info("STEP 6 complete — %.1fs", time.perf_counter() - t0)
    return all_shap


# =============================================================================
# STEP 7 — Final Evaluation & Quality Summary Table
# =============================================================================

def _step7_evaluate(
    clf1_eval: dict,
    clf2_eval: dict,
    horizon_results: list,
    metrics_yield_af: dict,
    metrics_yield_ta: dict,
    campaign: Any,
) -> pd.DataFrame:
    """
    Build the quality summary table and persist pipeline-level metrics.

    Quality thresholds (see QUALITY_THRESHOLDS):
        CLF1 F1 macro >= 0.70  |  CLF1 ROC-AUC >= 0.85
        CLF2 Accuracy >= 0.80
        carbonStem  R² >= 0.60  |  rho >= 0.85
        yield_AF    R² >= 0.55
        yield_TA    R² >= 0.55

    Parameters
    ----------
    clf1_eval        : dict  — from STEP 5a
    clf2_eval        : dict  — from STEP 5a
    horizon_results  : list  — from STEP 5b (per-horizon metric dicts)
    metrics_yield_af : dict  — from STEP 5c {test_r2, test_rmse, …}
    metrics_yield_ta : dict  — from STEP 5c {test_r2, test_rmse, …}
    campaign         : CampaignPaths

    Returns
    -------
    df_summary : pd.DataFrame
    """
    t0 = time.perf_counter()
    log.info("=" * 70)
    log.info("STEP 7 — Final Evaluation & Quality Summary")
    log.info("=" * 70)

    rows: list = []

    # CLF1
    rows.append({
        "stage":     "CLF1",
        "target":    "tree_status (binary)",
        "metric":    "F1 macro | ROC-AUC",
        "value":     (
            f"F1={clf1_eval['f1_macro']:.3f} | "
            f"AUC={clf1_eval.get('roc_auc', float('nan')):.3f}"
        ),
        "threshold": (
            f"F1>={QUALITY_THRESHOLDS['clf1_f1_macro']:.2f} | "
            f"AUC>={QUALITY_THRESHOLDS['clf1_roc_auc']:.2f}"
        ),
        "status": (
            "OK"
            if (
                clf1_eval["f1_macro"] >= QUALITY_THRESHOLDS["clf1_f1_macro"]
                and clf1_eval.get("roc_auc", 0.0) >= QUALITY_THRESHOLDS["clf1_roc_auc"]
            )
            else "WARN"
        ),
    })

    # CLF2
    rows.append({
        "stage":     "CLF2",
        "target":    "yield_failed",
        "metric":    "Accuracy",
        "value":     f"{clf2_eval.get('accuracy', 0.0):.3f}",
        "threshold": f">= {QUALITY_THRESHOLDS['clf2_accuracy']:.2f}",
        "status": (
            "OK"
            if clf2_eval.get("accuracy", 0.0) >= QUALITY_THRESHOLDS["clf2_accuracy"]
            else "WARN"
        ),
    })

    # Stage 1 — carbonStem horizons
    for row in horizon_results:
        r2, rho, h = row["r2"], row["spearman_rho"], row["horizon"]
        ok = (
            r2  >= QUALITY_THRESHOLDS["carbon_r2"]
            and rho >= QUALITY_THRESHOLDS["carbon_rho"]
        )
        rows.append({
            "stage":     "Stage1",
            "target":    f"{row['target']} h={h}",
            "metric":    "R² | rho",
            "value":     f"R²={r2:.3f} | rho={rho:.3f}",
            "threshold": (
                f"R²>={QUALITY_THRESHOLDS['carbon_r2']} | "
                f"rho>={QUALITY_THRESHOLDS['carbon_rho']}"
            ),
            "status":    "OK" if ok else "WARN",
        })

    # Stage 2 — yield_AF row-by-row
    r2_af = metrics_yield_af.get("test_r2", float("nan"))
    rows.append({
        "stage":     "Stage2",
        "target":    "yield_AF (row-by-row)",
        "metric":    "R² test",
        "value":     f"R²={r2_af:.3f}",
        "threshold": f">= {QUALITY_THRESHOLDS['yield_af_r2']:.2f}",
        "status":    "OK" if r2_af >= QUALITY_THRESHOLDS["yield_af_r2"] else "WARN",
    })

    # Stage 2 — yield_TA row-by-row
    r2_ta = metrics_yield_ta.get("test_r2", float("nan"))
    rows.append({
        "stage":     "Stage2",
        "target":    "yield_TA (row-by-row)",
        "metric":    "R² test",
        "value":     f"R²={r2_ta:.3f}",
        "threshold": f">= {QUALITY_THRESHOLDS['yield_ta_r2']:.2f}",
        "status":    "OK" if r2_ta >= QUALITY_THRESHOLDS["yield_ta_r2"] else "WARN",
    })

    df_summary = pd.DataFrame(rows)
    log.info("\n--- QUALITY SUMMARY TABLE ---\n%s", df_summary.to_string(index=False))

    n_ok  = (df_summary["status"] == "OK").sum()
    n_tot = len(df_summary)
    log.info("\n%d / %d models meet their quality thresholds.", n_ok, n_tot)

    save_metrics(
        {"pipeline_validated": int(n_ok == n_tot), "n_ok": int(n_ok), "n_total": n_tot},
        campaign.metrics_dir / "pipeline_summary.json",
    )

    print_campaign_summary(campaign)
    log.info("STEP 7 complete — %.1fs", time.perf_counter() - t0)
    return df_summary


# =============================================================================
# STEP 8 — End-to-End Inference Test
# =============================================================================

def _step8_inference_test(
    horizon_models: dict,
    model_yield_af: Any,
    model_yield_ta: Any,
    clf1_fitted: Any,
    clf2_fitted: Any,
    stunted_model: dict,
    campaign: Any,
    example_params: dict,
) -> dict:
    """
    Run a complete end-to-end inference pass on a single example scenario.

    Validates that the predict_single_sim pipeline executes without error
    and produces non-trivial outputs for a nominal routing path. Saves a
    3-panel trajectory figure (carbonStem / yield_AF / yield_TA).

    Parameters
    ----------
    horizon_models  : dict  — {(target, h): model}  (Stage 1)
    model_yield_af  : fitted LGBMRegressor           (Stage 2, row-by-row)
    model_yield_ta  : fitted LGBMRegressor           (Stage 2, row-by-row)
    clf1_fitted, clf2_fitted : classifiers
    stunted_model   : dict  — conditional median fallback
    campaign        : CampaignPaths
    example_params  : dict  — user-supplied parameter set

    Returns
    -------
    result : dict  — output of predict_single_sim()
    """
    t0 = time.perf_counter()
    log.info("=" * 70)
    log.info("STEP 8 — End-to-End Inference Test (example scenario)")
    log.info("=" * 70)

    # Build unified models dict matching keys expected by predictor.py v5.0
    models_inf: dict = {}
    for (target, h), model_h in horizon_models.items():
        models_inf[f"{target}_h{h}"] = model_h     # Stage 1: "carbonStem_AF_h{h}"
    models_inf["yield_AF"] = model_yield_af         # Stage 2: flat key (row-by-row)
    models_inf["yield_TA"] = model_yield_ta         # Stage 2: flat key (row-by-row)

    log.info(
        "models_inf: %d total — Stage1(carbonStem): %d | "
        "Stage2(yield_AF): %d | Stage2(yield_TA): %d",
        len(models_inf),
        sum(1 for k in models_inf if k.startswith("carbonStem")),
        sum(1 for k in models_inf if k == "yield_AF"),
        sum(1 for k in models_inf if k == "yield_TA"),
    )

    result = predict_single_sim(
        params=example_params,
        models=models_inf,
        clf1=clf1_fitted,
        clf2=clf2_fitted,
        stunted_model=stunted_model,
        n_years=40,
        return_routing=True,
        log_transform_stage1=True,
    )

    log.info("Inference result:")
    log.info("  Population  : %s", result["population"])
    log.info("  tree_failed : %d", result["tree_failed"])
    log.info("  yield_failed: %d", result["yield_failed"])

    preds = result["predictions"]
    for target, arr in preds.items():
        if arr is not None and len(arr) > 0 and arr.max() > 0:
            log.info(
                "  %-25s: min=%.3f | mean=%.3f | max=%.3f",
                target, arr.min(), arr.mean(), arr.max(),
            )
        else:
            log.info(
                "  %-25s: all zeros (routing bypass or missing model)", target,
            )

    # 3-panel trajectory figure
    years = result["years"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1 — Stage 1: carbonStem trajectories
    for cs_t, color, label in [
        ("carbonStem_AF", "#2E86AB", "carbonStem AF"),
        ("carbonStem_TF", "#52B788", "carbonStem TF"),
    ]:
        if cs_t in preds and preds[cs_t].max() > 0:
            axes[0].plot(years, preds[cs_t], "-", color=color, label=label, lw=2)
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("carbonStem (kgC/tree)")
    axes[0].set_title(f"Stage 1 — carbonStem\n{result['population']}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Panel 2 — Stage 2: yield_AF
    if "yield_AF" in preds and preds["yield_AF"].max() > 0:
        axes[1].plot(
            years, preds["yield_AF"], "-",
            color="#E63946", label="yield_AF (row-by-row)", lw=2,
        )
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Yield (t/ha)")
    axes[1].set_title(f"Stage 2 — yield_AF\n{result['population']}")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Panel 3 — Stage 2: yield_TA
    if "yield_TA" in preds and preds["yield_TA"].max() > 0:
        axes[2].plot(
            years, preds["yield_TA"], "-",
            color="#F4A261", label="yield_TA (row-by-row)", lw=2,
        )
    axes[2].set_xlabel("Year")
    axes[2].set_ylabel("Yield (t/ha)")
    axes[2].set_title(f"Stage 2 — yield_TA\n{result['population']}")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(
        f"40-year trajectory predictions — {result['population']}",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    fig_path = campaign.diagnostics_dir / "step8_inference_test_trajectories.png"
    plt.savefig(fig_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Trajectory figure saved → %s", fig_path)

    df_out = format_output(result)
    log.info("format_output: %d rows × %d cols", *df_out.shape)

    log.info("STEP 8 complete — %.1fs", time.perf_counter() - t0)
    return result


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

def run_full_training(
    campaign_name: str,
    config_overrides: dict | None = None,
    run_shap: bool = True,
    run_inference: bool = True,
    excluded_crops: list[str] | None = None,
) -> dict[str, Any]:
    """
    MetAIsAFe full training pipeline — v5.0 (Stage 2 row-by-row).

    Executes STEP 1 through STEP 8 sequentially for a given campaign.
    All intermediate outputs are returned as a dict for notebook inspection.

    Parameters
    ----------
    campaign_name : str
        Campaign folder name (must exist under the 03_Models directory).
    config_overrides : dict, optional
        Override any default hyperparameter dict. Accepted keys:
            "params_clf1"                     — CLF1 hyperparameters
            "params_carbon_short_horizons"    — Stage 1 (h ≤ 10)
            "params_carbon_long_horizons"     — Stage 1 (h > 10)
            "params_yield_af"                 — Stage 2 yield_AF row-by-row
            "params_yield_ta"                 — Stage 2 yield_TA row-by-row
            "example_params"                  — STEP 8 inference scenario
    run_shap : bool, default True
        Enable STEP 6 (SHAP). Set False for fast debug runs.
    run_inference : bool, default True
        Enable STEP 8 (inference test).
    excluded_crops : list of str, optional
        Crop types to exclude in STEP 1. Default: ["rape"].

    Returns
    -------
    dict with keys:
        campaign, df, df_nominal, df_horizon_pop,
        train_df, test_df, train_wins, test_wins, X_train, X_test,
        win_bounds, clf1, clf2, clf1_eval, clf2_eval, stunted_model,
        horizon_models, horizon_datasets, horizon_results,
        model_yield_af, model_yield_ta, metrics_yield_af, metrics_yield_ta,
        df_summary, shap_results, inference_result
    """
    t_start = time.perf_counter()

    # ── Resolve configuration ─────────────────────────────────────────────
    cfg = config_overrides or {}
    params_clf1         = cfg.get("params_clf1",                PARAMS_CLF1_DEFAULT)
    params_carbon_short = cfg.get("params_carbon_short_horizons", PARAMS_CARBON_SHORT)
    params_carbon_long  = cfg.get("params_carbon_long_horizons",  PARAMS_CARBON_LONG)
    params_yield_af     = cfg.get("params_yield_af",            PARAMS_YIELD_AF_ROW)
    params_yield_ta     = cfg.get("params_yield_ta",            PARAMS_YIELD_TA_ROW)
    example_params      = cfg.get("example_params",             EXAMPLE_PARAMS_DEFAULT)

    # ── Campaign paths ────────────────────────────────────────────────────
    campaign = get_campaign_paths(campaign_name)
    setup_file_logging(reports_dir=campaign.reports_dir, campaign_name=campaign_name)

    log.info("MetAIsAFe full_training.py — v%s", __version__)
    log.info("Campaign : %s", campaign_name)
    log.info("Models   : %s", campaign.metamodels_dir)
    log.info("SHAP     : %s | Inference: %s", run_shap, run_inference)

    # Static features: non-climate, non-temporal — shared across Stage 1
    static_feats = [
        f for f in ACTIVE_FEATURES_B2
        if f != "Harvest_Year_Absolute" and f not in CLIMATE_FEATURES
    ]

    # ─────────────────────────────────────────────────────────────────────
    # STEP 1 — Data Loading & Preparation
    # ─────────────────────────────────────────────────────────────────────
    df = _step1_load_and_prepare(campaign, excluded_crops=excluded_crops)

    # ─────────────────────────────────────────────────────────────────────
    # STEP 2 — Population Filtering
    # ─────────────────────────────────────────────────────────────────────
    df_all_pops, df_nominal = _step2_filter_populations(df)

    # Horizon population: nominal SimIDs with sufficient final carbon stock
    # (MIN_CARBON_HORIZON filter applies to Stage 1 only)
    cs_final_nom   = (
        df_nominal.sort_values(["SimID", "Harvest_Year_Absolute"])
        .groupby("SimID")["carbonStem_AF"].last()
    )
    simids_horizon = cs_final_nom[cs_final_nom >= MIN_CARBON_HORIZON].index
    df_horizon_pop = df_nominal[df_nominal["SimID"].isin(simids_horizon)].copy()

    n_nom     = df_nominal["SimID"].nunique()
    n_horizon = len(simids_horizon)
    log.info(
        "Stage 1 horizon population: %d / %d SimIDs "
        "(carbonStem_AF(t=40) >= %.1f kgC/tree).",
        n_horizon, n_nom, MIN_CARBON_HORIZON,
    )
    if n_horizon < n_nom:
        log.info(
            "  %d SimIDs excluded from Stage 1 horizon training "
            "(carbonStem_AF(t=40) < %.1f — stunted fallback applied at inference).",
            n_nom - n_horizon, MIN_CARBON_HORIZON,
        )

    # ─────────────────────────────────────────────────────────────────────
    # STEP 3 — Train/Test Split + CV Groups
    # ─────────────────────────────────────────────────────────────────────
    train_df, test_df, train_sims, test_sims, groups, cv = _step3_split(df_nominal)

    log.info(
        "Stage 1 horizon population — effective split: "
        "train %d (%.1f%%) | test %d (%.1f%%)",
        sum(s in train_sims for s in simids_horizon),
        100.0 * sum(s in train_sims for s in simids_horizon) / max(n_horizon, 1),
        sum(s in test_sims  for s in simids_horizon),
        100.0 * sum(s in test_sims  for s in simids_horizon) / max(n_horizon, 1),
    )

    # ─────────────────────────────────────────────────────────────────────
    # STEP 4 — Categorical Encoding & Winsorization
    # ─────────────────────────────────────────────────────────────────────
    train_wins, test_wins, win_bounds, X_train, X_test = _step4_encode_and_winsorize(
        train_df, test_df, campaign,
    )

    # ── Build nominal row-by-row DataFrames for Stage 2 (STEP 5c) ────────
    # train_wins / test_wins are already filtered to the nominal population
    # (derived from train_df / test_df via _step3_split on df_nominal).
    # These are the authoritative row-level datasets for yield_AF / yield_TA.
    df_train_nominal = train_wins.copy()
    df_test_nominal  = test_wins.copy()

    # ─────────────────────────────────────────────────────────────────────
    # STEP 5a — Classifiers + Stunted Fallback Model
    # ─────────────────────────────────────────────────────────────────────
    (clf1, clf2,
     clf1_eval, clf2_eval,
     stunted_model,
     df_clf_train, df_clf_test) = _step5a_classifiers(
        df, df_nominal, campaign, params_clf1,
    )

    # Build CLF test feature matrices for STEP 6 SHAP
    X_clf1_test = build_classifier_features(df_clf_test, CLF1_FEATURES)
    X_clf2_test = build_classifier_features(df_clf_test, CLF2_FEATURES)
    y_clf1_test = build_tree_fail_labels_multiclass(df_clf_test)
    y_clf2_test = build_yield_fail_labels(df_clf_test)
    X_clf1_test = X_clf1_test.loc[X_clf1_test.index.intersection(y_clf1_test.index)]
    X_clf2_test = X_clf2_test.loc[X_clf2_test.index.intersection(y_clf2_test.index)]
    assert len(X_clf1_test) > 0, "X_clf1_test is empty after alignment."
    assert len(X_clf2_test) > 0, "X_clf2_test is empty after alignment."

    # ─────────────────────────────────────────────────────────────────────
    # STEP 5b — Stage 1: carbonStem Horizon Models
    # ─────────────────────────────────────────────────────────────────────
    horizon_models, horizon_datasets, horizon_results = _step5b_stage1_carbon(
        df_horizon_pop, train_sims, test_sims,
        campaign, params_carbon_short, params_carbon_long, static_feats,
    )

    # ─────────────────────────────────────────────────────────────────────
    # STEP 5c — Stage 2: yield_AF and yield_TA (row-by-row)
    # ─────────────────────────────────────────────────────────────────────
    model_yield_af, model_yield_ta, metrics_yield_af, metrics_yield_ta = \
        _step5c_stage2_yield(
            df_train_nominal, df_test_nominal,
            campaign, params_yield_af, params_yield_ta,
        )

    # Build Stage 2 test feature matrices for STEP 6 SHAP
    X_test_af = df_test_nominal[STEP2_AF_FEATURES]
    X_test_ta = df_test_nominal[STEP2_TA_FEATURES]

    # ─────────────────────────────────────────────────────────────────────
    # STEP 6 — SHAP Analysis
    # ─────────────────────────────────────────────────────────────────────
    shap_results: dict = {}
    if run_shap:
        shap_results = _step6_shap(
            horizon_models, horizon_datasets,
            model_yield_af, X_test_af,
            model_yield_ta, X_test_ta,
            clf1, clf2,
            X_clf1_test, X_clf2_test,
            test_sims, campaign,
        )
    else:
        log.info("STEP 6 skipped (run_shap=False).")

    # ─────────────────────────────────────────────────────────────────────
    # STEP 7 — Final Evaluation
    # ─────────────────────────────────────────────────────────────────────
    df_summary = _step7_evaluate(
        clf1_eval, clf2_eval,
        horizon_results,
        metrics_yield_af,
        metrics_yield_ta,
        campaign,
    )

    # ─────────────────────────────────────────────────────────────────────
    # STEP 8 — End-to-End Inference Test
    # ─────────────────────────────────────────────────────────────────────
    inference_result: dict = {}
    if run_inference:
        inference_result = _step8_inference_test(
            horizon_models,
            model_yield_af,
            model_yield_ta,
            clf1, clf2,
            stunted_model,
            campaign,
            example_params,
        )
    else:
        log.info("STEP 8 skipped (run_inference=False).")

    # ── Pipeline complete ─────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    log.info("=" * 70)
    log.info("PIPELINE COMPLETE — %.1fs (%.1f min)", elapsed, elapsed / 60)
    log.info("=" * 70)

    cleanup_empty_campaign_dirs(campaign=campaign)
    print_campaign_summary(campaign)

    return {
        # Infrastructure
        "campaign":          campaign,
        # Raw & filtered data
        "df":                df,
        "df_nominal":        df_nominal,
        "df_horizon_pop":    df_horizon_pop,
        # Train / test splits
        "train_df":          train_df,
        "test_df":           test_df,
        "train_wins":        train_wins,
        "test_wins":         test_wins,
        "X_train":           X_train,
        "X_test":            X_test,
        "win_bounds":        win_bounds,
        # Classifiers & fallback
        "clf1":              clf1,
        "clf2":              clf2,
        "clf1_eval":         clf1_eval,
        "clf2_eval":         clf2_eval,
        "stunted_model":     stunted_model,
        # Stage 1 — carbonStem horizon models
        "horizon_models":    horizon_models,
        "horizon_datasets":  horizon_datasets,
        "horizon_results":   horizon_results,
        # Stage 2 — yield row-by-row models
        "model_yield_af":    model_yield_af,
        "model_yield_ta":    model_yield_ta,
        "metrics_yield_af":  metrics_yield_af,
        "metrics_yield_ta":  metrics_yield_ta,
        # Outputs
        "df_summary":        df_summary,
        "shap_results":      shap_results,
        "inference_result":  inference_result,
    }


# =============================================================================
# CLI
# =============================================================================

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "MetAIsAFe — Full training pipeline B2 "
            "(Stage 2 row-by-row, v5.0)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python full_training.py --campaign sobol_training_1_n2048
  python full_training.py --campaign sobol_training_1_n2048 --no-shap
  python full_training.py --campaign sobol_training_1_n2048 --no-shap --no-inference
        """,
    )
    p.add_argument(
        "--campaign", "-c",
        required=True,
        help="Campaign name (folder under 03_Models/).",
    )
    p.add_argument(
        "--no-shap",
        action="store_true",
        default=False,
        help="Skip STEP 6 (SHAP analysis) — useful for fast debug runs.",
    )
    p.add_argument(
        "--no-inference",
        action="store_true",
        default=False,
        help="Skip STEP 8 (end-to-end inference test).",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console logging verbosity (default: INFO).",
    )
    return p


if __name__ == "__main__":
    parser = _build_arg_parser()
    args   = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    results = run_full_training(
        campaign_name=args.campaign,
        run_shap=not args.no_shap,
        run_inference=not args.no_inference,
    )

    print("\n" + "=" * 70)
    print("FINAL QUALITY SUMMARY")
    print("=" * 70)
    print(results["df_summary"].to_string(index=False))
    print("=" * 70)

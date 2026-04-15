"""
MetAIsAFe — full_training.py
==============================
Full training pipeline B2 — horizon architecture (v4.1).

Runs the complete workflow in a single pass:
    STEP 1  — Data loading & preparation
    STEP 2  — Population filtering
    STEP 3  — Train/test split + CV groups
    STEP 4  — Categorical encoding & winsorization
    STEP 5a — Classifiers: CLF1 (3-class tree status) + CLF2 (binary yield fail) + Stunted model
    STEP 5b — Stage 1  : carbonStem_AF_h{h} + carbonStem_TF_h{h}  (log1p, SimID-level)
    STEP 5c — Stage 2a : yield_AF_h{h}      (no log1p, carbonStem_AF_h{h} injected)
    STEP 5d — Stage 2b : yield_TA_h{h}      (no log1p, no tree injection — pure crop control)
    STEP 6  — SHAP analysis (h=40 for all horizon targets + classifiers)
    STEP 7  — Final evaluation + summary table
    STEP 8  — Inference test on an example scenario

Saved model artifacts
---------------------
    clf1_tree_fail.joblib              CLF1 — 3-class tree status
    clf2_yield_fail.joblib             CLF2 — binary yield failure
    stunted_model.joblib               Conditional median fallback for stunted trees
    winsorization_bounds.joblib        Winsorization bounds (fit on train only)
    lgbm_carbonStem_AF_h{h}.joblib     Stage 1  — × 5 horizons
    lgbm_carbonStem_TF_h{h}.joblib     Stage 1  — × 5 horizons
    lgbm_yield_AF_h{h}.joblib          Stage 2a — × 5 horizons (carbonStem_AF injected)
    lgbm_yield_TA_h{h}.joblib          Stage 2b — × 5 horizons (no tree features)

Architecture notes
------------------
    All regression targets are predicted at discrete horizons (h = 5, 10, 20, 30, 40)
    and interpolated to full 40-year trajectories via PCHIP in predictor.py.

    yield_TA is a pure-crop agricultural control (no agroforestry interactions).
    carbonStem features are therefore NEVER injected into yield_TA models.

CLI usage
---------
    python full_training.py --campaign sobol_training_1_n2048
    python full_training.py --campaign sobol_training_1_n2048 --no-shap --no-inference
    python full_training.py --help

Author  : Etienne SABY
Created : 2026-04
Version : 4.1 — full horizon alignment (carbonStem + yield_AF + yield_TA)
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
matplotlib.use("Agg")   # non-interactive backend for script execution
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
    MIN_ENRICH_HORIZON,
    YIELD_FAIL_THRESHOLD,
    YIELD_FAIL_RATE,
    CARBON_HORIZONS,
    LGB_PARAMS,
    CV_N_FOLDS,
    FIGURE_DPI,
    NO_INJECT_HORIZON,
)
from column_taxonomy import (
    ACTIVE_FEATURES_B2,
    CATEGORICAL_FEATURES_B2,
    CLIMATE_FEATURES,
    STOCK_TARGETS_MINIMAL,
    NOMINAL_POPULATION,
    CLIMATE_HORIZON_FEATURES,
)
from data.loader import load_data, encode_categoricals, build_dataset
from data.preparation import (
    add_derived_columns, filter_crops, clean,
    compute_effective_vars, filter_population, build_horizon_dataset,
)
from data.preprocessing import apply_winsorization
from data.splitter import (
    stratified_split_by_rotation, split_by_simid,
    build_cv_groups, make_group_kfold,
    summarise_rotations, get_rotation_signature,
)
from modeling.models import build_lgb, build_lgb_classifier, get_feature_importances
from modeling.trainer import train_final_model, train_classifier
from modeling.classifiers import (
    build_tree_fail_classifier,
    build_yield_fail_classifier,
    build_classifier_features,
    build_tree_fail_labels_multiclass,
    build_tree_degraded_labels,
    build_yield_fail_labels,
    evaluate_clf1_binary,
    evaluate_classifier,
    evaluate_classifier_multiclass,
    apply_geographic_rule,
    predict_routing,
    save_classifiers,
    CLF1_FEATURES, CLF2_FEATURES,
    TREE_STATUS_LABELS,
    TREE_BINARY_LABELS,
    TREE_BINARY_OK,
    TREE_BINARY_DEGRADED,
    STUNTED_PROBA_THRESHOLD,

)
from modeling.evaluator import (
    compute_metrics, plot_pred_vs_obs, plot_residuals,
)
from modeling.shap_analysis import (
    run_shap_analysis, run_shap_classifiers,
    plot_shap_by_target, export_shap_for_shiny,
    compute_shap_values, summarise_shap,
)
from modeling.predictor import (
    build_inference_grid, predict_single_sim, format_output,
)
from utils.io_utils import (
    save_model, save_metrics, save_cv_results,
    save_predictions, save_feature_importances,
    print_campaign_summary, setup_file_logging,
)

log = logging.getLogger("metaisafe.training")


# =============================================================================
# DEFAULT HYPERPARAMETERS
# All dicts are overridable via run_full_training(config_overrides=...).
# =============================================================================

# CLF1 — multiclass tree-status classifier
PARAMS_CLF1_DEFAULT: dict = {
    **LGB_PARAMS,
    "objective":         "multiclass",
    "num_class":         3,
    "metric":            "multi_logloss",
    "min_child_samples": 20,
    "reg_alpha":         0.3,
    "reg_lambda":        0.5,
}

PARAMS_CLF1_DEFAULT_V41: dict = {
    # Binary objective — tree_degraded (0) vs tree_ok (1)
    # No num_class needed for binary LightGBM
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
PARAMS_CARBON_HORIZON_DEFAULT: dict = {
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

PARAMS_CARBON_SHORT = {   # h = 5, 10
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

PARAMS_CARBON_LONG = {# h = 20, 30, 40
    **PARAMS_CARBON_HORIZON_DEFAULT
}

# Stage 2a — yield_AF horizon models (raw target space, carbonStem injected)
PARAMS_YIELD_AF_DEFAULT: dict = {
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

# Stage 2b — yield_TA horizon models (raw target space, no tree features)
PARAMS_YIELD_TA_DEFAULT: dict = {
    "n_estimators":      400,
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

# Quality thresholds per step — used in STEP 7 summary table
QUALITY_THRESHOLDS: dict = {
    "clf1_f1_macro":     0.70,
    "clf1_roc_auc":      0.85,
    "clf2_accuracy":     0.80,
    "carbon_r2":         0.60,
    "carbon_rho":        0.85,
    "yield_af_r2":       0.55,
    "yield_af_rho_h40":  0.80,
    "yield_ta_r2":       0.55,
    "yield_ta_rho":      0.75,
}

# =============================================================================
# STEP 1 — Data Loading & Preparation
# =============================================================================

def _step1_load_and_prepare(
    campaign: Any,
    excluded_crops: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load raw meta-table and apply preparation pipeline.

    Operations:
        load_data()           → raw parquet/csv
        add_derived_columns() → Harvest_Year_Absolute, Rotation
        filter_crops()        → drop excluded crop types
        clean()               → conservative NA / duplicate handling
        compute_effective_vars() → _eff_ analytical variables

    Parameters
    ----------
    campaign : CampaignPaths
    excluded_crops : list of str, optional
        Crop types to exclude. Default: ["rape"].

    Returns
    -------
    df : pd.DataFrame
        Fully prepared meta-table (all populations, all rows).
    """
    if excluded_crops is None:
        excluded_crops = ["rape"]

    t0 = time.perf_counter()
    log.info("=" * 70)
    log.info("STEP 1 — Data Loading & Preparation")
    log.info("=" * 70)

    # 1a. Raw data loading
    df_raw = load_data(campaign.raw_meta)
    log.info("Raw shape: %d rows × %d cols", *df_raw.shape)

    required_cols = (
        ["SimID", "Harvest_Year_AF",
         "carbonStem_AF", "carbonStem_TF", "yield_AF", "yield_TA"]
        + [f for f in ACTIVE_FEATURES_B2
           if f not in ["SimID", "Harvest_Year_Absolute"]]
    )
    missing = [c for c in required_cols if c not in df_raw.columns]
    if missing:
        raise ValueError(f"STEP 1: missing columns in meta-table: {missing}")

    # 1b. Derived columns
    df = add_derived_columns(df_raw, verbose=True)
    assert "Harvest_Year_Absolute" in df.columns, \
        "add_derived_columns did not create Harvest_Year_Absolute"
    assert df["Harvest_Year_Absolute"].isna().sum() == 0, \
        "Harvest_Year_Absolute contains NaN values"
    assert df["Harvest_Year_Absolute"].min() >= 1, \
        "Harvest_Year_Absolute minimum < 1"
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
        log.warning("Columns with >5%% NA:\n%s", na_high.to_string())
    else:
        log.info("No key column with >5%% NA.")

    # 1e. Effective (analytical) variables
    df = compute_effective_vars(df, verbose=True)
    eff_cols = [c for c in df.columns if "_eff_" in c]
    assert eff_cols, "compute_effective_vars produced no _eff_ columns"
    for col in eff_cols:
        assert (df[col] < 0).sum() == 0, \
            f"Negative values in effective variable {col}"

    log.info(
        "STEP 1 done in %.1fs — %d rows × %d cols",
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
    Classify all SimIDs into populations and extract the nominal population.

    Populations (defined by carbonStem_AF(t=40) and yield failure rate):
        yield_ok × tree_ok      → NOMINAL population (training target)
        yield_ok × tree_stunted → stunted fallback
        yield_ok × tree_failed  → CLF1 training data
        yield_fail × *          → CLF2 training data

    Parameters
    ----------
    df : pd.DataFrame  — full prepared meta-table

    Returns
    -------
    df_all_pops : pd.DataFrame  — full df with 'population' column attached
    df_nominal  : pd.DataFrame  — nominal population rows only
    """
    t0 = time.perf_counter()
    log.info("=" * 70)
    log.info("STEP 2 — Population Filtering")
    log.info("=" * 70)

    # Full population distribution (all SimIDs)
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

    # carbonStem_AF at t=40 — diagnostic thresholds
    cs40 = (
        df.sort_values(["SimID", "Harvest_Year_Absolute"])
        .groupby("SimID")["carbonStem_AF"].last()
    )
    log.info("carbonStem_AF(t=40) — %d SimIDs", len(cs40))
    log.info(
        "  < %.1f kgC  (tree_failed)    : %d (%.1f%%)",
        TREE_FAIL_THRESHOLD,
        (cs40 < TREE_FAIL_THRESHOLD).sum(),
        100 * (cs40 < TREE_FAIL_THRESHOLD).mean(),
    )
    log.info(
        "  [%.1f, %.1f[ (tree_stunted)  : %d (%.1f%%)",
        TREE_FAIL_THRESHOLD, TREE_STUNT_THRESHOLD,
        ((cs40 >= TREE_FAIL_THRESHOLD) & (cs40 < TREE_STUNT_THRESHOLD)).sum(),
        100 * ((cs40 >= TREE_FAIL_THRESHOLD) & (cs40 < TREE_STUNT_THRESHOLD)).mean(),
    )
    log.info(
        "  >= %.1f kgC  (tree_ok)       : %d (%.1f%%)",
        TREE_STUNT_THRESHOLD,
        (cs40 >= TREE_STUNT_THRESHOLD).sum(),
        100 * (cs40 >= TREE_STUNT_THRESHOLD).mean(),
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

    # Anti-contamination check: no tree_failed SimIDs in nominal population
    cs_last = (
        df_nominal.sort_values(["SimID", "Harvest_Year_Absolute"])
        .groupby("SimID")["carbonStem_AF"].last()
    )
    assert (cs_last >= TREE_FAIL_THRESHOLD).all(), (
        f"tree_failed SimIDs detected in nominal population! "
        f"min(carbonStem_AF)={cs_last.min():.3f}"
    )

    log.info(
        "Nominal population: %d rows | %d SimIDs",
        len(df_nominal), df_nominal["SimID"].nunique(),
    )
    log.info("STEP 2 done in %.1fs", time.perf_counter() - t0)
    return df_all_pops, df_nominal


# =============================================================================
# STEP 3 — Train/Test Split + CV Groups
# =============================================================================

def _step3_split(
    df_nominal: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, set, set, Any, Any]:
    """
    Split nominal population into train/test sets and build CV groups.

    Uses stratified split by rotation type (maize / wheat) to ensure
    balanced representation in both sets. GroupKFold on SimID prevents
    temporal leakage within cross-validation.

    Parameters
    ----------
    df_nominal : pd.DataFrame

    Returns
    -------
    train_df   : pd.DataFrame
    test_df    : pd.DataFrame
    train_sims : set of SimIDs
    test_sims  : set of SimIDs
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

    # Leakage guard — SimID-level
    overlap = train_sims & test_sims
    assert not overlap, f"SimID leakage: {len(overlap)} SimIDs shared between train and test"

    total_sims = df_nominal["SimID"].nunique()
    actual_test_ratio = len(test_sims) / total_sims
    assert abs(actual_test_ratio - 0.20) < 0.05, (
        f"Test ratio {actual_test_ratio:.2f} outside [0.15, 0.25]"
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
        log.info("Stratification check passed:\n%s", strat_df.to_string())

    # CV groups
    groups = build_cv_groups(train_df)
    cv     = make_group_kfold(n_splits=CV_N_FOLDS)
    assert len(groups) == len(train_df), \
        "groups length mismatch with train_df"
    assert groups.nunique() == len(train_sims), \
        "groups unique count mismatch with train SimIDs"

    # Per-fold leakage guard
    X_dummy = train_df[["SimID"]].reset_index(drop=True)
    y_dummy = pd.Series(0, index=range(len(train_df)))
    for fold_i, (tr_idx, val_idx) in enumerate(
        cv.split(X_dummy, y_dummy, groups=groups)
    ):
        sims_tr  = set(groups.iloc[tr_idx])
        sims_val = set(groups.iloc[val_idx])
        assert not (sims_tr & sims_val), \
            f"SimID leakage in CV fold {fold_i + 1}"

    log.info(
        "Train: %d SimIDs | Test: %d SimIDs | %d CV folds — no leakage detected",
        len(train_sims), len(test_sims), CV_N_FOLDS,
    )
    log.info("STEP 3 done in %.1fs", time.perf_counter() - t0)
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

    Encoding  : LightGBM native category dtype (no statistical fit).
    Winsorize : quantile [0.01, 0.99] clipping, fit on train, applied to test.
    Both operations strictly follow fit-on-train / apply-on-test to prevent
    any data leakage from the test set into preprocessing.

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
    X_train    : pd.DataFrame  — feature matrix (base, no stage1 preds)
    X_test     : pd.DataFrame  — feature matrix (base, no stage1 preds)
    """
    t0 = time.perf_counter()
    log.info("=" * 70)
    log.info("STEP 4 — Encoding & Winsorization")
    log.info("=" * 70)

    # Categorical encoding — fit on train, apply on test
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
                f"{col} not category dtype in train"
            assert test_df[col].dtype.name == "category", \
                f"{col} not category dtype in test"
    log.info("Categorical encoding (LightGBM native) validated.")

    # Winsorization — fit on train, apply on test
    train_wins, win_bounds = apply_winsorization(
        train_df, quantiles=(0.01, 0.99), fit=True, verbose=True,
    )
    test_wins, _ = apply_winsorization(
        test_df, quantiles=(0.01, 0.99), fit=False,
        bounds=win_bounds, verbose=True,
    )

    # Verify bounds are respected in train
    for col, (lo, hi) in win_bounds.items():
        if col in train_wins.columns:
            assert train_wins[col].min() >= lo - 1e-9, \
                f"{col}: min below lower bound after winsorization"
            assert train_wins[col].max() <= hi + 1e-9, \
                f"{col}: max above upper bound after winsorization"

    # Persist winsorization bounds (needed for inference)
    win_path = campaign.metamodels_dir / "winsorization_bounds.joblib"
    joblib.dump(win_bounds, win_path)
    log.info(
        "Winsorization bounds saved: %d columns → %s", len(win_bounds), win_path
    )

    # Base feature matrices (no stage1 predictions injected yet)
    X_train, _, _, _ = build_dataset(
        train_wins, features=ACTIVE_FEATURES_B2,
        targets=STOCK_TARGETS_MINIMAL, verbose=True,
    )
    X_test, _, _, _ = build_dataset(
        test_wins, features=ACTIVE_FEATURES_B2,
        targets=STOCK_TARGETS_MINIMAL, verbose=True,
    )
    assert X_train.shape[1] == X_test.shape[1], \
        "Feature count mismatch between X_train and X_test"
    assert list(X_train.columns) == list(X_test.columns), \
        "Feature column order mismatch between X_train and X_test"
    log.info(
        "X_train: %d × %d | X_test: %d × %d",
        *X_train.shape, *X_test.shape,
    )

    log.info("STEP 4 done in %.1fs", time.perf_counter() - t0)
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
    Train CLF1 (3-class tree status) + CLF2 (binary yield failure).
    Build conditional median fallback model for stunted trees.

    Both classifiers are trained on the FULL dataset (all populations),
    using a dedicated SimID-level random split (not the nominal split from STEP 3).

    Parameters
    ----------
    df         : pd.DataFrame  — full prepared meta-table (all populations)
    df_nominal : pd.DataFrame  — nominal population (for routing sanity check)
    campaign   : CampaignPaths
    params_clf1 : dict         — LightGBM hyperparameters for CLF1

    Returns
    -------
    clf1_fitted     : fitted LGBMClassifier (3-class)
    clf2_fitted     : fitted LGBMClassifier (binary) or None
    clf1_eval       : dict  — evaluation metrics for CLF1
    clf2_eval       : dict  — evaluation metrics for CLF2
    stunted_model   : dict  — conditional median fallback
    df_clf_train    : pd.DataFrame  — classifier training split (for stunted model)
    df_clf_test     : pd.DataFrame  — classifier test split (for SHAP in STEP 6)
    """
    t0 = time.perf_counter()
    log.info("=" * 70)
    log.info("STEP 5a — Classifiers + Stunted Fallback Model")
    log.info("=" * 70)

    # Dedicated SimID-level split on full dataset (all populations)
    df_clf_train, df_clf_test, _, _ = split_by_simid(
        df, test_size=0.20, random_state=RANDOM_STATE, verbose=True,
    )
    log.info(
        "Classifier split — train: %d SimIDs | test: %d SimIDs",
        df_clf_train["SimID"].nunique(), df_clf_test["SimID"].nunique(),
    )

    # ── CLF1 — Tree Status (3 classes: failed / stunted / ok) ─────────────
    log.info("\\n--- CLF1: Tree Status (binary v4.1 — tree_degraded / tree_ok) ---")
    clf1_model, _, _ = build_tree_fail_classifier(
        df, params=params_clf1, multiclass=False, verbose=True,  # ← False = binary v4.1
    )

    X_clf1_train = build_classifier_features(df_clf_train, CLF1_FEATURES)
    X_clf1_test  = build_classifier_features(df_clf_test,  CLF1_FEATURES)
    y_clf1_train = build_tree_degraded_labels(df_clf_train)   # ← binary v4.1
    y_clf1_test  = build_tree_degraded_labels(df_clf_test)    # ← binary v4.1

    # Index alignment
    X_clf1_train = X_clf1_train.loc[X_clf1_train.index.intersection(y_clf1_train.index)]
    y_clf1_train = y_clf1_train.loc[X_clf1_train.index]
    X_clf1_test  = X_clf1_test.loc[X_clf1_test.index.intersection(y_clf1_test.index)]
    y_clf1_test  = y_clf1_test.loc[X_clf1_test.index]

    assert len(X_clf1_test) > 0, "X_clf1_test empty after alignment"

    for i, label in enumerate(TREE_BINARY_LABELS):
        n = (y_clf1_train == i).sum()
        log.info("  CLF1 train — %-16s: %d (%.1f%%)", label, n, 100*n/len(y_clf1_train))

    clf1_fitted, _ = train_classifier(
        clf1_model, X_clf1_train, y_clf1_train,
        X_clf1_test, y_clf1_test, verbose=True,
    )
    y_clf1_pred  = clf1_fitted.predict(X_clf1_test)
    y_clf1_proba = clf1_fitted.predict_proba(X_clf1_test)[:, 1]  # P(tree_ok)
    clf1_eval    = evaluate_clf1_binary(
        y_clf1_test, y_clf1_pred, y_clf1_proba,
        classifier_name="CLF1 — Tree Status (binary v4.1)",
        verbose=True,
    )

    thr_f1  = QUALITY_THRESHOLDS.get("clf1_f1_macro", 0.70)
    thr_auc = QUALITY_THRESHOLDS.get("clf1_roc_auc",  0.85)
    if clf1_eval["f1_macro"] < thr_f1:
        log.warning(
            "CLF1 F1 macro below threshold (%.3f < %.2f)",
            clf1_eval["f1_macro"], thr_f1,
        )
    if clf1_eval.get("roc_auc", 1.0) < thr_auc:
        log.warning(
            "CLF1 ROC-AUC below threshold (%.3f < %.2f)",
            clf1_eval.get("roc_auc", 0.0), thr_auc,
        )
    else:
        log.info(
            "CLF1 — Accuracy: %.1f%%  F1 macro: %.3f  ROC-AUC: %.3f",
            clf1_eval["accuracy"] * 100,
            clf1_eval["f1_macro"],
            clf1_eval.get("roc_auc", float("nan")),
        )

    # ── CLF2 — Yield Failure (binary) ─────────────────────────────────────
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

    assert len(X_clf2_test) > 0, \
        "X_clf2_test is empty after index alignment"

    log.info(
        "CLF2 — train: %d SimIDs (%.1f%% failed) | test: %d SimIDs (%.1f%% failed)",
        len(y_clf2_train), 100.0 * y_clf2_train.mean(),
        len(y_clf2_test),  100.0 * y_clf2_test.mean(),
    )

    if clf2_model is None:
        # Geographic deterministic fallback
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

    thr2 = QUALITY_THRESHOLDS["clf2_accuracy"]
    if clf2_eval.get("accuracy", 1.0) < thr2:
        log.warning(
            "CLF2 accuracy below threshold (%.1f%% < %.0f%%)",
            clf2_eval.get("accuracy", 0) * 100, thr2 * 100,
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
            "n_classes":     3,
        },
        clf2_meta={
            "accuracy_test": clf2_eval.get("accuracy"),
            "f1_test":       clf2_eval.get("f1"),
            "roc_auc_test":  clf2_eval.get("roc_auc"),
            "n_train":       len(y_clf2_train),
            "n_test":        len(y_clf2_test),
        } if clf2_fitted is not None else None,
    )
    log.info("CLF1 + CLF2 saved.")

    # ── Routing sanity check on nominal population ─────────────────────────
    log.info("\n--- Routing sanity check (nominal population) ---")
    all_clf_feats = CLF1_FEATURES + [c for c in CLF2_FEATURES if c not in CLF1_FEATURES]
    X_nominal_clf = build_classifier_features(df_nominal, feature_cols=all_clf_feats)
    routing_check = predict_routing(X_nominal_clf, clf1_fitted, clf2_fitted)
    for i, label in enumerate(TREE_BINARY_LABELS):
        n   = (routing_check["tree_status"] == i).sum()
        pct = 100 * n / len(routing_check)
        log.info("  CLF1 predicts %-18s: %d (%.1f%%)", label, n, pct)

    # Misrouted = prédit tree_degraded alors que SimID est nominal (tree_ok ground truth)
    n_misrouted   = (routing_check["tree_status"] == TREE_BINARY_DEGRADED).sum()
    pct_misrouted = 100.0 * n_misrouted / len(routing_check)
    if pct_misrouted > 10.0:
        log.warning(
            "High misrouting rate on nominal population: %d / %d (%.1f%%)",
            n_misrouted, len(routing_check), pct_misrouted,
        )
    else:
        log.info(
            "Misrouted nominal SimIDs (predicted tree_degraded): %d / %d (%.1f%%)",
            n_misrouted, len(routing_check), pct_misrouted,
        )

    # Log moyen de P(tree_ok) sur la population nominale
    if "tree_ok_proba" in routing_check.columns:
        mean_ok_proba = routing_check["tree_ok_proba"].mean()
        log.info(
            "Mean P(tree_ok) on nominal population: %.3f  "
            "[stunted sub-routing threshold: %.2f]",
            mean_ok_proba, STUNTED_PROBA_THRESHOLD,
        )

    # ── Stunted fallback model ─────────────────────────────────────────────
    log.info("\n--- Stunted fallback model ---")
    cs_final_all = (
        df.sort_values(["SimID", "Harvest_Year_Absolute"])
        .groupby("SimID")["carbonStem_AF"].last()
    )
    mask_stunted   = (
        (cs_final_all >= TREE_FAIL_THRESHOLD) & (cs_final_all < TREE_STUNT_THRESHOLD)
    )
    simids_stunted = cs_final_all[mask_stunted].index
    df_stunted     = df[df["SimID"].isin(simids_stunted)].copy()
    log.info(
        "Stunted population: %d SimIDs  (carbonStem_AF(t=40) in [%.1f, %.1f[)",
        len(simids_stunted), TREE_FAIL_THRESHOLD, TREE_STUNT_THRESHOLD,
    )

    # Fit conditional median exclusively on train split of CLF (no test leakage)
    stunted_train_sims = set(df_clf_train["SimID"].unique()) & set(simids_stunted)
    df_stunted_train   = df_stunted[df_stunted["SimID"].isin(stunted_train_sims)]

    cs_final_stunted = (
        df_stunted_train.sort_values(["SimID", "Harvest_Year_Absolute"])
        .groupby("SimID").last()[["carbonStem_AF", "main_crop"]]
    )
    median_by_crop = cs_final_stunted.groupby("main_crop")["carbonStem_AF"].median()
    global_median  = float(cs_final_stunted["carbonStem_AF"].median())

    for crop, med in median_by_crop.items():
        log.info("  Stunted median  %-10s: %.3f kgC/tree", crop, med)
    log.info("  Stunted median  %-10s: %.3f kgC/tree", "global", global_median)

    # Normalized temporal profile (ratio = carbonStem_AF(t) / carbonStem_AF(t=40))
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
    log.info("Stunted model saved: %s", stunted_path)

    log.info("STEP 5a done in %.1fs", time.perf_counter() - t0)
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
) -> tuple[dict, dict, list, dict]:
    """
    Train 10 SimID-level carbonStem horizon models (5 horizons × 2 targets).

    Targets : carbonStem_AF, carbonStem_TF
    Horizons: CARBON_HORIZONS (default: 5, 10, 20, 30, 40)

    Target distribution is right-skewed (cumulative biomass over years),
    so targets are log1p-transformed before training and expm1-back-transformed
    for metric computation and downstream injection.

    Parameters
    ----------
    df_horizon_pop : pd.DataFrame  — nominal pop with min carbon filter applied
    train_sims     : set of SimIDs
    test_sims      : set of SimIDs
    campaign       : CampaignPaths
    params_carbon_short  : dict  — LightGBM hyperparameters for short Horizons (<=10)
    params_carbon_long   : dict  — LightGBM hyperparameters for long Horizons (>10)
    static_feats   : list[str]  — non-climate, non-temporal features

    Returns
    -------
    horizon_models      : dict  {(target, h): fitted LGBMRegressor}
    horizon_datasets    : dict  {(target, h): (X_h, y_h)}
    horizon_results     : list of metric dicts
    cs_af_preds_by_simid: dict  {simid_str: {h: pred_in_original_space}}
                          Used to inject carbonStem_AF into yield_AF models (STEP 5c).
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
            # Encode categoricals — fit on full horizon dataset
            # (LightGBM native: astype("category"), no statistical leakage)
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
                log.warning("  Insufficient data — skipping.")
                continue

            # log1p transform (right-skewed cumulative biomass distribution)
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
            thr_r2  = QUALITY_THRESHOLDS["carbon_r2"]
            thr_rho = QUALITY_THRESHOLDS["carbon_rho"]
            if m["test_r2"] < thr_r2:
                log.warning("  R²=%.3f below threshold (%.2f)", m["test_r2"], thr_r2)
            if h == 40 and rho < thr_rho:
                log.warning(
                    "  rho=%.3f below threshold (%.2f) at h=40", rho, thr_rho
                )

            model_key  = f"{target}_h{h}"
            model_path = campaign.metamodels_dir / f"lgbm_{model_key}.joblib"
            save_model(
                model_h_fitted, model_path,
                metadata={
                    "target":        target,
                    "horizon":       h,
                    "model_key":     model_key,
                    "log_transform": True,
                    "cs_af_injected": False,
                    "feature_names": list(X_h.columns),
                    "n_features":    X_h.shape[1],
                    "r2_test":       m["test_r2"],
                    "spearman_rho":  rho,
                },
            )

            horizon_models[(target, h)]   = model_h_fitted
            horizon_datasets[(target, h)] = (X_h, y_h)
            horizon_results.append({
                "target": target, "horizon": h,
                "n_train": n_train, "n_test": n_test,
                "r2": m["test_r2"], "rmse": m["test_rmse"],
                "spearman_rho": rho, "spearman_p": p,
            })

    df_hr = pd.DataFrame(horizon_results).sort_values(["target", "horizon"])
    log.info(
        "\nStage 1 — %d carbonStem models trained:\n%s",
        len(horizon_results),
        df_hr[["target", "horizon", "r2", "rmse", "spearman_rho"]].to_string(index=False),
    )

    # Reconstruct Stage 1 predictions in original space for yield_AF injection (STEP 5c)
    cs_af_preds_by_simid: dict = {}
    for h in CARBON_HORIZONS:
        if ("carbonStem_AF", h) not in horizon_models:
            continue
        model_h      = horizon_models[("carbonStem_AF", h)]
        X_h_full, _  = horizon_datasets[("carbonStem_AF", h)]
        preds_orig   = np.clip(np.expm1(model_h.predict(X_h_full)), 0.0, None)
        for sim_id, pred_val in zip(X_h_full.index, preds_orig):
            sim_str = str(sim_id)
            if sim_str not in cs_af_preds_by_simid:
                cs_af_preds_by_simid[sim_str] = {}
            cs_af_preds_by_simid[sim_str][h] = float(pred_val)

    log.info(
        "Stage 1 predictions reconstructed: %d SimIDs × %d horizons",
        len(cs_af_preds_by_simid), len(CARBON_HORIZONS),
    )
    log.info("STEP 5b done in %.1fs", time.perf_counter() - t0)
    return horizon_models, horizon_datasets, horizon_results, cs_af_preds_by_simid


# =============================================================================
# STEP 5c — Stage 2a: yield_AF Horizon Models (carbonStem_AF injected)
# =============================================================================

def _step5c_stage2a_yield_af(
    df_horizon_pop: pd.DataFrame,
    cs_af_preds_by_simid: dict,
    train_sims: set,
    test_sims: set,
    campaign: Any,
    params_yield_af: dict,
    static_feats: list[str],
    no_inject_horizon: int = NO_INJECT_HORIZON,
) -> tuple[dict, dict, list]:
    """
    Train 5 SimID-level yield_AF horizon models.

    Architecture:
        Features = static + CLIMATE aggregated over [1 → h]
                   + carbonStem_AF_h{h}  (Stage 1 prediction, original space)
        Target   = yield_AF at exact year h  (raw, no log1p)

    carbonStem_AF injection rationale: in agroforestry systems, tree biomass
    drives light and water competition with the associated crop. Injecting the
    Stage 1 carbonStem_AF prediction provides the model with an explicit
    mechanistic signal that static features alone cannot capture.

    Parameters
    ----------
    df_horizon_pop       : pd.DataFrame
    cs_af_preds_by_simid : dict  — {simid_str: {h: carbonStem_AF pred}}
    train_sims, test_sims: sets of SimIDs
    campaign             : CampaignPaths
    params_yield_af      : dict  — LightGBM hyperparameters
    static_feats         : list[str]

    Returns
    -------
    yield_af_models   : dict  {h: fitted LGBMRegressor}
    yield_af_datasets : dict  {h: (X_h, y_h)}
    yield_af_results  : list of metric dicts
    """
    t0 = time.perf_counter()
    log.info("=" * 70)
    log.info("STEP 5c — Stage 2a: yield_AF Horizon Models (carbonStem_AF injected)")
    log.info("=" * 70)

    yield_af_models:   dict = {}
    yield_af_datasets: dict = {}
    yield_af_results:  list = []

    for h in CARBON_HORIZONS:
        log.info("\n--- yield_AF | h=%d ---", h)

        # Extract yield_AF at exact year h (SimID-level target)
        df_at_h = (
            df_horizon_pop[df_horizon_pop["Harvest_Year_Absolute"] == h]
            [["SimID", "yield_AF"]]
            .dropna(subset=["yield_AF"])
            .set_index("SimID")
        )
        df_at_h.index = df_at_h.index.astype(str)
        y_h = df_at_h["yield_AF"].copy()

        if len(y_h) < 30:
            log.warning("  Too few SimIDs (n=%d) at h=%d — skipping.", len(y_h), h)
            continue

        log.info(
            "  yield_AF at h=%d — %d SimIDs | mean=%.3f  std=%.3f",
            h, len(y_h), y_h.mean(), y_h.std(),
        )

        # Build base feature matrix via build_horizon_dataset
        X_base_h, _ = build_horizon_dataset(
            df_horizon_pop,
            horizon=h,
            target_col="yield_AF",
            feature_cols=static_feats,
            min_final_carbon=None,
            min_enrich_horizon=MIN_ENRICH_HORIZON,
        )
        X_base_h.index = X_base_h.index.astype(str)

        # Inject carbonStem_AF_h{h} (Stage 1 output — original space)
        # Only for h > no_inject_horizon
        X_h = X_base_h.copy()

        inject_cs = (h > no_inject_horizon)
        if inject_cs:
            cs_col  = f"carbonStem_AF_h{h}"
            cs_vals = pd.Series(
                {sim: cs_af_preds_by_simid.get(sim, {}).get(h, 0.0)
                 for sim in X_base_h.index},
                name=cs_col,
            )
            X_h[cs_col] = cs_vals
            log.info(
                "  h=%d: carbonStem_AF injected (%s mean=%.3f  std=%.3f)",
                h, cs_col, X_h[cs_col].mean(), X_h[cs_col].std(),
            )
        else:
            cs_col   = None
            inject_cs = False
            log.info(
                "  h=%d: carbonStem_AF injection SKIPPED "
                "(h <= no_inject_horizon=%d) — Levier 3 ablation",
                h, no_inject_horizon,
            )

        # Encode categoricals
        X_h, _ = encode_categoricals(
            X_h, features=CATEGORICAL_FEATURES_B2,
            fit=True, method="lightgbm", verbose=False,
        )

        # Align SimIDs
        common = X_h.index.intersection(y_h.index)
        X_h    = X_h.loc[common]
        y_h    = y_h.loc[common]
        log.info(
            "  After alignment: %d SimIDs | cs_injected=%s",
            len(common), inject_cs,
        )

        # Train/test split
        X_train_h = X_h[X_h.index.isin([str(s) for s in train_sims])]
        y_train_h = y_h[y_h.index.isin([str(s) for s in train_sims])]
        X_test_h  = X_h[X_h.index.isin([str(s) for s in test_sims])]
        y_test_h  = y_h[y_h.index.isin([str(s) for s in test_sims])]
        n_train, n_test = len(y_train_h), len(y_test_h)
        log.info("  Train: %d | Test: %d", n_train, n_test)

        if n_train < 30 or n_test < 10:
            log.warning("  Insufficient data — skipping.")
            continue

        # Train without log1p (yield_AF distribution ~ symmetric in [0, 12.5] t/ha)
        cat_feats_h    = [c for c in CATEGORICAL_FEATURES_B2 if c in X_train_h.columns]
        model_yh       = build_lgb(params=params_yield_af, categorical_feature=cat_feats_h)
        model_yh_fitted, _ = train_final_model(
            model_yh, X_train_h, y_train_h,
            X_test_h, y_test_h, verbose=False,
        )

        y_pred_h = np.clip(model_yh_fitted.predict(X_test_h), 0.0, None)
        m_h      = compute_metrics(y_test_h.values, y_pred_h, prefix="test_")
        rho_h, p = spearmanr(y_test_h.values, y_pred_h)

        log.info(
            "  yield_AF h=%d — R²=%.3f | RMSE=%.4f | rho=%.3f (p=%.1e)",
            h, m_h["test_r2"], m_h["test_rmse"], rho_h, p,
        )
        if m_h["test_r2"] < QUALITY_THRESHOLDS["yield_af_r2"]:
            log.warning(
                "  R²=%.3f below threshold (%.2f)",
                m_h["test_r2"], QUALITY_THRESHOLDS["yield_af_r2"],
            )
        if h == 40 and rho_h < QUALITY_THRESHOLDS["yield_af_rho_h40"]:
            log.warning(
                "  rho=%.3f below h=40 threshold (%.2f)",
                rho_h, QUALITY_THRESHOLDS["yield_af_rho_h40"],
            )

        model_key  = f"yield_AF_h{h}"
        save_model(
            model_yh_fitted,
            campaign.metamodels_dir / f"lgbm_{model_key}.joblib",
            metadata={
                "target":         "yield_AF",
                "horizon":        h,
                "model_key":      model_key,
                "log_transform":  False,
                "cs_af_injected": inject_cs,
                "no_inject_horizon": no_inject_horizon,
                "feature_names":  list(X_h.columns),
                "n_features":     X_h.shape[1],
                "r2_test":        m_h["test_r2"],
                "spearman_rho":   rho_h,
            },
        )

        yield_af_models[h]   = model_yh_fitted
        yield_af_datasets[h] = (X_h, y_h)
        yield_af_results.append({
            "horizon": h, "n_train": n_train, "n_test": n_test,
            "r2": m_h["test_r2"], "rmse": m_h["test_rmse"],
            "mae": m_h["test_mae"], "bias": m_h["test_bias"],
            "spearman_rho": rho_h, "spearman_p": p,
            "cs_af_injected": inject_cs, "no_inject_horizon": no_inject_horizon,
        })

    df_res = pd.DataFrame(yield_af_results).sort_values("horizon")
    log.info(
        "\nStage 2a — %d yield_AF models trained:\n%s",
        len(yield_af_results),
        df_res[["horizon", "r2", "rmse", "spearman_rho"]].to_string(index=False),
    )
    log.info("STEP 5c done in %.1fs", time.perf_counter() - t0)
    return yield_af_models, yield_af_datasets, yield_af_results


# =============================================================================
# STEP 5d — Stage 2b: yield_TA Horizon Models (no tree features)
# =============================================================================

def _step5d_stage2b_yield_ta(
    df_horizon_pop: pd.DataFrame,
    train_sims: set,
    test_sims: set,
    campaign: Any,
    params_yield_ta: dict,
    static_feats: list[str],
) -> tuple[dict, dict, list]:
    """
    Train 5 SimID-level yield_TA horizon models.

    Architecture:
        Features = static + CLIMATE aggregated over [1 → h]
                   (NO carbonStem — yield_TA is a pure-crop agricultural control)
        Target   = yield_TA at exact year h  (raw, no log1p)

    yield_TA represents the sole-crop yield in the reference agricultural
    scenario (no agroforestry). It is mechanistically independent of tree
    biomass, so no tree-derived features are injected.

    Parameters
    ----------
    df_horizon_pop : pd.DataFrame
    train_sims, test_sims : sets of SimIDs
    campaign       : CampaignPaths
    params_yield_ta: dict  — LightGBM hyperparameters
    static_feats   : list[str]

    Returns
    -------
    yield_ta_models   : dict  {h: fitted LGBMRegressor}
    yield_ta_datasets : dict  {h: (X_h, y_h)}
    yield_ta_results  : list of metric dicts
    """
    t0 = time.perf_counter()
    log.info("=" * 70)
    log.info("STEP 5d — Stage 2b: yield_TA Horizon Models (no tree features)")
    log.info("=" * 70)

    yield_ta_models:   dict = {}
    yield_ta_datasets: dict = {}
    yield_ta_results:  list = []

    for h in CARBON_HORIZONS:
        log.info("\n--- yield_TA | h=%d ---", h)

        # Extract yield_TA at exact year h (SimID-level target)
        df_at_h = (
            df_horizon_pop[df_horizon_pop["Harvest_Year_Absolute"] == h]
            [["SimID", "yield_TA"]]
            .dropna(subset=["yield_TA"])
            .set_index("SimID")
        )
        df_at_h.index = df_at_h.index.astype(str)
        y_h = df_at_h["yield_TA"].copy()

        if len(y_h) < 30:
            log.warning("  Too few SimIDs (n=%d) at h=%d — skipping.", len(y_h), h)
            continue

        log.info(
            "  yield_TA at h=%d — %d SimIDs | mean=%.3f  std=%.3f",
            h, len(y_h), y_h.mean(), y_h.std(),
        )

        # Build feature matrix — static + aggregated climate, NO carbonStem
        X_base_h, _ = build_horizon_dataset(
            df_horizon_pop,
            horizon=h,
            target_col="yield_TA",
            feature_cols=static_feats,
            min_final_carbon=None,
            min_enrich_horizon=MIN_ENRICH_HORIZON,
        )
        X_base_h.index = X_base_h.index.astype(str)
        X_h = X_base_h.copy()

        # Encode categoricals
        X_h, _ = encode_categoricals(
            X_h, features=CATEGORICAL_FEATURES_B2,
            fit=True, method="lightgbm", verbose=False,
        )

        # Align SimIDs
        common = X_h.index.intersection(y_h.index)
        X_h    = X_h.loc[common]
        y_h    = y_h.loc[common]

        # Train/test split
        X_train_h = X_h[X_h.index.isin([str(s) for s in train_sims])]
        y_train_h = y_h[y_h.index.isin([str(s) for s in train_sims])]
        X_test_h  = X_h[X_h.index.isin([str(s) for s in test_sims])]
        y_test_h  = y_h[y_h.index.isin([str(s) for s in test_sims])]
        n_train, n_test = len(y_train_h), len(y_test_h)
        log.info("  Train: %d | Test: %d", n_train, n_test)

        if n_train < 30 or n_test < 10:
            log.warning("  Insufficient data — skipping.")
            continue

        # Train without log1p (yield_TA distribution ~ symmetric)
        cat_feats_h     = [c for c in CATEGORICAL_FEATURES_B2 if c in X_train_h.columns]
        model_ta_h      = build_lgb(params=params_yield_ta, categorical_feature=cat_feats_h)
        model_ta_fitted, _ = train_final_model(
            model_ta_h, X_train_h, y_train_h,
            X_test_h, y_test_h, verbose=False,
        )

        y_pred_h = np.clip(model_ta_fitted.predict(X_test_h), 0.0, None)
        m_h      = compute_metrics(y_test_h.values, y_pred_h, prefix="test_")
        rho_h, p = spearmanr(y_test_h.values, y_pred_h)

        log.info(
            "  yield_TA h=%d — R²=%.3f | RMSE=%.4f | rho=%.3f (p=%.1e)",
            h, m_h["test_r2"], m_h["test_rmse"], rho_h, p,
        )
        if m_h["test_r2"] < QUALITY_THRESHOLDS["yield_ta_r2"]:
            log.warning(
                "  R²=%.3f below threshold (%.2f)",
                m_h["test_r2"], QUALITY_THRESHOLDS["yield_ta_r2"],
            )
        if rho_h < QUALITY_THRESHOLDS["yield_ta_rho"]:
            log.warning(
                "  rho=%.3f below threshold (%.2f)",
                rho_h, QUALITY_THRESHOLDS["yield_ta_rho"],
            )

        model_key = f"yield_TA_h{h}"
        save_model(
            model_ta_fitted,
            campaign.metamodels_dir / f"lgbm_{model_key}.joblib",
            metadata={
                "target":         "yield_TA",
                "horizon":        h,
                "model_key":      model_key,
                "log_transform":  False,
                "cs_af_injected": False,
                "feature_names":  list(X_h.columns),
                "n_features":     X_h.shape[1],
                "r2_test":        m_h["test_r2"],
                "spearman_rho":   rho_h,
            },
        )

        yield_ta_models[h]   = model_ta_fitted
        yield_ta_datasets[h] = (X_h, y_h)
        yield_ta_results.append({
            "horizon": h, "n_train": n_train, "n_test": n_test,
            "r2": m_h["test_r2"], "rmse": m_h["test_rmse"],
            "spearman_rho": rho_h, "spearman_p": p,
            "cs_af_injected": False,
        })

    df_res = pd.DataFrame(yield_ta_results).sort_values("horizon")
    log.info(
        "\nStage 2b — %d yield_TA models trained:\n%s",
        len(yield_ta_results),
        df_res[["horizon", "r2", "rmse", "spearman_rho"]].to_string(index=False),
    )
    log.info("STEP 5d done in %.1fs", time.perf_counter() - t0)
    return yield_ta_models, yield_ta_datasets, yield_ta_results


# =============================================================================
# STEP 6 — SHAP Analysis
# =============================================================================

def _step6_shap(
    horizon_models: dict,
    horizon_datasets: dict,
    yield_af_models: dict,
    yield_af_datasets: dict,
    yield_ta_models: dict,
    yield_ta_datasets: dict,
    clf1_fitted: Any,
    clf2_fitted: Any,
    X_clf1_test: pd.DataFrame,
    X_clf2_test: pd.DataFrame,
    test_sims: set,
    campaign: Any,
) -> dict:
    """
    Compute and export SHAP values for all models at h=40.

    Targets analysed:
        - carbonStem_AF h=40  (Stage 1)
        - carbonStem_TF h=40  (Stage 1)
        - yield_AF h=40       (Stage 2a)
        - yield_TA h=40       (Stage 2b)
        - CLF1 + CLF2         (classifiers)

    Parameters
    ----------
    horizon_models    : dict  {(target, h): model}
    horizon_datasets  : dict  {(target, h): (X_h, y_h)}
    yield_af_models   : dict  {h: model}
    yield_af_datasets : dict  {h: (X_h, y_h)}
    yield_ta_models   : dict  {h: model}
    yield_ta_datasets : dict  {h: (X_h, y_h)}
    clf1_fitted, clf2_fitted : classifiers
    X_clf1_test, X_clf2_test : pd.DataFrame  — from STEP 5a
    test_sims  : set of SimIDs
    campaign   : CampaignPaths

    Returns
    -------
    all_shap : dict  — consolidated SHAP results for all targets
    """
    t0 = time.perf_counter()
    log.info("=" * 70)
    log.info("STEP 6 — SHAP Analysis (h=40 for all horizon targets)")
    log.info("=" * 70)

    all_shap: dict = {}

    # ── Stage 1 — carbonStem h=40 ─────────────────────────────────────────
    log.info("\n--- SHAP: Stage 1 carbonStem (h=40) ---")
    shap_stage1: dict = {}
    for cs_target in ["carbonStem_AF", "carbonStem_TF"]:
        key = f"{cs_target}_h40"
        if (cs_target, 40) not in horizon_models:
            log.warning("Model (%s, 40) not found — skipping SHAP.", cs_target)
            continue
        model_h40      = horizon_models[(cs_target, 40)]
        X_h40_full, _  = horizon_datasets[(cs_target, 40)]
        X_h40_test     = X_h40_full[X_h40_full.index.isin(test_sims)]
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
        plot_shap_by_target(shap_stage1, save_dir=campaign.shap_dir,
                            plot_type="both", max_display=15)
        all_shap.update(shap_stage1)

    # ── Stage 2a — yield_AF h=40 ──────────────────────────────────────────
    log.info("\n--- SHAP: Stage 2a yield_AF (h=40) ---")
    if 40 in yield_af_models:
        X_yaf40_full, _ = yield_af_datasets[40]
        X_yaf40_test    = X_yaf40_full[
            X_yaf40_full.index.isin([str(s) for s in test_sims])
        ]
        if len(X_yaf40_test) > 0:
            sv_af, Xs_af = compute_shap_values(
                yield_af_models[40], X_yaf40_test,
                max_samples=2000, random_state=RANDOM_STATE,
            )
            shap_yaf40 = {
                "yield_AF_h40": {
                    "shap_values":   sv_af,
                    "X_sample":      Xs_af,
                    "summary_df":    summarise_shap(sv_af, list(Xs_af.columns)),
                    "feature_names": list(Xs_af.columns),
                }
            }
            plot_shap_by_target(shap_yaf40, save_dir=campaign.shap_dir,
                                plot_type="both", max_display=15)
            all_shap.update(shap_yaf40)
        else:
            log.warning("No test samples for yield_AF h=40 — skipping SHAP.")
    else:
        log.warning("yield_AF h=40 model not available — skipping SHAP.")

    # ── Stage 2b — yield_TA h=40 ──────────────────────────────────────────
    log.info("\n--- SHAP: Stage 2b yield_TA (h=40) ---")
    if 40 in yield_ta_models:
        X_yta40_full, _ = yield_ta_datasets[40]
        X_yta40_test    = X_yta40_full[
            X_yta40_full.index.isin([str(s) for s in test_sims])
        ]
        if len(X_yta40_test) > 0:
            sv_ta, Xs_ta = compute_shap_values(
                yield_ta_models[40], X_yta40_test,
                max_samples=2000, random_state=RANDOM_STATE,
            )
            shap_yta40 = {
                "yield_TA_h40": {
                    "shap_values":   sv_ta,
                    "X_sample":      Xs_ta,
                    "summary_df":    summarise_shap(sv_ta, list(Xs_ta.columns)),
                    "feature_names": list(Xs_ta.columns),
                }
            }
            plot_shap_by_target(shap_yta40, save_dir=campaign.shap_dir,
                                plot_type="both", max_display=15)
            all_shap.update(shap_yta40)
        else:
            log.warning("No test samples for yield_TA h=40 — skipping SHAP.")
    else:
        log.warning("yield_TA h=40 model not available — skipping SHAP.")

    # ── Classifiers ───────────────────────────────────────────────────────
    log.info("\n--- SHAP: Classifiers ---")
    try:
        shap_clf = run_shap_classifiers(
            clf1=clf1_fitted, clf2=clf2_fitted,
            X_clf1=X_clf1_test, X_clf2=X_clf2_test,
            random_state=RANDOM_STATE,
        )
        plot_shap_by_target(shap_clf, save_dir=campaign.shap_dir,
                            plot_type="bar", max_display=10)
        all_shap.update(shap_clf)
    except Exception as exc:
        log.warning("Classifier SHAP failed: %s", exc)

    # ── Export for R/Shiny ────────────────────────────────────────────────
    export_path = export_shap_for_shiny(
        all_shap, campaign.shap_data_dir, include_raw=False,
    )
    log.info("SHAP exported: %s", export_path)
    log.info("STEP 6 done in %.1fs", time.perf_counter() - t0)
    return all_shap


# =============================================================================
# STEP 7 — Final Evaluation & Summary Table
# =============================================================================

def _step7_evaluate(
    clf1_eval: dict,
    clf2_eval: dict,
    horizon_results: list,
    yield_af_results: list,
    yield_ta_results: list,
    campaign: Any,
) -> pd.DataFrame:
    """
    Build quality summary table and persist pipeline-level metrics.

    Quality thresholds (from QUALITY_THRESHOLDS):
        CLF1 F1 macro     >= 0.50
        CLF2 Accuracy     >= 0.80
        carbonStem R²     >= 0.60  rho >= 0.85
        yield_AF R²       >= 0.55  rho >= 0.80 (h=40 only)
        yield_TA R²       >= 0.55  rho >= 0.75

    Parameters
    ----------
    clf1_eval        : dict  — from STEP 5a
    clf2_eval        : dict  — from STEP 5a
    horizon_results  : list  — from STEP 5b
    yield_af_results : list  — from STEP 5c
    yield_ta_results : list  — from STEP 5d
    campaign         : CampaignPaths

    Returns
    -------
    df_summary : pd.DataFrame
    """
    t0 = time.perf_counter()
    log.info("=" * 70)
    log.info("STEP 7 — Final Evaluation")
    log.info("=" * 70)

    rows: list = []

    # CLF1
    rows.append({
        "stage":     "CLF1",
        "target":    "tree_status (binary v4.1)",
        "metric":    "F1 macro | ROC-AUC",
        "value":     (
            f"F1={clf1_eval['f1_macro']:.3f} | "
            f"AUC={clf1_eval.get('roc_auc', float('nan')):.3f}"
        ),
        "threshold": (
            f"F1>={QUALITY_THRESHOLDS.get('clf1_f1_macro', 0.70):.2f} | "
            f"AUC>={QUALITY_THRESHOLDS.get('clf1_roc_auc', 0.85):.2f}"
        ),
        "status": (
            "OK"
            if clf1_eval["f1_macro"] >= QUALITY_THRESHOLDS.get("clf1_f1_macro", 0.70)
            and clf1_eval.get("roc_auc", 0.0) >= QUALITY_THRESHOLDS.get("clf1_roc_auc", 0.85)
            else "WARN"
        ),
    })

    # CLF2
    rows.append({
        "stage":     "CLF2",
        "target":    "yield_failed",
        "metric":    "Accuracy",
        "value":     f"{clf2_eval.get('accuracy', 0):.3f}",
        "threshold": f">= {QUALITY_THRESHOLDS['clf2_accuracy']:.2f}",
        "status":    "OK" if clf2_eval.get("accuracy", 0) >= QUALITY_THRESHOLDS["clf2_accuracy"] else "WARN",
    })

    # Stage 1 — carbonStem horizons
    for row in horizon_results:
        r2, rho = row["r2"], row["spearman_rho"]
        ok = (r2 >= QUALITY_THRESHOLDS["carbon_r2"] and
              rho >= QUALITY_THRESHOLDS["carbon_rho"])
        rows.append({
            "stage":     "Stage1",
            "target":    f"{row['target']} h={row['horizon']}",
            "metric":    "R2 | rho",
            "value":     f"R2={r2:.3f} | rho={rho:.3f}",
            "threshold": f"R2>={QUALITY_THRESHOLDS['carbon_r2']} | rho>={QUALITY_THRESHOLDS['carbon_rho']}",
            "status":    "OK" if ok else "WARN",
        })

    # Stage 2a — yield_AF horizons
    for row in yield_af_results:
        r2, rho, h = row["r2"], row["spearman_rho"], row["horizon"]
        rho_thr = QUALITY_THRESHOLDS["yield_af_rho_h40"] if h == 40 else 0.0
        ok = (r2 >= QUALITY_THRESHOLDS["yield_af_r2"] and
              (rho >= rho_thr if h == 40 else True))
        rows.append({
            "stage":     "Stage2a",
            "target":    f"yield_AF h={h}",
            "metric":    "R2 | rho",
            "value":     f"R2={r2:.3f} | rho={rho:.3f}",
            "threshold": (
                f"R2>={QUALITY_THRESHOLDS['yield_af_r2']} | rho>={rho_thr} (h=40)"
                if h == 40 else
                f"R2>={QUALITY_THRESHOLDS['yield_af_r2']}"
            ),
            "status":    "OK" if ok else "WARN",
        })

    # Stage 2b — yield_TA horizons
    for row in yield_ta_results:
        r2, rho = row["r2"], row["spearman_rho"]
        ok = (r2  >= QUALITY_THRESHOLDS["yield_ta_r2"] and
              rho >= QUALITY_THRESHOLDS["yield_ta_rho"])
        rows.append({
            "stage":     "Stage2b",
            "target":    f"yield_TA h={row['horizon']}",
            "metric":    "R2 | rho",
            "value":     f"R2={r2:.3f} | rho={rho:.3f}",
            "threshold": f"R2>={QUALITY_THRESHOLDS['yield_ta_r2']} | rho>={QUALITY_THRESHOLDS['yield_ta_rho']}",
            "status":    "OK" if ok else "WARN",
        })

    df_summary = pd.DataFrame(rows)
    log.info("\n--- QUALITY SUMMARY TABLE ---\n%s", df_summary.to_string(index=False))

    n_ok  = (df_summary["status"] == "OK").sum()
    n_tot = len(df_summary)
    log.info("\n%d / %d models meet their quality thresholds.", n_ok, n_tot)

    save_metrics(
        {"pipeline_validated": int(n_ok == n_tot), "n_ok": n_ok, "n_total": n_tot},
        campaign.metrics_dir / "pipeline_summary.csv",
        run_id="full_pipeline",
    )

    print_campaign_summary(campaign)
    log.info("STEP 7 done in %.1fs", time.perf_counter() - t0)
    return df_summary


# =============================================================================
# STEP 8 — Inference Test (Example Scenario)
# =============================================================================

def _step8_inference_test(
    horizon_models: dict,
    yield_af_models: dict,
    yield_ta_models: dict,
    clf1_fitted: Any,
    clf2_fitted: Any,
    stunted_model: dict,
    campaign: Any,
    example_params: dict,
) -> dict:
    """
    Run end-to-end inference on a single example parameter set.

    Validates that the complete predict_single_sim pipeline executes
    without error and produces non-trivial outputs for a routed nominal
    scenario. Saves a 3-panel trajectory figure.

    Parameters
    ----------
    horizon_models    : dict  {(target, h): model}  — Stage 1
    yield_af_models   : dict  {h: model}             — Stage 2a
    yield_ta_models   : dict  {h: model}             — Stage 2b
    clf1_fitted, clf2_fitted : classifiers
    stunted_model     : dict  — conditional median fallback
    campaign          : CampaignPaths
    example_params    : dict  — user-supplied parameter set

    Returns
    -------
    result : dict  — output of predict_single_sim()
    """
    t0 = time.perf_counter()
    log.info("=" * 70)
    log.info("STEP 8 — Inference Test (example scenario)")
    log.info("=" * 70)

    # Build unified models dict with aligned keys expected by predictor.py v4.1
    models_inf: dict = {}
    for (target, h), model_h in horizon_models.items():
        models_inf[f"{target}_h{h}"] = model_h          # "carbonStem_AF_h{h}"
    for h, model_yh in yield_af_models.items():
        models_inf[f"yield_AF_h{h}"] = model_yh         # "yield_AF_h{h}"
    for h, model_ta in yield_ta_models.items():
        models_inf[f"yield_TA_h{h}"] = model_ta         # "yield_TA_h{h}"

    log.info(
        "models_inf: %d total — "
        "Stage1(carbonStem): %d | Stage2a(yield_AF): %d | Stage2b(yield_TA): %d",
        len(models_inf),
        sum(1 for k in models_inf if k.startswith("carbonStem")),
        sum(1 for k in models_inf if k.startswith("yield_AF")),
        sum(1 for k in models_inf if k.startswith("yield_TA")),
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
            log.info("  %-25s: all zeros (routing or missing model)", target)

    # 3-panel trajectory figure
    years = result["years"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1 — Stage 1: carbonStem
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

    # Panel 2 — Stage 2a: yield_AF
    if "yield_AF" in preds and preds["yield_AF"].max() > 0:
        axes[1].plot(years, preds["yield_AF"], "-", color="#E63946",
                     label="yield_AF (PCHIP)", lw=2)
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Yield (t/ha)")
    axes[1].set_title(f"Stage 2a — yield_AF\n{result['population']}")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Panel 3 — Stage 2b: yield_TA
    if "yield_TA" in preds and preds["yield_TA"].max() > 0:
        axes[2].plot(years, preds["yield_TA"], "-", color="#F4A261",
                     label="yield_TA (PCHIP)", lw=2)
    axes[2].set_xlabel("Year")
    axes[2].set_ylabel("Yield (t/ha)")
    axes[2].set_title(f"Stage 2b — yield_TA\n{result['population']}")
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
    log.info("Trajectory figure saved: %s", fig_path)

    df_out = format_output(result)
    log.info("format_output: %d rows × %d cols", *df_out.shape)

    log.info("STEP 8 done in %.1fs", time.perf_counter() - t0)
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
    MetAIsAFe full training pipeline — horizon architecture v4.1.

    Executes STEP 1 through STEP 8 sequentially for a given campaign.
    All intermediate outputs are returned as a dict for notebook inspection.

    Parameters
    ----------
    campaign_name : str
        Campaign folder name (must exist under the 03_Models directory).
    config_overrides : dict, optional
        Override any default hyperparameter dict. Accepted keys:
            "params_clf1"          — CLF1 hyperparameters
            "params_carbon_short/long_horizons"— Stage 1 (carbonStem) hyperparameters
            "params_yield_af"      — Stage 2a (yield_AF) hyperparameters
            "params_yield_ta"      — Stage 2b (yield_TA) hyperparameters
            "example_params"       — inference test scenario
    run_shap : bool, default True
        Enable STEP 6 (SHAP). Disable for fast debug runs.
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
        yield_af_models, yield_af_datasets, yield_af_results,
        yield_ta_models, yield_ta_datasets, yield_ta_results,
        df_summary, shap_results, inference_result
    """
    t_start = time.perf_counter()

    # ── Resolve configuration ─────────────────────────────────────────────
    cfg = config_overrides or {}
    params_clf1         = cfg.get("params_clf1",           PARAMS_CLF1_DEFAULT_V41)
    params_carbon_short = cfg.get("params_carbon_short_horizons", PARAMS_CARBON_SHORT)
    params_carbon_long = cfg.get("params_carbon_long_horizons", PARAMS_CARBON_LONG)
    params_yield_af     = cfg.get("params_yield_af",      PARAMS_YIELD_AF_DEFAULT)
    params_yield_ta     = cfg.get("params_yield_ta",      PARAMS_YIELD_TA_DEFAULT)
    example_params      = cfg.get("example_params",       EXAMPLE_PARAMS_DEFAULT)

    # ── Campaign paths ────────────────────────────────────────────────────
    campaign = get_campaign_paths(campaign_name)
    setup_file_logging(reports_dir=campaign.reports_dir, campaign_name=campaign_name)

    log.info("MetAIsAFe full_training.py — v%s", __version__)
    log.info("Campaign : %s", campaign_name)
    log.info("Models   : %s", campaign.metamodels_dir)
    log.info("SHAP     : %s | Inference: %s", run_shap, run_inference)

    # Static features shared across all stages (no climate, no time axis)
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
    cs_final_nom  = (
        df_nominal.sort_values(["SimID", "Harvest_Year_Absolute"])
        .groupby("SimID")["carbonStem_AF"].last()
    )
    simids_horizon = cs_final_nom[cs_final_nom >= MIN_CARBON_HORIZON].index
    df_horizon_pop = df_nominal[df_nominal["SimID"].isin(simids_horizon)].copy()

    n_nom     = df_nominal["SimID"].nunique()
    n_horizon = len(simids_horizon)
    log.info(
        "Horizon population: %d / %d SimIDs (carbonStem_AF(t=40) >= %.1f kgC/tree)",
        n_horizon, n_nom, MIN_CARBON_HORIZON,
    )
    if n_horizon < n_nom:
        log.info(
            "  %d SimIDs excluded from horizon training "
            "(carbonStem_AF(t=40) < %.1f — will use stunted fallback at inference).",
            n_nom - n_horizon, MIN_CARBON_HORIZON,
        )

    # ─────────────────────────────────────────────────────────────────────
    # STEP 3 — Train/Test Split + CV Groups
    # ─────────────────────────────────────────────────────────────────────
    train_df, test_df, train_sims, test_sims, groups, cv = _step3_split(df_nominal)

    # Log effective horizon split ratios
    n_train_h = sum(s in train_sims for s in simids_horizon)
    n_test_h  = sum(s in test_sims  for s in simids_horizon)
    log.info(
        "Horizon population effective split — train: %d (%.1f%%) | test: %d (%.1f%%)",
        n_train_h, 100 * n_train_h / n_horizon,
        n_test_h,  100 * n_test_h  / n_horizon,
    )

    # ─────────────────────────────────────────────────────────────────────
    # STEP 4 — Encoding & Winsorization
    # ─────────────────────────────────────────────────────────────────────
    train_wins, test_wins, win_bounds, X_train, X_test = _step4_encode_and_winsorize(
        train_df, test_df, campaign,
    )

    # ─────────────────────────────────────────────────────────────────────
    # STEP 5a — Classifiers + Stunted Fallback Model
    # ─────────────────────────────────────────────────────────────────────
    (clf1, clf2,
     clf1_eval, clf2_eval,
     stunted_model,
     df_clf_train, df_clf_test) = _step5a_classifiers(
        df, df_nominal, campaign, params_clf1,
    )

    # Build CLF test feature matrices for STEP 6 (SHAP on classifiers)
    # Derived from the SAME split used in STEP 5a — no re-splitting
    X_clf1_test = build_classifier_features(df_clf_test, CLF1_FEATURES)
    X_clf2_test = build_classifier_features(df_clf_test, CLF2_FEATURES)
    y_clf1_test = build_tree_fail_labels_multiclass(df_clf_test)
    y_clf2_test = build_yield_fail_labels(df_clf_test)
    X_clf1_test = X_clf1_test.loc[X_clf1_test.index.intersection(y_clf1_test.index)]
    X_clf2_test = X_clf2_test.loc[X_clf2_test.index.intersection(y_clf2_test.index)]
    assert len(X_clf1_test) > 0, "X_clf1_test empty after alignment"
    assert len(X_clf2_test) > 0, "X_clf2_test empty after alignment"

    # ─────────────────────────────────────────────────────────────────────
    # STEP 5b — Stage 1: carbonStem Horizon Models
    # ─────────────────────────────────────────────────────────────────────
    horizon_models, horizon_datasets, horizon_results, cs_af_preds = \
        _step5b_stage1_carbon(
            df_horizon_pop, train_sims, test_sims,
            campaign, params_carbon_short, params_carbon_long, static_feats,
        )

    # ─────────────────────────────────────────────────────────────────────
    # STEP 5c — Stage 2a: yield_AF Horizon Models
    # ─────────────────────────────────────────────────────────────────────
    yield_af_models, yield_af_datasets, yield_af_results = \
        _step5c_stage2a_yield_af(
            df_horizon_pop, cs_af_preds,
            train_sims, test_sims,
            campaign, params_yield_af, static_feats,
        )

    # ─────────────────────────────────────────────────────────────────────
    # STEP 5d — Stage 2b: yield_TA Horizon Models
    # ─────────────────────────────────────────────────────────────────────
    yield_ta_models, yield_ta_datasets, yield_ta_results = \
        _step5d_stage2b_yield_ta(
            df_horizon_pop, train_sims, test_sims,
            campaign, params_yield_ta, static_feats,
        )

    # ─────────────────────────────────────────────────────────────────────
    # STEP 6 — SHAP Analysis
    # ─────────────────────────────────────────────────────────────────────
    shap_results: dict = {}
    if run_shap:
        shap_results = _step6_shap(
            horizon_models, horizon_datasets,
            yield_af_models, yield_af_datasets,
            yield_ta_models, yield_ta_datasets,
            clf1, clf2, X_clf1_test, X_clf2_test,
            test_sims, campaign,
        )
    else:
        log.info("STEP 6 skipped (run_shap=False).")

    # ─────────────────────────────────────────────────────────────────────
    # STEP 7 — Final Evaluation
    # ─────────────────────────────────────────────────────────────────────
    df_summary = _step7_evaluate(
        clf1_eval, clf2_eval,
        horizon_results, yield_af_results, yield_ta_results,
        campaign,
    )

    # ─────────────────────────────────────────────────────────────────────
    # STEP 8 — Inference Test
    # ─────────────────────────────────────────────────────────────────────
    inference_result: dict = {}
    if run_inference:
        inference_result = _step8_inference_test(
            horizon_models, yield_af_models, yield_ta_models,
            clf1, clf2, stunted_model, campaign, example_params,
        )
    else:
        log.info("STEP 8 skipped (run_inference=False).")

    # ── Final summary ─────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    log.info("=" * 70)
    log.info("PIPELINE COMPLETE — %.1fs (%.1f min)", elapsed, elapsed / 60)
    log.info("=" * 70)

    cleanup_empty_campaign_dirs(campaign=campaign)
    print_campaign_summary(campaign)

    return {
        # Infrastructure
        "campaign":          campaign,
        # Data
        "df":                df,
        "df_nominal":        df_nominal,
        "df_horizon_pop":    df_horizon_pop,
        "train_df":          train_df,
        "test_df":           test_df,
        "train_wins":        train_wins,
        "test_wins":         test_wins,
        "X_train":           X_train,
        "X_test":            X_test,
        "win_bounds":        win_bounds,
        # Classifiers
        "clf1":              clf1,
        "clf2":              clf2,
        "clf1_eval":         clf1_eval,
        "clf2_eval":         clf2_eval,
        "stunted_model":     stunted_model,
        # Stage 1 — carbonStem
        "horizon_models":    horizon_models,
        "horizon_datasets":  horizon_datasets,
        "horizon_results":   horizon_results,
        # Stage 2a — yield_AF
        "yield_af_models":   yield_af_models,
        "yield_af_datasets": yield_af_datasets,
        "yield_af_results":  yield_af_results,
        # Stage 2b — yield_TA
        "yield_ta_models":   yield_ta_models,
        "yield_ta_datasets": yield_ta_datasets,
        "yield_ta_results":  yield_ta_results,
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
            "(horizon architecture v4.1)"
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
        help="Campaign name (folder under 03_Models/)",
    )
    p.add_argument(
        "--no-shap",
        action="store_true",
        default=False,
        help="Skip STEP 6 (SHAP) — useful for fast debug runs",
    )
    p.add_argument(
        "--no-inference",
        action="store_true",
        default=False,
        help="Skip STEP 8 (inference test)",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console logging level (default: INFO)",
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

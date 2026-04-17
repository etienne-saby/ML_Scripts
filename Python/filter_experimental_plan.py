"""
MetAIsAFe — filter_experimental_plan.py
=========================================
Pre-simulation cascade filter for experimental plans.

Applies trained CLF1 (tree status) and CLF2 (yield failure) classifiers
to an experimental plan CSV *before* launching HiSAFe simulations, in order
to discard parameter combinations predicted as failures (tree_degraded or
yield_fail). Only rows classified as 'yield_ok × tree_ok' are retained.

CONTROL PLAN PROPAGATION (NEW in v1.1)
---------------------------------------
By default (without --without-controls), the script assumes a triplet design:
  - {target_campaign}_AF : reference plan for cascade filtering
  - {target_campaign}_TA : control plan A
  - {target_campaign}_TF : control plan F

SimIDs identified as defective on the _AF plan are excluded from _TA and _TF
plans as well (same SimID across all three plans).

Use --without-controls to disable propagation and filter a single plan.

PRE-FILTER CLASSIFIER TRAINING (v1.2)
--------------------------------------
Standard CLF1/CLF2 models are trained with CLIMATE_FEATURES (7 cycle-level
climate aggregates) that are HiSAFe simulation *outputs* — unavailable at
plan generation time.

This script automatically trains specialized **prefilter classifiers** that
replace CLIMATE_FEATURES with 'longitude' as a geographic proxy. These models
are cached as:
    - clf1_tree_fail_prefilter.joblib
    - clf2_yield_fail_prefilter.joblib

Training reuses the robust `_step5a_classifiers()` pipeline from
`full_training.py` with adapted feature sets. If models already exist,
training is skipped unless --retrain-prefilter is specified.

PATH CONVENTIONS (from config.py)
-----------------------------------
Source classifiers : <source_campaign>/MetaModels/clf1_tree_fail_prefilter.joblib
                     <source_campaign>/MetaModels/clf2_yield_fail_prefilter.joblib
Input plan CSV     : 02_Simulations/<ref_campaign>/<ref_campaign>_Plan.csv
Outputs            : 02_Simulations/<ref_campaign>/<ref_campaign>_Plan_filtered.csv
                     02_Simulations/<ref_campaign>/<ref_campaign>_Plan_rejected.csv
                     03_Models/<ref_campaign>/Data/<ref_campaign>_Plan_annotated.csv
                     03_Models/<ref_campaign>/Data/Reports/filter_report_<ref>.json

USAGE
-----
    # Default — triplet mode (AF/TA/TF)
    python filter_experimental_plan.py \\
        --source-campaign sobol_training_1_n2048 \\
        --target-campaign lhs_training_3

    # Single plan mode (no controls)
    python filter_experimental_plan.py \\
        --source-campaign sobol_training_1_n2048 \\
        --target-campaign lhs_training_3 \\
        --without-controls

    # Force prefilter classifier retraining
    python filter_experimental_plan.py \\
        --source-campaign sobol_training_1_n2048 \\
        --target-campaign lhs_training_3 \\
        --retrain-prefilter

Author  : Étienne SABY
Version : 1.2 — prefilter classifiers via _step5a (2026-05)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path regardless of invocation directory
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    get_campaign_paths,
    get_hisafe_paths,
    find_project_root,
    SOBOL_FIXED_PARAMS,
    CampaignPaths,
    LGB_PARAMS,
    RANDOM_STATE,
)
from column_taxonomy import (
    CLIMATE_FEATURES,
    CATEGORICAL_FEATURES_B2,
)
from modeling.classifiers import (
    build_tree_fail_classifier,
    build_yield_fail_classifier,
    build_classifier_features,
    build_tree_degraded_labels,
    build_yield_fail_labels,
    evaluate_clf1_binary,
    evaluate_classifier,
    apply_geographic_rule,
    CLF1_FEATURES,
    CLF2_FEATURES,
    TREE_BINARY_OK,
    TREE_BINARY_LABELS,
    STUNTED_PROBA_THRESHOLD,
)
from modeling.trainer import train_classifier
from data.loader import load_data, encode_categoricals
from data.preparation import add_derived_columns, filter_crops, clean
from data.splitter import split_by_simid
from utils.io_utils import save_model

log = logging.getLogger(__name__)


# ===========================================================================
# CONSTANTS
# ===========================================================================

_SIMID_COL = "SimID"

# Column names written into the output DataFrames
_COL_TREE_STATUS   = "clf1_tree_status"       # 0 = tree_degraded, 1 = tree_ok
_COL_TREE_PROBA    = "clf1_p_tree_ok"         # P(tree_ok) from CLF1
_COL_YIELD_STATUS  = "clf2_yield_failed"      # 0 = yield_ok, 1 = yield_failed
_COL_YIELD_PROBA   = "clf2_p_yield_ok"        # 1 - P(yield_fail) from CLF2
_COL_KEEP          = "keep"                   # bool — True = send to simulation
_COL_POPULATION    = "predicted_population"   # routing label string


# ===========================================================================
# PREFILTER FEATURE SETS (longitude replaces CLIMATE_FEATURES)
# ===========================================================================

def _build_prefilter_features(base_features: list[str]) -> list[str]:
    """
    Replace all CLIMATE_FEATURES with a single 'longitude' proxy.
    
    The first occurrence of a climate feature is replaced by 'longitude';
    subsequent climate features are dropped. Non-climate features are
    preserved in their original order.
    """
    climate_set = set(CLIMATE_FEATURES)
    result = []
    proxy_added = False
    for feat in base_features:
        if feat in climate_set:
            if not proxy_added:
                result.append("longitude")
                proxy_added = True
            # else: drop subsequent climate features
        else:
            result.append(feat)
    return result


CLF1_FEATURES_PREFILTER = _build_prefilter_features(CLF1_FEATURES)
CLF2_FEATURES_PREFILTER = _build_prefilter_features(CLF2_FEATURES)


# ===========================================================================
# PREFILTER CLASSIFIER TRAINING
# ===========================================================================

def train_prefilter_classifiers(
    source_campaign: str,
    root_dir: Path | None = None,
    force_retrain: bool = False,
) -> tuple[Path, Path | None]:
    """
    Train CLF1 and CLF2 specialized for pre-filtering (longitude replaces CLIMATE_FEATURES).

    This function reuses the robust training logic from full_training.py
    (_step5a_classifiers pattern) with adapted feature sets.

    Models are cached in <source_campaign>/MetaModels/ as:
        - clf1_tree_fail_prefilter.joblib
        - clf2_yield_fail_prefilter.joblib

    If these models already exist and force_retrain=False, training is skipped.

    Parameters
    ----------
    source_campaign : str
        Campaign name containing the training meta-table (e.g. 'sobol_training_1_n2048')
    root_dir : Path, optional
        Project root override
    force_retrain : bool, default False
        If True, retrain even if models exist

    Returns
    -------
    tuple[Path, Path | None]
        (clf1_path, clf2_path) — saved model paths
    """
    src = get_campaign_paths(source_campaign, root_dir=root_dir, create=False)

    clf1_path = src.metamodels_dir / "clf1_tree_fail_prefilter.joblib"
    clf2_path = src.metamodels_dir / "clf2_yield_fail_prefilter.joblib"

    # ── Check cache ────────────────────────────────────────────────────────
    if not force_retrain and clf1_path.exists():
        log.info("Prefilter classifiers already trained — skipping re-train.")
        log.info("  CLF1: %s", clf1_path)
        if clf2_path.exists():
            log.info("  CLF2: %s", clf2_path)
        else:
            log.info("  CLF2: None (geographic fallback)")
        return clf1_path, (clf2_path if clf2_path.exists() else None)

    # ── Load and prepare meta-table ────────────────────────────────────────
    log.info("=" * 70)
    log.info("Training PREFILTER classifiers (longitude proxy)")
    log.info("=" * 70)
    log.info("Source campaign: %s", source_campaign)
    log.info("Meta-table path: %s", src.raw_meta)

    if not src.raw_meta.exists():
        raise FileNotFoundError(
            f"Meta-table not found: {src.raw_meta}\n"
            f"Run data preparation for {source_campaign} first."
        )

    df = load_data(src.raw_meta)
    log.info("Loaded meta-table: %d rows × %d cols", *df.shape)

    # Minimal preparation (full_training.py _step1 pattern, simplified)
    df = add_derived_columns(df, verbose=False)
    df, _ = filter_crops(df, excluded_crops=["rape"], verbose=False)
    df = clean(df, verbose=False)
    log.info("After preparation: %d rows × %d cols", *df.shape)

    # ── SimID-level split (like _step5a in full_training.py) ──────────────
    df_clf_train, df_clf_test, _, _ = split_by_simid(
        df, test_size=0.20, random_state=RANDOM_STATE, verbose=True,
    )
    log.info(
        "Prefilter split — train: %d SimIDs | test: %d SimIDs",
        df_clf_train["SimID"].nunique(),
        df_clf_test["SimID"].nunique(),
    )

    # ── CLF1 — Binary tree status (tree_degraded vs tree_ok) ──────────────
    log.info("\n" + "─" * 70)
    log.info("CLF1: Tree Status (binary — tree_degraded / tree_ok)")
    log.info("Features (%d): %s", len(CLF1_FEATURES_PREFILTER), CLF1_FEATURES_PREFILTER)
    log.info("─" * 70)

    X_clf1_train = build_classifier_features(df_clf_train, CLF1_FEATURES_PREFILTER)
    X_clf1_test  = build_classifier_features(df_clf_test,  CLF1_FEATURES_PREFILTER)
    y_clf1_train = build_tree_degraded_labels(df_clf_train)
    y_clf1_test  = build_tree_degraded_labels(df_clf_test)

    # Index alignment
    X_clf1_train = X_clf1_train.loc[X_clf1_train.index.intersection(y_clf1_train.index)]
    y_clf1_train = y_clf1_train.loc[X_clf1_train.index]
    X_clf1_test  = X_clf1_test.loc[X_clf1_test.index.intersection(y_clf1_test.index)]
    y_clf1_test  = y_clf1_test.loc[X_clf1_test.index]

    for i, label in enumerate(TREE_BINARY_LABELS):
        n = (y_clf1_train == i).sum()
        log.info(
            "  CLF1 train — %-20s: %d (%.1f%%)",
            label, n, 100.0 * n / len(y_clf1_train),
        )

    # Build and train model
    params_clf1 = LGB_PARAMS.copy()
    params_clf1.update({
        "objective": "binary",
        "metric": "binary_logloss",
        "random_state": RANDOM_STATE,
    })

    from modeling.models import build_lgb_classifier
    cat_feats_clf1 = [c for c in CLF1_FEATURES_PREFILTER if c in CATEGORICAL_FEATURES_B2 and c in X_clf1_train.columns]
    clf1_model = build_lgb_classifier(params=params_clf1, categorical_feature=cat_feats_clf1 or None)

    clf1_fitted, _ = train_classifier(
        clf1_model, X_clf1_train, y_clf1_train,
        X_clf1_test, y_clf1_test, verbose=True,
    )

    # Evaluate
    y_clf1_pred  = clf1_fitted.predict(X_clf1_test)
    y_clf1_proba = clf1_fitted.predict_proba(X_clf1_test)[:, 1]   # P(tree_ok)
    clf1_eval    = evaluate_clf1_binary(
        y_clf1_test, y_clf1_pred, y_clf1_proba,
        classifier_name="CLF1 — Prefilter (longitude proxy)",
        verbose=True,
    )

    # Save
    save_model(
        clf1_fitted,
        clf1_path,
        metadata={
            "classifier": "CLF1_tree_fail_prefilter",
            "version": "v1.2_prefilter",
            "features": CLF1_FEATURES_PREFILTER,
            "climate_proxy": "longitude",
            "classes": TREE_BINARY_LABELS,
            "accuracy_test": clf1_eval["accuracy"],
            "f1_macro_test": clf1_eval["f1_macro"],
            "n_train": len(y_clf1_train),
            "n_test": len(y_clf1_test),
        },
    )
    log.info("✓ CLF1 prefilter saved: %s", clf1_path)

    # ── CLF2 — Binary yield failure ────────────────────────────────────────
    log.info("\n" + "─" * 70)
    log.info("CLF2: Yield Failure (binary)")
    log.info("Features (%d): %s", len(CLF2_FEATURES_PREFILTER), CLF2_FEATURES_PREFILTER)
    log.info("─" * 70)

    X_clf2_train = build_classifier_features(df_clf_train, CLF2_FEATURES_PREFILTER)
    X_clf2_test  = build_classifier_features(df_clf_test,  CLF2_FEATURES_PREFILTER)
    y_clf2_train = build_yield_fail_labels(df_clf_train)
    y_clf2_test  = build_yield_fail_labels(df_clf_test)

    X_clf2_train = X_clf2_train.loc[X_clf2_train.index.intersection(y_clf2_train.index)]
    y_clf2_train = y_clf2_train.loc[X_clf2_train.index]
    X_clf2_test  = X_clf2_test.loc[X_clf2_test.index.intersection(y_clf2_test.index)]
    y_clf2_test  = y_clf2_test.loc[X_clf2_test.index]

    log.info(
        "CLF2 — train: %d SimIDs (%.1f%% failed) | test: %d SimIDs (%.1f%% failed)",
        len(y_clf2_train), 100.0 * y_clf2_train.mean(),
        len(y_clf2_test),  100.0 * y_clf2_test.mean(),
    )

    n_minority = int(y_clf2_train.sum())
    if n_minority < 80:
        log.warning("CLF2: insufficient minority samples (%d) — skipping training.", n_minority)
        log.warning("Geographic fallback rule will be used at inference time.")
        clf2_path_final = None
    else:
        params_clf2 = LGB_PARAMS.copy()
        params_clf2.update({
            "objective": "binary",
            "metric": "binary_logloss",
            "random_state": RANDOM_STATE,
        })

        cat_feats_clf2 = [c for c in CLF2_FEATURES_PREFILTER if c in CATEGORICAL_FEATURES_B2 and c in X_clf2_train.columns]
        clf2_model = build_lgb_classifier(params=params_clf2, categorical_feature=cat_feats_clf2 or None)

        clf2_fitted, _ = train_classifier(
            clf2_model, X_clf2_train, y_clf2_train,
            X_clf2_test, y_clf2_test, verbose=True,
        )

        y_clf2_pred  = clf2_fitted.predict(X_clf2_test)
        y_clf2_proba = clf2_fitted.predict_proba(X_clf2_test)[:, 1]
        clf2_eval    = evaluate_classifier(
            y_clf2_test, y_clf2_pred, y_clf2_proba,
            classifier_name="CLF2 — Prefilter (longitude proxy)",
            verbose=True,
        )

        save_model(
            clf2_fitted,
            clf2_path,
            metadata={
                "classifier": "CLF2_yield_fail_prefilter",
                "version": "v1.2_prefilter",
                "features": CLF2_FEATURES_PREFILTER,
                "climate_proxy": "longitude",
                "accuracy_test": clf2_eval.get("accuracy"),
                "f1_test": clf2_eval.get("f1"),
                "roc_auc_test": clf2_eval.get("roc_auc"),
                "n_train": len(y_clf2_train),
                "n_test": len(y_clf2_test),
            },
        )
        log.info("✓ CLF2 prefilter saved: %s", clf2_path)
        clf2_path_final = clf2_path

    log.info("=" * 70)
    log.info("Prefilter classifier training complete.")
    log.info("=" * 70)
    return clf1_path, clf2_path_final


# ===========================================================================
# PLAN LOADER
# ===========================================================================

def load_plan(
    plan_path: Path,
    fixed_params: dict | None = None,
) -> pd.DataFrame:
    """
    Load experimental plan CSV and inject fixed parameters if provided.

    Parameters
    ----------
    plan_path    : Path to the CSV file generated by R
    fixed_params : dict of {column: value} to add as constant columns
                   (e.g. SOBOL_FIXED_PARAMS for B2/B3 campaigns).
                   Existing columns are NOT overwritten.

    Returns
    -------
    pd.DataFrame — plan with all required columns present
    """
    df = pd.read_csv(plan_path)
    log.info("Plan loaded: %d rows × %d cols from %s", *df.shape, plan_path.name)

    if fixed_params:
        for col, val in fixed_params.items():
            if col not in df.columns:
                df[col] = val
                log.debug("  Injected fixed param: %s = %s", col, val)
            else:
                log.debug("  Fixed param '%s' already present — not overwritten.", col)

    return df


# ===========================================================================
# FEATURE MATRIX BUILDER
# ===========================================================================

def _prepare_feature_matrix(
    df: pd.DataFrame,
    feature_set: list[str],
    categorical_features: list[str] = CATEGORICAL_FEATURES_B2,
) -> pd.DataFrame:
    """
    Build a feature matrix from the plan DataFrame for one classifier.

    Features absent from `df` are filled with NaN. Categorical columns
    are cast to pandas category dtype.

    Parameters
    ----------
    df                  : experimental plan DataFrame
    feature_set         : list of feature names
    categorical_features: columns to cast to category dtype

    Returns
    -------
    pd.DataFrame — shape (n_rows, len(feature_set))
    """
    # reindex inserts NaN for missing columns
    X = df.reindex(columns=feature_set).copy()

    missing = [c for c in feature_set if c not in df.columns]
    if missing:
        log.warning(
            "  %d features absent from plan (filled with NaN): %s",
            len(missing), missing,
        )

    cat_cols = [c for c in feature_set if c in categorical_features]
    for col in cat_cols:
        if col in X.columns and X[col].notna().any():
            X[col] = X[col].astype("category")

    return X


# ===========================================================================
# CASCADE FILTER — CORE
# ===========================================================================

def apply_cascade_filter(
    df: pd.DataFrame,
    clf1,
    clf2,
    proba_mode: bool = False,
    threshold_tree: float = 0.50,
    threshold_yield: float = 0.50,
) -> pd.DataFrame:
    """
    Apply CLF1 → CLF2 cascade and annotate every row with routing columns.

    Both classifiers run on the full plan (no early stopping between stages)
    so that all diagnostic columns are available for post-hoc analysis.

    Parameters
    ----------
    df              : experimental plan DataFrame (from load_plan)
    clf1            : fitted LGBMClassifier — CLF1 (tree status, binary)
    clf2            : fitted LGBMClassifier or None — CLF2 (yield failure)
    proba_mode      : if True, use probability thresholds instead of
                      majority vote for the keep/reject decision
    threshold_tree  : P(tree_ok) >= threshold to keep (proba_mode only)
    threshold_yield : P(yield_ok) >= threshold to keep (proba_mode only)

    Returns
    -------
    pd.DataFrame — original plan + diagnostic columns:
        clf1_tree_status, clf1_p_tree_ok,
        clf2_yield_failed, clf2_p_yield_ok,
        keep (bool), predicted_population (str)
    """
    result = df.copy()

    log.info("CLF1 prefilter feature set (%d features): %s", len(CLF1_FEATURES_PREFILTER), CLF1_FEATURES_PREFILTER)
    log.info("CLF2 prefilter feature set (%d features): %s", len(CLF2_FEATURES_PREFILTER), CLF2_FEATURES_PREFILTER)

    # ── CLF1 — Tree status ─────────────────────────────────────────────────
    X_clf1 = _prepare_feature_matrix(result, CLF1_FEATURES_PREFILTER)

    try:
        tree_preds = clf1.predict(X_clf1).astype(int)
        tree_proba = clf1.predict_proba(X_clf1)[:, 1]   # P(tree_ok)
    except Exception as exc:
        log.error("CLF1 prediction failed: %s", exc)
        log.error(
            "This typically means the saved model expects a different feature set. "
            "Try running with --retrain-prefilter to rebuild the classifiers."
        )
        raise

    result[_COL_TREE_STATUS] = tree_preds
    result[_COL_TREE_PROBA]  = tree_proba

    # ── CLF2 — Yield failure ───────────────────────────────────────────────
    X_clf2 = _prepare_feature_matrix(result, CLF2_FEATURES_PREFILTER)

    if clf2 is not None:
        try:
            yield_preds = clf2.predict(X_clf2).astype(int)
            yield_proba = clf2.predict_proba(X_clf2)[:, 1]   # P(yield_fail)
        except Exception as exc:
            log.error("CLF2 prediction failed: %s", exc)
            raise
        result[_COL_YIELD_STATUS] = yield_preds
        result[_COL_YIELD_PROBA]  = 1.0 - yield_proba       # → P(yield_ok)
    else:
        log.info("CLF2 is None — applying geographic fallback rule.")
        geo_preds = apply_geographic_rule(X_clf2)
        result[_COL_YIELD_STATUS] = geo_preds
        result[_COL_YIELD_PROBA]  = (1 - geo_preds).astype(float)

    # ── Keep / reject decision ─────────────────────────────────────────────
    if proba_mode:
        log.info(
            "Decision mode: PROBABILITY — threshold_tree=%.2f, threshold_yield=%.2f",
            threshold_tree, threshold_yield,
        )
        keep_tree  = result[_COL_TREE_PROBA]  >= threshold_tree
        keep_yield = result[_COL_YIELD_PROBA] >= threshold_yield
    else:
        log.info("Decision mode: MAJORITY VOTE")
        keep_tree  = result[_COL_TREE_STATUS]  == TREE_BINARY_OK
        keep_yield = result[_COL_YIELD_STATUS] == 0              # 0 = yield_ok

    result[_COL_KEEP] = keep_tree & keep_yield

    # ── Population label ───────────────────────────────────────────────────
    def _label(row: pd.Series) -> str:
        y = "yield_ok"       if row[_COL_YIELD_STATUS] == 0 else "yield_fail"
        t = "tree_ok"        if row[_COL_TREE_STATUS]  == TREE_BINARY_OK else "tree_degraded"
        return f"{y} × {t}"

    result[_COL_POPULATION] = result.apply(_label, axis=1)

    return result


# ===========================================================================
# CONTROL PLAN PROPAGATION
# ===========================================================================

def propagate_rejections_to_control_plans(
    rejected_simids: set[str],
    target_campaign: str,
    root_dir: Path | None = None,
    suffixes: tuple[str, ...] = ("_TA", "_TF"),
) -> dict[str, dict]:
    """
    Propagate SimID rejections from _AF plan to control plans (_TA, _TF).

    For each control plan:
      - Load {target_campaign}{suffix}/{target_campaign}{suffix}_Plan.csv
      - Split rows by SimID presence in rejected_simids
      - Write filtered/rejected CSVs to 02_Simulations/
      - Write annotated CSV to 03_Models/{campaign}/Data/ for diagnostics

    Parameters
    ----------
    rejected_simids : set of str — SimIDs rejected on _AF plan
    target_campaign : str — base campaign name (without _AF suffix)
    root_dir : Path, optional
    suffixes : tuple of str — control plan suffixes (default: ("_TA", "_TF"))

    Returns
    -------
    dict — {suffix: {n_total, n_kept, n_rejected, keep_rate_pct, skipped}}
    """
    results = {}
    hisafe = get_hisafe_paths(explicit_root=root_dir)

    for suffix in suffixes:
        camp_name = f"{target_campaign}{suffix}"
        log.info("\n" + "─" * 60)
        log.info("Propagating rejections to: %s", camp_name)
        log.info("─" * 60)

        sim_dir  = hisafe.simulations_dir / camp_name
        plan_csv = sim_dir / f"{camp_name}_Plan.csv"

        if not plan_csv.exists():
            log.warning("  Plan CSV not found: %s — skipping.", plan_csv)
            results[suffix] = {"skipped": True, "reason": "file_not_found"}
            continue

        df_ctrl = pd.read_csv(plan_csv)
        log.info("  Loaded: %d rows", len(df_ctrl))

        if _SIMID_COL not in df_ctrl.columns:
            log.warning("  Column '%s' not found — skipping.", _SIMID_COL)
            results[suffix] = {"skipped": True, "reason": "simid_column_missing"}
            continue

        # Split by SimID membership
        mask_rejected  = df_ctrl[_SIMID_COL].astype(str).isin(rejected_simids)
        df_ctrl_kept     = df_ctrl[~mask_rejected].reset_index(drop=True)
        df_ctrl_rejected = df_ctrl[mask_rejected].reset_index(drop=True)

        n_total    = len(df_ctrl)
        n_kept     = len(df_ctrl_kept)
        n_rejected = len(df_ctrl_rejected)
        keep_rate  = 100.0 * n_kept / n_total if n_total > 0 else 0.0

        log.info("  Kept: %d / %d (%.1f%%) | Rejected: %d", n_kept, n_total, keep_rate, n_rejected)

        # Write to 02_Simulations/ (source location)
        path_filtered = sim_dir / f"{camp_name}_Plan_filtered.csv"
        path_rejected = sim_dir / f"{camp_name}_Plan_rejected.csv"

        df_ctrl_kept.to_csv(path_filtered, index=False)
        df_ctrl_rejected.to_csv(path_rejected, index=False)

        log.info("  Filtered → %s", path_filtered)
        log.info("  Rejected → %s", path_rejected)

        # Write annotated to 03_Models/{campaign}/Data/ for diagnostics
        try:
            campaign_ctrl = get_campaign_paths(camp_name, root_dir=root_dir, create=True)
            path_annotated = campaign_ctrl.data_dir / f"{camp_name}_Plan_annotated.csv"

            df_ctrl_annot = df_ctrl.copy()
            df_ctrl_annot[_COL_KEEP] = ~mask_rejected
            df_ctrl_annot[_COL_POPULATION] = df_ctrl_annot[_COL_KEEP].map({
                True:  "propagated_ok",
                False: "propagated_rejected",
            })
            df_ctrl_annot.to_csv(path_annotated, index=False)
            log.info("  Annotated → %s", path_annotated)
        except Exception as exc:
            log.warning("  Could not write annotated CSV to 03_Models/: %s", exc)

        results[suffix] = {
            "skipped": False,
            "n_total": n_total,
            "n_kept": n_kept,
            "n_rejected": n_rejected,
            "keep_rate_pct": round(keep_rate, 2),
        }

    return results


# ===========================================================================
# REPORT BUILDER
# ===========================================================================

def build_filter_report(
    df_annotated: pd.DataFrame,
    source_campaign: str,
    target_campaign: str,
    ref_campaign: str,
    plan_path: Path,
    proba_mode: bool,
    threshold_tree: float,
    threshold_yield: float,
    elapsed_s: float,
    without_controls: bool,
    control_results: dict | None = None,
) -> dict:
    """
    Build a structured JSON-serialisable report of the filtering run.

    Parameters
    ----------
    df_annotated    : full plan with cascade annotation columns
    source_campaign : name of the campaign that trained the classifiers
    target_campaign : base campaign name (without _AF suffix)
    ref_campaign    : actual reference campaign filtered (e.g. target_campaign_AF)
    plan_path       : path to the input plan CSV
    proba_mode      : decision mode used
    threshold_tree  : threshold_tree used (relevant in proba_mode)
    threshold_yield : threshold_yield used (relevant in proba_mode)
    elapsed_s       : wall-clock seconds
    without_controls: bool — single plan mode flag
    control_results : dict — propagation results for _TA/_TF (or None)

    Returns
    -------
    dict — JSON-serialisable report
    """
    n_total    = len(df_annotated)
    n_kept     = int(df_annotated[_COL_KEEP].sum())
    n_rejected = n_total - n_kept

    pop_counts = (
        df_annotated[_COL_POPULATION]
        .value_counts()
        .rename_axis("population")
        .reset_index(name="n")
    )
    pop_counts["pct"] = (pop_counts["n"] / n_total * 100).round(2)
    pop_dict = {
        row["population"]: {"n": int(row["n"]), "pct": float(row["pct"])}
        for _, row in pop_counts.iterrows()
    }

    proba_stats: dict = {}
    for col, label in [
        (_COL_TREE_PROBA,  "P(tree_ok)"),
        (_COL_YIELD_PROBA, "P(yield_ok)"),
    ]:
        if col in df_annotated.columns:
            s = df_annotated[col]
            proba_stats[label] = {
                "mean":   round(float(s.mean()), 4),
                "std":    round(float(s.std()),  4),
                "p10":    round(float(s.quantile(0.10)), 4),
                "median": round(float(s.median()), 4),
                "p90":    round(float(s.quantile(0.90)), 4),
            }

    report = {
        "metaisafe_version": "filter_experimental_plan v1.2",
        "source_campaign":   source_campaign,
        "target_campaign":   target_campaign,
        "reference_campaign": ref_campaign,
        "plan_file":         str(plan_path.name),
        "decision_mode":     "proba" if proba_mode else "majority_vote",
        "threshold_tree":    threshold_tree  if proba_mode else None,
        "threshold_yield":   threshold_yield if proba_mode else None,
        "without_controls":  without_controls,
        "n_total":           n_total,
        "n_kept":            n_kept,
        "n_rejected":        n_rejected,
        "keep_rate_pct":     round(100.0 * n_kept / n_total, 2),
        "population_distribution": pop_dict,
        "proba_statistics":  proba_stats,
        "elapsed_seconds":   round(elapsed_s, 2),
        "control_plans":     control_results if not without_controls else None,
        "note": (
            "Prefilter classifiers trained with 'longitude' as a geographic proxy "
            "for CLIMATE_FEATURES. Classifier precision is lower than post-simulation "
            "models. Monitor batch failure rates post-simulation to assess filter quality."
        ),
    }
    return report


# ===========================================================================
# MAIN ENTRYPOINT
# ===========================================================================

def run_filter(
    source_campaign: str,
    target_campaign: str,
    plan_file: str | None = None,
    proba_mode: bool = False,
    threshold_tree: float = 0.50,
    threshold_yield: float = 0.50,
    inject_fixed_params: bool = True,
    without_controls: bool = False,
    force_retrain: bool = False,
    root_dir: Path | None = None,
) -> dict:
    """
    Full filtering pipeline: load → classify → export (± control propagation).

    Parameters
    ----------
    source_campaign     : campaign name that contains trained CLF1/CLF2 prefilter
                          (e.g. 'sobol_training_1_n2048')
    target_campaign     : base campaign name whose plan is to be filtered
                          (e.g. 'lhs_training_3' → references 'lhs_training_3_AF')
    plan_file           : CSV filename inside 02_Simulations/<ref_campaign>/
                          (ignored in triplet mode — auto-resolved)
    proba_mode          : use probability thresholds instead of majority vote
    threshold_tree      : P(tree_ok) >= threshold to keep (proba_mode only)
    threshold_yield     : P(yield_ok) >= threshold to keep (proba_mode only)
    inject_fixed_params : if True, add SOBOL_FIXED_PARAMS as constant columns
    without_controls    : if True, filter single plan (target_campaign directly)
                          if False (default), filter triplet (AF/TA/TF)
    force_retrain       : if True, force prefilter classifier retraining
    root_dir            : project root override (for tests)

    Returns
    -------
    dict with keys:
        df_filtered   : pd.DataFrame — kept rows (without annotation columns)
        df_rejected   : pd.DataFrame — rejected rows (without annotation columns)
        df_annotated  : pd.DataFrame — full plan with all diagnostic columns
        report        : dict         — JSON report
        paths         : dict         — output file paths
    """
    t_start = time.perf_counter()

    # ── Resolve reference campaign ─────────────────────────────────────────
    if without_controls:
        ref_campaign = target_campaign
        log.info("Mode: SINGLE PLAN (--without-controls) — reference: %s", ref_campaign)
    else:
        ref_campaign = f"{target_campaign}_AF"
        log.info("Mode: TRIPLET (AF/TA/TF) — reference plan: %s", ref_campaign)
        if plan_file is not None:
            log.warning(
                "--plan-file is ignored in triplet mode (auto-resolved to %s_Plan.csv).",
                ref_campaign,
            )
            plan_file = None

    # ── Resolve campaign paths ─────────────────────────────────────────────
    src = get_campaign_paths(source_campaign, root_dir=root_dir, create=False)
    ref = get_campaign_paths(ref_campaign,    root_dir=root_dir, create=True)
    hisafe = get_hisafe_paths(explicit_root=root_dir)

    log.info("=" * 65)
    log.info("MetAIsAFe — Pre-simulation cascade filter v1.2")
    log.info("=" * 65)
    log.info("Source campaign (classifiers) : %s", source_campaign)
    log.info("  CLF1/CLF2 dir : %s", src.metamodels_dir)
    log.info("Target campaign (base)        : %s", target_campaign)
    log.info("Reference campaign (filtered) : %s", ref_campaign)

    # ── Train or load prefilter classifiers ────────────────────────────────
    log.info("\n" + "─" * 65)
    log.info("Step 1/3: Load or train prefilter classifiers")
    log.info("─" * 65)

    clf1_path, clf2_path = train_prefilter_classifiers(
        source_campaign,
        root_dir=root_dir,
        force_retrain=force_retrain,
    )

    clf1 = joblib.load(clf1_path)
    log.info("CLF1 prefilter loaded: %s", clf1_path)

    clf2 = None
    if clf2_path is not None and clf2_path.exists():
        clf2 = joblib.load(clf2_path)
        log.info("CLF2 prefilter loaded: %s", clf2_path)
    else:
        log.warning("CLF2 prefilter not found — geographic fallback rule will be used.")

    # ── Resolve plan CSV path ──────────────────────────────────────────────
    if plan_file is None:
        plan_file = f"{ref_campaign}_Plan.csv"

    sim_dir   = hisafe.simulations_dir / ref_campaign
    plan_path = sim_dir / plan_file

    log.info("  Plan CSV      : %s", plan_path)

    if not plan_path.exists():
        raise FileNotFoundError(
            f"Plan CSV not found: {plan_path}\n"
            f"Expected location: 02_Simulations/{ref_campaign}/{plan_file}\n"
            f"Verify that the experimental plan exists."
        )

    # ── Load plan + inject fixed params ───────────────────────────────────
    log.info("\n" + "─" * 65)
    log.info("Step 2/3: Load and filter experimental plan")
    log.info("─" * 65)

    fixed = SOBOL_FIXED_PARAMS if inject_fixed_params else None
    df = load_plan(plan_path, fixed_params=fixed)

    # ── Apply cascade ──────────────────────────────────────────────────────
    log.info("\nApplying cascade filter...")
    df_annotated = apply_cascade_filter(
        df, clf1, clf2,
        proba_mode=proba_mode,
        threshold_tree=threshold_tree,
        threshold_yield=threshold_yield,
    )

    # ── Split kept / rejected ──────────────────────────────────────────────
    annotation_cols = [
        _COL_TREE_STATUS, _COL_TREE_PROBA,
        _COL_YIELD_STATUS, _COL_YIELD_PROBA,
        _COL_KEEP, _COL_POPULATION,
    ]
    plan_cols = [c for c in df_annotated.columns if c not in annotation_cols]

    df_filtered = df_annotated.loc[df_annotated[_COL_KEEP],  plan_cols].reset_index(drop=True)
    df_rejected = df_annotated.loc[~df_annotated[_COL_KEEP], plan_cols].reset_index(drop=True)

    n_total = len(df_annotated)
    n_kept  = len(df_filtered)
    log.info(
        "\nFilter result: %d / %d kept (%.1f%%) | %d rejected (%.1f%%)",
        n_kept, n_total, 100.0 * n_kept / n_total,
        n_total - n_kept, 100.0 * (n_total - n_kept) / n_total,
    )

    pop_summary = df_annotated[_COL_POPULATION].value_counts()
    log.info("\nPredicted population distribution:")
    for pop, n in pop_summary.items():
        log.info("  %-35s : %5d  (%.1f%%)", pop, n, 100.0 * n / n_total)

    # ── Propagate to control plans (if triplet mode) ──────────────────────
    control_results = None
    if not without_controls:
        log.info("\n" + "─" * 65)
        log.info("Step 3/3: Propagate rejections to control plans (_TA, _TF)")
        log.info("─" * 65)

        if _SIMID_COL not in df_rejected.columns:
            log.warning(
                "SimID column not found in rejected set — skipping propagation."
            )
        else:
            rejected_simids = set(df_rejected[_SIMID_COL].astype(str).unique())
            log.info("Rejected SimIDs from _AF: %d", len(rejected_simids))

            control_results = propagate_rejections_to_control_plans(
                rejected_simids,
                target_campaign,
                root_dir=root_dir,
                suffixes=("_TA", "_TF"),
            )
    else:
        log.info("\nStep 3/3: Skipped (--without-controls)")

    # ── Export outputs ─────────────────────────────────────────────────────
    stem = plan_path.stem   # e.g. 'lhs_training_3_AF_Plan'

    path_filtered  = sim_dir / f"{stem}_filtered.csv"
    path_rejected  = sim_dir / f"{stem}_rejected.csv"
    path_annotated = ref.data_dir / f"{stem}_annotated.csv"
    path_report    = ref.reports_dir / f"filter_report_{stem}.json"

    df_filtered.to_csv(path_filtered,   index=False)
    df_rejected.to_csv(path_rejected,   index=False)
    df_annotated.to_csv(path_annotated, index=False)

    log.info("\nOutputs written:")
    log.info("  Filtered  → %s  (%d rows)", path_filtered,  len(df_filtered))
    log.info("  Rejected  → %s  (%d rows)", path_rejected,  len(df_rejected))
    log.info("  Annotated → %s  (%d rows, diagnostics)", path_annotated, len(df_annotated))

    # ── Build and save report ──────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    report  = build_filter_report(
        df_annotated, source_campaign, target_campaign, ref_campaign,
        plan_path, proba_mode, threshold_tree, threshold_yield, elapsed,
        without_controls, control_results,
    )
    with open(path_report, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    log.info("  Report    → %s  (JSON)", path_report)

    log.info("\n%s", "=" * 65)
    log.info(
        "Done — %.2fs | kept %d / %d (%.1f%%)",
        elapsed, n_kept, n_total, 100.0 * n_kept / n_total,
    )
    log.info("%s", "=" * 65)

    return {
        "df_filtered":  df_filtered,
        "df_rejected":  df_rejected,
        "df_annotated": df_annotated,
        "report":       report,
        "paths": {
            "filtered":  path_filtered,
            "rejected":  path_rejected,
            "annotated": path_annotated,
            "report":    path_report,
        },
    }


# ===========================================================================
# CLI
# ===========================================================================

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "MetAIsAFe — Pre-simulation cascade filter v1.2.\n"
            "Applies CLF1 (tree status) + CLF2 (yield failure) to an experimental\n"
            "plan CSV and retains only 'yield_ok × tree_ok' rows."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default — triplet mode (AF/TA/TF)
  python filter_experimental_plan.py \\
      --source-campaign sobol_training_1_n2048 \\
      --target-campaign lhs_training_3

  # Single plan mode (no controls)
  python filter_experimental_plan.py \\
      --source-campaign sobol_training_1_n2048 \\
      --target-campaign lhs_training_3 \\
      --without-controls

  # Force prefilter classifier retraining
  python filter_experimental_plan.py \\
      --source-campaign sobol_training_1_n2048 \\
      --target-campaign lhs_training_3 \\
      --retrain-prefilter

  # Probability-threshold mode (strict budget)
  python filter_experimental_plan.py \\
      --source-campaign sobol_training_1_n2048 \\
      --target-campaign lhs_training_3 \\
      --proba-mode --threshold-tree 0.65 --threshold-yield 0.65
        """,
    )
    p.add_argument(
        "--source-campaign", "-s",
        required=True,
        help=(
            "Campaign name containing training data for prefilter classifiers "
            "(e.g. 'sobol_training_1_n2048'). "
            "Prefilter models are cached in <source>/MetaModels/."
        ),
    )
    p.add_argument(
        "--target-campaign", "-t",
        required=True,
        help=(
            "Base campaign name whose experimental plan is to be filtered "
            "(e.g. 'lhs_training_3'). In triplet mode (default), filters "
            "'lhs_training_3_AF' and propagates to '_TA', '_TF'."
        ),
    )
    p.add_argument(
        "--plan-file", "-f",
        default=None,
        help=(
            "CSV filename inside 02_Simulations/<ref_campaign>/. "
            "Ignored in triplet mode (auto-resolved). "
            "Default: {ref_campaign}_Plan.csv"
        ),
    )
    p.add_argument(
        "--proba-mode",
        action="store_true",
        default=False,
        help=(
            "Use probability thresholds instead of majority vote. "
            "Stricter filter — recommended when simulation budget is tight."
        ),
    )
    p.add_argument(
        "--threshold-tree",
        type=float,
        default=0.50,
        metavar="FLOAT",
        help="P(tree_ok) >= threshold to keep a row (proba-mode only). Default: 0.50",
    )
    p.add_argument(
        "--threshold-yield",
        type=float,
        default=0.50,
        metavar="FLOAT",
        help="P(yield_ok) >= threshold to keep a row (proba-mode only). Default: 0.50",
    )
    p.add_argument(
        "--no-fixed-params",
        action="store_true",
        default=False,
        help=(
            "Do NOT inject SOBOL_FIXED_PARAMS as constant columns. "
            "Use this if fixed parameters are already present in the CSV."
        ),
    )
    p.add_argument(
        "--without-controls",
        action="store_true",
        default=False,
        help=(
            "Filter a single plan (target_campaign) instead of triplet mode. "
            "Disables _TA/_TF propagation."
        ),
    )
    p.add_argument(
        "--retrain-prefilter",
        action="store_true",
        default=False,
        help=(
            "Force prefilter classifier retraining even if models already exist. "
            "Use this after retraining the source campaign with updated data."
        ),
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console logging verbosity. Default: INFO",
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

    results = run_filter(
        source_campaign     = args.source_campaign,
        target_campaign     = args.target_campaign,
        plan_file           = args.plan_file,
        proba_mode          = args.proba_mode,
        threshold_tree      = args.threshold_tree,
        threshold_yield     = args.threshold_yield,
        inject_fixed_params = not args.no_fixed_params,
        without_controls    = args.without_controls,
        force_retrain       = args.retrain_prefilter,
    )

    print("\n" + "=" * 65)
    print("FILTER SUMMARY")
    print("=" * 65)
    r = results["report"]
    print(f"  Plan            : {r['plan_file']}")
    print(f"  Decision mode   : {r['decision_mode']}")
    print(f"  Total rows      : {r['n_total']}")
    print(f"  Kept (simulated): {r['n_kept']}  ({r['keep_rate_pct']:.1f}%)")
    print(f"  Rejected        : {r['n_rejected']}")
    print("\n  Population distribution:")
    for pop, stats in r["population_distribution"].items():
        print(f"    {pop:<35}  {stats['n']:5d}  ({stats['pct']:.1f}%)")

    if not r["without_controls"] and r["control_plans"]:
        print("\n  Control plan propagation:")
        for suffix, ctrl in r["control_plans"].items():
            if ctrl.get("skipped"):
                print(f"    {suffix}: SKIPPED ({ctrl.get('reason', 'unknown')})")
            else:
                print(
                    f"    {suffix}: kept {ctrl['n_kept']}/{ctrl['n_total']} "
                    f"({ctrl['keep_rate_pct']:.1f}%)"
                )

    if r.get("proba_statistics"):
        print("\n  Probability statistics:")
        for label, stats in r["proba_statistics"].items():
            print(
                f"    {label:<15}  mean={stats['mean']:.3f}  "
                f"p10={stats['p10']:.3f}  median={stats['median']:.3f}  "
                f"p90={stats['p90']:.3f}"
            )
    print("=" * 65)

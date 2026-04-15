"""
MetAIsAFe — xgb_fallback.py
=============================
XGBoost fallback pipeline for validation and benchmarking purposes.

.. warning::
    This script is **NOT** part of the main production pipeline.
    LightGBM is the primary surrogate model (see ``MODEL_CHOICE_RATIONALE``
    in ``pipeline.py``). This script exists to:

    1. Validate LightGBM results against an independent implementation.
    2. Provide a fallback in edge cases where LightGBM convergence is unstable.
    3. Support academic benchmarking / reporting.

Usage
-----
Run this script directly::

    python xgb_fallback.py --campaign <campaign_name> [--targets yield_AF yield_TA]

Or import and call::

    from xgb_fallback import run_xgb_pipeline
    results = run_xgb_pipeline(campaign_name="my_campaign")

When to use
-----------
- LightGBM validation R² is unexpectedly low for a specific target.
- You need a second opinion for publication / peer review.
- Checking for target-specific model family sensitivity.

Author : MetAIsAFe team
Version: 3.2
"""
from __future__ import annotations

import argparse
import logging
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ── Taxonomy & config ────────────────────────────────────────────────────────
from column_taxonomy import (
    ALL_FEATURES,
    STOCK_TARGETS_CROP,
    STOCK_TARGETS_TREE,
    CATEGORICAL_FEATURES,
)
from config import (
    RANDOM_STATE,
    FIGURE_FORMAT,
    get_campaign_paths,
    CampaignPaths,
)

# ── Data layer ───────────────────────────────────────────────────────────────
from data.loader import load_data, encode_categoricals
from data.preparation import add_derived_columns, filter_crops, clean, compute_effective_vars
from data.preprocessing import apply_winsorization, compute_ratios_from_stocks, build_dataset
from data.splitter import has_rotations, stratified_split_by_rotation, split_by_simid

# ── Modeling layer ───────────────────────────────────────────────────────────
from modeling.models import build_xgb, get_feature_importances
from modeling.trainer import cross_validate, train_final_model
from modeling.evaluator import compute_metrics, plot_pred_vs_obs, plot_residuals

# ── Utils ────────────────────────────────────────────────────────────────────
from utils.io_utils import save_metrics, save_predictions, save_cv_results

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================

STOCK_TARGETS: list[str] = list(dict.fromkeys(STOCK_TARGETS_CROP + STOCK_TARGETS_TREE))

_FALLBACK_BANNER = """
╔══════════════════════════════════════════════════════════════╗
║          MetAIsAFe — XGBoost Fallback Pipeline               ║
║  ⚠  NOT the production pipeline — for validation only  ⚠    ║
╚══════════════════════════════════════════════════════════════╝
"""


# ============================================================================
# INTERNAL HELPERS
# ============================================================================

def _encode_xgb(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    winsorize_quantiles: tuple[float, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Encode + winsorise for XGBoost (``enable_categorical=True``).

    XGBoost >= 1.6 supports pandas ``category`` dtype natively,
    so encoding strategy is identical to LightGBM.

    Parameters
    ----------
    train_df : pd.DataFrame
    test_df : pd.DataFrame
    features : list[str]
    winsorize_quantiles : tuple[float, float]

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Encoded + winsorised (train, test).
    """
    # Category encoding (fit on train, transform test)
    train_enc, encoders = encode_categoricals(train_df, features, fit=True, verbose=False)
    test_enc, _ = encode_categoricals(test_df, features, fit=False, encoders=encoders, verbose=False)

    # Winsorisation (fit on train only)
    train_enc, win_bounds = apply_winsorization(
        train_enc, quantiles=winsorize_quantiles, fit=True, verbose=False
    )
    test_enc, _ = apply_winsorization(
        test_enc, bounds=win_bounds, fit=False, verbose=False
    )

    return train_enc, test_enc


# ============================================================================
# MAIN PUBLIC FUNCTION
# ============================================================================

def run_xgb_pipeline(
    campaign_name: str,
    stock_targets: list[str] | None = None,
    features: list[str] | None = None,
    excluded_crops: list[str] | None = None,
    test_size: float = 0.20,
    cv_folds: int = 5,
    winsorize_quantiles: tuple[float, float] = (0.01, 0.99),
    verbose: bool = True,
) -> dict[str, Any]:
    """
    XGBoost fallback pipeline for benchmarking against LightGBM.

    Runs the same data preparation and split logic as ``run_pipeline()``,
    but uses XGBoost as the surrogate model. No Optuna tuning, no CART
    analysis, no model saving — evaluation only.

    Parameters
    ----------
    campaign_name : str
        Campaign identifier.
    stock_targets : list[str], optional
        Defaults to all STOCK_TARGETS.
    features : list[str], optional
        Defaults to ALL_FEATURES.
    excluded_crops : list[str], optional
    test_size : float
    cv_folds : int
    winsorize_quantiles : tuple[float, float]
    verbose : bool

    Returns
    -------
    dict
        ``{target: {"model": ..., "metrics_train": ..., "metrics_test": ...,
                    "cv_results": ...}}``

    Examples
    --------
    >>> from xgb_fallback import run_xgb_pipeline
    >>> results = run_xgb_pipeline("sobol_training_1_n2048")
    >>> for t, r in results.items():
    ...     print(f"{t}: R²={r['metrics_test']['test_r2']:.3f}")
    """
    print(_FALLBACK_BANNER)

    if stock_targets is None:
        stock_targets = STOCK_TARGETS
    if features is None:
        features = ALL_FEATURES

    campaign = get_campaign_paths(campaign_name)
    t_start = time.time()

    logger.info("XGBoost fallback | campaign=%s | targets=%s", campaign_name, stock_targets)

    # ── Load ─────────────────────────────────────────────────────────────────
    try:
        df = load_data(campaign)
        logger.info("Data loaded: %d rows × %d cols", *df.shape)
    except Exception as exc:
        logger.error("Data loading failed: %s", exc)
        raise

    # ── Prepare ───────────────────────────────────────────────────────────────
    df = add_derived_columns(df)
    if excluded_crops:
        df = filter_crops(df, excluded_crops=excluded_crops)
    df = clean(df, verbose=verbose)
    df = compute_effective_vars(df, verbose=verbose)

    # ── Split ─────────────────────────────────────────────────────────────────
    if has_rotations(df):
        train_df, test_df = stratified_split_by_rotation(
            df, test_size=test_size, random_state=RANDOM_STATE
        )
    else:
        train_df, test_df = split_by_simid(
            df, test_size=test_size, random_state=RANDOM_STATE
        )

    # ── Encode + Winsorise ────────────────────────────────────────────────────
    train_df, test_df = _encode_xgb(train_df, test_df, features, winsorize_quantiles)
    train_df = compute_ratios_from_stocks(train_df)
    test_df = compute_ratios_from_stocks(test_df)

    # ── XGBoost fallback output directory ────────────────────────────────────
    xgb_dir = campaign.figures_dir / "_xgb_fallback"
    xgb_metrics_dir = campaign.metrics_dir / "_xgb_fallback"
    xgb_dir.mkdir(parents=True, exist_ok=True)
    xgb_metrics_dir.mkdir(parents=True, exist_ok=True)

    # ── Train per target ──────────────────────────────────────────────────────
    results: dict[str, Any] = {}

    for target in stock_targets:
        logger.info("─── XGB | Target: %s ───", target)
        try:
            X_train, y_train = build_dataset(train_df, features, target)
            X_test, y_test = build_dataset(test_df, features, target)

            mask_tr = y_train.notna()
            mask_te = y_test.notna()
            X_train, y_train = X_train[mask_tr], y_train[mask_tr]
            X_test, y_test = X_test[mask_te], y_test[mask_te]

            if len(X_train) == 0:
                logger.warning("Target '%s': empty train set — skipped.", target)
                results[target] = {}
                continue

            groups_train = (
                train_df.loc[mask_tr, "SimID"]
                if "SimID" in train_df.columns else None
            )

            # CV
            model_cv = build_xgb(enable_categorical=True)
            cv_results = cross_validate(
                model_cv, X_train, y_train,
                groups=groups_train,
                n_splits=cv_folds,
                verbose=verbose,
            )
            logger.info(
                "XGB | %s | CV R²=%.3f ± %.3f",
                target, cv_results["cv_r2_mean"], cv_results["cv_r2_std"],
            )

            # Final model
            final_model = build_xgb(enable_categorical=True)
            final_model = train_final_model(final_model, X_train, y_train, verbose=verbose)

            y_pred_train = final_model.predict(X_train)
            y_pred_test = final_model.predict(X_test)

            metrics_train = compute_metrics(y_train.values, y_pred_train, prefix="train_")
            metrics_test = compute_metrics(y_test.values, y_pred_test, prefix="test_")

            logger.info(
                "✅ XGB | %s | R²_test=%.3f | RMSE=%.4f",
                target,
                metrics_test.get("test_r2", float("nan")),
                metrics_test.get("test_rmse", float("nan")),
            )

            # Plots
            plot_pred_vs_obs(
                y_test.values, y_pred_test,
                save_path=xgb_dir / f"xgb_pred_vs_obs_{target}.{FIGURE_FORMAT}",
                title=f"[XGB Fallback] Predictions vs Observations — {target}",
            )
            plot_residuals(
                y_test.values, y_pred_test,
                save_path=xgb_dir / f"xgb_residuals_{target}.{FIGURE_FORMAT}",
                title=f"[XGB Fallback] Residuals — {target}",
            )

            # Metrics
            save_metrics(
                {**metrics_train, **metrics_test, "cv_r2_mean": cv_results["cv_r2_mean"]},
                save_path=xgb_metrics_dir / f"xgb_metrics_{target}.csv",
                run_id=f"xgb_{target}",
            )
            save_cv_results(
                cv_results,
                save_path=xgb_metrics_dir / f"xgb_cv_{target}.csv",
            )

            results[target] = {
                "model":         final_model,
                "metrics_train": metrics_train,
                "metrics_test":  metrics_test,
                "cv_results":    cv_results,
            }

        except Exception as exc:
            logger.error("XGB | Target '%s' failed: %s", target, exc, exc_info=True)
            results[target] = {}

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print(f"\n{'─'*60}")
    print(f"  XGBoost Fallback complete — {elapsed:.1f}s")
    print(f"{'─'*60}")

    for target, res in results.items():
        if res:
            r2_lgb = res.get("metrics_test", {}).get("test_r2", float("nan"))
            print(f"  {target:30s}  XGB R²_test = {r2_lgb:.3f}")

    print("\n💡 Compare these results with LGB output in campaign.metrics_dir")
    print(f"   XGB outputs saved in: {xgb_metrics_dir}")

    return results


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MetAIsAFe XGBoost Fallback — validation only, not production."
    )
    parser.add_argument("--campaign", required=True, help="Campaign name")
    parser.add_argument(
        "--targets", nargs="+", default=None,
        help="Stock targets to evaluate (default: all)"
    )
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    run_xgb_pipeline(
        campaign_name=args.campaign,
        stock_targets=args.targets,
        test_size=args.test_size,
        cv_folds=args.cv_folds,
        verbose=not args.quiet,
    )

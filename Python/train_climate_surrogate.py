"""
MetAIsAFe — train_climate_surrogate.py
========================================
Trains 7 LightGBM regressors to predict climate features from (lat, lon, crop, year).

Usage:
    python train_climate_surrogate.py --campaign sobol_training_1_n2048
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupKFold

from config import get_campaign_paths, RANDOM_STATE, CV_N_FOLDS
from column_taxonomy import CLIMATE_FEATURES, CATEGORICAL_FEATURES_B2
from data.loader import load_data
from data.preparation import add_derived_columns
from modeling.evaluator import compute_metrics

log = logging.getLogger("metaisafe.climate_surrogate")

# Features for the climate surrogate
SURROGATE_FEATURES = ["latitude", "longitude", "main_crop", "Harvest_Year_Absolute"]
SURROGATE_TARGETS = CLIMATE_FEATURES  # 7 climate variables

# Lighter hyperparameters — climate is smooth, doesn't need heavy models
SURROGATE_PARAMS = {
    "n_estimators":      300,
    "learning_rate":     0.05,
    "num_leaves":        31,
    "min_child_samples": 30,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "reg_alpha":         0.1,
    "reg_lambda":        0.5,
    "random_state":      RANDOM_STATE,
    "n_jobs":            -1,
    "verbose":           -1,
    "importance_type":   "gain",
}


def train_climate_surrogate(campaign_name: str) -> dict:
    """Train 7 climate surrogate models."""
    campaign = get_campaign_paths(campaign_name)
    
    log.info("=" * 60)
    log.info("Climate Surrogate Training")
    log.info("=" * 60)
    
    # ── 1. Load data ─────────────────────────────────────────────
    df_raw = load_data(campaign.raw_meta)
    df = add_derived_columns(df_raw, verbose=False)
    
    # Verify longitude is available
    if "longitude" not in df.columns:
        raise ValueError(
            "Column 'longitude' not found in meta-table. "
            "It must be present for climate surrogate training."
        )
    
    # ── 2. Prepare dataset ───────────────────────────────────────
    # Keep all SimIDs (no population filter — climate doesn't depend on tree/yield status)
    required = SURROGATE_FEATURES + SURROGATE_TARGETS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    # Deduplicate: one row per (SimID, Harvest_Year_Absolute)
    df_clim = (
        df[["SimID"] + required]
        .drop_duplicates(subset=["SimID", "Harvest_Year_Absolute"])
        .dropna(subset=required)
        .reset_index(drop=True)
    )
    
    log.info("Training data: %d rows | %d SimIDs", 
             len(df_clim), df_clim["SimID"].nunique())
    
    # Encode categoricals
    df_clim["main_crop"] = df_clim["main_crop"].astype("category")
    
    X = df_clim[SURROGATE_FEATURES].copy()
    groups = df_clim["SimID"]
    
    # ── 3. Train one model per climate target ────────────────────
    models = {}
    metrics_all = {}
    
    cv = GroupKFold(n_splits=CV_N_FOLDS)
    
    for target in SURROGATE_TARGETS:
        log.info("\n--- %s ---", target)
        y = df_clim[target].values
        
        # Cross-validation
        cv_scores = []
        for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y, groups)):
            model_cv = LGBMRegressor(**SURROGATE_PARAMS)
            model_cv.fit(X.iloc[tr_idx], y[tr_idx])
            y_pred = model_cv.predict(X.iloc[val_idx])
            m = compute_metrics(y[val_idx], y_pred, prefix="val_")
            cv_scores.append(m["val_r2"])
        
        mean_r2 = np.mean(cv_scores)
        std_r2 = np.std(cv_scores)
        log.info("  CV R²: %.4f ± %.4f", mean_r2, std_r2)
        
        # Final model on all data
        model = LGBMRegressor(**SURROGATE_PARAMS)
        model.fit(X, y)
        
        # Feature importances
        fi = dict(zip(SURROGATE_FEATURES, model.feature_importances_))
        log.info("  Feature importances: %s", 
                 {k: round(v, 1) for k, v in sorted(fi.items(), key=lambda x: -x[1])})
        
        # Save
        model_key = f"climate_surrogate_{target}"
        model_path = campaign.metamodels_dir / f"{model_key}.joblib"
        joblib.dump(model, model_path)
        
        models[target] = model
        metrics_all[target] = {"cv_r2_mean": mean_r2, "cv_r2_std": std_r2}
        
        log.info("  Saved: %s", model_path.name)
    
    # ── 4. Summary ───────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("CLIMATE SURROGATE SUMMARY")
    log.info("=" * 60)
    for target, m in metrics_all.items():
        status = "✅" if m["cv_r2_mean"] > 0.7 else "⚠️"
        log.info("  %s %-35s R² = %.4f ± %.4f", 
                 status, target, m["cv_r2_mean"], m["cv_r2_std"])
    
    return {"models": models, "metrics": metrics_all}


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", "-c", required=True)
    args = parser.parse_args()
    
    train_climate_surrogate(args.campaign)
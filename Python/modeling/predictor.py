"""
MetAIsAFe — modeling/predictor.py
===================================
Inference pipeline: from user parameters to 40-year prediction curves.

Architecture horizon v4.1 — FULL ALIGNMENT
-------------------------------------------
ALL regression targets are now predicted at discrete horizons (h=5,10,20,30,40)
and interpolated to 40-year curves via PCHIP.

    Stage 1  : carbonStem_AF_h{h}, carbonStem_TF_h{h}
               └─ log1p-trained → expm1 back-transform
               └─ features: static + CLIMATE aggregated over [1→h]

    Stage 2a : yield_AF_h{h}
               └─ NO log1p (distribution ~symmetric)
               └─ features: static + CLIMATE + carbonStem_AF_h{h} (Stage 1 injection)

    Stage 2b : yield_TA_h{h}
               └─ NO log1p
               └─ features: static + CLIMATE only (pure crop, NO tree interaction)

CASCADE ROUTING
---------------
    CLF1 → tree_status  : 0=failed | 1=stunted | 2=ok
    CLF2 → yield_failed : 0=ok     | 1=failed

    Routing outcomes:
        tree_ok    × yield_ok   → Stage 1 + Stage 2a + Stage 2b
        tree_stunted × yield_ok → stunted fallback (carbon) + Stage 2a + Stage 2b
        tree_failed × yield_ok  → carbon=0 + Stage 2b only
        any        × yield_fail → yield_AF=0, yield_TA=0 (carbon still predicted)
        tree_failed × yield_fail → all outputs = 0

MODEL KEYS (disk / in-memory dict)
-----------------------------------
    "carbonStem_AF_h{h}"  — Stage 1
    "carbonStem_TF_h{h}"  — Stage 1
    "yield_AF_h{h}"       — Stage 2a
    "yield_TA_h{h}"       — Stage 2b
    (no more "yield_AF" or "yield_TA" flat keys)

Author  : Étienne SABY
Updated : 2026-04  (v4.1 — full horizon alignment)
"""
from __future__ import annotations
from _version import __version__

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from config import (
    CampaignPaths,
    RANDOM_STATE,
    CARBON_HORIZONS,
    MIN_ENRICH_HORIZON
)
from column_taxonomy import (
    ACTIVE_FEATURES_B2,
    CATEGORICAL_FEATURES_B2,
    CLIMATE_FEATURES,
    CLIMATE_RECENT_WINDOW,
    CLIMATE_MEAN_VARS, 
    CLIMATE_SUM_VARS,
    NOMINAL_POPULATION,
    STEP2_AF_FEATURES,
    STEP2_TA_FEATURES,
    STEP2_TARGETS,
)
from modeling.classifiers import (
    CLF1_FEATURES, CLF2_FEATURES,
    apply_geographic_rule, predict_routing,
    TREE_STATUS_FAILED, TREE_STATUS_OK, TREE_STATUS_STUNTED,  # compat diagnostics
    TREE_BINARY_DEGRADED, TREE_BINARY_OK,                      # v4.1 routing
    STUNTED_PROBA_THRESHOLD,                                    # v4.1 sub-routing
)

from utils.io_utils import load_model

log = logging.getLogger(__name__)

N_YEARS: int = 40

# Horizon targets — ALL in horizon mode now
STAGE1_TARGETS  = ["carbonStem_AF", "carbonStem_TF"]
STAGE2_TARGETS  = STEP2_TARGETS
ALL_HORIZON_TARGETS = STAGE1_TARGETS


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_all_models(
    campaign: CampaignPaths,
    horizons: list[int] | None = None,
) -> dict[str, Any]:
    """
    Load all horizon-based regression models from a campaign directory.

    Expected filenames on disk:
        lgbm_carbonStem_AF_h{h}.joblib
        lgbm_carbonStem_TF_h{h}.joblib
        lgbm_yield_AF_h{h}.joblib
        lgbm_yield_TA_h{h}.joblib

    Returns
    -------
    dict  keys = "<target>_h<horizon>"  e.g. "carbonStem_AF_h40"
    """
    if horizons is None:
        horizons = CARBON_HORIZONS

    models: dict[str, Any] = {}
    for target in ALL_HORIZON_TARGETS:
        for h in horizons:
            key  = f"{target}_h{h}"
            path = campaign.metamodels_dir / f"lgbm_{key}.joblib"
            if path.exists():
                models[key] = load_model(path)
                log.debug("Loaded: %s", key)
            else:
                log.warning("Model not found: '%s' → %s", key, path)
    
    # ── Stage 2 : row-by-row (v5.0) ─────────────────────────────────────
    for target in STAGE2_TARGETS:
        key  = target
        path = campaign.metamodels_dir / f"lgbm_{target}_rowwise.joblib"
        if path.exists():
            models[key] = load_model(path)
            log.debug("Loaded Stage 2 rowwise: %s", key)
        else:
            log.warning("Stage 2 model not found: '%s' → %s", key, path)

    log.info(
        "Loaded %d Stage 1 + %d Stage 2 models.",
        len([k for k in models if "_h" in k]),
        len([k for k in models if "_h" not in k and k in STAGE2_TARGETS]),
    )

    log.info(
        "Loaded %d / %d expected horizon models.",
        len(models), len(ALL_HORIZON_TARGETS) * len(horizons),
    )
    return models


def load_all_classifiers(campaign: CampaignPaths) -> tuple[Any, Any]:
    from modeling.classifiers import load_classifiers
    return load_classifiers(campaign)


def _resolve_models(
    campaign: CampaignPaths | None,
    models: dict[str, Any] | None,
    clf1: Any | None,
    clf2: Any | None,
) -> tuple[dict[str, Any], Any, Any]:
    if models is None:
        if campaign is None:
            raise ValueError(
                "Either 'models' (in-memory dict) or 'campaign' must be provided."
            )
        models = load_all_models(campaign)
    if clf1 is None:
        if campaign is None:
            raise ValueError(
                "Either 'clf1' or 'campaign' must be provided."
            )
        clf1, clf2 = load_all_classifiers(campaign)
    return models, clf1, clf2


# ============================================================================
# INFERENCE GRID BUILDER
# ============================================================================

def build_inference_grid(
    params: dict[str, Any],
    n_years: int = N_YEARS,
    horizons: list[int] | None = None,
) -> dict[int, pd.DataFrame]:
    """
    Generate one feature DataFrame per horizon for inference.

    For each horizon h, produces a 1-row DataFrame that EXACTLY mirrors
    the feature set generated by build_horizon_dataset() during training:
        - Static features  (pedology, geometry, crop type)
        - Climate aggregates over years [1 → h] — v4.2 enriched:

          For CLIMATE_SUM_VARS  (GDD, ETP, precip, radiation):
              {var}_cum         : sum over [1→h]
              {var}_std         : std over [1→h]  (population, ddof=0)
              {var}_p10         : 10th percentile over [1→h]
              {var}_p90         : 90th percentile over [1→h]
              {var}_trend       : OLS slope over [1→h]
              {var}_recent_mean : mean over [h-RECENT_WINDOW+1, h]

          For CLIMATE_MEAN_VARS (frost, Tmax_extreme, Tmin_extreme):
              {var}_mean        : mean over [1→h]
              {var}_std         : std over [1→h]  (population, ddof=0)
              {var}_p10         : 10th percentile over [1→h]
              {var}_p90         : 90th percentile over [1→h]
              {var}_trend       : OLS slope over [1→h]
              {var}_recent_mean : mean over [h-RECENT_WINDOW+1, h]

    Parameters
    ----------
    params : dict
        Required keys:
        - All ACTIVE_FEATURES_B2 static features (excl. Harvest_Year_Absolute)
        - Climate features as scalar (broadcast over all n_years)
          or array of length n_years (annual values).
    n_years : int, default 40
    horizons : list of int, optional
        Defaults to CARBON_HORIZONS from config.

    Returns
    -------
    dict[int, pd.DataFrame]
        Keys = horizon values. Each DataFrame has exactly 1 row.
        Column set matches build_horizon_dataset() output exactly.

    Raises
    ------
    ValueError
        If required static features are missing from params, or if a
        climate array has wrong length.

    Notes
    -----
    - Scalar climate params are broadcast to n_years annual values.
      This matches the training assumption that climate is stationary
      (constant yearly conditions = representative mean scenario).
    - _trend computed via OLS (np.polyfit degree 1) over year index [0, h-1].
      Returns 0.0 if h < 2 (i.e., h=1 edge case, not used in practice).
    - _recent_mean uses window [h-RECENT_WINDOW+1, h].
      For h <= RECENT_WINDOW, this equals the full-period mean — correct.
    - ddof=0 for std to avoid NaN when h=1 (matches build_horizon_dataset).
    """
    if horizons is None:
        horizons = CARBON_HORIZONS

    year_col     = "Harvest_Year_Absolute"
    static_feats = [
        f for f in ACTIVE_FEATURES_B2
        if f != year_col and f not in CLIMATE_FEATURES
    ]

    missing = [f for f in static_feats if f not in params]
    if missing:
        raise ValueError(f"build_inference_grid: missing parameters {missing}")

    # ── Arrays climatiques annuels (longueur n_years) ─────────────────────
    climate_arrays: dict[str, np.ndarray] = {}
    for var in CLIMATE_FEATURES:
        val = params.get(var)
        if val is None:
            log.warning("Climate feature '%s' absent from params — set to 0.", var)
            climate_arrays[var] = np.zeros(n_years)
        elif isinstance(val, (list, np.ndarray)):
            arr = np.asarray(val, dtype=float)
            if len(arr) != n_years:
                raise ValueError(
                    f"'{var}': array length {len(arr)}, expected {n_years}."
                )
            climate_arrays[var] = arr
        else:
            climate_arrays[var] = np.full(n_years, float(val))

    # ── Helpers vectorisés (miroir exact de build_horizon_dataset) ────────
    def _trend_arr(arr: np.ndarray) -> float:
        if len(arr) < 2:
            return 0.0
        yr   = np.arange(len(arr), dtype=float)
        yr_m = yr.mean()
        denom = ((yr - yr_m) ** 2).sum()
        if denom == 0:
            return 0.0
        return float(((yr - yr_m) * (arr - arr.mean())).sum() / denom)

    def _p10_arr(arr: np.ndarray) -> float:
        return float(np.nanpercentile(arr, 10)) if len(arr) > 0 else float("nan")

    def _p90_arr(arr: np.ndarray) -> float:
        return float(np.nanpercentile(arr, 90)) if len(arr) > 0 else float("nan")

    # ── Construction d'une ligne par horizon ──────────────────────────────
    grid: dict[int, pd.DataFrame] = {}

    for h in horizons:
        use_enriched = (h > MIN_ENRICH_HORIZON)
        row: dict[str, Any] = {}

        # Features statiques
        for feat in static_feats:
            row[feat] = params[feat]

        # Fenêtre long-terme [1, h] → indices [0, h-1]
        idx_lt  = slice(0, h)

        for var in CLIMATE_SUM_VARS:
            arr_lt = climate_arrays[var][idx_lt]
            row[f"{var}_cum"] = float(arr_lt.sum())
            row[f"{var}_std"] = float(arr_lt.std())    # ddof=0 (numpy default)
            if use_enriched:
                row[f"{var}_p10"]   = _p10_arr(arr_lt)
                row[f"{var}_p90"]   = _p90_arr(arr_lt)
                row[f"{var}_trend"] = _trend_arr(arr_lt)
                if h > CLIMATE_RECENT_WINDOW:
                    arr_rt = climate_arrays[var][max(0, h - CLIMATE_RECENT_WINDOW):h]
                    row[f"{var}_recent_mean"] = float(arr_rt.mean())

        for var in CLIMATE_MEAN_VARS:
            arr_lt = climate_arrays[var][idx_lt]
            row[f"{var}_mean"] = float(arr_lt.mean()) if len(arr_lt) > 0 else 0.0
            row[f"{var}_std"]  = float(arr_lt.std())   # ddof=0
            if use_enriched:
                row[f"{var}_p10"]   = _p10_arr(arr_lt)
                row[f"{var}_p90"]   = _p90_arr(arr_lt)
                row[f"{var}_trend"] = _trend_arr(arr_lt)
                if h > CLIMATE_RECENT_WINDOW:
                    arr_rt = climate_arrays[var][max(0, h - CLIMATE_RECENT_WINDOW):h]
                    row[f"{var}_recent_mean"] = float(arr_rt.mean())

        # Encodage catégoriel
        df_h = pd.DataFrame([row])
        for col in CATEGORICAL_FEATURES_B2:
            if col in df_h.columns:
                df_h[col] = df_h[col].astype("category")

        grid[h] = df_h
        log.debug(
            "grid[h=%d] built — %d features (enriched=%s)",
            h, len(df_h.columns), use_enriched,
        )

    return grid


# ============================================================================
# INFERENCE ROW BUILDER — Step 2 row-by-row (v5.0)
# ============================================================================

def build_inference_rows(
    params: dict[str, Any],
    cs_af_trajectory: np.ndarray,
    n_years: int = N_YEARS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Génère les DataFrames d'inférence row-by-row pour yield_AF et yield_TA.

    Chaque ligne = 1 année (t = 1 → n_years).
    Remplace la logique d'agrégation horizon + PCHIP pour Step 2 (v5.0).

    Parameters
    ----------
    params : dict
        Paramètres utilisateur. Doit contenir :
          - toutes les features statiques (DESIGN, SOIL, GEO, CATEGORICAL)
          - les CLIMATE_FEATURES comme scalaire (broadcast) ou array (n_years,)
    cs_af_trajectory : np.ndarray, shape (n_years,)
        Trajectoire carbonStem_AF prédite par Step 1 (expm1 back-transformée).
        Injectée comme feature dans df_af uniquement.
    n_years : int, default 40

    Returns
    -------
    df_af : pd.DataFrame, shape (n_years, len(STEP2_AF_FEATURES))
        Feature matrix pour yield_AF — inclut carbonStem_AF[t].
    df_ta : pd.DataFrame, shape (n_years, len(STEP2_TA_FEATURES))
        Feature matrix pour yield_TA — sans carbonStem.

    Notes
    -----
    - Les CLIMATE_FEATURES scalaires sont broadcastés sur n_years années.
      Cela suppose un climat stationnaire (scénario moyen représentatif).
      Pour un scénario annualisé, passer des arrays de longueur n_years.
    - Harvest_Year_Absolute = 1 → n_years (entiers).
    - Les colonnes catégorielles sont castées en dtype='category' (LightGBM natif).
    """
    years = np.arange(1, n_years + 1)

    # ── 1. Résolution des variables climatiques ──────────────────────────
    climate_arrays: dict[str, np.ndarray] = {}
    for var in CLIMATE_FEATURES:
        val = params.get(var)
        if val is None:
            log.warning("Climate feature '%s' absent de params → broadcast 0.0", var)
            climate_arrays[var] = np.zeros(n_years)
        elif np.isscalar(val):
            climate_arrays[var] = np.full(n_years, float(val))
        else:
            arr = np.asarray(val, dtype=float)
            if len(arr) != n_years:
                raise ValueError(
                    f"Climate array '{var}' a une longueur {len(arr)}, "
                    f"attendu {n_years}."
                )
            climate_arrays[var] = arr

    # ── 2. Features statiques (design, soil, geo, categorical) ──────────
    static_feats = [
        f for f in STEP2_TA_FEATURES
        if f not in CLIMATE_FEATURES
        and f != "Harvest_Year_Absolute"
    ]
    missing_static = [f for f in static_feats if f not in params]
    if missing_static:
        raise ValueError(
            f"Features statiques manquantes dans params : {missing_static}"
        )

    # ── 3. Construction des DataFrames row-by-row ────────────────────────
    rows_af: list[dict[str, Any]] = []
    rows_ta: list[dict[str, Any]] = []

    for i, t in enumerate(years):
        row_base: dict[str, Any] = {
            "Harvest_Year_Absolute": int(t),
        }
        # Features statiques
        for feat in static_feats:
            row_base[feat] = params[feat]
        # Climat annuel à l'année t
        for var in CLIMATE_FEATURES:
            row_base[var] = float(climate_arrays[var][i])

        # yield_AF : + carbonStem_AF[t]
        row_af = {**row_base, "carbonStem_AF": float(cs_af_trajectory[i])}
        rows_af.append(row_af)

        # yield_TA : pas de carbonStem
        rows_ta.append(row_base.copy())

    df_af = pd.DataFrame(rows_af)[STEP2_AF_FEATURES]
    df_ta = pd.DataFrame(rows_ta)[STEP2_TA_FEATURES]

    # ── 4. Encodage catégoriel ───────────────────────────────────────────
    for col in CATEGORICAL_FEATURES_B2:
        if col in df_af.columns:
            df_af[col] = df_af[col].astype("category")
        if col in df_ta.columns:
            df_ta[col] = df_ta[col].astype("category")

    log.debug(
        "build_inference_rows: df_af %s, df_ta %s",
        df_af.shape, df_ta.shape,
    )
    return df_af, df_ta

# ============================================================================
# CORE HORIZON PREDICTOR (internal)
# ============================================================================

def _predict_horizon_target(
    target: str,
    grid: dict[int, pd.DataFrame],
    models: dict[str, Any],
    horizons: list[int],
    cs_af_by_h: dict[int, float] | None = None,
    log_transform: bool = False,
) -> np.ndarray:
    """
    Predict a single target at all horizons → PCHIP → 40-year trajectory.

    Parameters
    ----------
    target : str
        e.g. "carbonStem_AF", "yield_AF", "yield_TA"
    grid : dict[int, pd.DataFrame]
        Output of build_inference_grid().
    models : dict
        All loaded models, keys = "<target>_h<h>".
    horizons : list of int
    cs_af_by_h : dict[int, float] or None
        Predicted carbonStem_AF values per horizon (for yield_AF injection).
        Required when target == "yield_AF".  Ignored for all other targets.
    log_transform : bool, default False
        Apply expm1() back-transform to model output.
        True for carbonStem targets (trained on log1p).
        False for yield targets (trained on raw values).

    Returns
    -------
    np.ndarray  shape (N_YEARS,)
    """
    years = np.arange(1, N_YEARS + 1, dtype=float)

    h_points: list[int]   = []
    h_values: list[float] = []

    for h in horizons:
        key = f"{target}_h{h}"
        if key not in models:
            log.debug("Model '%s' absent — skipping horizon %d.", key, h)
            continue
        if h not in grid:
            log.warning("grid[h=%d] missing — skipping '%s'.", h, key)
            continue

        model_h = models[key]

        # Build feature row for this horizon
        df_h = grid[h].copy()

        # Align to model's feature names
        try:
            feat_names = list(model_h.feature_name_)
        except AttributeError:
            feat_names = list(df_h.columns)

        avail = [f for f in feat_names if f in df_h.columns]
        missing_feats = [f for f in feat_names if f not in df_h.columns]
        if missing_feats:
            log.warning(
                "'%s' h=%d: %d features missing from grid → %s",
                key, h, len(missing_feats), missing_feats[:5],
            )

        pred_raw = model_h.predict(df_h[avail])
        pred_val = float(np.clip(
            np.expm1(pred_raw[0]) if log_transform else pred_raw[0],
            0.0, None,
        ))

        h_points.append(h)
        h_values.append(pred_val)
        log.debug("'%s' h=%d → %.4f", target, h, pred_val)

    # PCHIP interpolation → 40-year trajectory
    if len(h_points) >= 2:
        pchip = PchipInterpolator(
            np.array(h_points, dtype=float),
            np.array(h_values),
            extrapolate=False,
        )
        trajectory = np.clip(pchip(years), 0.0, None)
        trajectory = np.where(np.isnan(trajectory), 0.0, trajectory)
    elif len(h_points) == 1:
        trajectory = np.full(N_YEARS, h_values[0])
    else:
        log.warning("No valid horizon predictions for '%s' — returning zeros.", target)
        trajectory = np.zeros(N_YEARS)

    return trajectory


# ============================================================================
# CASCADE PREDICTOR
# ============================================================================

def predict_cascade(
    grid: dict[int, pd.DataFrame],
    params: dict[str, Any],
    models: dict[str, Any],
    clf1: Any,
    clf2: Any | None,
    stunted_model: dict[str, Any] | None = None,
    horizons: list[int] | None = None,
    return_routing: bool = False,
    log_transform_stage1: bool = True,
) -> dict[str, Any]:
    """
    Full cascade prediction — Architecture horizon v4.1.

    Stage 1 → Stage 2a (yield_AF + carbonStem injection)
             → Stage 2b (yield_TA — no tree features)

    All targets predicted at discrete horizons + PCHIP interpolation.

    Parameters
    ----------
    grid : dict[int, pd.DataFrame]
        Output of build_inference_grid(). Keys = horizon values.
    params : dict
        Raw user parameters (used for routing + stunted crop lookup).
    models : dict[str, estimator]
        All models. Expected keys: "<target>_h<h>"
        e.g. "carbonStem_AF_h40", "yield_AF_h20", "yield_TA_h10"
    clf1 : LGBMClassifier (3-class)
    clf2 : LGBMClassifier or None (geographic fallback)
    stunted_model : dict or None
    horizons : list of int, optional
    return_routing : bool, default False
    log_transform_stage1 : bool, default True
        expm1() for carbonStem targets only.

    Returns
    -------
    dict:
        predictions  : dict[str, np.ndarray]  one per target (length N_YEARS)
        population   : str
        tree_failed  : int
        yield_failed : int
        routing      : pd.DataFrame  (if return_routing=True)
    """
    if horizons is None:
        horizons = CARBON_HORIZONS

    # ── 1. Routing ────────────────────────────────────────────────────────
    # Build routing features
    X_clf_dict = {}
    for col in CLF1_FEATURES + CLF2_FEATURES:
        if col not in params:
            continue
        val = params[col]
        # Si c'est un array, prendre la première valeur (année 1)
        if hasattr(val, '__len__') and not isinstance(val, str):
            X_clf_dict[col] = float(val[0])
        else:
            X_clf_dict[col] = val

    X_clf = pd.DataFrame([X_clf_dict])
    
    for col in CATEGORICAL_FEATURES_B2:
        if col in X_clf.columns:
            X_clf[col] = X_clf[col].astype("category")

    routing      = predict_routing(X_clf, clf1, clf2, return_proba=False)
    tree_status  = int(routing["tree_status"].iloc[0])   # 0=degraded, 1=ok (binary v4.1)
    tree_ok_proba = float(routing["tree_ok_proba"].iloc[0])  # P(tree_ok) for sub-routing
    yield_failed = int(routing["yield_failed"].iloc[0])
    population   = str(routing["population"].iloc[0])

    # tree_failed flag: True when tree_status=degraded AND low P(tree_ok)
    # → used in output dict for downstream consumers
    is_tree_degraded = (tree_status == TREE_BINARY_DEGRADED)
    # Sub-routing within tree_degraded:
    #   P(tree_ok) > STUNTED_PROBA_THRESHOLD → apply stunted_model (some carbon)
    #   P(tree_ok) <= STUNTED_PROBA_THRESHOLD → zero-carbon (fully failed)
    use_stunted_model = is_tree_degraded and (tree_ok_proba > STUNTED_PROBA_THRESHOLD)
    tree_failed = int(is_tree_degraded and not use_stunted_model)

    log.info(
        "Routing → %s (tree_status=%d, P(tree_ok)=%.3f, yield_failed=%d)",
        population, tree_status, tree_ok_proba, yield_failed,
    )
    if is_tree_degraded:
        if use_stunted_model:
            log.info(
                "  Sub-routing: tree_degraded + P(tree_ok)=%.3f > %.2f → stunted_model",
                tree_ok_proba, STUNTED_PROBA_THRESHOLD,
            )
        else:
            log.info(
                "  Sub-routing: tree_degraded + P(tree_ok)=%.3f <= %.2f → zero-carbon",
                tree_ok_proba, STUNTED_PROBA_THRESHOLD,
            )

    # ── 2. Init zero predictions ──────────────────────────────────────────
    preds: dict[str, np.ndarray] = {
        t: np.zeros(N_YEARS) for t in ALL_HORIZON_TARGETS
    }

    # ── 3. Full failure shortcut ──────────────────────────────────────────
    if is_tree_degraded and yield_failed == 1:
        log.info("Full failure (tree_degraded × yield_fail) — all outputs = 0.")
        out = {
            "predictions": preds,
            "population":  population,
            "tree_failed": 1,
            "yield_failed": yield_failed,
        }
        if return_routing:
            out["routing"] = routing
        return out

    # ── 4. Stage 1 — carbonStem ───────────────────────────────────────────
    cs_af_by_h: dict[int, float] = {}

    if tree_status == TREE_BINARY_OK:
        # tree_ok → predict via horizon models (unchanged)
        for s1_target in STAGE1_TARGETS:
            preds[s1_target] = _predict_horizon_target(
                target=s1_target,
                grid=grid,
                models=models,
                horizons=horizons,
                log_transform=log_transform_stage1,
            )
        for h in horizons:
            key = f"carbonStem_AF_h{h}"
            if key in models and h in grid:
                try:
                    feat_names = list(models[key].feature_name_)
                except AttributeError:
                    feat_names = list(grid[h].columns)
                avail    = [f for f in feat_names if f in grid[h].columns]
                pred_raw = models[key].predict(grid[h][avail])
                cs_af_by_h[h] = float(np.clip(
                    np.expm1(pred_raw[0]) if log_transform_stage1 else pred_raw[0],
                    0.0, None,
                ))

    elif use_stunted_model:
        # tree_degraded + P(tree_ok) > threshold → stunted_model fallback
        if stunted_model is None:
            log.warning("use_stunted_model=True but stunted_model not provided → carbonStem = 0.")
        else:
            main_crop   = str(params.get("main_crop", ""))
            pred_final  = float(
                stunted_model.get("median_by_crop", {}).get(
                    main_crop, stunted_model.get("global_median", 0.0)
                )
            )
            profile_raw = stunted_model.get("profile_median", {})
            p_years     = np.array(sorted(int(k) for k in profile_raw))
            p_ratios    = np.array([profile_raw[k]
                                    for k in sorted(profile_raw, key=int)])
            if len(p_years) >= 2:
                pchip_s = PchipInterpolator(
                    p_years.astype(float), p_ratios, extrapolate=True
                )
                ratios = np.clip(
                    pchip_s(np.arange(1, N_YEARS + 1, dtype=float)), 0, 1
                )
            else:
                ratios = np.ones(N_YEARS)

            preds["carbonStem_AF"] = np.clip(ratios * pred_final, 0.0, None)
            preds["carbonStem_TF"] = np.zeros(N_YEARS)
            log.info(
                "Stunted fallback (sub-routing) — '%s' pred_final=%.3f kgC/tree  "
                "P(tree_ok)=%.3f",
                main_crop, pred_final, tree_ok_proba,
            )
            for h in horizons:
                cs_af_by_h[h] = float(preds["carbonStem_AF"][h - 1]) \
                    if h <= N_YEARS else 0.0

    else:
        # tree_degraded + P(tree_ok) <= threshold → zero-carbon (fully failed)
        log.info(
            "Zero-carbon fallback (tree_degraded, P(tree_ok)=%.3f <= %.2f).",
            tree_ok_proba, STUNTED_PROBA_THRESHOLD,
        )
        # cs_af_by_h stays empty → yield_AF will receive 0 for carbonStem
        
    # ── 5. Stage 2 — yield_AF & yield_TA (row-by-row, v5.0) ────────────
    if yield_failed:
        preds["yield_AF"] = np.zeros(N_YEARS)
        preds["yield_TA"] = np.zeros(N_YEARS)
        log.info("yield_failed=True → yield_AF = yield_TA = 0.")

    else:
        cs_af_trajectory = preds.get("carbonStem_AF", np.zeros(N_YEARS))

        df_af, df_ta = build_inference_rows(
            params=params,
            cs_af_trajectory=cs_af_trajectory,
            n_years=N_YEARS,
        )

        # yield_AF
        model_yield_af = models.get("yield_AF")
        if model_yield_af is not None:
            preds["yield_AF"] = np.clip(
                model_yield_af.predict(df_af), 0.0, None
            )
        else:
            log.warning("Modèle 'yield_AF' absent → zeros.")
            preds["yield_AF"] = np.zeros(N_YEARS)

        # yield_TA
        model_yield_ta = models.get("yield_TA")
        if model_yield_ta is not None:
            preds["yield_TA"] = np.clip(
                model_yield_ta.predict(df_ta), 0.0, None
            )
        else:
            log.warning("Modèle 'yield_TA' absent → zeros.")
            preds["yield_TA"] = np.zeros(N_YEARS)

    out: dict[str, Any] = {
        "predictions":    preds,
        "population":     population,
        "tree_failed":    tree_failed,
        "tree_degraded":  int(is_tree_degraded),
        "tree_ok_proba":  tree_ok_proba,
        "yield_failed":   yield_failed,
    }
    if return_routing:
        out["routing"] = routing
    return out


# ============================================================================
# SINGLE SIMULATION
# ============================================================================

def predict_single_sim(
    params: dict[str, Any],
    models: dict[str, Any] | None = None,
    clf1: Any | None = None,
    clf2: Any | None = None,
    campaign: CampaignPaths | None = None,
    stunted_model: dict[str, Any] | None = None,
    n_years: int = N_YEARS,
    return_routing: bool = False,
    log_transform_stage1: bool = True,
) -> dict[str, Any]:
    """Single parameter set → 40-year prediction curves (horizon architecture)."""
    models, clf1, clf2 = _resolve_models(campaign, models, clf1, clf2)
    grid   = build_inference_grid(params, n_years=n_years)
    result = predict_cascade(
        grid=grid,
        params=params,
        models=models,
        clf1=clf1,
        clf2=clf2,
        stunted_model=stunted_model,
        return_routing=return_routing,
        log_transform_stage1=log_transform_stage1,
    )
    result["years"] = np.arange(1, n_years + 1)
    return result


# ============================================================================
# BATCH PREDICTION
# ============================================================================

def predict_batch(
    params_list: list[dict[str, Any]],
    models: dict[str, Any] | None = None,
    clf1: Any | None = None,
    clf2: Any | None = None,
    campaign: CampaignPaths | None = None,
    stunted_model: dict[str, Any] | None = None,
    n_years: int = N_YEARS,
    verbose: bool = True,
    log_transform_stage1: bool = True,
) -> pd.DataFrame:
    """N parameter sets → long-format DataFrame (scenario grid)."""
    models, clf1, clf2 = _resolve_models(campaign, models, clf1, clf2)
    records: list[pd.DataFrame] = []

    for i, params in enumerate(params_list):
        if verbose and (i % max(1, len(params_list) // 10) == 0):
            log.info("Batch inference: %d / %d", i + 1, len(params_list))
        try:
            grid   = build_inference_grid(params, n_years=n_years)
            result = predict_cascade(
                grid=grid,
                params=params,
                models=models,
                clf1=clf1,
                clf2=clf2,
                stunted_model=stunted_model,
                log_transform_stage1=log_transform_stage1,
            )
        except Exception as exc:
            log.error("Scenario %d failed: %s", i, exc, exc_info=True)
            continue
        records.append(format_output(result, scenario_id=i))

    if not records:
        log.warning("predict_batch: no successful predictions.")
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True)


# ============================================================================
# OUTPUT FORMATTING
# ============================================================================

def format_output(
    result: dict[str, Any],
    scenario_id: int | str | None = None,
) -> pd.DataFrame:
    """
    Tidy long-format DataFrame — R/Shiny compatible.

    Columns: [scenario_id (opt), year, target, value,
               population, tree_failed, yield_failed]
    """
    preds = result["predictions"]
    years = result.get("years", np.arange(1, N_YEARS + 1))

    rows: list[dict] = []
    for target, values in preds.items():
        for yr, val in zip(years, values):
            row: dict[str, Any] = {
                "year":        int(yr),
                "target":      target,
                "value":       float(val),
                "population":  result["population"],
                "tree_failed": result["tree_failed"],
                "yield_failed": result["yield_failed"],
            }
            if scenario_id is not None:
                row["scenario_id"] = scenario_id
            rows.append(row)

    df = pd.DataFrame(rows)
    base_cols = ["year", "target", "value",
                 "population", "tree_failed", "yield_failed"]
    cols = (["scenario_id"] + base_cols) if scenario_id is not None else base_cols
    return df[cols]


# # ============================================================================
# # STAGE 1 PREDS — ROW-LEVEL RECONSTRUCTION (training helper)
# # ============================================================================

# def build_stage1_preds_rowlevel(
#     df_split: pd.DataFrame,
#     horizon_models: dict,
#     horizon_datasets: dict,
#     horizons: list[int] | None = None,
#     log_transform: bool = True,
# ) -> pd.DataFrame:
#     """
#     Reconstruct Stage 1 predictions at SimID level for training injection.

#     Used in full_training.py to feed carbonStem_AF_h{h} into
#     yield_AF_h{h} training (Stage 2a). NOT used for yield_TA.

#     Parameters
#     ----------
#     df_split : pd.DataFrame  — must contain 'SimID' column
#     horizon_models : dict    — {(target, h): fitted_model}
#     horizon_datasets : dict  — {(target, h): (X_h, y_h)}
#     horizons : list of int
#     log_transform : bool, default True  — must match Stage 1 training

#     Returns
#     -------
#     pd.DataFrame
#         Columns: ["carbonStem_AF_h{h}" for h in horizons]
#                  + ["carbonStem_TF_h{h}" for h in horizons]
#         Index aligned to df_split.index.
#     """
#     if horizons is None:
#         horizons = CARBON_HORIZONS

#     stage1_cols = (
#         [f"carbonStem_AF_h{h}" for h in horizons] +
#         [f"carbonStem_TF_h{h}" for h in horizons]
#     )
#     preds_by_simid: dict[str, dict[str, float]] = {}

#     for (target, h), model_h in horizon_models.items():
#         col_name = f"{target}_h{h}"
#         key      = (target, h)
#         if key not in horizon_datasets:
#             log.warning("horizon_datasets missing key %s — skipping.", key)
#             continue

#         X_h_full, _ = horizon_datasets[key]
#         assert len(X_h_full) > 0, f"Empty X for {key}"

#         preds_raw = model_h.predict(X_h_full)
#         preds_val = (
#             np.clip(np.expm1(preds_raw), 0.0, None)
#             if log_transform
#             else np.clip(preds_raw, 0.0, None)
#         )

#         for sim, pred in zip(X_h_full.index, preds_val):
#             sim_str = str(sim)
#             if sim_str not in preds_by_simid:
#                 preds_by_simid[sim_str] = {}
#             preds_by_simid[sim_str][col_name] = float(pred)

#     rows = [
#         {col: preds_by_simid.get(str(sim), {}).get(col, 0.0)
#          for col in stage1_cols}
#         for sim in df_split["SimID"].values
#     ]

#     df_out = pd.DataFrame(rows, index=df_split.index, columns=stage1_cols)

#     # Validation
#     n_zero_rows = (df_out == 0.0).all(axis=1).sum()
#     if n_zero_rows > 0:
#         log.warning(
#             "build_stage1_preds_rowlevel: %d rows fully zero "
#             "(SimID absent from horizon_datasets).", n_zero_rows
#         )
#     return df_out
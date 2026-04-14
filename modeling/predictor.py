"""
MetAIsAFe — modeling/predictor.py
===================================
Inference pipeline: from user parameters to 40-year prediction curves.

CASCADE ROUTING
---------------
    CLF1 → tree_failed  (pedological)
    CLF2 → yield_failed (geographic)

    Routing outcomes:
        tree_ok  × yield_ok   → Stage 1 (carbonStem) + Stage 2 (yield)
        tree_fail × yield_ok  → yield regressors only; carbon = 0
        tree_ok  × yield_fail → carbon regressors only; yield = 0
        tree_fail × yield_fail → all outputs = 0

SEQUENTIAL PIPELINE (stage 2 uses stage 1 predictions as features)
---------------
    Stage 1 : carbonStem_AF, carbonStem_TF
    Stage 2 : yield_AF, yield_TA  (+ stage 1 preds injected as features)

INFERENCE MODES
---------------
    predict_single_sim()  — 1 parameter set → 40-year curves
    predict_batch()       — N parameter sets → stacked DataFrame
    predict_cascade()     — core routing + regression logic (internal)

MODEL LOADING
-------------
    Models can be passed in-memory (dict) or loaded from disk
    (CampaignPaths). Both modes are supported and combinable.

Author  : Étienne SABY
Created : 2026-04
"""
from __future__ import annotations
from _version import __version__

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import (
    CampaignPaths,
    RANDOM_STATE,
    SEQUENTIAL_TARGETS_STAGE1,
    SEQUENTIAL_TARGETS_STAGE2,
)
from column_taxonomy import (
    ACTIVE_FEATURES_B2,
    CATEGORICAL_FEATURES_B2,
    NOMINAL_POPULATION,
)
from modeling.classifiers import (
    CLF1_FEATURES,
    CLF2_FEATURES,
    apply_geographic_rule,
    predict_routing,
)
from utils.io_utils import load_model

log = logging.getLogger(__name__)

# Number of simulation years (fixed HiSAFe horizon)
N_YEARS: int = 40

# All regression targets in pipeline order
ALL_REGRESSION_TARGETS: list[str] = SEQUENTIAL_TARGETS_STAGE1 + SEQUENTIAL_TARGETS_STAGE2


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_all_models(
    campaign: CampaignPaths,
    targets: list[str] | None = None,
) -> dict[str, Any]:
    """
    Load all regression models from a campaign directory.

    Parameters
    ----------
    campaign : CampaignPaths
    targets : list of str, optional
        Targets to load. Defaults to ALL_REGRESSION_TARGETS.

    Returns
    -------
    dict[str, estimator]
        Keys = target names. Missing models are logged and skipped.
    """
    if targets is None:
        targets = ALL_REGRESSION_TARGETS

    models: dict[str, Any] = {}
    for target in targets:
        path = campaign.metamodels_dir / f"lgbm_{target}.joblib"
        if path.exists():
            models[target] = load_model(path)
        else:
            log.warning("Model not found for target '%s': %s", target, path)

    log.info(
        "Loaded %d / %d regression models: %s",
        len(models), len(targets), list(models.keys()),
    )
    return models


def load_all_classifiers(
    campaign: CampaignPaths,
) -> tuple[Any, Any]:
    """
    Load CLF1 and CLF2 from a campaign directory.

    Returns
    -------
    clf1 : fitted LGBMClassifier
    clf2 : fitted LGBMClassifier or None (geographic fallback)
    """
    from modeling.classifiers import load_classifiers
    return load_classifiers(campaign)


def _resolve_models(
    campaign: CampaignPaths | None,
    models: dict[str, Any] | None,
    clf1: Any | None,
    clf2: Any | None,
) -> tuple[dict[str, Any], Any, Any]:
    """
    Resolve regression models and classifiers from memory or disk.

    Priority: in-memory > disk. Mixed mode supported
    (e.g. models in memory, classifiers from disk).

    Parameters
    ----------
    campaign : CampaignPaths or None
    models : dict or None
    clf1, clf2 : fitted classifiers or None

    Returns
    -------
    models, clf1, clf2
    """
    if models is None:
        if campaign is None:
            raise ValueError(
                "Either 'models' (in-memory dict) or 'campaign' (disk path) "
                "must be provided."
            )
        models = load_all_models(campaign)

    if clf1 is None:
        if campaign is None:
            raise ValueError(
                "Either 'clf1' or 'campaign' must be provided to load classifiers."
            )
        clf1, clf2 = load_all_classifiers(campaign)

    return models, clf1, clf2


# ============================================================================
# INFERENCE GRID BUILDER
# ============================================================================

def build_inference_grid(
    params: dict[str, Any],
    n_years: int = N_YEARS,
    year_col: str = "Harvest_Year_Absolute",
) -> pd.DataFrame:
    """
    Generate a 40-row feature DataFrame from a single parameter set.

    Each row corresponds to one crop year (1 → n_years). All Sobol
    plan parameters are broadcast across all rows; only the temporal
    feature (Harvest_Year_Absolute) varies.

    Parameters
    ----------
    params : dict
        User-supplied parameter set. Must contain all ACTIVE_FEATURES_B2
        except Harvest_Year_Absolute (added automatically).
        Categorical values should be strings (e.g. "wheat", "CONST").
    n_years : int, default 40
    year_col : str

    Returns
    -------
    pd.DataFrame
        Shape (n_years, n_features). Categorical columns cast to
        ``category`` dtype for LightGBM.

    Raises
    ------
    ValueError
        If required features are missing from params.
    """
    # Required features excluding the temporal one (added below)
    required = [f for f in ACTIVE_FEATURES_B2 if f != year_col]
    missing  = [f for f in required if f not in params]
    if missing:
        raise ValueError(
            f"build_inference_grid: missing parameters {missing}. "
            f"Required: {required}"
        )

    # Broadcast scalar params across all years
    grid = pd.DataFrame({col: [params[col]] * n_years for col in required})
    grid[year_col] = np.arange(1, n_years + 1, dtype=float)

    # Reorder columns to match ACTIVE_FEATURES_B2
    ordered_cols = [c for c in ACTIVE_FEATURES_B2 if c in grid.columns]
    grid = grid[ordered_cols]

    # Cast categoricals for LightGBM native handling
    for col in CATEGORICAL_FEATURES_B2:
        if col in grid.columns:
            grid[col] = grid[col].astype("category")

    log.debug("Inference grid built: %d rows × %d features", len(grid), len(grid.columns))
    return grid


# ============================================================================
# CORE CASCADE PREDICTOR
# ============================================================================

def predict_cascade(
    grid: pd.DataFrame,
    params: dict[str, Any],
    models: dict[str, Any],
    clf1: Any,
    clf2: Any | None,
    return_routing: bool = False,
) -> dict[str, Any]:
    """
    Apply the full cascade pipeline to a pre-built inference grid.

    Routing → Stage 1 → Stage 2 (with stage 1 predictions injected).

    Parameters
    ----------
    grid : pd.DataFrame
        Output of ``build_inference_grid()`` — shape (n_years, n_features).
    params : dict
        Original scalar params (used to build SimID-level classifier input).
    models : dict[str, estimator]
        Regression models keyed by target name.
    clf1 : LGBMClassifier
        Tree failure classifier.
    clf2 : LGBMClassifier or None
        Yield failure classifier (None → geographic rule).
    return_routing : bool, default False
        If True, include routing metadata in output.

    Returns
    -------
    dict with keys:
        predictions  : dict[str, np.ndarray]  — one array per target (length n_years)
        population   : str                    — routing label
        tree_failed  : int                    — 0 or 1
        yield_failed : int                    — 0 or 1
        routing      : pd.DataFrame           — (if return_routing=True)
    """
    # ── 1. Routing — SimID level (one row = one simulation) ──────────────
    # Build single-row feature matrix for classifiers from scalar params
    X_sim = pd.DataFrame([{
        col: params.get(col) for col in CLF1_FEATURES + CLF2_FEATURES
        if col in params
    }])
    # Cast categoricals
    for col in CATEGORICAL_FEATURES_B2:
        if col in X_sim.columns:
            X_sim[col] = X_sim[col].astype("category")

    routing = predict_routing(X_sim, clf1, clf2, return_proba=False)
    tree_failed  = int(routing["tree_failed"].iloc[0])
    yield_failed = int(routing["yield_failed"].iloc[0])
    population   = str(routing["population"].iloc[0])

    log.info(
        "Routing → %s (tree_failed=%d, yield_failed=%d)",
        population, tree_failed, yield_failed,
    )

    # ── 2. Initialise predictions to zero ────────────────────────────────
    n_years = len(grid)
    preds: dict[str, np.ndarray] = {
        t: np.zeros(n_years) for t in ALL_REGRESSION_TARGETS
    }

    # ── 3. Full failure — return zeros immediately ────────────────────────
    if tree_failed == 1 and yield_failed == 1:
        log.info("Full failure — all outputs = 0.")
        out = {"predictions": preds, "population": population,
               "tree_failed": tree_failed, "yield_failed": yield_failed}
        if return_routing:
            out["routing"] = routing
        return out

    # ── 4. Stage 1 — tree carbon stocks ──────────────────────────────────
    stage1_preds: dict[str, np.ndarray] = {}

    if tree_failed == 0:
        for target in SEQUENTIAL_TARGETS_STAGE1:
            if target not in models:
                log.warning("Stage 1 model missing: '%s' — predicting zeros.", target)
                stage1_preds[target] = np.zeros(n_years)
                continue
            stage1_preds[target] = models[target].predict(grid)
            # Physical constraint: carbon stocks are non-negative
            stage1_preds[target] = np.clip(stage1_preds[target], 0.0, None)
            log.debug(
                "Stage 1 '%s': min=%.3f max=%.3f",
                target, stage1_preds[target].min(), stage1_preds[target].max(),
            )
    else:
        # tree_failed=1, yield_failed=0 → carbon = 0, yield predicted below
        log.info("tree_failed=1 — carbonStem outputs set to 0.")

    preds.update(stage1_preds)

    # ── 5. Stage 2 — crop yields (+ stage 1 predictions injected) ────────
    if yield_failed == 0:
        # Build augmented grid: original features + stage 1 predictions
        grid_aug = grid.copy()
        for s1_target, s1_arr in stage1_preds.items():
            grid_aug[s1_target] = s1_arr

        for target in SEQUENTIAL_TARGETS_STAGE2:
            if target not in models:
                log.warning("Stage 2 model missing: '%s' — predicting zeros.", target)
                preds[target] = np.zeros(n_years)
                continue

            # Select only features the model was trained on
            try:
                feat_names = models[target].feature_name_
            except AttributeError:
                feat_names = list(grid_aug.columns)

            avail = [f for f in feat_names if f in grid_aug.columns]
            preds[target] = np.clip(
                models[target].predict(grid_aug[avail]),
                0.0, None,
            )
            log.debug(
                "Stage 2 '%s': min=%.3f max=%.3f",
                target, preds[target].min(), preds[target].max(),
            )
    else:
        log.info("yield_failed=1 — yield outputs set to 0.")

    out: dict[str, Any] = {
        "predictions": preds,
        "population":  population,
        "tree_failed": tree_failed,
        "yield_failed": yield_failed,
    }
    if return_routing:
        out["routing"] = routing

    return out


# ============================================================================
# SINGLE SIMULATION PREDICTION
# ============================================================================

def predict_single_sim(
    params: dict[str, Any],
    models: dict[str, Any] | None = None,
    clf1: Any | None = None,
    clf2: Any | None = None,
    campaign: CampaignPaths | None = None,
    n_years: int = N_YEARS,
    return_routing: bool = False,
) -> dict[str, Any]:
    """
    Full prediction pipeline for a single parameter set.

    Supports in-memory models, disk loading, or mixed mode.

    Parameters
    ----------
    params : dict
        User-supplied parameters (all ACTIVE_FEATURES_B2 except
        Harvest_Year_Absolute, which is generated automatically).
    models : dict, optional
        In-memory regression models {target: estimator}.
    clf1, clf2 : optional
        In-memory classifiers.
    campaign : CampaignPaths, optional
        Used to load models from disk if not provided in-memory.
    n_years : int, default 40
    return_routing : bool, default False

    Returns
    -------
    dict with keys:
        predictions  : dict[str, np.ndarray]  (length n_years each)
        years        : np.ndarray             (1 … n_years)
        population   : str
        tree_failed  : int
        yield_failed : int
        routing      : pd.DataFrame           (if return_routing=True)

    Examples
    --------
    >>> result = predict_single_sim(params, campaign=campaign)
    >>> df = format_output(result)
    """
    models, clf1, clf2 = _resolve_models(campaign, models, clf1, clf2)

    grid = build_inference_grid(params, n_years=n_years)
    result = predict_cascade(grid, params, models, clf1, clf2, return_routing)
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
    n_years: int = N_YEARS,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Predict for multiple parameter sets (scenario grid).

    Parameters
    ----------
    params_list : list of dict
        Each dict is a complete parameter set (one simulation scenario).
    models, clf1, clf2 : optional
        In-memory models/classifiers.
    campaign : CampaignPaths, optional
        Used to load from disk if models/classifiers not provided.
    n_years : int, default 40
    verbose : bool

    Returns
    -------
    pd.DataFrame
        Long format: one row per (scenario_id × year × target).
        Columns: scenario_id, year, target, value, population,
                 tree_failed, yield_failed.

    Examples
    --------
    >>> df = predict_batch(scenarios, campaign=campaign)
    >>> df.pivot(index=["scenario_id","year"], columns="target", values="value")
    """
    models, clf1, clf2 = _resolve_models(campaign, models, clf1, clf2)

    records: list[pd.DataFrame] = []
    n_scenarios = len(params_list)

    for i, params in enumerate(params_list):
        if verbose and (i % max(1, n_scenarios // 10) == 0):
            log.info("Batch inference: %d / %d", i + 1, n_scenarios)

        try:
            grid   = build_inference_grid(params, n_years=n_years)
            result = predict_cascade(grid, params, models, clf1, clf2)
        except Exception as exc:
            log.error("Scenario %d failed: %s", i, exc)
            continue

        df_sc = format_output(result, scenario_id=i)
        records.append(df_sc)

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
    Convert ``predict_cascade`` / ``predict_single_sim`` output to a
    tidy long-format DataFrame compatible with R/Shiny.

    Parameters
    ----------
    result : dict
        Output of ``predict_single_sim()`` or ``predict_cascade()``.
        Must contain: predictions, population, tree_failed, yield_failed.
        Optional: years.
    scenario_id : int or str, optional
        Added as identifier column when batching.

    Returns
    -------
    pd.DataFrame
        Columns: [scenario_id (opt), year, target, value,
                  population, tree_failed, yield_failed]

    Examples
    --------
    >>> df = format_output(result)
    >>> # In R: df %>% pivot_wider(names_from=target, values_from=value)
    """
    preds  = result["predictions"]
    years  = result.get("years", np.arange(1, len(next(iter(preds.values()))) + 1))

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

    # Column order: scenario_id first if present
    base_cols = ["year", "target", "value", "population", "tree_failed", "yield_failed"]
    if scenario_id is not None:
        df = df[["scenario_id"] + base_cols]
    else:
        df = df[base_cols]

    return df
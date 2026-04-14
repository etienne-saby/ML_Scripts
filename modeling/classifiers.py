"""
MetAIsAFe — modeling/classifiers.py
=====================================
Cascade binary classifiers for simulation routing.

ARCHITECTURE OVERVIEW
---------------------
The MetAIsAFe surrogate operates a two-stage cascade classifier that routes
each simulation to the appropriate prediction regime before any regression
model is applied. This design is motivated by the structural heterogeneity
identified in the Hi-sAFe simulation outputs (Batch 1 analysis):

    Stage 1 — Tree failure classifier (CLF1)
    -----------------------------------------
    Predicts whether the agroforestry tree component fails over the 40-year
    simulation horizon (carbonStem_AF < 1.0 kgC/tree at final cycle).

    Failure mechanism : PEDOLOGICAL
    Key features      : clay, sand, stone, latitude, longitude, plotHeight
    Accuracy B2       : ~66.7% (LightGBM) — intentionally soft boundary
                        (failure zone is diffuse in soil texture space)

    Stage 2 — Yield failure classifier (CLF2)
    ------------------------------------------
    Predicts whether crop yields fail across the majority of simulation
    cycles (>50% of cycles with yield < 0.5 t/ha for AF or TA).

    Failure mechanism : GEOGRAPHIC / CLIMATIC
    Key features      : latitude, longitude, clay, waterTable
    Accuracy B2       : ~91.5% (LightGBM) — sharp geographic boundary

ROUTING LOGIC
-------------
The cascade produces four routing outcomes, each mapped to a specific
prediction regime:

    tree_ok  × yield_ok   → Main meta-model (Stage 1 + Stage 2 regressors)
    tree_fail × yield_ok  → Cultural-only model (yield_AF, yield_TA only;
                             carbonStem_AF = carbonStem_TF = 0)
    tree_ok  × yield_fail → Geographic rejection: yield = 0;
                             carbonStem predicted by tree regressors
    tree_fail × yield_fail → Full rejection: all outputs = 0

DESIGN DECISIONS
----------------
- LightGBM is used for both classifiers (same rationale as regressors:
  native category support, leaf-wise growth, fast inference).
- A deterministic geographic fallback rule is provided for yield_fail
  classification when LightGBM training data is insufficient (<80 samples
  in minority class). The rule achieves ~90% accuracy based on B1/B2
  decision tree analysis (latitude < 44°N AND longitude > 3.5°E).
- Classifiers are trained on the FULL dataset (all 4 populations),
  not just the nominal population — they need exposure to failure cases.
- Features are restricted to Sobol plan parameters only (no temporal
  features, no climate outputs) to ensure classifiers generalise to
  new parameter combinations at inference time.

CLASSIFIER FEATURES
-------------------
CLF1 (tree_fail):
    ['clay', 'sand', 'stone', 'latitude', 'longitude',
     'plotHeight', 'plotWidth', 'soilDepth', 'main_crop']

CLF2 (yield_fail):
    ['latitude', 'longitude', 'clay', 'waterTable',
     'sand', 'stone', 'soilDepth']

Author  : Étienne SABY
Updated : 2026-04
"""
from __future__ import annotations
from _version import __version__

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

import lightgbm as lgb

from config import (
    RANDOM_STATE,
    TREE_FAIL_THRESHOLD,
    YIELD_FAIL_THRESHOLD,
    YIELD_FAIL_RATE,
    LGB_PARAMS,
    CampaignPaths,
)
from column_taxonomy import CATEGORICAL_FEATURES_B2, POPULATION_LABELS
from modeling.models import build_lgb_classifier
from utils.io_utils import save_model, load_model

log = logging.getLogger(__name__)

# ============================================================================
# CLASSIFIER FEATURE SETS
# ============================================================================

# CLF1 — tree failure is a pedological + geographic phenomenon
# Features selected from B1/B2 decision tree importance analysis:
#   clay (25.6%), longitude (18.8%), sand (18.7%), stone (16.7%),
#   latitude (11.1%), plotHeight (22.1% in B2)
CLF1_FEATURES: list[str] = [
    "clay",
    "sand",
    "stone",
    "latitude",
    "longitude",
    "plotHeight",
    "plotWidth",
    "soilDepth",
    "main_crop",    # category — captures crop-specific root competition effects
]

# CLF2 — yield failure is a geographic / climatic phenomenon
# Features selected from B1/B2 decision tree importance analysis:
#   latitude (44.5-53.0%), longitude (24.2-41.6%), clay (5.9-15.0%),
#   waterTable (5.8-6.6%)
CLF2_FEATURES: list[str] = [
    "latitude",
    "longitude",
    "clay",
    "waterTable",
    "sand",
    "stone",
    "soilDepth",
]

# Geographic fallback rule thresholds (from B1/B2 decision tree analysis)
# Achieves ~90% accuracy for yield_fail identification
_GEO_RULE_LAT_THRESHOLD: float = 44.0   # °N — below = Mediterranean/continental risk zone
_GEO_RULE_LON_THRESHOLD: float = 3.5    # °E — above = continental dry risk zone
_MIN_MINORITY_SAMPLES:   int   = 80     # Minimum samples in minority class for LGB training


# ============================================================================
# LABEL BUILDERS
# ============================================================================

def build_tree_fail_labels(
    df: pd.DataFrame,
    threshold: float = TREE_FAIL_THRESHOLD,
    simid_col: str = "SimID",
    carbon_col: str = "carbonStem_AF",
    year_col: str = "Harvest_Year_Absolute",
) -> pd.Series:
    """
    Build binary tree failure labels at SimID level.

    A simulation is classified as tree-failed if the stem carbon stock
    at the LAST observed cycle falls below ``threshold``.

    The last cycle is identified via ``year_col`` (preferred) or as the
    last row per SimID in the DataFrame order (fallback).

    Parameters
    ----------
    df : pd.DataFrame
        Full meta-table (all cycles, all populations).
        Must contain: SimID, carbonStem_AF, Harvest_Year_Absolute.
    threshold : float, default TREE_FAIL_THRESHOLD (1.0 kgC/tree)
        carbonStem_AF < threshold at last cycle → tree_failed = 1.
    simid_col : str
    carbon_col : str
    year_col : str

    Returns
    -------
    pd.Series
        Binary labels (0/1), index = SimID.
        1 = tree failed, 0 = tree functional.

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    required = {simid_col, carbon_col}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"build_tree_fail_labels: missing columns {sorted(missing)}"
        )

    if year_col in df.columns:
        last = (
            df.sort_values([simid_col, year_col])
            .groupby(simid_col, as_index=True)[carbon_col]
            .last()
        )
    else:
        log.warning(
            "   ⚠ '%s' not found — using last row per SimID as proxy for last cycle.",
            year_col,
        )
        last = df.groupby(simid_col, as_index=True)[carbon_col].last()

    labels = (last < threshold).astype(int).rename("tree_failed")
    labels.index = pd.Index(labels.index.to_numpy(dtype=str))   # ← fix PyArrow index

    log.info(
        "CLF1 labels — tree_failed=1: %d / %d  (%.1f%%)",
        labels.sum(), len(labels),
        100.0 * labels.mean(),
    )
    return labels


def build_yield_fail_labels(
    df: pd.DataFrame,
    yield_threshold: float = YIELD_FAIL_THRESHOLD,
    fail_rate: float = YIELD_FAIL_RATE,
    simid_col: str = "SimID",
    yield_af_col: str = "yield_AF",
    yield_ta_col: str = "yield_TA",
) -> pd.Series:
    """
    Build binary yield failure labels at SimID level.

    A simulation is classified as yield-failed if the fraction of cycles
    where EITHER yield_AF OR yield_TA falls below ``yield_threshold``
    exceeds ``fail_rate``.

    This joint criterion (AF OR TA) captures both agroforestry-specific
    crop failure and sole-crop reference failure, which share the same
    geographic determinism (B1/B2 analysis: 88.7% co-occurrence).

    Parameters
    ----------
    df : pd.DataFrame
        Full meta-table (all cycles, all populations).
    yield_threshold : float, default YIELD_FAIL_THRESHOLD (0.5 t/ha)
    fail_rate : float, default YIELD_FAIL_RATE (0.5)
        SimID is yield_failed if > fail_rate fraction of cycles fail.
    simid_col : str
    yield_af_col : str
    yield_ta_col : str

    Returns
    -------
    pd.Series
        Binary labels (0/1), index = SimID.
        1 = yield failed, 0 = yield functional.

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    required = {simid_col, yield_af_col, yield_ta_col}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"build_yield_fail_labels: missing columns {sorted(missing)}"
        )

    cycle_fail = (
        (df[yield_af_col] < yield_threshold) |
        (df[yield_ta_col] < yield_threshold)
    )

    labels = (
        df.assign(_fail=cycle_fail)
        .groupby(simid_col)["_fail"]
        .mean()
        .gt(fail_rate)
        .astype(int)
        .rename("yield_failed")
    )
    labels.index = pd.Index(labels.index.to_numpy(dtype=str))   # ← fix PyArrow index

    log.info(
        "CLF2 labels — yield_failed=1: %d / %d  (%.1f%%)",
        labels.sum(), len(labels),
        100.0 * labels.mean(),
    )
    return labels


# ============================================================================
# FEATURE MATRIX BUILDER
# ============================================================================

def build_classifier_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    simid_col: str = "SimID",
) -> pd.DataFrame:
    """
    Extract classifier feature matrix at SimID level.

    Since classifiers operate at SimID level (one prediction per simulation,
    not per cycle), features are aggregated by taking the first row per SimID.
    This is valid because all Sobol plan parameters are constant within a
    SimID — only output variables (yield, carbonStem) vary across cycles.

    Parameters
    ----------
    df : pd.DataFrame
        Meta-table (any granularity — SimID level extracted internally).
    feature_cols : list of str
        Feature columns to extract (e.g. CLF1_FEATURES or CLF2_FEATURES).
    simid_col : str

    Returns
    -------
    pd.DataFrame
        One row per SimID, columns = feature_cols (intersected with df).
        Index = SimID.

    Notes
    -----
    Categorical columns (``main_crop``, ``w_type``) are cast to
    ``category`` dtype for LightGBM native handling.
    """
    avail = [c for c in feature_cols if c in df.columns]
    missing = set(feature_cols) - set(avail)
    if missing:
        log.warning(
            "   ⚠ Classifier features not found in DataFrame: %s", sorted(missing)
        )

    X = (
        df.groupby(simid_col)[avail]
        .first()
    )

    # Cast categoricals to LightGBM-native dtype
    cat_cols = [c for c in avail if c in CATEGORICAL_FEATURES_B2]
    for col in cat_cols:
        X[col] = X[col].astype("category")

    X.index = pd.Index(X.index.to_numpy(dtype=str))   # ← fix PyArrow index (pandas >= 2.0 + Parquet)
    
    log.info(
        "Classifier feature matrix: %d SimIDs × %d features",
        len(X), len(avail),
    )
    return X


# ============================================================================
# CLASSIFIER TRAINING
# ============================================================================

def build_tree_fail_classifier(
    df: pd.DataFrame,
    params: dict[str, Any] | None = None,
    simid_col: str = "SimID",
    verbose: bool = True,
) -> tuple[lgb.LGBMClassifier, pd.DataFrame, pd.Series]:
    """
    Build and return training data for the tree failure classifier (CLF1).

    This function prepares the feature matrix and labels only.
    Actual model fitting is delegated to ``trainer.train_classifier()``
    to preserve separation of concerns and enable cross-validation.

    Failure mechanism (from B1/B2 structural analysis)
    ---------------------------------------------------
    Tree failure is primarily pedological:
        - clay > 18.9% AND stone > 30.4% → strong failure signal
        - longitude (18.8%) and sand (18.7%) modulate the pedological signal
        - Geographic location acts as a climate stress amplifier

    B2-specific finding
    -------------------
    ``plotHeight`` (22.1%) and ``main_crop`` (18.3%) gain importance in B2
    as the texture space shifts toward higher clay content, making geometry
    and crop type more discriminating.

    Parameters
    ----------
    df : pd.DataFrame
        Full meta-table (all cycles, all populations). NOT pre-filtered.
    params : dict, optional
        LightGBM hyperparameter overrides.
    simid_col : str
    verbose : bool

    Returns
    -------
    model : lgb.LGBMClassifier
        Unfitted classifier instance (ready for trainer.train_classifier()).
    X : pd.DataFrame
        Feature matrix at SimID level (index = SimID).
    y : pd.Series
        Binary labels (index = SimID).
    """
    if verbose:
        log.info("=" * 60)
        log.info("CLF1 — Tree Failure Classifier")
        log.info("=" * 60)
        log.info("Features: %s", CLF1_FEATURES)

    # Build labels and features at SimID level
    y = build_tree_fail_labels(df, simid_col=simid_col)
    X = build_classifier_features(df, CLF1_FEATURES, simid_col=simid_col)

    # Align on common SimIDs
    common = X.index.intersection(y.index)
    X, y   = X.loc[common], y.loc[common]

    cat_feats = [c for c in CLF1_FEATURES if c in CATEGORICAL_FEATURES_B2 and c in X.columns]
    model = build_lgb_classifier(params=params, categorical_feature=cat_feats or None)

    if verbose:
        log.info(
            "CLF1 dataset: %d SimIDs | class balance: %.1f%% failed",
            len(y), 100.0 * y.mean(),
        )

    return model, X, y


def build_yield_fail_classifier(
    df: pd.DataFrame,
    params: dict[str, Any] | None = None,
    simid_col: str = "SimID",
    use_geographic_fallback: bool = True,
    verbose: bool = True,
) -> tuple[lgb.LGBMClassifier | None, pd.DataFrame, pd.Series]:
    """
    Build and return training data for the yield failure classifier (CLF2).

    This function prepares the feature matrix and labels only.
    Actual model fitting is delegated to ``trainer.train_classifier()``.

    Failure mechanism (from B1/B2 structural analysis)
    ---------------------------------------------------
    Yield failure is primarily geographic/climatic:
        - latitude < 44°N → Mediterranean/continental dry zone
        - longitude > 3.5°E → amplifies the southern climate signal
        - Accuracy: ~91.5% on B2 with LightGBM

    Geographic fallback rule
    ------------------------
    If the minority class (yield_failed=1) has fewer than
    ``_MIN_MINORITY_SAMPLES`` samples, the LightGBM classifier is replaced
    by the deterministic geographic rule:
        yield_failed = (latitude < 44.0°N) AND (longitude > 3.5°E)
    This avoids overfitting on very imbalanced datasets while maintaining
    ~90% accuracy.

    Parameters
    ----------
    df : pd.DataFrame
        Full meta-table (all cycles, all populations). NOT pre-filtered.
    params : dict, optional
        LightGBM hyperparameter overrides.
    simid_col : str
    use_geographic_fallback : bool, default True
        If True, check minority class size and switch to rule-based
        classification if insufficient.
    verbose : bool

    Returns
    -------
    model : lgb.LGBMClassifier or None
        Unfitted classifier, or None if geographic fallback is used.
    X : pd.DataFrame
        Feature matrix at SimID level.
    y : pd.Series
        Binary labels.
    """
    if verbose:
        log.info("=" * 60)
        log.info("CLF2 — Yield Failure Classifier")
        log.info("=" * 60)
        log.info("Features: %s", CLF2_FEATURES)

    y = build_yield_fail_labels(df, simid_col=simid_col)
    X = build_classifier_features(df, CLF2_FEATURES, simid_col=simid_col)

    common = X.index.intersection(y.index)
    X, y   = X.loc[common], y.loc[common]

    n_minority = int(y.sum())

    if use_geographic_fallback and n_minority < _MIN_MINORITY_SAMPLES:
        log.warning(
            "CLF2: minority class has only %d samples (< %d). "
            "Switching to deterministic geographic fallback rule.",
            n_minority, _MIN_MINORITY_SAMPLES,
        )
        log.info(
            "Geographic rule: yield_failed = (latitude < %.1f°N) AND (longitude > %.1f°E)",
            _GEO_RULE_LAT_THRESHOLD, _GEO_RULE_LON_THRESHOLD,
        )
        return None, X, y

    cat_feats = [c for c in CLF2_FEATURES if c in CATEGORICAL_FEATURES_B2 and c in X.columns]
    model = build_lgb_classifier(params=params, categorical_feature=cat_feats or None)

    if verbose:
        log.info(
            "CLF2 dataset: %d SimIDs | class balance: %.1f%% failed",
            len(y), 100.0 * y.mean(),
        )

    return model, X, y


# ============================================================================
# GEOGRAPHIC FALLBACK RULE
# ============================================================================

def apply_geographic_rule(
    X: pd.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    lat_threshold: float = _GEO_RULE_LAT_THRESHOLD,
    lon_threshold: float = _GEO_RULE_LON_THRESHOLD,
) -> np.ndarray:
    """
    Deterministic geographic rule for yield failure classification.

    Activated when LightGBM CLF2 training data is insufficient
    (minority class < ``_MIN_MINORITY_SAMPLES`` samples).

    Rule derived from B1/B2 decision tree analysis:
        yield_failed = (latitude < lat_threshold) AND (longitude > lon_threshold)

    Accuracy on B2: ~90% (latitude importance: 44.5-53.0%,
    longitude importance: 24.2-41.6% in decision trees).

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix with at least ``lat_col`` and ``lon_col`` columns.
    lat_col, lon_col : str
    lat_threshold : float, default 44.0
    lon_threshold : float, default 3.5

    Returns
    -------
    np.ndarray
        Binary predictions (0/1), shape (n_samples,).
    """
    if lat_col not in X.columns or lon_col not in X.columns:
        raise ValueError(
            f"apply_geographic_rule: '{lat_col}' and/or '{lon_col}' "
            f"not found in X.columns."
        )
    return (
        (X[lat_col].values < lat_threshold) &
        (X[lon_col].values > lon_threshold)
    ).astype(int)


# ============================================================================
# CLASSIFIER EVALUATION
# ============================================================================

def evaluate_classifier(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    classifier_name: str = "Classifier",
    verbose: bool = True,
) -> dict[str, float]:
    """
    Compute and log classification metrics.

    Metrics
    -------
    - Accuracy
    - F1 score (binary, positive class = failure)
    - ROC-AUC (if probabilities available)
    - Confusion matrix (logged)

    Parameters
    ----------
    y_true : array-like
        True binary labels (0/1).
    y_pred : np.ndarray
        Predicted binary labels (0/1).
    y_proba : np.ndarray, optional
        Predicted probabilities for positive class (shape: n_samples,).
        Required for ROC-AUC computation.
    classifier_name : str
        Label for logging.
    verbose : bool

    Returns
    -------
    dict[str, float]
        Keys: accuracy, f1, roc_auc (if y_proba provided).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1":       float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_proba is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            log.warning("ROC-AUC could not be computed (single class in y_true).")

    if verbose:
        log.info("\n%s", "=" * 60)
        log.info("%s — Evaluation", classifier_name)
        log.info("%s", "=" * 60)
        for k, v in metrics.items():
            log.info("  %-12s : %.4f", k, v)
        log.info("\nConfusion matrix:\n%s", confusion_matrix(y_true, y_pred))
        log.info(
            "\nClassification report:\n%s",
            classification_report(y_true, y_pred, target_names=["OK", "Failed"]),
        )

    return metrics


# ============================================================================
# PERSISTENCE
# ============================================================================

def save_classifiers(
    clf1: lgb.LGBMClassifier,
    clf2: lgb.LGBMClassifier | None,
    campaign: CampaignPaths,
    clf1_meta: dict[str, Any] | None = None,
    clf2_meta: dict[str, Any] | None = None,
) -> tuple[Path, Path | None]:
    """
    Save CLF1 and CLF2 to the campaign MetaModels directory.

    CLF2 may be None if the geographic fallback rule is used — in that
    case, only CLF1 is saved and the return value for CLF2 is None.

    Parameters
    ----------
    clf1 : lgb.LGBMClassifier
        Fitted tree failure classifier.
    clf2 : lgb.LGBMClassifier or None
        Fitted yield failure classifier, or None (geographic rule).
    campaign : CampaignPaths
    clf1_meta, clf2_meta : dict, optional
        Metadata to persist alongside each model (metrics, features, etc.).

    Returns
    -------
    path_clf1 : Path
    path_clf2 : Path or None
    """
    path_clf1 = save_model(
        clf1,
        campaign.metamodels_dir / "clf1_tree_fail.joblib",
        metadata={
            "classifier": "CLF1_tree_fail",
            "features":   CLF1_FEATURES,
            "threshold":  TREE_FAIL_THRESHOLD,
            **(clf1_meta or {}),
        },
    )

    path_clf2: Path | None = None
    if clf2 is not None:
        path_clf2 = save_model(
            clf2,
            campaign.metamodels_dir / "clf2_yield_fail.joblib",
            metadata={
                "classifier": "CLF2_yield_fail",
                "features":   CLF2_FEATURES,
                "thresholds": {
                    "yield_threshold": YIELD_FAIL_THRESHOLD,
                    "fail_rate":       YIELD_FAIL_RATE,
                },
                **(clf2_meta or {}),
            },
        )
    else:
        log.info(
            "CLF2 not saved — geographic fallback rule in use "
            "(lat < %.1f°N AND lon > %.1f°E).",
            _GEO_RULE_LAT_THRESHOLD, _GEO_RULE_LON_THRESHOLD,
        )

    return path_clf1, path_clf2


def load_classifiers(
    campaign: CampaignPaths,
) -> tuple[lgb.LGBMClassifier, lgb.LGBMClassifier | None]:
    """
    Load CLF1 and CLF2 from the campaign MetaModels directory.

    If CLF2 was not saved (geographic rule in use), returns None for CLF2.
    The caller (``predictor.predict_cascade()``) must handle the None case
    by calling ``apply_geographic_rule()`` instead.

    Parameters
    ----------
    campaign : CampaignPaths

    Returns
    -------
    clf1 : lgb.LGBMClassifier
    clf2 : lgb.LGBMClassifier or None
    """
    clf1 = load_model(campaign.metamodels_dir / "clf1_tree_fail.joblib")

    clf2_path = campaign.metamodels_dir / "clf2_yield_fail.joblib"
    if clf2_path.exists():
        clf2 = load_model(clf2_path)
    else:
        log.info(
            "CLF2 model not found at %s — geographic fallback rule will be used.",
            clf2_path,
        )
        clf2 = None

    return clf1, clf2


# ============================================================================
# ROUTING PREDICTION (single SimID or batch)
# ============================================================================

def predict_routing(
    X_sim: pd.DataFrame,
    clf1: lgb.LGBMClassifier,
    clf2: lgb.LGBMClassifier | None,
    return_proba: bool = False,
) -> pd.DataFrame:
    """
    Apply the two-stage cascade to produce routing labels.

    For each input row (one SimID), produces:
        - ``tree_failed``  (0/1)
        - ``yield_failed`` (0/1)
        - ``population``   (str label from POPULATION_LABELS)

    Used by ``predictor.predict_cascade()`` to select the correct
    regression model for each simulation.

    Parameters
    ----------
    X_sim : pd.DataFrame
        Feature matrix at SimID level. Must contain CLF1_FEATURES
        and CLF2_FEATURES (columns not present are silently ignored
        with a warning).
    clf1 : lgb.LGBMClassifier
        Fitted tree failure classifier.
    clf2 : lgb.LGBMClassifier or None
        Fitted yield failure classifier. If None, the geographic
        fallback rule is applied.
    return_proba : bool, default False
        If True, also return ``tree_fail_proba`` and ``yield_fail_proba``
        columns (useful for uncertainty quantification).

    Returns
    -------
    pd.DataFrame
        Index aligned with X_sim. Columns:
            tree_failed, yield_failed, population
            [tree_fail_proba, yield_fail_proba]  (if return_proba=True)
    """
    result = pd.DataFrame(index=X_sim.index)

    # ── CLF1 : tree failure ───────────────────────────────────────────────
    clf1_feats  = [c for c in CLF1_FEATURES if c in X_sim.columns]
    X_clf1      = X_sim[clf1_feats].copy()

    # Ensure category dtype for LightGBM
    cat_cols = [c for c in clf1_feats if c in CATEGORICAL_FEATURES_B2]
    for col in cat_cols:
        if X_clf1[col].dtype.name != "category":
            X_clf1[col] = X_clf1[col].astype("category")

    result["tree_failed"]     = clf1.predict(X_clf1).astype(int)
    if return_proba:
        result["tree_fail_proba"] = clf1.predict_proba(X_clf1)[:, 1]

    # ── CLF2 : yield failure ──────────────────────────────────────────────
    clf2_feats = [c for c in CLF2_FEATURES if c in X_sim.columns]
    X_clf2     = X_sim[clf2_feats].copy()

    if clf2 is not None:
        result["yield_failed"]     = clf2.predict(X_clf2).astype(int)
        if return_proba:
            result["yield_fail_proba"] = clf2.predict_proba(X_clf2)[:, 1]
    else:
        log.debug("CLF2 is None — applying geographic fallback rule.")
        result["yield_failed"]     = apply_geographic_rule(X_clf2)
        if return_proba:
            # Geographic rule is deterministic — proba is 0.0 or 1.0
            result["yield_fail_proba"] = result["yield_failed"].astype(float)

    # ── Population label ──────────────────────────────────────────────────
    def _label(row: pd.Series) -> str:
        y = "yield_ok"   if row["yield_failed"] == 0 else "yield_fail"
        t = "tree_ok"    if row["tree_failed"]  == 0 else "tree_failed"
        return f"{y} × {t}"

    result["population"] = result.apply(_label, axis=1)

    return result

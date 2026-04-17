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

    Stage 1 — Tree failure classifier (CLF1) — BINARY (v4.1)
    ----------------------------------------------------------
    Predicts whether the tree component succeeds or is degraded over
    the 40-year simulation horizon based on carbonStem_AF at final cycle:

        Class 0 (tree_degraded) : carbonStem_AF < TREE_STUNT_THRESHOLD (50 kgC)
                                  Fuses former tree_failed + tree_stunted.
        Class 1 (tree_ok)       : carbonStem_AF >= TREE_STUNT_THRESHOLD

    Rationale for binary fusion (v4.1):
        The 3-class variant had recall=20% on tree_stunted (the minority
        class, 12.7% of dataset) due to its inherently ambiguous frontier
        with tree_failed and tree_ok.  Fusing both degraded states into a
        single class raises the macro F1 without loss of routing utility,
        since the fine-grained stunted/failed distinction is handled
        downstream by the stunted_model fallback (which uses carbonStem
        magnitude to apply the correct profile).

    Stage 2 — Yield failure classifier (CLF2) — BINARY
    ---------------------------------------------------
    Predicts whether crop yields fail across the majority of simulation
    cycles (>50% of cycles with yield < 0.5 t/ha for AF or TA).
    Unchanged from v4.0.

ROUTING LOGIC (v4.1 — BINARY CLF1)
------------------------------------
The cascade produces four routing outcomes (down from six in v4.0):

    yield_ok  × tree_ok       → Stage 1 + Stage 2a + Stage 2b
    yield_ok  × tree_degraded → stunted/failed fallback (carbon) + Stage 2a + Stage 2b
    yield_fail × tree_ok      → carbon predicted, yield_AF=0, yield_TA=0
    yield_fail × tree_degraded → Full rejection: all outputs = 0

    Within tree_degraded routing, a secondary sub-criterion distinguishes
    the carbon profile to apply:
        carbonStem_AF_h40 predicted by CLF1 proba → stunted_model if
        P(tree_ok) > STUNTED_PROBA_THRESHOLD, else zero-carbon fallback.

DESIGN DECISIONS
----------------
- LightGBM is used for both classifiers (same rationale as regressors:
  native category support, leaf-wise growth, fast inference).
- A deterministic geographic fallback rule is provided for yield_fail
  classification when LightGBM training data is insufficient.
- Classifiers are trained on the FULL dataset (all populations).
- Features are restricted to Sobol plan parameters + climate aggregates
  to ensure classifiers generalise to new parameter combinations.

BACKWARD COMPATIBILITY
----------------------
- TREE_STATUS_FAILED, TREE_STATUS_STUNTED, TREE_STATUS_OK constants kept.
- build_tree_fail_labels_multiclass() kept for diagnostics.
- CLF1 v4.0 (3-class) can still be trained via multiclass=True in
  build_tree_fail_classifier().
- New default: multiclass=False → binary (tree_degraded / tree_ok).

Author  : Étienne SABY
Updated : 2026-05 (v4.1 — CLF1 binary fusion, Levier 6 Option B)
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
    TREE_STUNT_THRESHOLD,
    YIELD_FAIL_THRESHOLD,
    YIELD_FAIL_RATE,
    LGB_PARAMS,
    CampaignPaths,
)
from column_taxonomy import CATEGORICAL_FEATURES_B2, ACTIVE_FEATURES_B2, CLIMATE_FEATURES
from modeling.models import build_lgb_classifier
from utils.io_utils import save_model, load_model

log = logging.getLogger(__name__)

# ============================================================================
# CLASS LABELS & CONSTANTS
# ============================================================================

# CLF1 v4.0 — 3-class labels (kept for backward compat & diagnostics)
TREE_STATUS_FAILED:  int = 0
TREE_STATUS_STUNTED: int = 1
TREE_STATUS_OK:      int = 2
TREE_STATUS_LABELS: list[str] = ["tree_failed", "tree_stunted", "tree_ok"]

# CLF1 v4.1 — binary labels (tree_degraded = failed + stunted fused)
TREE_BINARY_DEGRADED: int = 0
TREE_BINARY_OK:       int = 1
TREE_BINARY_LABELS: list[str] = ["tree_degraded", "tree_ok"]

# Sub-routing threshold within tree_degraded:
# If P(tree_ok) > this threshold, apply stunted_model (some carbon expected).
# Otherwise apply zero-carbon fallback (fully failed).
STUNTED_PROBA_THRESHOLD: float = 0.30

# ============================================================================
# CLASSIFIER FEATURE SETS
# ============================================================================

CLF1_FEATURES = [
    "clay", "sand", "stone",
    "latitude",
    "plotHeight", "plotWidth", "soilDepth",
    "main_crop",
] + CLIMATE_FEATURES

CLF2_FEATURES = [
    "latitude",
    "clay", "waterTable",
    "sand", "stone", "soilDepth",
] + CLIMATE_FEATURES

_GEO_RULE_LAT_THRESHOLD: float = 44.0
_GEO_RULE_LON_THRESHOLD: float = 3.5
_MIN_MINORITY_SAMPLES:   int   = 80


# ============================================================================
# LABEL BUILDERS — BINARY tree fail (backward compat)
# ============================================================================

def build_tree_fail_labels(
    df: pd.DataFrame,
    threshold: float = TREE_FAIL_THRESHOLD,
    simid_col: str = "SimID",
    carbon_col: str = "carbonStem_AF",
    year_col: str = "Harvest_Year_Absolute",
) -> pd.Series:
    """
    Build binary tree failure labels at SimID level (threshold=TREE_FAIL_THRESHOLD).
    Kept for backward compatibility. For CLF1 v4.1, use
    build_tree_degraded_labels() instead.
    """
    required = {simid_col, carbon_col}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"build_tree_fail_labels: missing columns {sorted(missing)}")

    if year_col in df.columns:
        last = (
            df.sort_values([simid_col, year_col])
            .groupby(simid_col, as_index=True)[carbon_col]
            .last()
        )
    else:
        log.warning("'%s' not found — using last row per SimID.", year_col)
        last = df.groupby(simid_col, as_index=True)[carbon_col].last()

    labels = (last < threshold).astype(int).rename("tree_failed")
    labels.index = pd.Index(labels.index.to_numpy(dtype=str))
    log.info(
        "CLF1 labels (binary fail) — tree_failed=1: %d / %d  (%.1f%%)",
        labels.sum(), len(labels), 100.0 * labels.mean(),
    )
    return labels


def build_tree_degraded_labels(
    df: pd.DataFrame,
    threshold_ok: float = TREE_STUNT_THRESHOLD,
    simid_col: str = "SimID",
    carbon_col: str = "carbonStem_AF",
    year_col: str = "Harvest_Year_Absolute",
) -> pd.Series:
    """
    Build binary tree-degraded labels for CLF1 v4.1.

    Fuses tree_failed and tree_stunted into a single class:
        0 (tree_degraded) : carbonStem_AF(t=40) < threshold_ok  (< 50 kgC)
        1 (tree_ok)       : carbonStem_AF(t=40) >= threshold_ok (>= 50 kgC)

    This is the PRIMARY label builder for CLF1 in v4.1.
    The threshold_ok=TREE_STUNT_THRESHOLD (50 kgC) is the same boundary
    used to define NOMINAL_POPULATION — ensuring full consistency between
    the classifier and the regression training set.

    Parameters
    ----------
    df           : pd.DataFrame — full meta-table (all populations, all cycles)
    threshold_ok : float — carbonStem_AF below this → tree_degraded
    simid_col    : str
    carbon_col   : str
    year_col     : str

    Returns
    -------
    pd.Series — int (0/1), index = SimID (str dtype)
    """
    required = {simid_col, carbon_col}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"build_tree_degraded_labels: missing columns {sorted(missing)}"
        )

    if year_col in df.columns:
        last = (
            df.sort_values([simid_col, year_col])
            .groupby(simid_col, as_index=True)[carbon_col]
            .last()
        )
    else:
        log.warning("'%s' not found — using last row per SimID.", year_col)
        last = df.groupby(simid_col, as_index=True)[carbon_col].last()

    # 1 = tree_ok, 0 = tree_degraded
    labels = (last >= threshold_ok).astype(int).rename("tree_ok_binary")
    labels.index = pd.Index(labels.index.to_numpy(dtype=str))

    n_ok       = int(labels.sum())
    n_degraded = len(labels) - n_ok
    log.info("CLF1 labels (binary v4.1 — tree_degraded/tree_ok):")
    log.info(
        "   0 (tree_degraded): %d / %d  (%.1f%%)  [failed+stunted fused]",
        n_degraded, len(labels), 100.0 * n_degraded / len(labels),
    )
    log.info(
        "   1 (tree_ok)      : %d / %d  (%.1f%%)",
        n_ok, len(labels), 100.0 * n_ok / len(labels),
    )
    return labels


# ============================================================================
# LABEL BUILDERS — YIELD FAIL (unchanged)
# ============================================================================

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
    Unchanged from v4.0.
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
    labels.index = pd.Index(labels.index.to_numpy(dtype=str))
    log.info(
        "CLF2 labels — yield_failed=1: %d / %d  (%.1f%%)",
        labels.sum(), len(labels), 100.0 * labels.mean(),
    )
    return labels


# ============================================================================
# LABEL BUILDERS — MULTICLASS (v4.0 — kept for diagnostics)
# ============================================================================

def build_tree_fail_labels_multiclass(
    df: pd.DataFrame,
    threshold_fail: float = TREE_FAIL_THRESHOLD,
    threshold_stunt: float = TREE_STUNT_THRESHOLD,
    simid_col: str = "SimID",
    carbon_col: str = "carbonStem_AF",
    year_col: str = "Harvest_Year_Absolute",
) -> pd.Series:
    """
    Build 3-class tree status labels (v4.0 — kept for diagnostics).
    Use build_tree_degraded_labels() for CLF1 v4.1 training.
    """
    required = {simid_col, carbon_col}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"build_tree_fail_labels_multiclass: missing columns {sorted(missing)}"
        )
    if threshold_fail >= threshold_stunt:
        raise ValueError(
            f"threshold_fail ({threshold_fail}) must be < threshold_stunt ({threshold_stunt})"
        )

    if year_col in df.columns:
        last = (
            df.sort_values([simid_col, year_col])
            .groupby(simid_col, as_index=True)[carbon_col]
            .last()
        )
    else:
        log.warning("'%s' not found — using last row per SimID.", year_col)
        last = df.groupby(simid_col, as_index=True)[carbon_col].last()

    labels = pd.cut(
        last,
        bins=[-np.inf, threshold_fail, threshold_stunt, np.inf],
        labels=[TREE_STATUS_FAILED, TREE_STATUS_STUNTED, TREE_STATUS_OK],
        right=False,
    ).astype(int).rename("tree_status")
    labels.index = pd.Index(labels.index.to_numpy(dtype=str))

    counts = labels.value_counts().sort_index()
    log.info("CLF1 labels (3-class — diagnostic):")
    for status_code, status_name in enumerate(TREE_STATUS_LABELS):
        n = counts.get(status_code, 0)
        pct = 100.0 * n / len(labels) if len(labels) > 0 else 0.0
        log.info("   %d (%s): %d / %d  (%.1f%%)", status_code, status_name, n, len(labels), pct)
    return labels


# ============================================================================
# FEATURE MATRIX BUILDER (unchanged)
# ============================================================================

def build_classifier_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    simid_col: str = "SimID",
) -> pd.DataFrame:
    """
    Extract classifier feature matrix at SimID level (first row per SimID).
    Valid because all Sobol plan parameters are constant within a SimID.
    """
    avail = [c for c in feature_cols if c in df.columns]
    missing = set(feature_cols) - set(avail)
    if missing:
        log.warning("Classifier features not found in DataFrame: %s", sorted(missing))

    X = df.groupby(simid_col)[avail].first()

    cat_cols = [c for c in avail if c in CATEGORICAL_FEATURES_B2]
    for col in cat_cols:
        X[col] = X[col].astype("category")

    X.index = pd.Index(X.index.to_numpy(dtype=str))
    log.info("Classifier feature matrix: %d SimIDs × %d features", len(X), len(avail))
    return X


# ============================================================================
# CLASSIFIER TRAINING
# ============================================================================

def build_tree_fail_classifier(
    df: pd.DataFrame,
    params: dict[str, Any] | None = None,
    simid_col: str = "SimID",
    multiclass: bool = False,
    verbose: bool = True,
) -> tuple[lgb.LGBMClassifier, pd.DataFrame, pd.Series]:
    """
    Build and return training data for the tree failure classifier (CLF1).

    v4.1 default: multiclass=False → binary (tree_degraded / tree_ok).
    Set multiclass=True to use the v4.0 3-class variant for comparison.

    Parameters
    ----------
    df         : pd.DataFrame — full meta-table (all populations, all cycles)
    params     : dict — LightGBM hyperparameter overrides
    simid_col  : str
    multiclass : bool, default False
        False → binary v4.1 (tree_degraded / tree_ok)  ← NEW DEFAULT
        True  → 3-class v4.0 (failed / stunted / ok)   ← backward compat
    verbose    : bool

    Returns
    -------
    model : lgb.LGBMClassifier — unfitted
    X     : pd.DataFrame — feature matrix (SimID-level)
    y     : pd.Series — labels (0/1 binary or 0/1/2 multiclass)
    """
    if verbose:
        log.info("=" * 60)
        log.info("CLF1 — Tree Failure Classifier (%s)",
                 "3-class [v4.0 compat]" if multiclass else "binary [v4.1]")
        log.info("=" * 60)
        log.info("Features: %s", CLF1_FEATURES)

    if multiclass:
        y = build_tree_fail_labels_multiclass(df, simid_col=simid_col)
    else:
        # v4.1 default: binary (tree_degraded / tree_ok)
        y = build_tree_degraded_labels(df, simid_col=simid_col)

    X = build_classifier_features(df, CLF1_FEATURES, simid_col=simid_col)
    common = X.index.intersection(y.index)
    X, y   = X.loc[common], y.loc[common]

    cat_feats = [c for c in CLF1_FEATURES if c in CATEGORICAL_FEATURES_B2 and c in X.columns]

    if multiclass:
        mc_params = params.copy() if params else LGB_PARAMS.copy()
        mc_params.update({
            "objective": "multiclass",
            "num_class": 3,
            "metric":    "multi_logloss",
        })
        model = build_lgb_classifier(params=mc_params, categorical_feature=cat_feats or None)
    else:
        # Binary: standard binary objective
        bin_params = params.copy() if params else LGB_PARAMS.copy()
        bin_params.update({
            "objective": "binary",
            "metric":    "binary_logloss",
        })
        model = build_lgb_classifier(params=bin_params, categorical_feature=cat_feats or None)

    if verbose:
        if multiclass:
            counts = y.value_counts().sort_index()
            log.info("CLF1 dataset: %d SimIDs", len(y))
            for code, name in enumerate(TREE_STATUS_LABELS):
                n = counts.get(code, 0)
                log.info("   %s: %d (%.1f%%)", name, n, 100.0 * n / len(y))
        else:
            counts = y.value_counts().sort_index()
            log.info("CLF1 dataset: %d SimIDs", len(y))
            for code, name in enumerate(TREE_BINARY_LABELS):
                n = counts.get(code, 0)
                log.info("   %s: %d (%.1f%%)", name, n, 100.0 * n / len(y))

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
    Unchanged from v4.0.
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
# GEOGRAPHIC FALLBACK RULE (unchanged)
# ============================================================================

def apply_geographic_rule(
    X: pd.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    lat_threshold: float = _GEO_RULE_LAT_THRESHOLD,
    lon_threshold: float = _GEO_RULE_LON_THRESHOLD,
) -> np.ndarray:
    """Deterministic geographic rule for yield failure. Unchanged from v4.0."""
    if lat_col not in X.columns:
        raise ValueError(f"apply_geographic_rule: '{lat_col}' not found in X.columns.")
    # lon_col is optional — rule degrades gracefully to lat-only
    if lon_col in X.columns:
        return (
            (X[lat_col].values < lat_threshold) &
            (X[lon_col].values > lon_threshold)
        ).astype(int)
    else:
        log.warning("'%s' not found — applying latitude-only fallback rule.", lon_col)
        return (X[lat_col].values < lat_threshold).astype(int)


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
    """Compute and log binary classification metrics. Unchanged from v4.0."""
    n_classes = len(np.unique(y_true))
    if n_classes > 2:
        raise ValueError("evaluate_classifier is binary only. Use evaluate_classifier_multiclass.")

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
            classification_report(
                y_true, y_pred,
                target_names=TREE_BINARY_LABELS,   # v4.1: degraded / ok
                zero_division=0,
            ),
        )
    return metrics


def evaluate_clf1_binary(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    classifier_name: str = "CLF1 — Tree Status (binary v4.1)",
    verbose: bool = True,
) -> dict[str, float]:
    """
    Evaluate CLF1 v4.1 binary classifier (tree_degraded / tree_ok).

    Reports accuracy, F1 (both classes), ROC-AUC, and confusion matrix.
    This is the primary evaluation function for CLF1 in v4.1.

    Parameters
    ----------
    y_true       : true binary labels (0=tree_degraded, 1=tree_ok)
    y_pred       : predicted binary labels
    y_proba      : predicted probabilities for class 1 (tree_ok), shape (n,)
    classifier_name : str
    verbose      : bool

    Returns
    -------
    dict with keys: accuracy, f1_degraded, f1_ok, f1_macro, roc_auc (if proba)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics: dict[str, float] = {
        "accuracy":     float(accuracy_score(y_true, y_pred)),
        "f1_degraded":  float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "f1_ok":        float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_macro":     float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if y_proba is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            log.warning("ROC-AUC could not be computed.")

    if verbose:
        log.info("\n%s", "=" * 60)
        log.info("%s — Evaluation", classifier_name)
        log.info("%s", "=" * 60)
        for k, v in metrics.items():
            log.info("  %-14s : %.4f", k, v)
        log.info("\nConfusion matrix:\n%s", confusion_matrix(y_true, y_pred))
        log.info(
            "\nClassification report:\n%s",
            classification_report(
                y_true, y_pred,
                target_names=TREE_BINARY_LABELS,
                zero_division=0,
            ),
        )
    return metrics


def evaluate_classifier_multiclass(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    classifier_name: str = "Classifier",
    class_names: list[str] | None = None,
    verbose: bool = True,
) -> dict[str, float]:
    """Compute multiclass metrics. Kept for v4.0 backward compat / diagnostics."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if class_names is None:
        class_names = TREE_STATUS_LABELS

    metrics: dict[str, float] = {
        "accuracy":    float(accuracy_score(y_true, y_pred)),
        "f1_macro":    float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if verbose:
        log.info("\n%s", "=" * 60)
        log.info("%s — Evaluation (Multiclass)", classifier_name)
        log.info("%s", "=" * 60)
        for k, v in metrics.items():
            log.info("  %-15s : %.4f", k, v)
        log.info("\nConfusion matrix:\n%s", confusion_matrix(y_true, y_pred))
        log.info(
            "\nClassification report:\n%s",
            classification_report(y_true, y_pred, target_names=class_names, zero_division=0),
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
    """Save CLF1 and CLF2 to campaign MetaModels directory."""
    path_clf1 = save_model(
        clf1,
        campaign.metamodels_dir / "clf1_tree_fail.joblib",
        metadata={
            "classifier": "CLF1_tree_fail",
            "version":    "4.1_binary",
            "features":   CLF1_FEATURES,
            "thresholds": {
                "fail":  TREE_FAIL_THRESHOLD,
                "stunt": TREE_STUNT_THRESHOLD,
            },
            "classes": TREE_BINARY_LABELS,
            "stunted_proba_threshold": STUNTED_PROBA_THRESHOLD,
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
        log.info("CLF2 not saved — geographic fallback rule in use.")
    return path_clf1, path_clf2


def load_classifiers(
    campaign: CampaignPaths,
) -> tuple[lgb.LGBMClassifier, lgb.LGBMClassifier | None]:
    """Load CLF1 and CLF2 from campaign MetaModels directory."""
    clf1 = load_model(campaign.metamodels_dir / "clf1_tree_fail.joblib")
    clf2_path = campaign.metamodels_dir / "clf2_yield_fail.joblib"
    if clf2_path.exists():
        clf2 = load_model(clf2_path)
    else:
        log.info("CLF2 model not found — geographic fallback rule will be used.")
        clf2 = None
    return clf1, clf2


# ============================================================================
# ROUTING PREDICTION
# ============================================================================

def predict_routing(
    X_sim: pd.DataFrame,
    clf1: lgb.LGBMClassifier,
    clf2: lgb.LGBMClassifier | None,
    return_proba: bool = False,
) -> pd.DataFrame:
    """
    Apply the two-stage cascade to produce routing labels (v4.1 binary CLF1).

    For each input row (one SimID), produces:
        tree_status      : 0=tree_degraded | 1=tree_ok  (binary v4.1)
        yield_failed     : 0=ok | 1=failed
        population       : str label (4 possible outcomes)
        tree_ok_proba    : P(tree_ok) from CLF1  [always added, used for sub-routing]

    The tree_ok_proba column is always returned (even if return_proba=False)
    because it drives the stunted/failed sub-routing within tree_degraded.

    Parameters
    ----------
    X_sim        : pd.DataFrame — feature matrix at SimID level
    clf1         : fitted binary LGBMClassifier (tree_degraded / tree_ok)
    clf2         : fitted binary LGBMClassifier or None (geographic fallback)
    return_proba : bool — if True, also return yield_fail_proba

    Returns
    -------
    pd.DataFrame — index aligned with X_sim
        Columns: tree_status, tree_ok_proba, yield_failed, population
                 [yield_fail_proba] (if return_proba=True)
    """
    result = pd.DataFrame(index=X_sim.index)

    # ── CLF1 : binary tree status (v4.1) ─────────────────────────────────
    clf1_feats = [c for c in CLF1_FEATURES if c in X_sim.columns]
    X_clf1     = X_sim[clf1_feats].copy()
    cat_cols   = [c for c in clf1_feats if c in CATEGORICAL_FEATURES_B2]
    for col in cat_cols:
        if X_clf1[col].dtype.name != "category":
            X_clf1[col] = X_clf1[col].astype("category")

    tree_preds = clf1.predict(X_clf1).astype(int)
    tree_proba = clf1.predict_proba(X_clf1)  # shape (n, 2): [:, 1] = P(tree_ok)

    result["tree_status"]   = tree_preds              # 0=degraded, 1=ok
    result["tree_ok_proba"] = tree_proba[:, 1]        # always stored for sub-routing

    if return_proba:
        result["tree_status_proba"] = tree_proba[:, 1]

    # ── CLF2 : yield failure ──────────────────────────────────────────────
    clf2_feats = [c for c in CLF2_FEATURES if c in X_sim.columns]
    X_clf2     = X_sim[clf2_feats].copy()

    if clf2 is not None:
        result["yield_failed"] = clf2.predict(X_clf2).astype(int)
        if return_proba:
            result["yield_fail_proba"] = clf2.predict_proba(X_clf2)[:, 1]
    else:
        log.debug("CLF2 is None — applying geographic fallback rule.")
        result["yield_failed"] = apply_geographic_rule(X_clf2)
        if return_proba:
            result["yield_fail_proba"] = result["yield_failed"].astype(float)

    # ── Population label (4 outcomes for binary CLF1) ────────────────────
    def _label(row: pd.Series) -> str:
        y = "yield_ok"   if row["yield_failed"] == 0 else "yield_fail"
        t = "tree_ok"    if row["tree_status"]  == TREE_BINARY_OK else "tree_degraded"
        return f"{y} × {t}"

    result["population"] = result.apply(_label, axis=1)
    return result

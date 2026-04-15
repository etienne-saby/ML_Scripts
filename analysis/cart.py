"""
MetAIsAFe — analysis/cart.py
==============================
CART surrogate analysis: fits a shallow Decision Tree on top of the trained
LightGBM surrogate to produce human-readable decision rules and interpretable
feature rankings.

Training mode
-------------
CART is trained on **surrogate predictions over the full Sobol sampling plan**
(N × (2k+2) synthetic points). This captures the global response surface of the
surrogate over the entire parameter domain — the most informative approach for
an agronomy audience (thresholds, rules, actionable insights).

Outputs
-------
- Decision rules (text, printed and returned)
- Feature importance (Gini) DataFrame
- Tree plot          → ``campaign.cart_dir / cart_tree_{target}.png``
- Importance chart   → ``campaign.cart_dir / cart_importance_{target}.png``
- Metrics CSV        → ``campaign.metrics_dir / cart_metrics_{target}.csv``

Author : MetAIsAFe team
Version: 3.2

Notes
-----
Imports corrected vs v3.1:
  - Removed ``CUMULATIVE_TARGETS``, ``ANNUAL_TARGETS``, ``DIRECT_TARGETS``
    (no longer exist in column_taxonomy v3.2).
  - Uses ``STOCK_TARGETS_CROP``, ``STOCK_TARGETS_TREE`` instead.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr

from column_taxonomy import (
    ALL_FEATURES,
    SOBOL_FEATURES,
    STOCK_TARGETS_CROP,
    STOCK_TARGETS_TREE,
    TEMPORAL_FEATURES,
    CLIMATE_FEATURES,
    DESIGN_FEATURES,
    SOIL_FEATURES,
    GEO_FEATURES,
    CATEGORICAL_FEATURES,
)
from config import (
    RANDOM_STATE,
    FIGURE_FORMAT,
    SOBOL_BOUNDS,
    CART_DEFAULT_DEPTH,
    CART_MIN_SAMPLES_SOBOL,
    CART_MIN_SAMPLES_TEST,
)
from utils.plot_utils import save_figure


# ============================================================================
# CONSTANTS
# ============================================================================

# All stock targets (v3.2)
ALL_STOCK_TARGETS: list[str] = list(dict.fromkeys(STOCK_TARGETS_CROP + STOCK_TARGETS_TREE))

# Feature group colour map for the importance chart
_GROUP_COLORS: dict[str, str] = {
    **{f: "#E63946" for f in TEMPORAL_FEATURES},
    **{f: "#2E86AB" for f in CLIMATE_FEATURES},
    **{f: "#F4A261" for f in DESIGN_FEATURES},
    **{f: "#52B788" for f in SOIL_FEATURES},
    **{f: "#9B5DE5" for f in GEO_FEATURES},
    **{f: "#F7B731" for f in CATEGORICAL_FEATURES},
}
_GROUP_LABELS: dict[str, str] = {
    "#E63946": "Temporal",
    "#2E86AB": "Climate",
    "#F4A261": "Plot design",
    "#52B788": "Soil",
    "#9B5DE5": "Geography",
    "#F7B731": "Species",
}


# ============================================================================
# INTERNAL HELPERS
# ============================================================================

def _encode_for_cart(X: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """
    Label-encode categorical/boolean columns for sklearn CART.

    Parameters
    ----------
    X : pd.DataFrame
        Input features (may contain ``dtype='category'``, ``object``, or ``bool``).

    Returns
    -------
    X_enc : pd.DataFrame
        Encoded copy (numeric dtypes only).
    encoders : dict[str, LabelEncoder]
        Fitted encoders per column (pass to :func:`_apply_encoders` at predict time).
    """
    X_enc = X.copy()
    encoders: dict[str, LabelEncoder] = {}

    for col in X_enc.columns:
        if X_enc[col].dtype.name in ("category", "object"):
            le = LabelEncoder()
            X_enc[col] = le.fit_transform(X_enc[col].astype(str))
            encoders[col] = le
        elif X_enc[col].apply(lambda v: isinstance(v, str)).any():
            le = LabelEncoder()
            X_enc[col] = le.fit_transform(X_enc[col].astype(str))
            encoders[col] = le
        elif X_enc[col].dtype == bool:
            X_enc[col] = X_enc[col].astype(int)

    return X_enc, encoders


def _apply_encoders(X: pd.DataFrame, encoders: dict[str, LabelEncoder]) -> pd.DataFrame:
    """
    Apply pre-fitted LabelEncoders for consistent CART predictions.

    Parameters
    ----------
    X : pd.DataFrame
    encoders : dict[str, LabelEncoder]
        Output of :func:`_encode_for_cart`.

    Returns
    -------
    pd.DataFrame
    """
    X_enc = X.copy()
    for col, le in encoders.items():
        if col in X_enc.columns:
            X_enc[col] = le.transform(X_enc[col].astype(str))
    for col in X_enc.columns:
        if col not in encoders and X_enc[col].dtype == bool:
            X_enc[col] = X_enc[col].astype(int)
    return X_enc


# ============================================================================
# FACTORY
# ============================================================================

def build_cart(
    max_depth: int = CART_DEFAULT_DEPTH,
    min_samples_leaf: int = CART_MIN_SAMPLES_SOBOL,
    random_state: int = RANDOM_STATE,
) -> DecisionTreeRegressor:
    """
    Build an unfitted CART regressor.

    Parameters
    ----------
    max_depth : int
        Tree depth. ``4`` → readable (~15 rules); ``6`` → more faithful.
    min_samples_leaf : int
        Minimum samples per leaf. Use ~500 for Sobol plans, ~20 for test sets.
    random_state : int

    Returns
    -------
    DecisionTreeRegressor (unfitted)
    """
    return DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )


# ============================================================================
# FITTING
# ============================================================================

def fit_cart_on_sobol(
    sobol_plan: pd.DataFrame,
    sobol_preds: np.ndarray,
    sobol_features: list[str],
    max_depth: int = CART_DEFAULT_DEPTH,
    min_samples_leaf: int = CART_MIN_SAMPLES_SOBOL,
    verbose: bool = True,
) -> tuple[DecisionTreeRegressor, dict[str, LabelEncoder]]:
    """
    Train CART on surrogate predictions over the Sobol plan.

    The CART is fitted on *predicted* values (ŷ_surrogate), not on raw
    HiSAFe observations — approximating the global response surface of
    the surrogate over the full parameter domain.

    Parameters
    ----------
    sobol_plan : pd.DataFrame
        Full Sobol sampling plan.
    sobol_preds : np.ndarray
        Surrogate predictions on ``sobol_plan``.
    sobol_features : list[str]
        Varied parameters (keys of ``SOBOL_BOUNDS``); fixed features excluded.
    max_depth : int
    min_samples_leaf : int
    verbose : bool

    Returns
    -------
    cart : DecisionTreeRegressor (fitted)
    encoders : dict[str, LabelEncoder]
        For consistent encoding at predict time.
    """
    missing = [f for f in sobol_features if f not in sobol_plan.columns]
    if missing:
        raise ValueError(f"fit_cart_on_sobol: features missing from sobol_plan: {missing}")

    X = sobol_plan[sobol_features].copy()
    y = np.asarray(sobol_preds)

    X_enc, encoders = _encode_for_cart(X)

    cart = build_cart(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    cart.fit(X_enc, y)

    y_pred_cart = cart.predict(X_enc)
    r2 = r2_score(y, y_pred_cart)
    rmse = float(np.sqrt(mean_squared_error(y, y_pred_cart)))

    if verbose:
        print(
            f"  🌳 CART (Sobol) fitted | depth={max_depth} | "
            f"leaves={cart.get_n_leaves()} | R²={r2:.3f} | RMSE={rmse:.4f}"
        )

    return cart, encoders


# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================

def get_cart_feature_importance(
    cart: DecisionTreeRegressor,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Extract and rank Gini-based feature importances from a fitted CART.

    Parameters
    ----------
    cart : DecisionTreeRegressor
    feature_names : list[str]

    Returns
    -------
    pd.DataFrame
        Columns: ``["feature", "importance"]``, sorted descending.
    """
    return (
        pd.DataFrame({"feature": feature_names, "importance": cart.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


# ============================================================================
# PLOTS
# ============================================================================

def plot_cart_tree(
    cart: DecisionTreeRegressor,
    feature_names: list[str],
    save_path: Path,
    target: str = "",
    show: bool | None = None,
) -> Path:
    """
    Visual plot of the CART tree structure.

    Parameters
    ----------
    cart : DecisionTreeRegressor
    feature_names : list[str]
    save_path : Path
    target : str
    show : bool or None

    Returns
    -------
    Path
    """
    n_leaves = cart.get_n_leaves()
    fig_width = max(16, n_leaves * 2)
    fig, ax = plt.subplots(figsize=(fig_width, 8))
    plot_tree(
        cart,
        feature_names=feature_names,
        filled=True,
        rounded=True,
        fontsize=9,
        ax=ax,
        impurity=False,
    )
    ax.set_title(
        f"CART Surrogate Tree — {target}" if target else "CART Surrogate Tree",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    return save_figure(fig, save_path, show=show)


def plot_cart_importance(
    importance_df: pd.DataFrame,
    save_path: Path,
    target: str = "",
    top_n: int = 15,
    show: bool | None = None,
) -> Path:
    """
    Horizontal bar chart of CART feature importances, colour-coded by group.

    Parameters
    ----------
    importance_df : pd.DataFrame
        Output of :func:`get_cart_feature_importance`.
    save_path : Path
    target : str
    top_n : int
    show : bool or None

    Returns
    -------
    Path
    """
    df = importance_df.head(top_n).sort_values("importance")
    colors = [_GROUP_COLORS.get(f, "#AAAAAA") for f in df["feature"]]

    fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.45)))
    ax.barh(df["feature"], df["importance"], color=colors, edgecolor="white")
    ax.set_xlabel("Gini importance", fontsize=12, fontweight="bold")
    ax.set_title(
        f"CART Feature Importance — {target}" if target else "CART Feature Importance",
        fontsize=13, fontweight="bold", pad=12,
    )

    # Legend (feature groups present in this chart)
    present_colors = set(colors)
    patches = [
        mpatches.Patch(color=c, label=_GROUP_LABELS[c])
        for c in _GROUP_LABELS
        if c in present_colors
    ]
    if patches:
        ax.legend(handles=patches, fontsize=9, loc="lower right")

    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    return save_figure(fig, save_path, show=show)


# ============================================================================
# PUBLIC ORCHESTRATOR
# ============================================================================

def run_cart_analysis(
    surrogate,
    sobol_plan: pd.DataFrame,
    sobol_preds: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    sobol_features: list[str],
    target: str,
    campaign,
    max_depth: int = CART_DEFAULT_DEPTH,
    verbose: bool = True,
) -> dict:
    """
    Full CART surrogate analysis pipeline.

    Steps
    -----
    1. Fit CART on Sobol surrogate predictions.
    2. Extract decision rules (text).
    3. Evaluate CART fidelity vs surrogate on test set.
    4. Plot tree structure and feature importances.
    5. Save metrics CSV.

    Parameters
    ----------
    surrogate : fitted LightGBM estimator
        Trained surrogate model.
    sobol_plan : pd.DataFrame
        Sobol sampling plan (output of :func:`analysis.sensitivity.generate_sobol_sample`).
    sobol_preds : np.ndarray
        Surrogate predictions on ``sobol_plan``.
    X_test : pd.DataFrame
        Test feature matrix (original model features, ``dtype='category'``).
    y_test : np.ndarray
        Test observations.
    sobol_features : list[str]
        Parameters varied in the Sobol plan (keys of ``SOBOL_BOUNDS``).
    target : str
        Target stock name (used in titles and file names).
    campaign : CampaignPaths
        Campaign paths object from ``config.get_campaign_paths()``.
    max_depth : int
        CART maximum depth.
    verbose : bool

    Returns
    -------
    dict
        Keys: ``"cart"``, ``"encoders"``, ``"rules"``, ``"importance"``,
        ``"metrics_sobol"``, ``"metrics_test"``.
    """
    print(f"\n🌳 CART analysis — target: {target}")
    campaign.cart_dir.mkdir(parents=True, exist_ok=True)
    campaign.metrics_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Fit CART on Sobol plan ──────────────────────────────────────────
    cart, encoders = fit_cart_on_sobol(
        sobol_plan=sobol_plan,
        sobol_preds=sobol_preds,
        sobol_features=sobol_features,
        max_depth=max_depth,
        verbose=verbose,
    )

    # ── 2. Extract decision rules ──────────────────────────────────────────
    rules = export_text(cart, feature_names=sobol_features)
    if verbose:
        print(f"\n📋 Decision rules for {target}:\n{rules}")

    # ── 3. CART fidelity on Sobol plan ────────────────────────────────────
    X_sobol_enc, _ = _encode_for_cart(sobol_plan[sobol_features])
    cart_preds_sobol = cart.predict(X_sobol_enc)
    metrics_sobol = {
        "r2_sobol":   float(r2_score(sobol_preds, cart_preds_sobol)),
        "rmse_sobol": float(np.sqrt(mean_squared_error(sobol_preds, cart_preds_sobol))),
    }

    # ── 4. CART fidelity on test set (vs surrogate predictions) ──────────
    # Use only sobol_features present in X_test
    test_features_available = [f for f in sobol_features if f in X_test.columns]
    surrogate_preds_test = surrogate.predict(X_test)
    X_test_enc = _apply_encoders(X_test[test_features_available], encoders)
    cart_preds_test = cart.predict(X_test_enc)

    metrics_test = {
        "r2_test":   float(r2_score(surrogate_preds_test, cart_preds_test)),
        "rmse_test": float(np.sqrt(mean_squared_error(surrogate_preds_test, cart_preds_test))),
        "r2_vs_obs": float(r2_score(y_test, cart_preds_test)),
    }

    if verbose:
        print(
            f"  📊 CART fidelity (vs surrogate) | "
            f"R²_sobol={metrics_sobol['r2_sobol']:.3f} | "
            f"R²_test={metrics_test['r2_test']:.3f} | "
            f"R²_vs_obs={metrics_test['r2_vs_obs']:.3f}"
        )

    # ── 5. Feature importance ─────────────────────────────────────────────
    importance_df = get_cart_feature_importance(cart, sobol_features)

    # ── 6. Plots ──────────────────────────────────────────────────────────
    plot_cart_tree(
        cart, sobol_features,
        save_path=campaign.cart_dir / f"cart_tree_{target}.{FIGURE_FORMAT}",
        target=target,
    )
    plot_cart_importance(
        importance_df,
        save_path=campaign.cart_dir / f"cart_importance_{target}.{FIGURE_FORMAT}",
        target=target,
    )

    # ── 7. Save metrics ───────────────────────────────────────────────────
    all_metrics = {"target": target, **metrics_sobol, **metrics_test}
    metrics_path = campaign.metrics_dir / f"cart_metrics_{target}.csv"
    pd.DataFrame([all_metrics]).to_csv(metrics_path, index=False)
    print(f"  💾 CART metrics saved: {metrics_path}")

    return {
        "cart":          cart,
        "encoders":      encoders,
        "rules":         rules,
        "importance":    importance_df,
        "metrics_sobol": metrics_sobol,
        "metrics_test":  metrics_test,
    }

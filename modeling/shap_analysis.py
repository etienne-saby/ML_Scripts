"""
MetAIsAFe — modeling/shap_analysis.py
=======================================
SHAP value computation, summarisation, and export for regression
and classification models.

DESIGN DECISIONS
----------------
- TreeExplainer is used for all models (LightGBM native, exact values).
- SHAP is computed on the TEST SET to avoid explaining memorised training
  patterns. A stratified subsample (max MAX_SHAP_SAMPLES rows) is drawn
  when the test set exceeds this threshold, to keep runtime under ~15s/model.
- Classifiers (CLF1, CLF2) operate at SimID level — no subsampling needed.
- Export format for R/Shiny: long-format CSV with columns
  [target, feature, shap_value, feature_value] — directly usable with
  pivot_longer() and ggplot2.

Author  : Étienne SABY
Created : 2026-04
"""
from __future__ import annotations
from _version import __version__

import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
import matplotlib
import matplotlib.pyplot as plt

from config import (
    CampaignPaths,
    RANDOM_STATE,
    FIGURE_DPI,
    FIGURE_FORMAT,
    CARBON_HORIZONS
)
from column_taxonomy import STOCK_TARGETS_MINIMAL
from utils.io_utils import save_shap_values
from utils.plot_utils import save_figure

log = logging.getLogger(__name__)

# Maximum number of test-set rows passed to TreeExplainer per model.
# Above this threshold, a stratified random subsample is drawn.
MAX_SHAP_SAMPLES: int = 2_000


# ============================================================================
# CORE — TreeExplainer
# ============================================================================

def compute_shap_values(
    model: Any,
    X: pd.DataFrame,
    max_samples: int = MAX_SHAP_SAMPLES,
    random_state: int = RANDOM_STATE,
    check_additivity: bool = False,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Compute exact SHAP values via TreeExplainer.

    Parameters
    ----------
    model : LGBMRegressor or LGBMClassifier
        Fitted LightGBM model.
    X : pd.DataFrame
        Feature matrix (test set recommended). Subsampled internally
        if ``len(X) > max_samples``.
    max_samples : int, default MAX_SHAP_SAMPLES
        Row cap for TreeExplainer (performance guard).
    random_state : int
    check_additivity : bool, default False
        SHAP additivity check — disable for speed on large datasets.

    Returns
    -------
    shap_values : np.ndarray
        Shape (n_samples, n_features). For classifiers, values correspond
        to the POSITIVE class (failure = 1).
    X_sample : pd.DataFrame
        The (possibly subsampled) feature matrix aligned with shap_values.

    Raises
    ------
    ValueError
        If X is empty after subsample.
    """
    if len(X) == 0:
        raise ValueError("compute_shap_values: X is empty.")

    # Subsample if needed (stratification not applicable here — random)
    if len(X) > max_samples:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X), size=max_samples, replace=False)
        X_sample = X.iloc[idx].copy()
        log.info(
            "SHAP: subsampled %d → %d rows (max_samples=%d)",
            len(X), max_samples, max_samples,
        )
    else:
        X_sample = X.copy()

    # Cast object columns to category for LightGBM compatibility
    for col in X_sample.select_dtypes(include="object").columns:
        X_sample[col] = X_sample[col].astype("category")

    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(X_sample, check_additivity=check_additivity)

    # LGBMClassifier returns a list [shap_class0, shap_class1] — keep class 1
    if isinstance(raw, list):
        # Binaire : [class0, class1] → garder class1
        if len(raw) == 2:
            shap_vals = raw[1]
        else:
            # Multiclasse : mean(|SHAP|) sur toutes les classes
            # raw[k] shape : (n_samples, n_features) — stack sur axis=0
            shap_vals = np.mean(
                np.abs(np.stack(raw, axis=0)),   # (n_classes, n_samples, n_features)
                axis=0,                           # → (n_samples, n_features)
            )
    elif isinstance(raw, np.ndarray) and raw.ndim == 3:
        # SHAP retourne parfois (n_samples, n_features, n_classes)
        # On normalise vers (n_samples, n_features)
        if raw.shape[2] <= raw.shape[1]:
            # Dernier axe = classes : mean sur axis=2
            shap_vals = np.abs(raw).mean(axis=2)
        else:
            # Premier axe = classes (n_classes, n_samples, n_features)
            shap_vals = np.abs(raw).mean(axis=0)
    else:
        shap_vals = raw   # Régresseur → 2D directement

    # Validation finale — garantir 2D (n_samples, n_features)
    if shap_vals.ndim != 2:
        raise ValueError(
            f"compute_shap_values: unexpected shap_vals shape {shap_vals.shape} "
            f"after normalization. Expected 2D (n_samples, n_features)."
        )
    
    log.info(
        "SHAP computed: %d samples × %d features",
        shap_vals.shape[0], shap_vals.shape[1],
    )
    return shap_vals, X_sample


# ============================================================================
# SUMMARISATION
# ============================================================================

def summarise_shap(
    shap_values: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Compute mean(|SHAP|) per feature — global feature importance proxy.

    Parameters
    ----------
    shap_values : np.ndarray
        Shape (n_samples, n_features).
    feature_names : list of str

    Returns
    -------
    pd.DataFrame
        Columns: [feature, mean_abs_shap], sorted descending.
    """
    if isinstance(shap_values, list):
        shap_values = np.mean(
            np.abs(np.stack(shap_values, axis=0)), axis=0
        )
    mean_abs = np.abs(shap_values).mean(axis=0)
    
    # Validation longueur
    if len(mean_abs) != len(feature_names):
        raise ValueError(
            f"summarise_shap: mean_abs length ({len(mean_abs)}) "
            f"!= feature_names length ({len(feature_names)}). "
            f"shap_values shape: {np.array(shap_values).shape}"
        )
    return (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

def build_long_format(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    target: str,
) -> pd.DataFrame:
    """
    Convert SHAP matrix to long-format DataFrame for R/Shiny export.

    Output columns: [target, feature, shap_value, feature_value]
    This format is directly usable with tidyr::pivot_longer() and ggplot2.

    Parameters
    ----------
    shap_values : np.ndarray
        Shape (n_samples, n_features).
    X_sample : pd.DataFrame
        Feature values aligned with shap_values.
    target : str
        Target name added as identifier column.

    Returns
    -------
    pd.DataFrame (long format)
    """
    feature_names = list(X_sample.columns)
    n_samples, n_features = shap_values.shape

    records = []
    for j, feat in enumerate(feature_names):
        feat_vals = X_sample.iloc[:, j].values
        shap_vals = shap_values[:, j]
        try:
            feat_vals_float = feat_vals.astype(float)
        except (ValueError, TypeError):
            feat_vals_float = np.full(len(feat_vals), np.nan)
        
        records.append(pd.DataFrame({
            "target":        target,
            "feature":       feat,
            "shap_value":    shap_vals,
            "feature_value": feat_vals_float,
        }))

    return pd.concat(records, ignore_index=True)


# ============================================================================
# PLOTTING
# ============================================================================

def plot_shap_beeswarm(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    target: str,
    max_display: int = 15,
    save_dir: Path | None = None,
) -> None:
    """
    SHAP beeswarm summary plot for a single target.

    Parameters
    ----------
    shap_values : np.ndarray
    X_sample : pd.DataFrame
    target : str
        Used as plot title and filename.
    max_display : int, default 15
        Top N features displayed.
    save_dir : Path, optional
        Directory for saving the figure. If None, only displayed.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    plt.sca(ax)

    # shap.summary_plot writes to the current axes
    shap.summary_plot(
        shap_values,
        X_sample,
        max_display=max_display,
        show=False,
        plot_type="dot",
    )
    fig = plt.gcf()
    ax  = plt.gca()
    ax.set_title(f"SHAP — {target}", fontsize=12, pad=10)
    plt.tight_layout()

    fname = f"shap_beeswarm_{target}.{FIGURE_FORMAT}" if save_dir else None
    save_figure(fig, save_dir / fname if fname else None)
    plt.close(fig)


def plot_shap_bar(
    summary_df: pd.DataFrame,
    target: str,
    top_n: int = 15,
    save_dir: Path | None = None,
) -> None:
    """
    Horizontal bar chart of mean(|SHAP|) for a single target.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Output of ``summarise_shap()``.
    target : str
    top_n : int, default 15
    save_dir : Path, optional
    """
    df_plot = summary_df.head(top_n).iloc[::-1]   # ascending for barh

    fig, ax = plt.subplots(figsize=(8, max(3, top_n * 0.35)))
    bars = ax.barh(
        df_plot["feature"],
        df_plot["mean_abs_shap"],
        color="#2E86AB",
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xlabel("mean(|SHAP|)", fontsize=10)
    ax.set_title(f"SHAP feature importance — {target}", fontsize=11)
    ax.tick_params(axis="y", labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    fname = f"shap_bar_{target}.{FIGURE_FORMAT}" if save_dir else None
    save_figure(fig, save_dir / fname if fname else None)
    plt.close(fig)


def plot_shap_by_target(
    results: dict[str, dict],
    save_dir: Path | None = None,
    plot_type: str = "both",
    max_display: int = 15,
) -> None:
    """
    Generate SHAP plots for all targets in results dict.

    Parameters
    ----------
    results : dict
        Output of ``run_shap_analysis()``. Keys = target names.
        Each value must contain: shap_values, X_sample, summary_df.
    save_dir : Path, optional
    plot_type : {"beeswarm", "bar", "both"}
    max_display : int
    """
    valid_types = {"beeswarm", "bar", "both"}
    if plot_type not in valid_types:
        raise ValueError(f"plot_type must be one of {valid_types}, got '{plot_type}'")

    for target, res in results.items():
        log.info("Plotting SHAP — %s", target)
        if plot_type in ("beeswarm", "both"):
            plot_shap_beeswarm(
                res["shap_values"], res["X_sample"],
                target=target, max_display=max_display, save_dir=save_dir,
            )
        if plot_type in ("bar", "both"):
            plot_shap_bar(
                res["summary_df"], target=target,
                top_n=max_display, save_dir=save_dir,
            )


# ============================================================================
# ORCHESTRATION
# ============================================================================

def run_shap_analysis(
    models: dict[str, Any],
    X_test: pd.DataFrame,
    targets: list[str] | None = None,
    max_samples: int = MAX_SHAP_SAMPLES,
    random_state: int = RANDOM_STATE,
    horizons: list[int] = CARBON_HORIZONS,
) -> dict[str, dict]:
    """
    Run full SHAP analysis for a set of regression models.

    Parameters
    ----------
    models : dict[str, estimator]
        Mapping {target_name: fitted_model}. Targets not in ``targets``
        are silently skipped.
    X_test : pd.DataFrame
        Test-set feature matrix (full feature set — columns are selected
        internally per model based on model.feature_name_).
    targets : list of str, optional
        Targets to analyse. Defaults to STOCK_TARGETS_MINIMAL.
    max_samples : int
    random_state : int

    Returns
    -------
    dict[str, dict]
        Keys = target names. Each value:
            shap_values  : np.ndarray (n_samples × n_features)
            X_sample     : pd.DataFrame
            summary_df   : pd.DataFrame (mean_abs_shap, sorted)
            feature_names: list[str]
    """
    if targets is None:
        targets = STOCK_TARGETS_MINIMAL

    results: dict[str, dict] = {}

    for target in targets:
        
        if target not in models:
            log.warning("SHAP: model for '%s' not found — skipped.", target)
            continue

        model = models[target]
        log.info("=" * 55)
        log.info("SHAP analysis — %s", target)
        log.info("=" * 55)

        # Select only the features the model was trained on
        try:
            feat_names = model.feature_name_
        except AttributeError:
            feat_names = list(X_test.columns)
            log.warning(
                "SHAP: model.feature_name_ not available for '%s' — "
                "using all X_test columns.", target,
            )

        avail = [f for f in feat_names if f in X_test.columns]
        missing = set(feat_names) - set(avail)
        if missing:
            log.warning("SHAP: missing features for '%s': %s", target, sorted(missing))

        X_model = X_test[avail]

        shap_vals, X_sample = compute_shap_values(
            model, X_model,
            max_samples=max_samples,
            random_state=random_state,
        )

        summary_df = summarise_shap(shap_vals, list(X_sample.columns))

        results[target] = {
            "shap_values":   shap_vals,
            "X_sample":      X_sample,
            "summary_df":    summary_df,
            "feature_names": list(X_sample.columns),
        }

        log.info(
            "Top 3 features: %s",
            summary_df.head(3)["feature"].tolist(),
        )

    return results


def run_shap_classifiers(
    clf1: Any,
    clf2: Any | None,
    X_clf1: pd.DataFrame,
    X_clf2: pd.DataFrame,
    random_state: int = RANDOM_STATE,
) -> dict[str, dict]:
    """
    Run SHAP analysis on CLF1 and CLF2 (SimID-level, no subsampling).

    Since classifiers operate at SimID level (hundreds of rows, not tens
    of thousands), no subsampling is applied.

    Parameters
    ----------
    clf1 : LGBMClassifier
        Fitted tree failure classifier.
    clf2 : LGBMClassifier or None
        Fitted yield failure classifier. Skipped if None.
    X_clf1 : pd.DataFrame
        Feature matrix at SimID level for CLF1.
    X_clf2 : pd.DataFrame
        Feature matrix at SimID level for CLF2.
    random_state : int

    Returns
    -------
    dict with keys "clf1_tree_fail" and optionally "clf2_yield_fail".
    """
    results: dict[str, dict] = {}

    for name, model, X in [
        ("clf1_tree_fail",  clf1, X_clf1),
        ("clf2_yield_fail", clf2, X_clf2),
    ]:
        if model is None:
            log.info("SHAP: %s is None (geographic rule) — skipped.", name)
            continue

        log.info("SHAP analysis — %s (%d SimIDs)", name, len(X))

        # No subsampling: SimID-level data is small
        shap_vals, X_sample = compute_shap_values(
            model, X,
            max_samples=len(X),    # disable subsampling
            random_state=random_state,
        )

        summary_df = summarise_shap(shap_vals, list(X_sample.columns))
        results[name] = {
            "shap_values":   shap_vals,
            "X_sample":      X_sample,
            "summary_df":    summary_df,
            "feature_names": list(X_sample.columns),
        }

        log.info(
            "Top 3 features (%s): %s",
            name, summary_df.head(3)["feature"].tolist(),
        )

    return results


# ============================================================================
# EXPORT — R/Shiny compatible
# ============================================================================

def export_shap_for_shiny(
    results: dict[str, dict],
    save_dir: Path,
    include_raw: bool = False,
) -> Path:
    """
    Export SHAP results to long-format CSV for R/Shiny consumption.

    Produces two files:
        shap_long.csv    — long format: [target, feature, shap_value, feature_value]
        shap_summary.csv — wide summary: [target, feature, mean_abs_shap]

    Parameters
    ----------
    results : dict
        Output of ``run_shap_analysis()`` or ``run_shap_classifiers()``.
    save_dir : Path
        Destination directory (e.g. campaign.shap_data_dir).
    include_raw : bool, default False
        If True, also saves raw SHAP matrices as per-target CSVs
        (via ``io_utils.save_shap_values``). Large files — use sparingly.

    Returns
    -------
    Path
        Path to shap_long.csv.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    long_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []

    for target, res in results.items():
        # Long format
        long_df = build_long_format(res["shap_values"], res["X_sample"], target)
        long_frames.append(long_df)

        # Summary
        summary_df = res["summary_df"].copy()
        summary_df.insert(0, "target", target)
        summary_frames.append(summary_df)

        # Optional raw export
        if include_raw:
            save_shap_values(
                res["shap_values"],
                res["feature_names"],
                save_dir / f"shap_raw_{target}.csv",
                mean_abs=False,   # already in summary
            )

    # Write consolidated files
    long_path = save_dir / "shap_long.csv"
    pd.concat(long_frames, ignore_index=True).to_csv(long_path, index=False)
    log.info("💾 SHAP long format saved: %s", long_path)

    summary_path = save_dir / "shap_summary.csv"
    pd.concat(summary_frames, ignore_index=True).to_csv(summary_path, index=False)
    log.info("💾 SHAP summary saved: %s", summary_path)

    return long_path
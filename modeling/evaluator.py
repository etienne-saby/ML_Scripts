"""
MetAIsAFe — modeling/evaluator.py
==================================
Metrics computation and diagnostic visualisations for surrogate models.

Public API
----------
    compute_metrics(y_true, y_pred, prefix)  -> dict[str, float]
    plot_pred_vs_obs(...)                    -> Path
    plot_residuals(...)                      -> Path
    plot_error_distribution(...)             -> Path
    plot_feature_importances(...)            -> Path
    compare_models_plot(...)                 -> Path

Author  : Étienne SABY
Updated : 2026-04
"""
from __future__ import annotations
from _version import __version__

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from config import FIGURE_DPI, COLOR_PALETTE
from utils.plot_utils import save_figure

# ============================================================================
# GLOBAL PLOT STYLE
# ============================================================================

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = FIGURE_DPI
plt.rcParams["font.size"] = 10

_DEFAULT_COLOR = "#2E86AB"


# ============================================================================
# INTERNAL HELPERS
# ============================================================================

def _validate_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Validate input arrays before computing metrics or plotting.

    Raises
    ------
    ValueError
        If arrays are empty, mismatched, or contain non-finite values.
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("Empty arrays — y_true and y_pred must not be empty.")
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Size mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}."
        )
    if not np.all(np.isfinite(y_true)):
        raise ValueError("y_true contains NaN or Inf values.")
    if not np.all(np.isfinite(y_pred)):
        raise ValueError("y_pred contains NaN or Inf values.")


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute MAPE safely, excluding zero observations.

    Returns
    -------
    float
        MAPE in percent, or np.nan if all observations are zero.
    """
    mask = y_true != 0
    if not mask.any():
        warnings.warn("All y_true values are zero — MAPE returns NaN.")
        return np.nan
    if (~mask).any():
        warnings.warn(f"{(~mask).sum()} zero(s) in y_true excluded from MAPE.")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def _safe_pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson r, safe for zero-variance arrays."""
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        warnings.warn("Zero variance — Pearson r returns NaN.", UserWarning, stacklevel=3)
        return np.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _get_color(dataset_label: str) -> str:
    """Return palette colour for a dataset label, with safe fallback."""
    return COLOR_PALETTE.get(dataset_label, COLOR_PALETTE.get("test", _DEFAULT_COLOR))


# ============================================================================
# METRICS
# ============================================================================

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = "",
) -> dict[str, float]:
    """
    Compute a comprehensive set of regression metrics.

    Metrics
    -------
    R², RMSE, MAE, MAPE, NRMSE, Bias, Pearson r, Std residuals.

    Parameters
    ----------
    y_true : array-like
        Observed values.
    y_pred : array-like
        Predicted values.
    prefix : str, optional
        Prefix for metric keys (e.g. ``"train_"``, ``"test_"``).

    Returns
    -------
    dict[str, float]

    Examples
    --------
    >>> metrics = compute_metrics(y_test, y_pred, prefix="test_")
    >>> print(metrics["test_r2"])
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    _validate_arrays(y_true, y_pred)

    obs_range = y_true.max() - y_true.min()
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    return {
        f"{prefix}r2":            float(r2_score(y_true, y_pred)),
        f"{prefix}rmse":          rmse,
        f"{prefix}mae":           float(mean_absolute_error(y_true, y_pred)),
        f"{prefix}mape":          _safe_mape(y_true, y_pred),
        f"{prefix}nrmse":         rmse / obs_range if obs_range > 0 else np.nan,
        f"{prefix}bias":          float(np.mean(y_pred - y_true)),
        f"{prefix}pearson_r":     _safe_pearson_r(y_true, y_pred),
        f"{prefix}std_residuals": float(np.std(y_pred - y_true)),
    }


# ============================================================================
# PLOTS
# ============================================================================

def plot_pred_vs_obs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path,
    title: str = "Predictions vs Observations",
    dataset_label: str = "test",
    show: bool | None = None,
) -> Path:
    """
    Scatter plot of observations vs predictions with a 1:1 identity line.

    Annotates the plot with R², RMSE and MAE.

    Parameters
    ----------
    y_true : array-like
    y_pred : array-like
    save_path : Path
    title : str
    dataset_label : str
        Legend label and colour key (``"train"``, ``"test"``, …).
    show : bool or None
        ``None`` → auto-detect interactive environment.

    Returns
    -------
    Path
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    _validate_arrays(y_true, y_pred)
    metrics = compute_metrics(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(
        y_true, y_pred,
        alpha=0.5, s=30,
        c=_get_color(dataset_label),
        edgecolors="white", linewidths=0.5,
        label=dataset_label,
        zorder=3,
    )
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=2, label="1:1 line", zorder=2)
    ax.text(
        0.05, 0.95,
        f"R²   = {metrics['r2']:.3f}\nRMSE = {metrics['rmse']:.3f}\nMAE  = {metrics['mae']:.3f}",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        fontsize=11, family="monospace",
    )
    ax.set_xlabel("Observations", fontsize=12, fontweight="bold")
    ax.set_ylabel("Predictions", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return save_figure(fig, save_path, show=show)


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path,
    title: str = "Residuals vs Predicted",
    dataset_label: str = "test",
    show: bool | None = None,
) -> Path:
    """
    Residuals vs predicted values plot with a zero-reference line.

    Parameters
    ----------
    y_true : array-like
    y_pred : array-like
    save_path : Path
    title : str
    dataset_label : str
    show : bool or None

    Returns
    -------
    Path
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    _validate_arrays(y_true, y_pred)
    residuals = y_pred - y_true

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(
        y_pred, residuals,
        alpha=0.4, s=25,
        c=_get_color(dataset_label),
        edgecolors="white", linewidths=0.3,
        label=dataset_label,
    )
    ax.axhline(0, color="black", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Predicted values", fontsize=12, fontweight="bold")
    ax.set_ylabel("Residuals (pred − obs)", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return save_figure(fig, save_path, show=show)


def plot_error_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path,
    title: str = "Residual Distribution",
    show: bool | None = None,
) -> Path:
    """
    Histogram of residuals overlaid with a fitted Gaussian curve.

    Parameters
    ----------
    y_true : array-like
    y_pred : array-like
    save_path : Path
    title : str
    show : bool or None

    Returns
    -------
    Path
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    _validate_arrays(y_true, y_pred)
    residuals = y_pred - y_true

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residuals, bins=50, density=True, alpha=0.6, color=_DEFAULT_COLOR, label="Residuals")

    mu, sigma = norm.fit(residuals)
    x_range = np.linspace(residuals.min(), residuals.max(), 300)
    ax.plot(x_range, norm.pdf(x_range, mu, sigma), "r-", lw=2, label=f"N({mu:.3f}, {sigma:.3f})")

    ax.axvline(0, color="black", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Residual (pred − obs)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Density", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return save_figure(fig, save_path, show=show)


def plot_feature_importances(
    importances: pd.Series,
    save_path: Path,
    title: str = "Feature Importances",
    top_n: int = 20,
    show: bool | None = None,
) -> Path:
    """
    Horizontal bar chart of feature importances (top N).

    Parameters
    ----------
    importances : pd.Series
        Index = feature names, values = importance scores.
    save_path : Path
    title : str
    top_n : int
        Maximum number of features displayed.
    show : bool or None

    Returns
    -------
    Path
    """
    top = importances.nlargest(top_n)
    fig, ax = plt.subplots(figsize=(10, max(4, len(top) * 0.4)))
    top.sort_values().plot(kind="barh", ax=ax, color=_DEFAULT_COLOR, edgecolor="white")
    ax.set_xlabel("Importance", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    return save_figure(fig, save_path, show=show)

def plot_shap_summary(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    save_path: Path,
    title: str = "SHAP Summary",
    max_display: int = 15,
    show: bool | None = None,
) -> Path:
    """
    SHAP beeswarm summary plot.

    Parameters
    ----------
    shap_values : np.ndarray
        Raw SHAP values (n_samples × n_features).
    X : pd.DataFrame
        Feature matrix aligned with shap_values.
    save_path : Path
    title : str
    max_display : int
        Max number of features to display.
    show : bool or None

    Returns
    -------
    Path
    """
    try:
        import shap
    except ImportError:
        raise ImportError("shap is required: pip install shap")

    fig, ax = plt.subplots(figsize=(10, max(5, max_display * 0.4)))
    shap.summary_plot(
        shap_values, X,
        max_display=max_display,
        show=False,
        plot_size=None,
    )
    fig = plt.gcf()
    ax = plt.gca()
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    return save_figure(fig, save_path, show=show)

def compare_models_plot(
    comparison_df: pd.DataFrame,
    save_path: Path,
    metric: str = "cv_r2_mean",
    title: str = "Model Comparison",
    show: bool | None = None,
) -> Path:
    """
    Bar chart comparing models on a chosen metric.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        DataFrame with columns 'model_type' and the chosen metric.
        Can be built manually or from benchmark results.
        Must contain columns ``"model_type"`` and ``metric``.
    save_path : Path
    metric : str
        Column to plot.
    title : str
    show : bool or None

    Returns
    -------
    Path
    """
    if "model_type" not in comparison_df.columns or metric not in comparison_df.columns:
        raise ValueError(
            f"comparison_df must contain 'model_type' and '{metric}' columns."
        )
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [_DEFAULT_COLOR] * len(comparison_df)
    ax.bar(comparison_df["model_type"], comparison_df[metric], color=colors, edgecolor="white")
    ax.set_ylabel(metric, fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    return save_figure(fig, save_path, show=show)

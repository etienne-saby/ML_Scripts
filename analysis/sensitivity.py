"""
MetAIsAFe — analysis/sensitivity.py
=====================================
Global Sensitivity Analysis (GSA) via Sobol indices using SALib.

SHAP functions are preserved here for ad-hoc / publication use,
but are NOT called in the main pipeline (run_pipeline).
Use ``run_shap=True`` explicitly if needed.

Public API
----------
    generate_sobol_sample(...)     -> pd.DataFrame
    predict_on_sample(...)         -> np.ndarray
    compute_sobol_indices(...)     -> dict[str, pd.DataFrame]
    plot_sobol_bars(...)           -> Path
    compute_shap_values(...)       -> np.ndarray   [optional, not in pipeline]

Author : MetAIsAFe team
Version: 3.2
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from SALib.sample import saltelli
from SALib.analyze import sobol

from config import (
    SOBOL_BOUNDS,
    SOBOL_N_SAMPLES,
    SOBOL_FIXED_PARAMS,
    RANDOM_STATE,
)
from utils.plot_utils import save_figure


# ============================================================================
# SOBOL : GLOBAL SENSITIVITY ANALYSIS
# ============================================================================

def generate_sobol_sample(
    param_bounds: dict[str, list] | None = None,
    n_samples: int = SOBOL_N_SAMPLES,
    fixed_params: dict[str, float] | None = None,
    random_state: int = RANDOM_STATE,
    calc_second_order: bool = False,
) -> pd.DataFrame:
    """
    Generate a Saltelli Sobol sampling plan for sensitivity analysis.

    Handles mixed parameter types:

    - **Numeric** → Saltelli sampling via SALib
    - **Categorical** → uniform random sampling from modality list
    - **Fixed** → constant columns appended after sampling

    The Saltelli method generates ``N × (2k + 2)`` rows, where ``k`` is the
    number of *numeric* parameters.

    Parameters
    ----------
    param_bounds : dict, optional
        ``{param_name: [min, max]}`` for numeric parameters or
        ``{param_name: [mod1, mod2, ...]}`` for categorical ones.
        Defaults to ``SOBOL_BOUNDS`` from ``config.py``.
    n_samples : int
        Base sample size ``N``.
    fixed_params : dict, optional
        Parameters held constant (e.g. ``Harvest_Year_Absolute``).
        Defaults to ``SOBOL_FIXED_PARAMS``.
    random_state : int
        Seed for categorical sampling reproducibility.
    calc_second_order : bool
        Whether to include second-order Sobol indices.

    Returns
    -------
    pd.DataFrame
        Full Sobol design: numeric varied + categorical varied + fixed.

    Examples
    --------
    >>> plan = generate_sobol_sample(n_samples=1024)
    >>> print(f"Plan: {len(plan):,} rows × {plan.shape[1]} cols")
    """
    if param_bounds is None:
        param_bounds = SOBOL_BOUNDS
    if fixed_params is None:
        fixed_params = SOBOL_FIXED_PARAMS

    numeric_params: dict[str, list] = {}
    categorical_params: dict[str, list] = {}

    for name, bounds in param_bounds.items():
        if not isinstance(bounds, list) or len(bounds) < 2:
            raise ValueError(
                f"Parameter '{name}': bounds must be a list of ≥2 values, got {bounds!r}"
            )
        if any(isinstance(v, (str, bool)) for v in bounds):
            categorical_params[name] = bounds
        elif len(set(bounds)) <= 2 and all(isinstance(v, int) for v in bounds):
            categorical_params[name] = bounds
        else:
            if len(bounds) != 2:
                raise ValueError(
                    f"Numeric parameter '{name}' must have exactly [min, max], "
                    f"got {len(bounds)} values."
                )
            numeric_params[name] = bounds

    num_names = list(numeric_params.keys())
    num_bounds = [numeric_params[p] for p in num_names]

    problem = {
        "num_vars": len(num_names),
        "names": num_names,
        "bounds": num_bounds,
    }

    n_total = (
        n_samples * (2 * len(num_names) + 2)
        if calc_second_order
        else n_samples * (len(num_names) + 2)
    )

    print(
        f"🎲 Sobol (Saltelli) plan | N={n_samples} | "
        f"numeric={len(num_names)} | categorical={len(categorical_params)} | "
        f"fixed={len(fixed_params)} | total rows={n_total:,}"
    )

    sobol_samples = saltelli.sample(problem, n_samples, calc_second_order=calc_second_order)
    df = pd.DataFrame(sobol_samples, columns=num_names)

    rng = np.random.default_rng(random_state)
    for cat_name, modalities in categorical_params.items():
        df[cat_name] = rng.choice(modalities, size=n_total)

    for param, value in fixed_params.items():
        df[param] = value

    print(f"✅ Sobol plan ready: {len(df):,} rows × {len(df.columns)} cols")
    return df


def predict_on_sample(
    model,
    sobol_plan: pd.DataFrame,
    feature_names: list[str],
) -> np.ndarray:
    """
    Run surrogate predictions on a Sobol plan.

    Parameters
    ----------
    model : fitted estimator
        Trained LightGBM surrogate (sklearn-compatible API).
    sobol_plan : pd.DataFrame
        Plan generated by :func:`generate_sobol_sample`.
    feature_names : list[str]
        Feature order expected by the model.

    Returns
    -------
    np.ndarray
        1-D predictions array of length ``len(sobol_plan)``.

    Raises
    ------
    ValueError
        If any feature is missing from ``sobol_plan``.
    """
    missing = set(feature_names) - set(sobol_plan.columns)
    if missing:
        raise ValueError(f"Missing features in Sobol plan: {missing}")

    X = sobol_plan[feature_names]
    for col in X.select_dtypes(include="object").columns:
        X[col] = X[col].astype("category")
    print(f"🔮 Predicting on {len(X):,} Sobol samples…")
    preds = model.predict(X)
    print(f"✅ Predictions complete ({len(preds):,} values)")
    return preds


def compute_sobol_indices(
    param_bounds: dict[str, list],
    predictions: np.ndarray,
    n_samples: int,
    calc_second_order: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Compute first-order (S1) and total-order (ST) Sobol indices.

    Categorical parameters are automatically excluded (not amenable to
    variance-based decomposition).

    Interpretation guide
    --------------------
    - ``ST < 0.05`` → parameter has little influence (candidate for fixing)
    - ``ST > 0.50`` → parameter has dominant influence

    Parameters
    ----------
    param_bounds : dict
        Same dict as passed to :func:`generate_sobol_sample`.
    predictions : np.ndarray
        Surrogate predictions on the Sobol design.
    n_samples : int
        ``N`` used to generate the Sobol design.
    calc_second_order : bool
        Whether to compute S2 pairwise interactions.

    Returns
    -------
    dict
        Keys: ``"S1"``, ``"ST"`` (and optionally ``"S2"``).
        Each value is a :class:`pd.DataFrame` with columns
        ``[parameter, <index>, <index>_conf]``.

    Examples
    --------
    >>> indices = compute_sobol_indices(SOBOL_BOUNDS, preds, n_samples=1024)
    >>> print(indices["ST"].sort_values("ST", ascending=False))
    """
    numeric_bounds = {
        name: bounds for name, bounds in param_bounds.items()
        if not any(isinstance(v, (str, bool)) for v in bounds)
        and not (len(set(bounds)) <= 2 and all(isinstance(v, int) for v in bounds))
    }
    
    if not numeric_bounds:
        raise ValueError("No numeric parameters found — Sobol indices cannot be computed.")

    num_names = list(numeric_bounds.keys())
    problem = {
        "num_vars": len(num_names),
        "names": num_names,
        "bounds": [numeric_bounds[p] for p in num_names],
    }

    print(f"📐 Computing Sobol indices for {len(num_names)} numeric parameters…")
    si = sobol.analyze(problem, predictions, calc_second_order=calc_second_order, print_to_console=False)

    result: dict[str, pd.DataFrame] = {
        "S1": pd.DataFrame({"parameter": num_names, "S1": si["S1"], "S1_conf": si["S1_conf"]}),
        "ST": pd.DataFrame({"parameter": num_names, "ST": si["ST"], "ST_conf": si["ST_conf"]}),
    }

    if calc_second_order and "S2" in si:
        s2_rows = []
        for i, p1 in enumerate(num_names):
            for j, p2 in enumerate(num_names):
                if j > i:
                    s2_rows.append({"param1": p1, "param2": p2, "S2": si["S2"][i, j]})
        result["S2"] = pd.DataFrame(s2_rows)

    print(f"✅ Sobol indices computed | Top ST: "
          f"{result['ST'].nlargest(3, 'ST')[['parameter', 'ST']].to_dict('records')}")
    return result


def plot_sobol_bars(
    sobol_indices: dict[str, pd.DataFrame],
    save_path: Path,
    target: str = "",
    top_n: int = 15,
    show: bool | None = None,
) -> Path:
    """
    Horizontal bar chart of S1 and ST Sobol indices.

    Parameters
    ----------
    sobol_indices : dict
        Output of :func:`compute_sobol_indices`.
    save_path : Path
    target : str
        Target name used in the figure title.
    top_n : int
        Maximum number of parameters shown (ranked by ST).
    show : bool or None

    Returns
    -------
    Path
    """
    df_st = sobol_indices["ST"].copy()
    df_s1 = sobol_indices["S1"].copy()

    df_plot = (
        df_st.merge(df_s1, on="parameter")
        .nlargest(top_n, "ST")
        .sort_values("ST")
    )

    fig, ax = plt.subplots(figsize=(10, max(4, len(df_plot) * 0.45)))
    y_pos = np.arange(len(df_plot))
    bar_height = 0.35

    ax.barh(
        y_pos + bar_height / 2, df_plot["ST"], bar_height,
        label="ST (total order)", color="#2E86AB", alpha=0.85,
        xerr=df_plot["ST_conf"], error_kw={"elinewidth": 1, "capsize": 3},
    )
    ax.barh(
        y_pos - bar_height / 2, df_plot["S1"], bar_height,
        label="S1 (first order)", color="#E63946", alpha=0.85,
        xerr=df_plot["S1_conf"], error_kw={"elinewidth": 1, "capsize": 3},
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot["parameter"], fontsize=10)
    ax.set_xlabel("Sobol index", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Sobol Sensitivity Indices — {target}" if target else "Sobol Sensitivity Indices",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.axvline(0.05, color="grey", linestyle=":", linewidth=1, label="Threshold 0.05")
    ax.legend(fontsize=10)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    return save_figure(fig, save_path, show=show)


# ============================================================================
# SHAP — optional, NOT called in the main pipeline
# ============================================================================

def compute_shap_values(
    model,
    X: pd.DataFrame,
    n_samples: int | None = None,
    random_state: int = RANDOM_STATE,
) -> "np.ndarray":
    """
    Compute SHAP values using TreeExplainer (LightGBM-compatible).

    .. note::
        This function is **not** called in the main ``run_pipeline`` workflow.
        Use it explicitly for publication-quality interpretability analysis.

    Parameters
    ----------
    model : fitted LightGBM estimator
    X : pd.DataFrame
        Feature matrix (may be subsampled via ``n_samples``).
    n_samples : int, optional
        If provided, a random subsample of ``X`` is used (faster).
    random_state : int

    Returns
    -------
    np.ndarray
        SHAP values array of shape ``(n_rows, n_features)``.

    Raises
    ------
    ImportError
        If the ``shap`` package is not installed.
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "SHAP is not installed. Run `pip install shap` to use this function."
        ) from exc

    if n_samples is not None and n_samples < len(X):
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X), size=n_samples, replace=False)
        X = X.iloc[idx]

    print(f"🔍 Computing SHAP values on {len(X):,} samples…")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    print(f"✅ SHAP values computed — shape: {np.array(shap_values).shape}")
    return shap_values

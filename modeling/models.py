"""
MetAIsAFe — Model Factories v4.0
==================================
Factory functions for LightGBM (primary) and XGBoost (fallback only).

MODEL SELECTION RATIONALE
--------------------------
LightGBM is the sole production model. See ``config.py`` for the full
rationale. XGBoost is kept here for use by ``xgb_fallback.py`` only.
Random Forest has been retired from the pipeline.

Author  : Étienne SABY
Updated : 2026-04
"""
from __future__ import annotations
from _version import __version__

import warnings
from typing import Any

import pandas as pd
import lightgbm as lgb
import xgboost as xgb

from config import LGB_PARAMS, XGB_PARAMS

# ============================================================================
# LIGHTGBM — PRIMARY PRODUCTION MODEL
# ============================================================================

def build_lgb(
    params: dict[str, Any] | None = None,
    categorical_feature: list[str] | None = None,
    **kwargs: Any,
) -> lgb.LGBMRegressor:
    """
    Build a LightGBM regressor with default or custom parameters.

    LightGBM is the primary surrogate model for MetAIsAFe.
    See ``config.MODEL_SELECTION_RATIONALE`` for justification.

    Categorical handling
    --------------------
    LightGBM handles pandas ``category`` dtype natively.
    Pass ``categorical_feature`` to store the list on the model instance
    so ``trainer.py`` can inject it at ``.fit()`` time (LightGBM ≥ 4.x
    ignores categorical_feature if passed to the constructor).

    Parameters
    ----------
    params : dict, optional
        Hyperparameter overrides. Merged with ``LGB_PARAMS`` from ``config``.
    categorical_feature : list of str, optional
        Categorical column names for LightGBM native handling.
    **kwargs
        Additional parameter overrides (highest priority).

    Returns
    -------
    lgb.LGBMRegressor
        Unfitted model instance.

    Examples
    --------
    >>> model = build_lgb(categorical_feature=cat_feats)
    >>> model.fit(X_train, y_train, categorical_feature=cat_feats)
    """
    final_params = LGB_PARAMS.copy()
    if params is not None:
        final_params.update(params)
    final_params.update(kwargs)

    model = lgb.LGBMRegressor(**final_params)

    # Store categorical_feature as a custom attribute for trainer.py
    # LightGBM ≥ 4.x requires it to be passed at .fit() time, not __init__
    model._categorical_feature = categorical_feature

    return model

def build_lgb_classifier(
    params: dict[str, Any] | None = None,
    categorical_feature: list[str] | None = None,
    **kwargs: Any,
) -> lgb.LGBMClassifier:
    """
    Build a LightGBM binary classifier for cascade routing.

    Used by classifiers.py for tree_fail and yield_fail classification.
    Parameters mirror build_lgb() — see its docstring for details.

    Parameters
    ----------
    params : dict, optional
        Overrides for LGB_PARAMS. Note: objective is forced to
        'binary' and metric to 'binary_logloss'.
    categorical_feature : list of str, optional
    **kwargs

    Returns
    -------
    lgb.LGBMClassifier
    """
    base = LGB_PARAMS.copy()
    # Remove regression-specific keys irrelevant to classifier
    base.pop("importance_type", None)

    if params is not None:
        base.update(params)
    base.update(kwargs)

    model = lgb.LGBMClassifier(**base)
    model._categorical_feature = categorical_feature
    return model

# ============================================================================
# XGBOOST — FALLBACK MODEL (xgb_fallback.py only)
# ============================================================================

def build_xgb(
    params: dict[str, Any] | None = None,
    enable_categorical: bool = False,
    **kwargs: Any,
) -> xgb.XGBRegressor:
    """
    Build an XGBoost regressor.

    .. warning::
        XGBoost is a **fallback model** only, used in ``xgb_fallback.py``
        for validation purposes. Do NOT use in the main pipeline.

    Parameters
    ----------
    params : dict, optional
        Hyperparameter overrides. Merged with ``XGB_PARAMS`` from ``config``.
    enable_categorical : bool, default False
        Enable XGBoost native categorical support (requires dtype ``category``).
    **kwargs
        Additional overrides.

    Returns
    -------
    xgb.XGBRegressor
    """
    final_params = XGB_PARAMS.copy()
    if params is not None:
        final_params.update(params)
    final_params.update(kwargs)
    if enable_categorical:
        final_params["enable_categorical"] = True

    return xgb.XGBRegressor(**final_params)


# ============================================================================
# FEATURE IMPORTANCES
# ============================================================================

def get_feature_importances(
    model: lgb.LGBMRegressor | xgb.XGBRegressor,
    feature_names: list[str],
    importance_type: str = "gain",
) -> pd.Series:
    """
    Extract feature importances from a trained LightGBM or XGBoost model.

    Parameters
    ----------
    model : trained LGBMRegressor or XGBRegressor
    feature_names : list of str
        Feature names in the order of ``X_train.columns``.
    importance_type : str, default ``"gain"``
        ``"gain"``  — average reduction in loss (recommended).
        ``"split"`` — number of times the feature is used for splits.

    Returns
    -------
    pd.Series
        Index = feature names, values = importance scores — sorted descending.

    Raises
    ------
    ValueError
        If the model is not fitted.
    """
    if isinstance(model, lgb.LGBMRegressor):
        if not hasattr(model, "feature_importances_"):
            raise ValueError("LGBMRegressor is not fitted. Call model.fit() first.")
        importances = model.feature_importances_

    elif isinstance(model, xgb.XGBRegressor):
        if not hasattr(model, "feature_importances_"):
            raise ValueError("XGBRegressor is not fitted. Call model.fit() first.")
        importances = model.feature_importances_

    else:
        raise TypeError(
            f"Unsupported model type: {type(model).__name__}. "
            "Expected LGBMRegressor or XGBRegressor."
        )

    return (
        pd.Series(importances, index=feature_names, name="importance")
        .sort_values(ascending=False)
    )
"""
MetAIsAFe — Trainer v4.0
==========================
Cross-validation, final training, and Optuna hyperparameter optimisation.

CHANGES from v3.1
-----------------
- compare_models() removed (single model — LightGBM only).
- model_name in tune_optuna() restricted to ``"lightgbm"`` in main pipeline.
- RF support removed from tune_optuna().

Author  : Étienne SABY
Updated : 2026-04
"""
from __future__ import annotations

from _version import __version__

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    f1_score,
    roc_auc_score,
)
import optuna
from tqdm.auto import tqdm

from config import (
    CV_N_FOLDS,
    RANDOM_STATE,
    OPTUNA_N_TRIALS,
    OPTUNA_TIMEOUT,
    OPTUNA_N_JOBS,
    OPTUNA_SEARCH_SPACE,
    CARBON_HORIZONS
)

log = logging.getLogger(__name__)

# Suppress Optuna verbose output by default
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ============================================================================
# CROSS-VALIDATION
# ============================================================================

def cross_validate(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    cv: GroupKFold | None = None,
    n_folds: int = CV_N_FOLDS,
    extra_train_features: pd.DataFrame | None = None,  # ← AJOUT
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Cross-validation with GroupKFold (grouped by SimID).

    GroupKFold ensures entire SimIDs stay in a single fold, preventing
    temporal leakage (a model trained on cycle 15 cannot predict cycle 14
    of the same simulation in validation).

    Parameters
    ----------
    model : estimator
        Sklearn-compatible model (untrained). LightGBM recommended.
    X : pd.DataFrame
        Training features.
    y : pd.Series
        Training target (single stock).
    groups : pd.Series
        SimID values aligned with ``X`` (from ``splitter.build_cv_groups()``).
    cv : GroupKFold, optional
        Custom fold generator. Created from ``n_folds`` if ``None``.
    n_folds : int, default ``CV_N_FOLDS``
    verbose : bool, default True

    Returns
    -------
    dict with keys:
        ``fold_scores``    : list of dicts ``{fold, r2_train, r2_val, rmse_val, mae_val}``
        ``mean_r2_train``  : float
        ``mean_r2_val``    : float
        ``std_r2_val``     : float
        ``mean_rmse_val``  : float
        ``mean_mae_val``   : float

    Examples
    --------
    >>> groups = build_cv_groups(train_df)
    >>> cv_results = cross_validate(model, X_train, y_train, groups=groups)
    >>> print(f"R² val: {cv_results['mean_r2_val']:.3f} ± {cv_results['std_r2_val']:.3f}")
    """
    if cv is None:
        cv = GroupKFold(n_splits=n_folds)
    else:
        n_folds = cv.n_splits

    fold_scores: list[dict[str, Any]] = []
    cat_feat = getattr(model, "_categorical_feature", None)

    iterator = enumerate(cv.split(X, y, groups=groups))
    if verbose:
        iterator = tqdm(iterator, total=n_folds, desc="Cross-validation")

    for fold_idx, (train_idx, val_idx) in iterator:

        X_fold_train = X.iloc[train_idx]
        X_fold_val   = X.iloc[val_idx]

        # Inject stage-1 predictions if provided (sequential pipeline)
        if extra_train_features is not None:
            assert len(extra_train_features) == len(X), "extra_train_features is not aligned with X (lengths mismatch)."
            X_fold_train = pd.concat(
                [X_fold_train.reset_index(drop=True),
                extra_train_features.iloc[train_idx].reset_index(drop=True)],
                axis=1,
            )
            X_fold_val = pd.concat(
                [X_fold_val.reset_index(drop=True),
                extra_train_features.iloc[val_idx].reset_index(drop=True)],
                axis=1,
            )

        y_fold_train = y.iloc[train_idx]
        y_fold_val   = y.iloc[val_idx]

        # Clone preserves hyperparameters but not fitted state
        model_fold = clone(model)
        if cat_feat is not None:
            model_fold._categorical_feature = cat_feat

        fit_kwargs = {"categorical_feature": cat_feat} if cat_feat is not None else {}
        model_fold.fit(X_fold_train, y_fold_train, **fit_kwargs)

        y_train_pred = model_fold.predict(X_fold_train)
        y_val_pred   = model_fold.predict(X_fold_val)

        fold_scores.append({
            "fold":      fold_idx + 1,
            "r2_train":  float(r2_score(y_fold_train, y_train_pred)),
            "r2_val":    float(r2_score(y_fold_val,   y_val_pred)),
            "rmse_val":  float(np.sqrt(mean_squared_error(y_fold_val, y_val_pred))),
            "mae_val":   float(mean_absolute_error(y_fold_val, y_val_pred)),
        })

    df_scores = pd.DataFrame(fold_scores)
    results = {
        "fold_scores":    fold_scores,
        "mean_r2_train":  float(df_scores["r2_train"].mean()),
        "mean_r2_val":    float(df_scores["r2_val"].mean()),
        "std_r2_val":     float(df_scores["r2_val"].std()),
        "mean_rmse_val":  float(df_scores["rmse_val"].mean()),
        "mean_mae_val":   float(df_scores["mae_val"].mean()),
        "std_rmse_val":   float(df_scores["rmse_val"].std()),
        "std_mae_val":    float(df_scores["mae_val"].std()),
    }

    if verbose:
        log.info("\n%s", "=" * 60)
        log.info("CROSS-VALIDATION RESULTS (%d folds)", n_folds)
        log.info("%s", "=" * 60)
        log.info("Mean R² train : %.3f", results["mean_r2_train"])
        log.info("Mean R² val   : %.3f ± %.3f", results["mean_r2_val"], results["std_r2_val"])
        log.info("Mean RMSE val : %.3f", results["mean_rmse_val"])
        log.info("Mean MAE val  : %.3f", results["mean_mae_val"])
        log.info("%s\n", "=" * 60)

    return results


# ============================================================================
# FINAL TRAINING
# ============================================================================

def train_final_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame | None = None,
    y_test: pd.Series | None = None,
    verbose: bool = True,
) -> tuple[Any, dict[str, float]]:
    """
    Train the final model on the full training set and evaluate on test set.

    Parameters
    ----------
    model : estimator
        Untrained (or freshly built) model instance.
    X_train : pd.DataFrame
    y_train : pd.Series
    X_test : pd.DataFrame, optional
    y_test : pd.Series, optional
    verbose : bool, default True

    Returns
    -------
    model : fitted estimator
    metrics : dict
        Keys: ``train_r2``, ``train_rmse``, ``train_mae``
        and optionally ``test_r2``, ``test_rmse``, ``test_mae``.

    Examples
    --------
    >>> model, metrics = train_final_model(model, X_train, y_train, X_test, y_test)
    >>> print(f"Test R²: {metrics['test_r2']:.3f}")
    """
    if verbose:
        log.info("🚀 Training final model on %d examples...", len(X_train))

    cat_feat   = getattr(model, "_categorical_feature", None)
    fit_kwargs = {"categorical_feature": cat_feat} if cat_feat is not None else {}
    model.fit(X_train, y_train, **fit_kwargs)

    y_train_pred = model.predict(X_train)
    metrics: dict[str, float] = {
        "train_r2":   float(r2_score(y_train, y_train_pred)),
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, y_train_pred))),
        "train_mae":  float(mean_absolute_error(y_train, y_train_pred)),
    }

    if X_test is not None and y_test is not None:
        y_test_pred = model.predict(X_test)
        metrics.update({
            "test_r2":   float(r2_score(y_test, y_test_pred)),
            "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_test_pred))),
            "test_mae":  float(mean_absolute_error(y_test, y_test_pred)),
        })

    if verbose:
        log.info("\n%s", "=" * 60)
        log.info("FINAL MODEL METRICS")
        log.info("%s", "=" * 60)
        log.info("R² train  : %.3f", metrics["train_r2"])
        log.info("RMSE train: %.3f", metrics["train_rmse"])
        if "test_r2" in metrics:
            log.info("R² test   : %.3f", metrics["test_r2"])
            log.info("RMSE test : %.3f", metrics["test_rmse"])
            log.info("MAE test  : %.3f", metrics["test_mae"])
        log.info("%s\n", "=" * 60)

    return model, metrics

def train_classifier(
    model: lgb.LGBMClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame | None = None,
    y_test: pd.Series | None = None,
    verbose: bool = True,
) -> tuple[lgb.LGBMClassifier, dict[str, float]]:
    """
    Train a LightGBM classifier and compute metrics.

    Handles both binary and multiclass classification automatically.

    Parameters
    ----------
    model : lgb.LGBMClassifier
        Unfitted classifier instance.
    X_train, y_train : pd.DataFrame, pd.Series
        Training data.
    X_test, y_test : pd.DataFrame, pd.Series, optional
        Test data.
    verbose : bool, default True

    Returns
    -------
    model_fitted : lgb.LGBMClassifier
        Fitted classifier.
    metrics : dict
        Train and test metrics (accuracy, f1, roc_auc for binary;
        accuracy, f1_macro for multiclass).
    """
    if verbose:
        log.info("Training classifier: %s", model.__class__.__name__)
        log.info("  Train: %d samples | Test: %d samples",
                 len(y_train), len(y_test) if y_test is not None else 0)

    # Detect number of classes
    n_classes = y_train.nunique()
    is_binary = (n_classes == 2)

    if verbose:
        log.info("  Detected %d classes → %s classification",
                 n_classes, "binary" if is_binary else "multiclass")

    # Fit
    cat_feat = getattr(model, "_categorical_feature", None)
    fit_kwargs = {"categorical_feature": cat_feat} if cat_feat is not None else {}
    model.fit(X_train, y_train, **fit_kwargs)

    if verbose:
        log.info("  ✓ Training complete")

    # ── Metrics computation ─────────────────────────────────────────────────
    def _clf_metrics_binary(y_true, y_pred_proba, prefix):
        """Binary classification metrics."""
        y_pred = (y_pred_proba >= 0.5).astype(int)
        return {
            f"{prefix}accuracy": float(accuracy_score(y_true, y_pred)),
            f"{prefix}f1":       float(f1_score(y_true, y_pred, zero_division=0)),
            f"{prefix}roc_auc":  float(roc_auc_score(y_true, y_pred_proba)),
        }

    def _clf_metrics_multiclass(y_true, y_pred, prefix):
        """Multiclass classification metrics."""
        return {
            f"{prefix}accuracy": float(accuracy_score(y_true, y_pred)),
            f"{prefix}f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        }

    # ── Compute metrics ─────────────────────────────────────────────────────
    if is_binary:
        # Binary: use predict_proba[:, 1] for positive class
        metrics = _clf_metrics_binary(
            y_train, model.predict_proba(X_train)[:, 1], "train_"
        )
        if X_test is not None and y_test is not None:
            metrics.update(_clf_metrics_binary(
                y_test, model.predict_proba(X_test)[:, 1], "test_"
            ))
    else:
        # Multiclass: use predict() directly
        metrics = _clf_metrics_multiclass(
            y_train, model.predict(X_train), "train_"
        )
        if X_test is not None and y_test is not None:
            metrics.update(_clf_metrics_multiclass(
                y_test, model.predict(X_test), "test_"
            ))

    if verbose:
        log.info("  Metrics:")
        for k, v in metrics.items():
            log.info("    %-20s : %.4f", k, v)

    return model, metrics

# ============================================================================
# OPTUNA — BAYESIAN HYPERPARAMETER OPTIMISATION
# ============================================================================

def tune_optuna(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups: pd.Series,
    n_trials: int = OPTUNA_N_TRIALS,
    timeout: int | None = OPTUNA_TIMEOUT,
    n_jobs: int = OPTUNA_N_JOBS,
    cv_folds: int = CV_N_FOLDS,
    categorical_feature: list[str] | None = None,
    random_state: int = RANDOM_STATE,
    verbose: bool = True,
) -> tuple[dict[str, Any], "optuna.Study"]:
    """
    Bayesian hyperparameter optimisation with Optuna for LightGBM.

    Uses GroupKFold cross-validation (grouped by SimID) as the objective.
    Optimises mean validation R² across folds.

    Parameters
    ----------
    X_train : pd.DataFrame
    y_train : pd.Series
    groups : pd.Series
        SimID groups for GroupKFold.
    n_trials : int, default ``OPTUNA_N_TRIALS``
    timeout : int, optional
        Max optimisation time in seconds.
    n_jobs : int, default ``OPTUNA_N_JOBS``
        Parallel trials. Keep at 1 for LightGBM (callback issues with >1).
    cv_folds : int, default ``CV_N_FOLDS``
    categorical_feature : list of str, optional
    random_state : int, default ``RANDOM_STATE``
    verbose : bool, default True

    Returns
    -------
    best_params : dict
        Best hyperparameters found.
    study : optuna.Study
        Full Optuna study object (for further inspection).

    Examples
    --------
    >>> best_params, study = tune_optuna(X_train, y_train, groups)
    >>> model = build_lgb(params=best_params, categorical_feature=cat_feats)
    >>> model, metrics = train_final_model(model, X_train, y_train, X_test, y_test)
    """
    import lightgbm as lgb
    from config import OPTUNA_SEARCH_SPACE

    search_space = OPTUNA_SEARCH_SPACE
    cv = GroupKFold(n_splits=cv_folds)

    def _objective(trial: "optuna.Trial") -> float:
        params = {
            "n_estimators":      trial.suggest_int(
                "n_estimators", *search_space["n_estimators"]
            ),
            "learning_rate":     trial.suggest_float(
                "learning_rate", *search_space["learning_rate"], log=True
            ),
            "num_leaves":        trial.suggest_int(
                "num_leaves", *search_space["num_leaves"]
            ),
            "max_depth":         trial.suggest_int(
                "max_depth", *search_space["max_depth"]
            ),
            "min_child_samples": trial.suggest_int(
                "min_child_samples", *search_space["min_child_samples"]
            ),
            "subsample":         trial.suggest_float(
                "subsample", *search_space["subsample"]
            ),
            "colsample_bytree":  trial.suggest_float(
                "colsample_bytree", *search_space["colsample_bytree"]
            ),
            "reg_alpha":         trial.suggest_float(
                "reg_alpha", *search_space["reg_alpha"]
            ),
            "reg_lambda":        trial.suggest_float(
                "reg_lambda", *search_space["reg_lambda"]
            ),
            "random_state":      random_state,
            "n_jobs":            -1,
            "verbose":           -1,
        }

        model = lgb.LGBMRegressor(**params)
        if categorical_feature is not None:
            model._categorical_feature = categorical_feature

        r2_scores: list[float] = []
        for train_idx, val_idx in cv.split(X_train, y_train, groups=groups):
            X_tr = X_train.iloc[train_idx]
            y_tr = y_train.iloc[train_idx]
            X_vl = X_train.iloc[val_idx]
            y_vl = y_train.iloc[val_idx]

            fit_kw = {"categorical_feature": categorical_feature} if categorical_feature else {}
            try:
                model_fold = lgb.LGBMRegressor(**params)
                model_fold.fit(X_tr, y_tr, **fit_kw)
                r2_scores.append(float(r2_score(y_vl, model_fold.predict(X_vl))))
            except Exception:
                return -1.0   # Penalise failed trials

        return float(np.mean(r2_scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )

    if verbose:
        log.info(
            "🔍 Optuna — %d trials | timeout: %s s | cv: %d folds",
            n_trials, timeout, cv_folds,
        )

    study.optimize(
        _objective,
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=n_jobs,
        show_progress_bar=verbose,
    )

    best_params = study.best_params
    best_params.update({"random_state": random_state, 
                        "n_jobs": -1, "verbose": -1,
                        "importance_type": "gain", # For SHAP compatibility
                        })

    if verbose:
        log.info("✅ Optuna complete — best R² val: %.4f", study.best_value)
        log.info("   Best params: %s", best_params)

    return best_params, study
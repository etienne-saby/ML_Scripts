"""
MetAIsAFe — utils/io_utils.py
================================
Model persistence (joblib), results export (CSV), and logging helpers.

Public API
----------
    save_model(model, path, metadata)        -> Path
    load_model(path)                         -> estimator
    list_models(campaign)                    -> list[Path]
    save_metrics(metrics, path, ...)         -> Path
    save_predictions(...)                    -> Path
    save_cv_results(cv_results, path)        -> Path
    save_sobol_indices(indices, path)        -> Path
    save_feature_importances(...)            -> Path
    save_shap_values(...)                    -> Path
    create_run_summary(...)                  -> dict
    print_campaign_summary(campaign)         -> None
    setup_file_logging(reports_dir, name)    -> logging.FileHandler

Notes
-----
- save_cv_results() expects a dict with a "fold_scores" key, as returned
  by modeling.trainer.cross_validate().
- save_shap_values() requires Python >= 3.9 (uses Path.with_stem()).
- print_campaign_summary() uses log.info() intentionally (interactive helper).
  All other functions route through the module-level logger.

Author  : Étienne SABY
Updated : 2026-04
"""
from __future__ import annotations

import warnings

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from config import CampaignPaths
from _version import __version__

log = logging.getLogger(__name__)

# ============================================================================
# MODEL PERSISTENCE
# ============================================================================

def save_model(
    model: Any,
    save_path: Path,
    metadata: dict[str, Any] | None = None,
    compress: int = 3,
) -> Path:
    """
    Save a trained model to disk using joblib.

    Saves the model as a .joblib file. If metadata is provided, a companion
    .json file is written alongside it containing the metadata plus
    model type and save timestamp.

    Parameters
    ----------
    model : estimator
        Trained sklearn-compatible model (LightGBM, XGBoost, ...).
    save_path : Path
        Destination path (e.g. campaign.metamodels_dir / "lgbm_yield_AF.joblib").
    metadata : dict, optional
        Info to persist alongside the model.
        Example: {"target": "yield_AF", "r2_test": 0.87}.
    compress : int
        Joblib compression level (0-9). Default 3 (balanced).

    Returns
    -------
    Path
        Path to the saved .joblib file.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, save_path, compress=compress)
    log.info(f"💾 Model saved: {save_path}")

    if metadata is not None:
        meta_path = save_path.with_suffix(".json")
        payload = {
            "saved_at": datetime.now().isoformat(),
            "model_type": type(model).__name__,
            **metadata,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        log.info(f"💾 Metadata saved: {meta_path}")

    return save_path


def load_model(model_path: Path) -> Any:
    """
    Load a model from disk.

    If a companion .json metadata file exists alongside the .joblib file,
    its model_type and saved_at fields are printed to stdout.

    Parameters
    ----------
    model_path : Path
        Path to the .joblib file.

    Returns
    -------
    estimator
        Loaded model, ready for prediction.

    Raises
    ------
    FileNotFoundError
        If model_path does not exist.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = joblib.load(model_path)
    log.info(f"✅ Model loaded: {model_path}")

    meta_path = model_path.with_suffix(".json")
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        log.info(
            f"   └─ {meta.get('model_type', 'N/A')} | "
            f"saved {meta.get('saved_at', 'N/A')}"
        )

    return model


def list_models(campaign: CampaignPaths) -> list[Path]:
    """
    List all saved models in a campaign's metamodels directory.

    Parameters
    ----------
    campaign : CampaignPaths

    Returns
    -------
    list[Path]
        Sorted list of .joblib files found in campaign.metamodels_dir.
    """
    return sorted(campaign.metamodels_dir.glob("*.joblib"))


# ============================================================================
# RESULTS EXPORT
# ============================================================================

def save_metrics(
    metrics: dict[str, float],
    save_path: Path,
    append: bool = False,
    run_id: str | None = None,
) -> Path:
    """
    Save a metrics dictionary to CSV.

    Each call writes a single row containing a timestamp, an optional run_id,
    and all metrics keys. If append=True and the file already exists, the new
    row is concatenated to the existing file (full re-read + re-write).

    Parameters
    ----------
    metrics : dict[str, float]
    save_path : Path
    append : bool
        If True, appends to existing file (read-modify-write).
    run_id : str, optional
        Identifier added as first column (e.g. "lgbm_yield_AF").

    Returns
    -------
    Path
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    row: dict[str, Any] = {"timestamp": datetime.now().isoformat()}
    if run_id is not None:
        row["run_id"] = run_id
    row.update(metrics)

    df_new = pd.DataFrame([row])

    if append and save_path.exists():
        df_existing = pd.read_csv(save_path)
        df_new = pd.concat([df_existing, df_new], ignore_index=True)

    df_new.to_csv(save_path, index=False)
    log.info(f"💾 Metrics saved: {save_path}")
    return save_path


def save_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path,
    index: pd.Index | None = None,
    extra_columns: dict[str, np.ndarray] | None = None,
) -> Path:
    """
    Save predictions alongside true values to CSV.

    Output columns: y_true, y_pred, residual (= y_pred - y_true).
    Extra columns (e.g. SimID) are appended after residual.

    Parameters
    ----------
    y_true : array-like
    y_pred : array-like
    save_path : Path
    index : pd.Index, optional
        If provided, written as CSV index.
    extra_columns : dict, optional
        Additional columns, e.g. {"SimID": sim_ids}.

    Returns
    -------
    Path
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "y_true":   np.asarray(y_true).ravel(),
        "y_pred":   np.asarray(y_pred).ravel(),
        "residual": np.asarray(y_pred).ravel() - np.asarray(y_true).ravel(),
    })
    if extra_columns:
        for col, vals in extra_columns.items():
            df[col] = vals
    if index is not None:
        df.index = index

    df.to_csv(save_path, index=(index is not None))
    log.info(f"💾 Predictions saved: {save_path} ({len(df):,} rows)")
    return save_path


def save_cv_results(
    cv_results: dict[str, Any],
    save_path: Path,
) -> Path:
    """
    Save cross-validation fold-level results to CSV.

    Parameters
    ----------
    cv_results : dict
        Output of modeling.trainer.cross_validate(). Must contain a
        "fold_scores" key whose value is a list of per-fold metric dicts.
    save_path : Path

    Returns
    -------
    Path
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fold_scores = cv_results.get("fold_scores", [])
    if not fold_scores:
        warnings.warn("save_cv_results: 'fold_scores' is empty — nothing saved.")
        return save_path

    df = pd.DataFrame(fold_scores)
    df.to_csv(save_path, index=False)
    log.info(f"💾 CV results saved: {save_path}")
    return save_path


def save_sobol_indices(
    sobol_indices: dict[str, pd.DataFrame],
    save_path: Path,
) -> Path:
    """
    Save Sobol indices to CSV, one file per index type.

    For each key in sobol_indices (e.g. "S1", "ST"), writes a file named
    <save_path.stem>_<key>.csv in save_path.parent.

    Parameters
    ----------
    sobol_indices : dict[str, pd.DataFrame]
        Output of analysis.sensitivity.compute_sobol_indices().
        Keys are index types (e.g. "S1", "ST"); values are DataFrames.
    save_path : Path
        Base path — suffixes _S1, _ST, etc. are appended automatically.

    Returns
    -------
    Path
        Path to the ST file (primary index).
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for key, df in sobol_indices.items():
        out = save_path.parent / f"{save_path.stem}_{key}.csv"
        df.to_csv(out, index=True)
        log.info(f"💾 Sobol {key} saved: {out}")

    return save_path.parent / f"{save_path.stem}_ST.csv"


def save_feature_importances(
    importances: "pd.Series | pd.DataFrame",
    save_path: Path,
) -> Path:
    """
    Save feature importances to CSV.

    If a Series is passed, it is reset to a two-column DataFrame
    (feature, importance). DataFrames are written as-is.

    Parameters
    ----------
    importances : pd.Series or pd.DataFrame
    save_path : Path

    Returns
    -------
    Path
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(importances, pd.Series):
        df = importances.reset_index()
        df.columns = ["feature", "importance"]
    else:
        df = importances.copy()

    df.to_csv(save_path, index=False)
    log.info(f"💾 Feature importances saved: {save_path}")
    return save_path


def save_shap_values(
    shap_values: np.ndarray,
    feature_names: list[str],
    save_path: Path,
    mean_abs: bool = True,
) -> Path:
    """
    Save raw SHAP values to CSV, with optional mean(|SHAP|) summary.

    Requires Python >= 3.9 (uses Path.with_stem()).

    Parameters
    ----------
    shap_values : np.ndarray
        Raw SHAP values matrix (n_samples x n_features).
    feature_names : list of str
    save_path : Path
        Path for the raw SHAP CSV. The summary file is written alongside
        it with suffix "_summary" appended to the stem.
    mean_abs : bool, default True
        If True, also saves a mean(|SHAP|) summary CSV sorted descending.

    Returns
    -------
    Path
        Path to the raw SHAP values CSV.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    df_shap = pd.DataFrame(shap_values, columns=feature_names)
    df_shap.to_csv(save_path, index=False)
    log.info("💾 SHAP values saved: %s", save_path)

    if mean_abs:
        summary_path = save_path.with_stem(save_path.stem + "_summary")
        summary = pd.Series(
            np.abs(shap_values).mean(axis=0),
            index=feature_names,
            name="mean_abs_shap",
        ).sort_values(ascending=False)
        summary.to_csv(summary_path)
        log.info("💾 SHAP summary saved: %s", summary_path)

    return save_path


# ============================================================================
# RUN SUMMARY & REPORTING
# ============================================================================

def create_run_summary(
    target: str,
    mode: str,
    metrics_train: dict[str, float],
    metrics_test: dict[str, float],
    model_params: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build a structured run summary dictionary for logging or export.

    Train and test metrics are prefixed with "train_" and "test_"
    respectively. Extra fields are merged at the top level.

    Parameters
    ----------
    target : str
        Target stock name (e.g. "yield_AF").
    mode : str
        Pipeline mode: "sensitivity" or "metamodel".
    metrics_train : dict
    metrics_test : dict
    model_params : dict, optional
    extra : dict, optional
        Any additional fields to merge into the summary.

    Returns
    -------
    dict
    """
    summary: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "target": target,
        "mode": mode,
        **{f"train_{k}": v for k, v in metrics_train.items()},
        **{f"test_{k}": v for k, v in metrics_test.items()},
    }
    if model_params:
        summary["model_params"] = model_params
    if extra:
        summary.update(extra)
    return summary


def print_campaign_summary(campaign: CampaignPaths) -> None:
    """
    Print a summary of all models saved in the campaign directory.

    For each .joblib file found, reads the companion .json metadata file
    (if present) and prints target name and r2_test. Uses log.info() directly
    (intended as an interactive helper, not a logging call).

    Parameters
    ----------
    campaign : CampaignPaths
    """
    models = list_models(campaign)
    log.info(f"\n📦 Campaign: {campaign.campaign_name}")
    log.info(f"   Models dir : {campaign.metamodels_dir}")
    log.info(f"   Models found: {len(models)}")
    for p in models:
        meta_path = p.with_suffix(".json")
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            r2 = meta.get("r2_test", "N/A")
            label = meta.get("model_key", meta.get("target", p.stem))
            log.info(f"   └─ {label:30s} R²_test={r2}")
        else:
            log.info(f"   └─ {p.stem}")


def setup_file_logging(
    reports_dir: Path,
    campaign_name: str,
) -> logging.FileHandler:
    """
    Add a FileHandler to the root logger for the duration of a pipeline run.

    The log file is named pipeline_<campaign_name>_<timestamp>.log and
    written to reports_dir. The handler captures DEBUG level and above,
    catching output from all modules via the root logger.

    Parameters
    ----------
    reports_dir : Path
        Directory where the log file will be written (created if absent).
    campaign_name : str
        Used to name the log file.

    Returns
    -------
    logging.FileHandler
        The handler, so the caller can remove it after the run.
    """
    reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path  = reports_dir / f"pipeline_{campaign_name}_{timestamp}.log"

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    logging.getLogger().addHandler(fh)
    log.info("📝 Log file: %s", log_path)
    return fh

"""
MetAIsAFe — config.py
======================
Global parameters, hyperparameters, and path resolution for the MetAIsAFe
meta-modelling pipeline.

Contents
--------
- Path dataclasses : HiSafePaths, CampaignPaths
- Path resolution  : find_project_root(), get_hisafe_paths(), get_campaign_paths(),
                     cleanup_empty_campaign_dirs()
- Hyperparameters  : LGB_PARAMS, XGB_PARAMS, RF_PARAMS
- Optuna config    : OPTUNA_N_TRIALS, OPTUNA_TIMEOUT, OPTUNA_SEARCH_SPACE
- Sobol config     : SOBOL_BOUNDS_BATCH_1/2, SOBOL_FIXED_PARAMS, SOBOL_N_SAMPLES
- CART config      : CART_DEFAULT_DEPTH, CART_MIN_SAMPLES_*
- Classifier thresholds : TREE_FAIL_THRESHOLD, YIELD_FAIL_THRESHOLD, YIELD_FAIL_RATE
- Pipeline constants : SEQUENTIAL_TARGETS_STAGE1/2, LOG_TARGETS
- Visualisation    : FIGURE_DPI, FIGURE_FORMAT, COLOR_PALETTE

Model selection rationale
--------------------------
Following preliminary benchmarking across multiple stock targets
(yield_AF, yield_TA, carbonStem_AF, carbonStem_TF, biomass_AF, biomass_TA)
using GroupKFold cross-validation (k=5, grouped by SimID), LightGBM
consistently outperformed XGBoost and Random Forest on HiSAFe simulation
outputs. Mean validation R² scores showed LightGBM outperforming Random
Forest by 4-9 pp and XGBoost by 1-3 pp, with training time 3-5x faster than
XGBoost and up to 15x faster than Random Forest.

LightGBM's leaf-wise growth and histogram-based split finding are particularly
well-suited to tabular data with moderate feature count, mixed
numeric-categorical inputs, and high signal-to-noise ratio. LightGBM handles
pandas `category` dtype natively, eliminating explicit label encoding and
associated leakage risk.

XGBoost is maintained as an optional fallback in xgb_fallback.py.
Random Forest is retired from the main pipeline but RF_PARAMS is kept for
reference.

Path resolution
---------------
find_project_root() resolves the project root using the following priority:
  1. ON_CLUSTER=1  → CLUSTER_ROOT env variable (mandatory on HPC)
  2. Auto-detection by walking up the directory tree from config.py location,
     looking for marker files (.venv, .vscode, .git, pyproject.toml)
  3. Current working directory (last-resort fallback, emits UserWarning)

Author  : Étienne SABY
Updated : 2026-05
"""
from __future__ import annotations

import os
import warnings
import dataclasses
from dataclasses import dataclass
from pathlib import Path

from _version import __version__


# ============================================================================
# PATH DATACLASSES
# ============================================================================

@dataclass(frozen=True)
class HiSafePaths:
    """Resolved top-level project paths."""
    root_dir:        Path
    inputs_dir:      Path
    scripts_dir:     Path
    simulations_dir: Path
    models_dir:      Path
    templates_dir:   Path
    weather_dir:     Path
    base_pld:        Path
    base_sim:        Path
    base_export:     Path


@dataclass(frozen=True)
class CampaignPaths:
    """
    Resolved paths for a specific training campaign.

    Directory structure::

        models_dir / campaign_name /
            Data/
                Reports/
                Metrics/
                Predictions/
                CV/
                Feature_Importances/
                SHAP/
                Sobol/
            MetaModels/
            Plots/
                Diagnostics/
                Sobol/
                SHAP/
                Feature_Importances/
                Exclusion_Analysis/

    Notes
    -----
    cart_dir is defined as a field but not created by get_campaign_paths()
    (CART plots are written directly into diagnostics_dir).
    """
    campaign_name:   str
    sim_camp_dir:    Path
    campaign_dir:    Path
    data_dir:        Path
    metamodels_dir:  Path
    plots_dir:       Path

    # Raw meta-table
    raw_meta:        Path

    # Data subfolders
    reports_dir:     Path
    metrics_dir:     Path
    predictions_dir: Path
    cv_dir:          Path
    featimps_dir:    Path
    shap_data_dir:   Path
    sobol_data_dir:  Path

    # Plot subfolders
    diagnostics_dir: Path
    sobol_dir:       Path
    cart_dir:        Path
    shap_dir:        Path
    feature_imp_dir: Path
    exclusion_dir:   Path


# ============================================================================
# PATH RESOLUTION
# ============================================================================

def find_project_root(
    markers: tuple[str, ...] = (".venv", ".vscode", ".git", "pyproject.toml"),
    explicit_root: Path | None = None,
    start: Path | None = None,
) -> Path:
    """
    Walk up the directory tree from ``start`` until a directory containing
    at least one of ``markers`` is found.

    Priority order (first match wins):
      1. ``ON_CLUSTER=1``  → ``CLUSTER_ROOT`` env variable (mandatory on HPC)
      2. Auto-detection by walking up from ``start`` (default: location of config.py)
      3. Current working directory (last-resort fallback, emits UserWarning)

    Parameters
    ----------
    markers : tuple of str
        Files or directories whose presence signals the project root.
    explicit_root : Path, optional
        Hard-coded root override (useful for tests).
    start : Path, optional
        Directory to start walking up from.
        Defaults to the directory containing this file (config.py).

    Returns
    -------
    Path
        Resolved absolute project root.

    Raises
    ------
    EnvironmentError
        If ``ON_CLUSTER=1`` is set but ``CLUSTER_ROOT`` is missing.
    """
    # 1. Cluster override
    if os.getenv("ON_CLUSTER") == "1":
        cluster_root = os.getenv("CLUSTER_ROOT")
        if not cluster_root:
            raise EnvironmentError(
                "ON_CLUSTER=1 is set but CLUSTER_ROOT env variable is missing."
            )
        return Path(cluster_root).resolve()

    # 2. Explicit local override
    if explicit_root is not None:
        return Path(explicit_root).resolve()

    # 3. Auto-detection from config.py location
    if start is None:
        start = Path(__file__).resolve().parent

    current = start
    while True:
        if any((current / marker).exists() for marker in markers):
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent

    # 4. Last resort fallback
    fallback = Path(".").resolve()
    warnings.warn(
        f"Project root not found (markers={markers}). "
        f"Falling back to CWD: {fallback}. "
        "Set CLUSTER_ROOT (with ON_CLUSTER=1) or place a marker file to "
        "silence this warning.",
        UserWarning,
        stacklevel=2,
    )
    return fallback


def get_hisafe_paths(explicit_root: Path | None = None) -> HiSafePaths:
    """
    Resolve top-level HiSAFe project paths.

    Parameters
    ----------
    explicit_root : Path, optional
        Override for local development or tests.

    Returns
    -------
    HiSafePaths
    """
    root_dir = find_project_root(explicit_root=explicit_root)

    inputs      = root_dir / "00_Inputs"
    scripts     = root_dir / "01_Scripts"
    simulations = root_dir / "02_Simulations"
    models      = root_dir / "03_Models"
    templates   = inputs / "templates"
    weather     = inputs / "weather"

    return HiSafePaths(
        root_dir        = root_dir,
        inputs_dir      = inputs,
        scripts_dir     = scripts,
        simulations_dir = simulations,
        models_dir      = models,
        templates_dir   = templates,
        weather_dir     = weather,
        base_pld        = templates / "base_template/base_template_A2.pld",
        base_sim        = templates / "base_template/base_template_A2.sim",
        base_export     = templates / "base_template/export.out",
    )


def get_campaign_paths(
    campaign_name: str,
    root_dir: str | Path | None = None,
    create: bool = True,
) -> CampaignPaths:
    """
    Resolve all paths for a given campaign and optionally create directories.

    Directory structure::

        models_dir / campaign_name / Data/
                                     MetaModels/
                                     Plots/

    Parameters
    ----------
    campaign_name : str
        e.g. ``"sobol_training_1_n2048"``
    root_dir : str or Path, optional
        Project root override.
    create : bool, default True
        Create directories if they do not exist.

    Returns
    -------
    CampaignPaths

    Notes
    -----
    cart_dir is not created automatically. CART plots are written to
    diagnostics_dir directly.
    """
    base = get_hisafe_paths(root_dir)
    sim_dir        = base.simulations_dir / campaign_name
    campaign_dir   = base.models_dir / campaign_name
    data_dir       = campaign_dir / "Data"
    metamodels_dir = campaign_dir / "MetaModels"
    plots_dir      = campaign_dir / "Plots"

    cp = CampaignPaths(
        campaign_name  = campaign_name,
        sim_camp_dir   = sim_dir,
        campaign_dir   = campaign_dir,
        data_dir       = data_dir,
        metamodels_dir = metamodels_dir,
        plots_dir      = plots_dir,

        raw_meta        = data_dir / f"meta_table_{campaign_name}.parquet",

        # Data subfolders
        reports_dir     = data_dir / "Reports",
        metrics_dir     = data_dir / "Metrics",
        predictions_dir = data_dir / "Predictions",
        cv_dir          = data_dir / "CV",
        featimps_dir    = data_dir / "Feature_Importances",
        shap_data_dir   = data_dir / "SHAP",
        sobol_data_dir  = data_dir / "Sobol",

        # Plot subfolders
        diagnostics_dir = plots_dir / "Diagnostics",
        sobol_dir       = plots_dir / "Sobol",
        cart_dir        = plots_dir / "CART",
        shap_dir        = plots_dir / "SHAP",
        feature_imp_dir = plots_dir / "Feature_Importances",
        exclusion_dir   = plots_dir / "Exclusion_Analysis",
    )

    if create:
        for d in [
            cp.data_dir, cp.plots_dir, cp.metamodels_dir,
            cp.reports_dir, cp.metrics_dir,
            cp.predictions_dir, cp.cv_dir,
            cp.featimps_dir,
            cp.sobol_data_dir, cp.shap_data_dir,
            cp.diagnostics_dir, cp.sobol_dir,
            cp.shap_dir,
            cp.feature_imp_dir, cp.exclusion_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    return cp


def cleanup_empty_campaign_dirs(
    campaign: CampaignPaths,
    *,
    verbose: bool = True,
) -> list[Path]:
    """
    Delete every empty directory that belongs to a campaign.

    Iterates over all Path fields of :class:`CampaignPaths` and removes
    directories that exist and contain no files or subdirectories.

    Parameters
    ----------
    campaign : CampaignPaths
    verbose : bool, default True

    Returns
    -------
    list of Path
        Paths of directories that were removed.
    """
    removed: list[Path] = []
    for field in dataclasses.fields(campaign):
        d: Path = getattr(campaign, field.name)
        if not isinstance(d, Path):
            continue
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()
            removed.append(d)
            if verbose:
                print(f"  🗑  Removed empty dir: {d}")
    return removed


# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================

# --- Reproducibility & splits ---
RANDOM_STATE: int = 42
TEST_SIZE:    float = 0.20
CV_N_FOLDS:   int = 5

# --- LightGBM (primary model — production) ---
LGB_PARAMS: dict = {
    "n_estimators":      500,
    "learning_rate":     0.05,
    "max_depth":         -1,
    "num_leaves":        63,
    "min_child_samples": 20,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "reg_alpha":         0.1,
    "reg_lambda":        0.1,
    "random_state":      RANDOM_STATE,
    "n_jobs":            -1,
    "verbose":           -1,
    "importance_type":   "gain",    # required for SHAP compatibility
}

# --- XGBoost (fallback only — see xgb_fallback.py) ---
XGB_PARAMS: dict = {
    "n_estimators":    500,
    "learning_rate":   0.05,
    "max_depth":       6,
    "subsample":       0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":       0.1,
    "reg_lambda":      1.0,
    "random_state":    RANDOM_STATE,
    "n_jobs":          -1,
    "verbosity":       0,
    "tree_method":     "hist",
}

# --- Random Forest (retired from main pipeline — kept for reference) ---
RF_PARAMS: dict = {
    "n_estimators":     300,
    "max_depth":        None,
    "min_samples_leaf": 5,
    "max_features":     "sqrt",
    "random_state":     RANDOM_STATE,
    "n_jobs":           -1,
}

# ============================================================================
# OPTUNA HYPERPARAMETER SEARCH
# ============================================================================

OPTUNA_N_TRIALS: int = 80
OPTUNA_TIMEOUT:  int | None = 600   # seconds; None = no limit
OPTUNA_N_JOBS:   int = 1            # >1 can cause issues with LGB callbacks

# LightGBM search space for Optuna
OPTUNA_SEARCH_SPACE: dict = {
    "n_estimators":      (100, 1000),
    "learning_rate":     (0.01, 0.2),
    "num_leaves":        (15, 127),
    "max_depth":         (-1, 12),
    "min_child_samples": (10, 100),
    "subsample":         (0.6, 1.0),
    "colsample_bytree":  (0.6, 1.0),
    "reg_alpha":         (0.0, 1.0),
    "reg_lambda":        (0.0, 2.0),
}

# ============================================================================
# SOBOL SENSITIVITY ANALYSIS
# ============================================================================

SOBOL_N_SAMPLES: int = 1024   # Base N — total rows = N × (k_numeric + 2)

# Numeric parameter bounds [min, max] for Saltelli sampling.
# Categorical entries (lists of strings or booleans) are sampled uniformly
# outside the Saltelli matrix and must be handled separately in the
# sampling script — they are NOT passed to SALib directly.
SOBOL_BOUNDS_BATCH_1: dict = {
    "plotWidth":        [4.0,   24.0],
    "plotHeight":       [6.0,   30.0],
    "strip_width":      [1.0,    8.0],
    "northOrientation": [0.0,  180.0],
    "soilDepth":        [0.5,    2.5],
    "sand":             [10.0,  80.0],
    "clay":             [5.0,   50.0],
    "stone":            [0.0,   40.0],
    "w_amp":            [0.0,    3.0],
    "w_mean":           [-4.0,  -0.5],
    "w_peak_doy":       [1.0,  365.0],
    "latitude":         [43.0,  51.0],
    "longitude":        [-2.0,   8.0],
    # Categorical — sampled uniformly from modality list (not passed to SALib)
    "main_crop":        ["wheat", "maize", "rotation"],
    "period":           ["PRE", "FUT"],
    "w_type":           ["CONST", "VAR"],
    "Rotation":         [True, False],
    "waterTable":       [0, 1],
    "Harvest_Year_Absolute": [1.0, 40.0],
}

SOBOL_BOUNDS_BATCH_2: dict = {
    "plotWidth":             [4.0,   24.0],
    "plotHeight":            [6.0,   30.0],
    "soilDepth":             [0.5,    8.0],
    "sand":                  [10.0,  80.0],
    "clay":                  [5.0,   50.0],
    "stone":                 [0.0,   30.0],
    "waterTable":            [0,      1],
    "latitude":              [41.0,  51.0],
    "longitude":             [-5.0,   9.5],
    "Harvest_Year_Absolute": [1.0,   40.0],
    # Categorical — sampled uniformly from modality list (not passed to SALib)
    "main_crop":             ["wheat", "maize"],
    "w_type":                ["CONST", "VAR"],
}

# Parameters held constant in Sobol plan B2 (not varied)
SOBOL_FIXED_PARAMS: dict = {
    "northOrientation": 90,
    "strip_width":      3,
    "rotation":         False,
    "period":           "FUT",
    "w_amp":            4.0,
    "w_mean":          -7.0,
    "w_peak_doy":       60,
}

# ============================================================================
# CART SURROGATE
# ============================================================================

CART_DEFAULT_DEPTH:     int = 4     # Max tree depth — 4 → ~15 rules (human-readable)
CART_MIN_SAMPLES_SOBOL: int = 500   # Min leaf size on large Sobol plans
CART_MIN_SAMPLES_TEST:  int = 20    # Min leaf size on test set

# ============================================================================
# PIPELINE CONSTANTS — TARGETS & LOGGING
# ============================================================================

# Stocks for which log1p transformation may improve training (optional, not
# applied automatically — must be enabled explicitly in the training script).
LOG_TARGETS: list[str] = [
    "carbonStem_AF",
    "carbonStem_TF",
    "carbonBranches_AF",
    "carbonBranches_TF",
]

CARBON_HORIZONS: list[int] = [5, 10, 20, 30, 40]
MIN_ENRICH_HORIZON: int = 10
MIN_CARBON_HORIZON: float = 50.0      # kgC/tree — training threshold

SEQUENTIAL_TARGETS_STAGE1: list[str] = [
    f"carbonStem_AF_h{h}" for h in CARBON_HORIZONS
] + [
    f"carbonStem_TF_h{h}" for h in CARBON_HORIZONS
]

SEQUENTIAL_TARGETS_STAGE2: list[str] = ["yield_AF", "yield_TA"]

PARAMS_YIELD_AF_ROW: dict = {
    "n_estimators":      500,
    "learning_rate":     0.05,
    "num_leaves":        63,
    "min_child_samples": 15,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "reg_alpha":         0.3,
    "reg_lambda":        0.5,
    "random_state":      RANDOM_STATE,
    "n_jobs":            -1,
    "verbose":           -1,
    "importance_type":   "gain",
}

PARAMS_YIELD_TA_ROW: dict = {
    **PARAMS_YIELD_AF_ROW
}

# ============================================================================
# CLASSIFIER THRESHOLDS
# ============================================================================

# CLF1 — Tree failure
# A simulation is flagged tree_failed when carbonStem_AF at the last
# crop cycle is below this threshold.
TREE_FAIL_THRESHOLD: float  = 1.0     # kgC/tree — seuil failed/stunted
TREE_STUNT_THRESHOLD: float = 50.0    # kgC/tree — seuil stunted/ok
# CLF2 — Yield failure
# A simulation is flagged yield_failed when the fraction of crop cycles
# (rows in the meta-table for that SimID) with yield_AF below
# YIELD_FAIL_THRESHOLD exceeds YIELD_FAIL_RATE.
YIELD_FAIL_THRESHOLD: float = 0.5   # t/ha
YIELD_FAIL_RATE:      float = 0.5   # fraction of cycles below threshold

# ============================================================================
# VISUALISATION
# ============================================================================

FIGURE_DPI:    int = 150
FIGURE_FORMAT: str = "png"

COLOR_PALETTE: dict[str, str] = {
    "train": "#2E86AB",
    "test":  "#E63946",
    "val":   "#52B788",
}

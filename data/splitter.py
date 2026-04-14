"""
MetAIsAFe — Splitter v4.0
===========================
Train/test splits by SimID and GroupKFold cross-validation.

DESIGN PRINCIPLES
-----------------
1. NEVER split row-by-row.
   Each SimID is a time series of harvest cycles (cycle 1 → N).
   A row-level random split would let the model "see" cycle 15 in training
   and predict cycle 14 in test → massive temporal leakage.

2. ROTATION-AWARE stratification.
   A single SimID can contain MULTIPLE crops (e.g. cycle 1 = wheat, cycle 2 = maize).
   Functions that characterise a SimID by its crop content must aggregate all
   unique crops, not just the first row.

3. Split unit = SimID (whole simulation block, all cycles together).

Author  : Étienne SABY
Updated : 2026-04
"""
from __future__ import annotations

from _version import __version__
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from config import TEST_SIZE, RANDOM_STATE, CV_N_FOLDS

log = logging.getLogger(__name__)


# ============================================================================
# INTERNAL HELPERS
# ============================================================================

def _check_columns(data: pd.DataFrame, *cols: str) -> None:
    """
    Validate that all specified columns exist in data.

    Raises
    ------
    ValueError
        If any column in ``cols`` is not found in ``data.columns``.
    """
    missing = [c for c in cols if c not in data.columns]
    if missing:
        raise ValueError(
            f"Column(s) {missing} not found in DataFrame. "
            f"Available columns: {list(data.columns)}"
        )


# ============================================================================
# ROTATION HELPERS
# ============================================================================

def has_rotations(
    data: pd.DataFrame,
    simid_col: str = "SimID",
    crop_col: str = "Crop_Name",
) -> bool:
    """
    Detect whether any SimID contains more than one crop species (rotation).

    Parameters
    ----------
    data : pd.DataFrame
    simid_col : str, default ``"SimID"``
    crop_col : str, default ``"Crop_Name"``

    Returns
    -------
    bool
        ``True`` if at least one SimID has ≥ 2 distinct crops.
    """
    _check_columns(data, simid_col, crop_col)
    return bool((data.groupby(simid_col)[crop_col].nunique() > 1).any())


def get_rotation_signature(
    data: pd.DataFrame,
    simid_col: str = "SimID",
    crop_col: str = "Crop_Name",
) -> pd.Series:
    """
    Build a canonical rotation signature for every SimID.

    The signature is the alphabetically sorted, hyphen-joined list of all
    unique crops observed across all cycles of a SimID.

    Examples
    --------
    - SimID_001 : [wheat, maize, wheat]  →  ``"maize-wheat"``  (rotation)
    - SimID_002 : [wheat, wheat, wheat]  →  ``"wheat"``        (monoculture)

    Parameters
    ----------
    data : pd.DataFrame
    simid_col : str
    crop_col : str

    Returns
    -------
    pd.Series
        Index = SimID, values = rotation signature string.
    """
    _check_columns(data, simid_col, crop_col)

    def _signature(crops: pd.Series) -> str:
        return "-".join(sorted(crops.dropna().unique()))

    return data.groupby(simid_col)[crop_col].apply(_signature).rename("rotation_signature")


def summarise_rotations(
    data: pd.DataFrame,
    simid_col: str = "SimID",
    crop_col: str = "Crop_Name",
) -> pd.DataFrame:
    """
    Return a summary table of rotation types found in the dataset.

    Parameters
    ----------
    data : pd.DataFrame
    simid_col : str
    crop_col : str

    Returns
    -------
    pd.DataFrame
        Columns: ``rotation_signature``, ``n_sims``, ``pct_sims``.
    """
    sigs = get_rotation_signature(data, simid_col, crop_col)
    counts = sigs.value_counts().reset_index()
    counts.columns = ["rotation_signature", "n_sims"]
    counts["pct_sims"] = counts["n_sims"] / counts["n_sims"].sum() * 100
    return counts


# ============================================================================
# MAIN SPLIT — SimID integrity guaranteed
# ============================================================================

def split_by_simid(
    data: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    simid_col: str = "SimID",
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Split train/test while keeping every SimID entirely in one partition.

    ROTATION-SAFE: works correctly whether SimIDs are monocultures or
    multi-crop rotations because the split unit is always the SimID.

    Parameters
    ----------
    data : pd.DataFrame
    test_size : float, default ``TEST_SIZE``
    random_state : int, default ``RANDOM_STATE``
    simid_col : str, default ``"SimID"``
    verbose : bool, default True

    Returns
    -------
    train_data : pd.DataFrame
    test_data  : pd.DataFrame
    train_idx  : np.ndarray  — row indices (for sklearn compatibility)
    test_idx   : np.ndarray
    """
    _check_columns(data, simid_col)

    unique_sims   = data[simid_col].unique()
    n_sims        = len(unique_sims)
    rng           = np.random.RandomState(random_state)
    shuffled_sims = rng.permutation(unique_sims)
    n_test        = int(n_sims * test_size)
    test_sims     = set(shuffled_sims[:n_test])
    train_sims    = set(shuffled_sims[n_test:])

    test_mask  = data[simid_col].isin(test_sims)
    train_mask = ~test_mask

    train_data = data[train_mask].copy()
    test_data  = data[test_mask].copy()
    train_idx  = np.where(train_mask)[0]
    test_idx   = np.where(test_mask)[0]

    if verbose:
        _print_split_stats(
            label="TRAIN/TEST SPLIT BY SimID",
            data=data,
            train_data=train_data,
            test_data=test_data,
            train_sims=train_sims,
            test_sims=test_sims,
            n_sims=n_sims,
        )

    return train_data, test_data, train_idx, test_idx


# ============================================================================
# STRATIFIED SPLIT — balanced rotation types in train and test
# ============================================================================

def stratified_split_by_rotation(
    data: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    simid_col: str = "SimID",
    crop_col: str = "Crop_Name",
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Stratified train/test split that preserves rotation type proportions.

    Uses ``get_rotation_signature()`` to group SimIDs and ensures each
    rotation stratum (e.g. ``"wheat"``, ``"maize"``, ``"maize-wheat"``)
    appears in roughly the same proportions in train and test.

    Parameters
    ----------
    data : pd.DataFrame
    test_size : float, default ``TEST_SIZE``
    random_state : int, default ``RANDOM_STATE``
    simid_col : str, default ``"SimID"``
    crop_col : str, default ``"Crop_Name"``
    verbose : bool, default True

    Returns
    -------
    train_data : pd.DataFrame
    test_data  : pd.DataFrame
    train_idx  : np.ndarray
    test_idx   : np.ndarray
    """
    _check_columns(data, simid_col, crop_col)

    sigs = get_rotation_signature(data, simid_col, crop_col)

    if verbose:
        _print_rotation_summary(data, simid_col, crop_col)

    # Stratified sampling per rotation signature
    rng = np.random.RandomState(random_state)
    test_sims: set = set()

    for _sig, group_sims in sigs.groupby(sigs):
        sim_ids   = group_sims.index.tolist()
        n_test_g  = max(1, int(len(sim_ids) * test_size))
        chosen    = rng.choice(sim_ids, size=n_test_g, replace=False)
        test_sims.update(chosen)

    test_mask  = data[simid_col].isin(test_sims)
    train_mask = ~test_mask
    train_sims = set(data.loc[train_mask, simid_col].unique())

    train_data = data[train_mask].copy()
    test_data  = data[test_mask].copy()
    train_idx  = np.where(train_mask)[0]
    test_idx   = np.where(test_mask)[0]

    if verbose:
        _print_split_stats(
            label="STRATIFIED TRAIN/TEST SPLIT BY ROTATION",
            data=data,
            train_data=train_data,
            test_data=test_data,
            train_sims=train_sims,
            test_sims=test_sims,
            n_sims=sigs.shape[0],
        )

    return train_data, test_data, train_idx, test_idx


# ============================================================================
# GROUP K-FOLD (for cross-validation in trainer.py)
# ============================================================================

def build_cv_groups(
    train_data: pd.DataFrame,
    simid_col: str = "SimID",
) -> pd.Series:
    """
    Extract the SimID column as the GroupKFold group array.

    Parameters
    ----------
    train_data : pd.DataFrame
    simid_col : str, default ``"SimID"``

    Returns
    -------
    pd.Series
        SimID values aligned with ``train_data`` index.
    """
    _check_columns(train_data, simid_col)
    return train_data[simid_col].reset_index(drop=True)


def make_group_kfold(n_splits: int = CV_N_FOLDS) -> GroupKFold:
    """
    Construct a ``GroupKFold`` splitter.

    Parameters
    ----------
    n_splits : int, default ``CV_N_FOLDS``

    Returns
    -------
    GroupKFold
    """
    return GroupKFold(n_splits=n_splits)


# ============================================================================
# INTERNAL DISPLAY HELPERS
# ============================================================================

def _print_split_stats(
    label: str,
    data: pd.DataFrame,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    train_sims: set,
    test_sims: set,
    n_sims: int,
) -> None:
    """Print a standardised split summary."""
    log.info("\n%s", "=" * 60)
    log.info("%s", label)
    log.info("%s", "=" * 60)
    log.info(
        "Total     : %d sims  |  %d rows",
        n_sims, len(data),
    )
    log.info(
        "Train     : %d sims  |  %d rows  (%.1f%%)",
        len(train_sims), len(train_data),
        100.0 * len(train_sims) / max(n_sims, 1),
    )
    log.info(
        "Test      : %d sims  |  %d rows  (%.1f%%)",
        len(test_sims), len(test_data),
        100.0 * len(test_sims) / max(n_sims, 1),
    )
    log.info("%s\n", "=" * 60)


def _print_rotation_summary(
    data: pd.DataFrame,
    simid_col: str,
    crop_col: str,
) -> None:
    """Print rotation type counts before stratified split."""
    summary = summarise_rotations(data, simid_col, crop_col)
    log.info("Rotation types detected:\n%s\n", summary.to_string(index=False))

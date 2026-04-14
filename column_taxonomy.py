"""
MetAIsAFe — column_taxonomy.py
================================
Single source of truth for all column classifications in the meta-table.

Contents
--------
- FEATURES   : model inputs, grouped by type (temporal, climate, design,
               soil, geography, categorical)
- TARGETS    : ML targets (physical stocks only) and post-processing
               derived outputs (ratios, effective stocks, deltas)
- METADATA   : identifiers, flags, intermediate columns — never used as
               ML features or targets
- REGISTRY   : patterns used by data/ modules for cleaning logic
               (interpolation eligibility, winsorisation, NA diagnosis)
- POPULATION : cascade classifier routing labels

Design decision — stocks vs ratios
------------------------------------
The ML model predicts physical stocks only (yield_AF, carbonStem_AF, etc.).
Ratios (RR_*, LER_*) are computed analytically in post-processing via
compute_ratios_from_stocks() — never predicted directly.

Rationale:
  - Predicting ratios directly causes numerical instability in the juvenile
    phase (years 1-10), where tree stocks are near-zero (division by ~0).
  - Interpolating ratios to fill resulting NAs produces physically
    meaningless values and leads to overfitting (R² test < 0 observed).
  - Predicting stocks and deriving ratios analytically gives controlled
    error propagation: sigma(RR) ≈ 2 × sigma(stocks), always positive.

Author  : Étienne SABY
Updated : 2026-04
"""
from __future__ import annotations

from _version import __version__


# ==============================================================================
# FEATURES
# ==============================================================================

# ------------------------------------------------------------------------------
# Group 1 : TEMPORAL
# Harvest_Year_Absolute is the primary temporal feature.
# Rule : NEVER split rows within a SimID — split at SimID level only.
# ------------------------------------------------------------------------------
TEMPORAL_FEATURES = [
    "Harvest_Year_Absolute",   # Absolute crop year (1→40) — PRIMARY temporal feature
]

# ------------------------------------------------------------------------------
# Group 2 : CLIMATE
# HiSAFe outputs from weather generator — correlated with lat/lon.
# Fixed at median in Sobol plan B2 (not varied as Sobol parameters).
# ------------------------------------------------------------------------------
CLIMATE_FEATURES = [
    "GDD_cycle_AF",                # Growing Degree Days
    "ETP_cycle_AF",                # Potential Evapotranspiration
    "precipitation_AF",            # Cumulated precipitation
    "frost_events_cycle_AF",       # Number of frost events
    "globalRadiation_AF",          # Global solar radiation
    "maxTemperature_extreme_AF",   # Max extreme temperature
    "minTemperature_extreme_AF",   # Min extreme temperature
]

# ------------------------------------------------------------------------------
# Group 3 : PLOT DESIGN (Sobol parameters — numeric)
# Direct experimental plan parameters controlling agroforestry geometry.
# Varied in B1 Sobol analysis; partially fixed in B2.
# ------------------------------------------------------------------------------
DESIGN_FEATURES_B1 = [
    "plotWidth",        # Plot width (m) — inter-row distance
    "plotHeight",       # Plot height (m) — along-row distance
    "strip_width",      # Strip width (m) — tree alley width
    "northOrientation", # Plot orientation relative to North (degrees)
    "Rotation",         # Boolean : monoculture or rotation
]

DESIGN_FEATURES_B2 = [
    "plotWidth",
    "plotHeight",
    # "strip_width"      : fixed in B2 (=3)   — zero variance, excluded
    # "northOrientation" : fixed in B2 (=90)  — zero variance, excluded
    # "Rotation"         : fixed in B2 (=FALSE) — zero variance, excluded
]

# ------------------------------------------------------------------------------
# Group 4 : SOIL (Sobol parameters — numeric)
# Pedological parameters from experimental plan.
# sand+clay dominate LER sensitivity (ST ≈ 0.70 cumulated in B1).
# ------------------------------------------------------------------------------
SOIL_FEATURES_B1 = [
    "soilDepth",   # Total soil depth (m)
    "sand",        # Sand fraction (%)
    "clay",        # Clay fraction (%)
    "stone",       # Stone/gravel fraction (%)
    "waterTable",  # Water table presence (boolean)
    "w_peak_doy",  # Peak day-of-year of water table depth
    "w_amp",       # Water table amplitude (m)
    "w_mean",      # Water table mean depth (m, negative = below surface)
]

SOIL_FEATURES_B2 = [
    "soilDepth",
    "sand",
    "clay",
    "stone",
    "waterTable",
    # "w_peak_doy" : fixed in B2 (=60)   — zero variance, excluded
    # "w_amp"      : fixed in B2 (=4.0)  — zero variance, excluded
    # "w_mean"     : fixed in B2 (=-7.0) — zero variance, excluded
]

# ------------------------------------------------------------------------------
# Group 5 : GEOGRAPHY (Sobol parameters — numeric)
# Spatial coordinates driving climate patterns.
# ------------------------------------------------------------------------------
GEO_FEATURES = [
    "latitude",    # Site latitude (degrees)
    "longitude",   # Site longitude (degrees)
]

# ------------------------------------------------------------------------------
# Group 6 : CATEGORICAL FEATURES
# Encoded as dtype='category' for LightGBM (native support, no leakage).
# LabelEncoder (sklearn) encoding is used only in cart.py for CART fitting.
# ------------------------------------------------------------------------------
CATEGORICAL_FEATURES_B1 = [
    "main_crop",   # Dominant crop species ('wheat' / 'maize')
    "period",      # Climate period reference ('PRE' / 'FUT')
    "w_type",      # Water table type ('CONST' / 'VAR')
]

CATEGORICAL_FEATURES_B2 = [
    "main_crop",   # 2 classes in B2 (wheat/maize) — rotation excluded
    "w_type",      # 2 classes (CONST/VAR)
    # "period"     : fixed in B2 (='FUT') — zero variance, excluded
]

# ------------------------------------------------------------------------------
# ALL FEATURES — full sets used in model training
# Order: temporal → climate → design → soil → geo → categorical
#
# STRICT RULE : ALL_FEATURES must NEVER contain:
#   - yield_AF, yield_TA, carbonStem_AF/TF, biomass_* → ML targets
#   - _eff_AF, _eff_TA, _eff_TF                       → derived from targets
#   - RR_*, LER_*, _delta_*                            → post-processing outputs
#   - Stress variables (sticsWater*, sticsNitrogen*)   → simulator outputs, not inputs
# ------------------------------------------------------------------------------
ALL_FEATURES_B1 = (
    TEMPORAL_FEATURES +
    CLIMATE_FEATURES  +
    GEO_FEATURES      +
    DESIGN_FEATURES_B1 +
    SOIL_FEATURES_B1   +
    CATEGORICAL_FEATURES_B1
)

# Features used in B1 sensitivity analysis (true Sobol plan parameters only —
# no temporal, no climate outputs)
SOBOL_FEATURES = (
    DESIGN_FEATURES_B1 +
    SOIL_FEATURES_B1   +
    GEO_FEATURES       +
    CATEGORICAL_FEATURES_B1
)

# Features used in B2 meta-model training
ACTIVE_FEATURES_B2 = (
    TEMPORAL_FEATURES    +
    GEO_FEATURES         +
    DESIGN_FEATURES_B2   +
    SOIL_FEATURES_B2     +
    CATEGORICAL_FEATURES_B2
)


# ==============================================================================
# TARGETS — ML LEVEL 1 : PHYSICAL STOCKS (directly predicted by the model)
# ==============================================================================

# Crop stocks
# AF : agroforestry system  |  TA : sole-crop reference (Témoin Agricole)
STOCK_TARGETS_CROP = [
    "yield_AF",          # Grain yield in AF system [t/ha cell]
    "yield_TA",          # Grain yield in sole-crop reference [t/ha]
    "biomass_AF",        # Total aboveground biomass AF [t/ha cell]
    "biomass_TA",        # Total aboveground biomass TA [t/ha]
    "grainBiomass_AF",   # Grain biomass AF [t/ha cell]
    "grainBiomass_TA",   # Grain biomass TA [t/ha]
]

# Tree stocks
# AF : trees in agroforestry  |  TF : sole-forest reference (Témoin Forestier)
STOCK_TARGETS_TREE = [
    "carbonStem_AF",           # Stem carbon in AF [kgC/tree]
    "carbonStem_TF",           # Stem carbon in sole-forest reference [kgC/tree]
    "carbonBranches_AF",       # Branch carbon AF [kgC/tree]
    "carbonBranches_TF",       # Branch carbon TF [kgC/tree]
    "carbonCoarseRoots_AF",    # Coarse root carbon AF [kgC/tree]
    "carbonCoarseRoots_TF",    # Coarse root carbon TF [kgC/tree]
]

# Canonical full set of ML targets
ALL_TARGETS = list(dict.fromkeys(STOCK_TARGETS_CROP + STOCK_TARGETS_TREE))

# Minimal stock set used in sensitivity analysis (core indicators only)
STOCK_TARGETS_MINIMAL = [
    "yield_AF",
    "yield_TA",
    "carbonStem_AF",
    "carbonStem_TF",
]


# ==============================================================================
# POPULATION TAXONOMY — cascade classifier routing
# ==============================================================================

# Labels assigned by filter_population() in preparation.py
POPULATION_LABELS = [
    "yield_ok × tree_ok",       # Nominal population — main meta-model training set
    "yield_ok × tree_failed",   # Cultural-only model
    "yield_fail × tree_ok",     # Geographic rejection rule
    "yield_fail × tree_failed", # Full rejection (yield=0, carbon=0)
]

NOMINAL_POPULATION = "yield_ok × tree_ok"


# ==============================================================================
# TARGETS — ML LEVEL 2 : DERIVED (post-processing only — never predicted)
# ==============================================================================

DERIVED_TARGETS = [
    # Effective stocks (area-corrected, used to compute ratios)
    "yield_eff_AF",
    "yield_eff_TA",
    "biomass_eff_AF",
    "biomass_eff_TA",
    "carbonStem_eff_AF",
    "carbonStem_eff_TF",
    # Relative Ratios (RR) — computed analytically from _eff stocks
    "RR_crop_yield",
    "RR_crop_biomass",
    "RR_tree_carbonStem",
    # Land Equivalence Ratios (LER) — computed analytically
    "LER_yield_carbonStem",
    "LER_biomass_carbonStem",
    # Carbon deltas (diagnostic only)
    "carbonStem_delta_AF",
    "carbonStem_delta_TF",
]

CUMULATIVE_RR = [
    "RR_crop_yield",
    "RR_crop_biomass",
    "RR_tree_carbonStem",
]

CUMULATIVE_LER = [
    "LER_yield_carbonStem",
    "LER_biomass_carbonStem",
]


# ==============================================================================
# METADATA — identifiers, flags, intermediate columns
# NOT used as ML features or targets
# ==============================================================================

ID_COLUMNS = [
    "SimID",
    "Crop_Name",
    "rot_id",
    "sim_name",
]

DATE_COLUMNS = [
    "Harvest_Date_AF",
    "Sowing_Date_AF",
]

FLAG_COLUMNS = [
    "carbon_dead",
    "yield_failure",
    "high_na",
    "excluded",
]

REFERENCE_TA_OUTPUTS = [
    "yield_TA",
    "biomass_TA",
    "grainBiomass_TA",
]

REFERENCE_TF_OUTPUTS = [
    "carbonStem_TF",
    "carbonBranches_TF",
    "carbonCoarseRoots_TF",
]

DYNAMIC_AF_OUTPUTS = [
    "yield_AF",
    "biomass_AF",
    "grainBiomass_AF",
    "carbonStem_AF",
    "carbonBranches_AF",
    "carbonCoarseRoots_AF",
]

INTERMEDIATE_COMPUTED = [
    "cult_frac",
    "density_AF",
    "density_TF",
    "Harvest_Year_Absolute",
]


# ==============================================================================
# REGISTRY — cleaning patterns (used by data/ modules)
# ==============================================================================

# Stocks eligible for temporal interpolation (NaN-filling along time axis).
# Only physically monotone or slowly-varying stocks are eligible.
# Crop yields and ratios are NEVER interpolated (episodic, not continuous).
INTERPOLABLE_STOCKS = [
    "carbonStem_AF",
    "carbonStem_TF",
    "carbonBranches_AF",
    "carbonBranches_TF",
    "carbonCoarseRoots_AF",
    "carbonCoarseRoots_TF",
]

# Column name patterns that must NEVER be interpolated
NON_INTERPOLABLE_PATTERNS = [
    "yield_",
    "grainBiomass_",
    "biomass_",
    "_eff_",
    "_delta_",
    "RR_",
    "LER_",
]

# Stocks subject to winsorisation — physical stocks only (= all ML targets)
WINSORIZE_STOCKS = ALL_TARGETS

# Column name patterns that must NOT be winsorised
WINSORIZE_EXCLUDE_PATTERNS = [
    "RR_",
    "LER_",
    "_eff_",
    "_delta_",
    "latitude",
    "longitude",
    "sand",
    "clay",
    "stone",
    "soilDepth",
    "plotWidth",
    "plotHeight",
    "strip_width",
]

# Key columns used to assess NA rate per SimID (Step 2 diagnostic)
NA_KEY_COLUMNS = [
    "yield_AF",
    "yield_TA",
    "carbonStem_AF",
    "carbonStem_TF",
    "biomass_AF",
]

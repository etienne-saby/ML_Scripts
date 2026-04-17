# ==============================================================================
# MetAIsAFe — R/utils_formulas.R
# ==============================================================================
# Analytical formulas for post-processing agroforestry predictions.
#
# DESIGN PRINCIPLES
# -----------------
# - Physical stocks (yield_AF/TA, carbonStem_AF/TF) are predicted by the ML
#   models. Ratios (RYR_*, LER) are computed analytically from these stocks.
# - All functions are vectorized (operate on 40-year trajectories).
# - Division by zero → NA_real_ (coherent with R semantics).
# - Negative stock values → warning + coercion to 0 (physical constraint).
#
# UNITS
# -----
# - yield_AF/TA         : t/ha at cell level (AF) or total area (TA)
# - carbonStem_AF/TF    : kgC/tree (individual biomass)
# - density_AF/TF       : trees/ha
# - RYR_*, LER          : dimensionless ratios
#
# Author  : Étienne SABY + Claude
# Created : 2026-05
# ==============================================================================

# ==============================================================================
# CONSTANTS
# ==============================================================================

#' Fixed strip width in Sobol plan B2 (meters).
#' In B2, strip_width is not varied — always 3 m.
#' @export
STRIP_WIDTH_B2 <- 3.0

#' Fixed plot dimensions for TF (sole-forest reference) simulations (meters).
#' TF always uses plotWidth = plotHeight = 3 m → density_TF = 1111.11 trees/ha.
#' @export
TF_PLOT_WIDTH  <- 3.0
TF_PLOT_HEIGHT <- 3.0

#' Tree density in TF simulations (trees/ha).
#' Computed from 10000 / (TF_PLOT_WIDTH × TF_PLOT_HEIGHT).
#' @export
DENSITY_TF <- 10000 / (TF_PLOT_WIDTH * TF_PLOT_HEIGHT)


# ==============================================================================
# HELPER — AREA CORRECTION
# ==============================================================================

#' Compute cultivated fraction in an agroforestry plot.
#'
#' In agroforestry systems, the tree row strip is not cultivated, resulting in
#' a cultivated area penalty compared to monocultures.
#'
#' Diagram:
#' \preformatted{
#'   <---- plotWidth ---->     ^
#'   x x x x o o o x x x x     |
#'   x x x x o o o x x x x     |
#'   x x x x o T o x x x x plotHeight
#'   x x x x o o o x x x x     |
#'   x x x x o o o x x x x     v
#'          <----->
#'         stripWidth
#'
#'   Cultivated surface = (plotWidth - stripWidth) × plotHeight
#'   Total surface      = plotWidth × plotHeight
#'   Fraction           = (plotWidth - stripWidth) / plotWidth
#' }

# ==============================================================================
# EFFECTIVE STOCKS (area-corrected)
# ==============================================================================

#' Compute effective AF crop yield (area-corrected to total land basis).
#'
#' yield_AF is expressed per hectare of cultivated cell area. To convert to
#' per-hectare of TOTAL land (including the tree strip), multiply by the
#' cultivated fraction.
#'
#' @param yield_AF Numeric vector. AF crop yield (t/ha cell).
#' @param plotWidth Numeric. Plot width (m).
#' @param strip_width Numeric. Tree strip width (m). Default: \code{STRIP_WIDTH_B2}.
#' @return Numeric vector. Effective yield (t/ha total land).
#' @export
compute_yield_AF_eff <- function(yield_AF, plotWidth, strip_width = STRIP_WIDTH_B2) {
  # Input validation
  if (any(yield_AF < 0, na.rm = TRUE)) {
    warning("Negative yield_AF values detected — coercing to 0.")
    yield_AF[yield_AF < 0] <- 0.0
  }
  
  yield_AF * ((plotWidth - strip_width) / plotWidth)
}


#' Compute effective tree carbon stock (per-hectare basis).
#'
#' carbonStem_AF is expressed per individual tree (kgC/tree). To convert to
#' per-hectare basis, multiply by tree density.
#'
#' @param carbonStem Numeric vector. AF tree carbon stock (kgC/tree).
#' @param plotWidth Numeric. Plot width (m).
#' @param plotHeight Numeric. Plot height (m).
#' @return Numeric vector. Effective carbon stock (kgC/ha).
#' @export
compute_carbonStem_eff <- function(carbonStem, plotWidth, plotHeight) {
  if (any(carbonStem < 0, na.rm = TRUE)) {
    warning("Negative carbonStem values detected — coercing to 0.")
    carbonStem[carbonStem < 0] <- 0.0
  }
  
  carbonStem * (10000 / (plotWidth * plotHeight))
}

# ==============================================================================
# RELATIVE YIELD RATIOS (RYR)
# ==============================================================================

#' Compute Relative Yield Ratio for crop production (RYR_crop).
#'
#' Measures the ratio of effective AF crop yield (area-corrected) to TA
#' monoculture reference yield.
#'
#' Formula:
#' \deqn{
#'   RYR_{crop} = \frac{yield_{AF,eff}}{yield_{TA}}
#'              = \frac{yield_{AF} \times (plotWidth - stripWidth) / plotWidth}{yield_{TA}}
#' }
#'
#' @param yield_AF Numeric vector. AF crop yield (t/ha cell).
#' @param yield_TA Numeric vector. TA monoculture yield (t/ha total).
#' @param plotWidth Numeric. Plot width (m).
#' @param strip_width Numeric. Tree strip width (m). Default: \code{STRIP_WIDTH_B2}.
#' @return Numeric vector. RYR_crop (dimensionless). NA when yield_TA = 0.
#' @export
compute_RYR_crop <- function(yield_AF, yield_TA, plotWidth, 
                             strip_width = STRIP_WIDTH_B2) {
  yield_AF_eff <- compute_yield_AF_eff(yield_AF, plotWidth, strip_width)
  
  # Division by zero → NA
  ryr <- ifelse(yield_TA == 0, NA_real_, yield_AF_eff / yield_TA)
  
  ryr
}


#' Compute Relative Yield Ratio for tree carbon sequestration (RYR_tree).
#'
#' Measures the ratio of effective AF tree carbon stock (per-hectare) to TF
#' sole-forest reference stock (per-hectare).
#'
#' Formula:
#' \deqn{
#'   RYR_{tree} = \frac{carbonStem_{AF} \times density_{AF}}{carbonStem_{TF} \times density_{TF}}
#' }
#'
#' where:
#'   - \eqn{density_{AF} = 10000 / (plotWidth \times plotHeight)} (trees/ha)
#'   - \eqn{density_{TF} = 1111.11} trees/ha (fixed — TF always uses 3 m × 3 m)
#'
#' @param carbonStem_AF Numeric vector. AF tree carbon stock (kgC/tree).
#' @param carbonStem_TF Numeric vector. TF tree carbon stock (kgC/tree).
#' @param plotWidth Numeric. Plot width (m) — AF system.
#' @param plotHeight Numeric. Plot height (m) — AF system.
#' @return Numeric vector. RYR_tree (dimensionless). NA when carbonStem_TF = 0.
#' @export
compute_RYR_tree <- function(carbonStem_AF, carbonStem_TF, 
                             plotWidth, plotHeight) {
  cs_AF_eff <- compute_carbonStem_eff(carbonStem_AF, plotWidth, plotHeight)
  cs_TF_eff <- compute_carbonStem_eff(carbonStem_TF, TF_PLOT_WIDTH, TF_PLOT_HEIGHT)
  
  # Division by zero → NA
  ryr <- ifelse(cs_TF_eff == 0, NA_real_, cs_AF_eff / cs_TF_eff)
  
  ryr
}


# ==============================================================================
# LAND EQUIVALENT RATIO (LER)
# ==============================================================================

#' Compute Land Equivalent Ratio (LER).
#'
#' LER quantifies the total land-use efficiency of an agroforestry system by
#' summing the relative contributions of crop and tree production.
#'
#' Formula:
#' \deqn{
#'   LER = RYR_{crop} + RYR_{tree}
#' }
#'
#' Interpretation:
#'   - LER > 1 → agroforestry is more land-efficient than monocultures
#'   - LER = 1 → equivalent efficiency
#'   - LER < 1 → net land-use penalty
#'
#' @param yield_AF Numeric vector. AF crop yield (t/ha cell).
#' @param yield_TA Numeric vector. TA monoculture yield (t/ha total).
#' @param carbonStem_AF Numeric vector. AF tree carbon stock (kgC/tree).
#' @param carbonStem_TF Numeric vector. TF tree carbon stock (kgC/tree).
#' @param plotWidth Numeric. Plot width (m) — AF system.
#' @param plotHeight Numeric. Plot height (m) — AF system.
#' @param strip_width Numeric. Tree strip width (m). Default: \code{STRIP_WIDTH_B2}.
#' @return Numeric vector. LER (dimensionless). NA propagated from RYR components.
#' @export
compute_LER <- function(yield_AF, yield_TA, 
                        carbonStem_AF, carbonStem_TF,
                        plotWidth, plotHeight,
                        strip_width = STRIP_WIDTH_B2) {
  ryr_crop <- compute_RYR_crop(yield_AF, yield_TA, plotWidth, strip_width)
  ryr_tree <- compute_RYR_tree(carbonStem_AF, carbonStem_TF, plotWidth, plotHeight)
  
  ryr_crop + ryr_tree
}


#' Compute cumulative mean LER over N years.
#'
#' Averages LER values across the full trajectory (typically 40 years).
#'
#' @param ler Numeric vector. Annual LER values.
#' @return Numeric scalar. Mean LER (NA removed).
#' @export
compute_LER_cumul <- function(ler) {
  mean(ler, na.rm = TRUE)
}


# ==============================================================================
# BATCH PROCESSING (for full prediction DataFrame)
# ==============================================================================

#' Add all derived columns (RYR_*, LER) to a prediction DataFrame.
#'
#' Expects a long-format DataFrame output from \code{predictor$format_output()}.
#'
#' Required columns: year, target, value, plus scenario-level metadata
#' (plotWidth, plotHeight, strip_width if not default).
#'
#' @param df Data.frame. Long-format predictions (columns: year, target, value).
#' @param plotWidth Numeric. Scalar or vector aligned with \code{df}.
#' @param plotHeight Numeric. Scalar or vector aligned with \code{df}.
#' @param strip_width Numeric. Default: \code{STRIP_WIDTH_B2}.
#' @return Data.frame. Input \code{df} with added columns:
#'   - yield_AF_eff, carbonStem_AF_eff, carbonStem_TF_eff
#'   - RYR_crop, RYR_tree, LER
#' @export
add_derived_ratios <- function(df, plotWidth, plotHeight, 
                               strip_width = STRIP_WIDTH_B2) {
  # Pivot to wide format for vectorized computation
  df_wide <- tidyr::pivot_wider(
    df, 
    id_cols = c("year", dplyr::any_of(c("scenario_id", "population", "tree_failed", "yield_failed"))),
    names_from = "target",
    values_from = "value"
  )
  
  # Validate required columns
  required <- c("yield_AF", "yield_TA", "carbonStem_AF", "carbonStem_TF")
  missing <- setdiff(required, names(df_wide))
  if (length(missing) > 0) {
    stop("add_derived_ratios: missing required columns: ", paste(missing, collapse = ", "))
  }
  
  # Compute effective stocks
  df_wide$yield_AF_eff <- compute_yield_AF_eff(
    df_wide$yield_AF, plotWidth, strip_width
  )
  df_wide$carbonStem_AF_eff <- compute_carbonStem_eff(
    df_wide$carbonStem_AF, plotWidth, plotHeight
  )
  df_wide$carbonStem_TF_eff <- compute_carbonStem_eff(
    df_wide$carbonStem_TF, TF_PLOT_WIDTH, TF_PLOT_HEIGHT
  )
  
  # Compute RYR
  df_wide$RYR_crop <- compute_RYR_crop(
    df_wide$yield_AF, df_wide$yield_TA, plotWidth, strip_width
  )
  df_wide$RYR_tree <- compute_RYR_tree(
    df_wide$carbonStem_AF, df_wide$carbonStem_TF, plotWidth, plotHeight
  )
  
  # Compute LER
  df_wide$LER <- df_wide$RYR_crop + df_wide$RYR_tree
  
  # Pivot back to long format (preserve original structure)
  tidyr::pivot_longer(
    df_wide,
    cols = -c("year", dplyr::any_of(c("scenario_id", "population", "tree_failed", "yield_failed"))),
    names_to = "target",
    values_to = "value"
  )
}

# ==============================================================================
# PREDICTION POST-PROCESSING
# ==============================================================================
#' Compute analytical ratios from a prediction result dict.
#'
#' Indicators computed:
#'
#'   RR_crop_cumul       = sum(yield_AF_eff_t) / sum(yield_TA_t)   [t=1→T]
#'   RR_crop_mean_annual = mean(yield_AF_eff_t / yield_TA_t)        [yield_TA > YIELD_GUARD]
#'   RR_tree             = (cs_AF_T × d_AF) / (cs_TF_T × d_TF)    [stock at T]
#'   LER                 = RR_crop_cumul + RR_tree                  [hybrid]
#'
#' @param result  List. Output of PREDICTOR$predict_single_sim().
#' @param params  List. User params (must contain plotWidth, plotHeight).
#' @param strip_width  Numeric. Default: STRIP_WIDTH_B2 (3 m).
#' @param yield_guard  Numeric. Minimum yield_TA (t/ha) for annual ratio
#'   inclusion. Years below this are treated as fallow/failure and excluded
#'   from RR_crop_mean_annual. Default: 0.1 t/ha.
#' @return Named list of scalar indicators and full trajectory vectors.
#' @export
compute_ratios_from_predictions <- function(result, params,
                                            strip_width  = STRIP_WIDTH_B2,
                                            yield_guard  = 0.1) {
  preds <- result$predictions
  
  # ── Extract trajectories as R numeric vectors ──────────────────────────
  cs_af <- as.numeric(preds$carbonStem_AF)   # kgC/tree, length T
  cs_tf <- as.numeric(preds$carbonStem_TF)   # kgC/tree, length T
  y_af  <- as.numeric(preds$yield_AF)        # t/ha cell, length T
  y_ta  <- as.numeric(preds$yield_TA)        # t/ha, length T
  
  pw <- as.numeric(params$plotWidth)
  ph <- as.numeric(params$plotHeight)
  T  <- length(cs_af)
  
  # ── Area-corrected AF yield (per ha total land) ────────────────────────
  y_af_eff   <- y_af * ((pw - strip_width) / pw)             # t/ha total, length T
  
  # ── RR_crop_cumul : ratio of cumulated productions ─────────────────────
  # Robust to annual zeros (fallow, rotation gaps)
  sum_y_af_eff <- sum(y_af_eff, na.rm = TRUE)
  sum_y_ta     <- sum(y_ta,     na.rm = TRUE)
  
  RR_crop_cumul <- if (sum_y_ta > 0) sum_y_af_eff / sum_y_ta else NA_real_
  
  # ── RR_crop_mean_annual : mean of annual ratios (guarded) ─────────────
  # Exclude years where yield_TA ≈ 0 (fallow, failure)
  valid_years   <- which(y_ta > yield_guard)
  annual_ratios <- rep(NA_real_, T)
  
  if (length(valid_years) > 0) {
    annual_ratios[valid_years] <- y_af_eff[valid_years] / y_ta[valid_years]
  }
  
  RR_crop_mean_annual <- if (length(valid_years) > 0) {
    mean(annual_ratios[valid_years], na.rm = TRUE)
  } else NA_real_
  
  n_excluded <- T - length(valid_years)
  
  # ── RR_tree : instantaneous stock ratio at T ───────────────────────────
  ryr_tree_vec <- compute_RYR_tree(cs_af, cs_tf, pw, ph)
  RR_tree_T    <- ryr_tree_vec[T]
  
  # ── LER hybrid : cumulative crop + instantaneous tree ─────────────────
  LER_T <- if (!is.na(RR_crop_cumul) && !is.na(RR_tree_T)) {
    RR_crop_cumul + RR_tree_T
  } else NA_real_
  
  # ── Full trajectories (for plotting in mod_trajectories) ──────────────
  # RR_crop trajectory = cumulative ratio up to year t
  # Useful for "how does LER build up over time?"
  cum_y_af_eff  <- cumsum(y_af_eff)
  cum_y_ta      <- cumsum(y_ta)
  ryr_crop_cumul_trajectory <- ifelse(
    cum_y_ta > 0, cum_y_af_eff / cum_y_ta, NA_real_
  )
  
  ler_trajectory <- ifelse(
    !is.na(ryr_crop_cumul_trajectory) & !is.na(ryr_tree_vec),
    ryr_crop_cumul_trajectory + ryr_tree_vec,
    NA_real_
  )
  
  list(
    # ── Scalars displayed in UI ──────────────────────────────────────────
    RR_crop_cumul        = RR_crop_cumul,
    RR_crop_mean_annual  = RR_crop_mean_annual,
    RR_tree              = RR_tree_T,
    LER                  = LER_T,
    
    # ── Diagnostics ─────────────────────────────────────────────────────
    n_years_yield_excluded = n_excluded,
    cult_frac              = (pw - strip_width) / pw,
    
    # ── Stocks at T (for percentile positioning) ─────────────────────────
    yield_AF_t40       = y_af[T],
    yield_TA_t40       = y_ta[T],
    carbonStem_AF_t40  = cs_af[T],
    carbonStem_TF_t40  = cs_tf[T],
    
    # ── Full trajectories (for mod_trajectories plotting) ─────────────────
    annual_ratios_trajectory      = annual_ratios,
    ryr_crop_cumul_trajectory     = ryr_crop_cumul_trajectory,
    ryr_tree_trajectory           = ryr_tree_vec,
    ler_trajectory                = ler_trajectory
  )
}

#' Compute percentile position of user values within Sobol reference distribution.
#'
#' @param ratios Named list. Output of compute_ratios_from_predictions().
#' @param sobol_ref Data.frame. Long-format Sobol predictions
#'   (columns: year, target, value).
#' @return Named list of percentile values (0-100).
#' @export
compute_percentiles <- function(ratios, sobol_ref) {
  if (is.null(sobol_ref) || nrow(sobol_ref) == 0) return(NULL)
  
  # Focus on t=40 values from Sobol reference
  sobol_t40 <- sobol_ref %>%
    dplyr::filter(year == max(year))
  
  result <- list()
  
  # Map ratio names to Sobol target names
  mapping <- c(
    "yield_AF_t40"       = "yield_AF",
    "yield_TA_t40"       = "yield_TA",
    "carbonStem_AF_t40"  = "carbonStem_AF",
    "carbonStem_TF_t40"  = "carbonStem_TF"
  )
  
  for (ratio_name in names(mapping)) {
    sobol_target <- mapping[[ratio_name]]
    user_val     <- ratios[[ratio_name]]
    
    if (is.null(user_val) || is.na(user_val)) next
    
    ref_values <- sobol_t40 %>%
      dplyr::filter(target == sobol_target) %>%
      dplyr::pull(value)
    
    if (length(ref_values) == 0) next
    
    result[[ratio_name]] <- round(100 * mean(ref_values <= user_val), 1)
  }
  
  result
}
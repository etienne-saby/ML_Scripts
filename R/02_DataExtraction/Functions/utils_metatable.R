# ==============================================================================
# FILE    : utils_metatable.R
# PURPOSE : Assemble the final meta-table training table by joining AF / TA /
#           TF cycle aggregates and computing RYR and LER performance indicators.
# AUTHOR  : Étienne SABY
# DATE    : 2026
#' #
#' # ------------------------------------------------------------------------------
#' # INDICATOR DEFINITIONS
#' # ------------------------------------------------------------------------------
#' #
#' # All indicators are computed in two temporal variants:
#' #
#' #   _cyc  (per-cycle)  : performance of the agroforestry system relative to the
#' #                        monoculture control for the SAME crop cycle i only.
#' #                        Captures within-year interactions.
#' #
#' #   _cum  (cumulative) : performance accumulated from cycle 1 up to and
#' #                        including cycle i (within each simulation).  Less
#' #                        sensitive to inter-annual variability; captures the
#' #                        long-term trajectory of the system.
#' #
#' # ---- CROP RYR ----------------------------------------------------------------
#' #
#' # yield_AF is expressed in t/ha at cell level.  However, in an
#' # agroforestry system, the tree row strip is not cultivated, resulting in a 
#' # loss of cultivated surface compared to monocultures.
#' #
#' #   cultivated_fraction = (plotWidth - strip_width) / plotWidth
#' # 
#' #         <---- plotWidth ---->     ^
#' #         x x x x o o o x x x x     |  
#' #         x x x x o o o x x x x     |
#' #         x x x x o o o x x x x     |
#' #         x x x x o T o x x x x plotHeight
#' #         x x x x o o o x x x x     |
#' #         x x x x o o o x x x x     |
#' #         x x x x o o o x x x x     |
#' #                <----->      ^     v
#' #               stripWidth    |
#' #                             |
#' #                             Cultivated surface AF = (plotWidth - stripWidth) / plotHeight
#' # 
#' #                             Monoculture equivalent = plotWidth / plotHeight
#' #
#' #                             Corrective ratio = [(plotWidth - stripWidth) / plotHeight] / [plotWidth / plotHeight]
#' #                                              = (plotWidth - stripWidth) / plotWidth
#' #
#' #
#' # The effective AF production per hectare of TOTAL land is therefore:
#' #
#' #   yield_AF_effective = yield_AF × cultivated_fraction
#' #
#' # The TA monoculture reference (yield_TA) already represents 100 % cultivated
#' # area, so no correction is needed there.  The RYR captures both per-cell
#' # productivity and the area penalty:
#' #
#' #   RYR_crop_<p>_cyc_i =
#' #       (yield_AF_i × cultivated_fraction) / yield_TA_i
#' #
#' #   RYR_crop_<p>_cum_i =
#' #       (sum_{k=1}^{i} yield_AF_k × cultivated_fraction) / sum_{k=1}^{i} yield_TA_k
#' #
#' #   RYR < 1 : AF produces less per total ha than monoculture.
#' #   RYR > 1 : AF compensates via higher per-cell productivity.
#' #
#' # ---- TREE RYR ----------------------------------------------------------------
#' #
#' # Tree carbon stocks are expressed per individual tree.  To convert to a
#' # per-hectare basis, multiply by tree density (trees/ha):
#' #
#' #   RYR_tree_<p>_cyc_i =
#' #       (carbonStem_delta_AF_i × density_AF) /
#' #       (carbonStem_delta_TF_i × density_TF)
#' #
#' #   where carbonStem_delta is the INCREMENT over cycle i (last − first value
#' #   within the cycle window), produced by aggregate_cycles().
#' #
#' #   RYR_tree_<p>_cum_i =
#' #       (carbonStem_AF_i × density_AF) /
#' #       (carbonStem_TF_i × density_TF)
#' #
#' #   where carbonStem (no _delta suffix) is the cumulative stock at end of
#' #   cycle i, also produced by aggregate_cycles().
#' #
#' # ---- LER ---------------------------------------------------------------------
#' #
#' #   LER_<crop>_<tree>_cyc_i = RYR_crop_<crop>_cyc_i + RYR_tree_<tree>_cyc_i
#' #   LER_<crop>_<tree>_cum_i = RYR_crop_<crop>_cum_i + RYR_tree_<tree>_cum_i
#' #
#' #   LER > 1 : the agroforestry system is more land-efficient than monocultures.
#' #
#' 
#' 
#' # ==============================================================================
#' # 1. PER-CYCLE RYR HELPERS  (suffix _cyc)
#' # ==============================================================================
#' 
#' #' Compute the per-cycle crop Relative Yield Ratio (\code{RYR_crop_<p>_cyc}).
#' #'
#' #' Correction factor applied to the AF numerator:
#' #' \deqn{cultivated\_fraction = \frac{plotWidth - strip\_width}{plotWidth}}
#' #'
#' #' @param meta         data.frame.  Meta-table (one row per SimID × cycle).
#' #' @param crop_param   String.  Base variable name, e.g. \code{"yield"}.
#' #'   Looks for \code{<crop_param>_AF} and \code{<crop_param>_TA}.
#' #' @return \code{meta} with one additional column
#' #'   \code{RYR_crop_<crop_param>_cyc}.
#' compute_crop_RYR_cyc <- function(meta, crop_param) {
#'   
#'   param_AF <- paste0(crop_param, "_AF")
#'   param_TA <- paste0(crop_param, "_TA")
#'   RYR_name <- paste0("RYR_crop_", crop_param, "_cyc")
#'   
#'   required <- c(param_AF, param_TA, "strip_width", "plotWidth")
#'   if (!all(required %in% names(meta))) {
#'     message(sprintf(
#'       "  [compute_crop_RYR_cyc] Missing column(s): %s — skipped.",
#'       paste(setdiff(required, names(meta)), collapse = ", ")
#'     ))
#'     return(meta)
#'   }
#'   
#'   dplyr::mutate(
#'     meta,
#'     !!RYR_name := dplyr::if_else(
#'       !is.na(.data[[param_TA]])                       &
#'         .data[["plotWidth"]] >  .data[["strip_width"]],
#'       (.data[[param_AF]] *
#'          ((.data[["plotWidth"]] - .data[["strip_width"]]) /
#'             .data[["plotWidth"]])) /
#'         .data[[param_TA]],
#'       NA_real_
#'     )
#'   )
#' }
#' 
#' 
#' #' Compute the per-cycle tree Relative Yield Ratio (\code{RYR_tree_<p>_cyc}).
#' #'
#' #' Uses the \strong{delta} carbon variable (cycle increment produced by
#' #' \code{aggregate_cycles()}) to represent the tree's contribution during
#' #' cycle i, scaled to per-hectare via density.
#' #'
#' #' \deqn{RYR\_tree\_cyc = \frac{delta\_AF \times density\_AF}
#' #'                              {delta\_TF \times density\_TF}}
#' #'
#' #' @param meta       data.frame.
#' #' @param tree_param String.  Base variable name \strong{without} the
#' #'   \code{_delta} suffix, e.g. \code{"carbonStem"}.  Looks for
#' #'   \code{<tree_param>_delta_AF} and \code{<tree_param>_delta_TF}.
#' #' @return \code{meta} with one additional column
#' #'   \code{RYR_tree_<tree_param>_cyc}.
#' compute_tree_RYR_cyc <- function(meta, tree_param) {
#'   
#'   param_AF <- paste0(tree_param, "_delta_AF")
#'   param_TF <- paste0(tree_param, "_delta_TF")
#'   RYR_name <- paste0("RYR_tree_", tree_param, "_cyc")
#'   
#'   required <- c(param_AF, param_TF, "density_AF", "density_TF")
#'   if (!all(required %in% names(meta))) {
#'     message(sprintf(
#'       "  [compute_tree_RYR_cyc] Missing column(s): %s — skipped.",
#'       paste(setdiff(required, names(meta)), collapse = ", ")
#'     ))
#'     return(meta)
#'   }
#'   
#'   dplyr::mutate(
#'     meta,
#'     !!RYR_name := dplyr::if_else(
#'       !is.na(.data[[param_TF]])                        &
#'         !is.na(.data[["density_TF"]])                   &
#'         .data[["density_TF"]]  >  0,
#'       (.data[[param_AF]] * .data[["density_AF"]]) /
#'         (.data[[param_TF]] * .data[["density_TF"]]),
#'       NA_real_
#'     )
#'   )
#' }
#' 
#' 
#' # ==============================================================================
#' # 2. CUMULATIVE RYR HELPERS  (suffix _cum)
#' # ==============================================================================
#' 
#' #' Compute the cumulative crop Relative Yield Ratio (\code{RYR_crop_<p>_cum}).
#' #'
#' #' Accumulates AF and TA yields from cycle 1 to cycle i within each simulation,
#' #' applies the cultivated-area correction to the AF numerator, and forms the
#' #' ratio of cumulative sums.
#' #'
#' #' \deqn{RYR\_crop\_cum_i =
#' #'   \frac{\sum_{k=1}^{i} yield\_AF_k \times cultivated\_fraction}
#' #'        {\sum_{k=1}^{i} yield\_TA_k}}
#' #'
#' #' Although cumulative sums smooth out individual-cycle anomalies, a near-zero
#' #' \code{cum_TA} can still occur when the TA monoculture has consistently poor
#' #' yields over many cycles (e.g. repeated frost kill or drought).
#' #'
#' #' \strong{Prerequisite}: \code{meta} must be sorted by \code{SimID × Cycle_Nb}
#' #' before calling this function.  Guaranteed by \code{build_meta_table()}.
#' #'
#' #' @param meta         data.frame.  Must be sorted by \code{SimID × Cycle_Nb}.
#' #' @param crop_param   String.  Base variable name, e.g. \code{"yield"}.
#' #' @return \code{meta} with one additional column
#' #'   \code{RYR_crop_<crop_param>_cum}.
#' compute_crop_RYR_cum <- function(meta, crop_param) {
#'   
#'   param_AF <- paste0(crop_param, "_AF")
#'   param_TA <- paste0(crop_param, "_TA")
#'   RYR_name <- paste0("RYR_crop_", crop_param, "_cum")
#'   
#'   required <- c(param_AF, param_TA, "strip_width", "plotWidth", "SimID")
#'   if (!all(required %in% names(meta))) {
#'     message(sprintf(
#'       "  [compute_crop_RYR_cum] Missing column(s): %s — skipped.",
#'       paste(setdiff(required, names(meta)), collapse = ", ")
#'     ))
#'     return(meta)
#'   }
#'   
#'   meta |>
#'     dplyr::group_by(SimID) |>
#'     dplyr::mutate(
#'       .cum_AF      = cumsum(dplyr::if_else(is.na(.data[[param_AF]]), 0, .data[[param_AF]])),
#'       .cum_TA      = cumsum(dplyr::if_else(is.na(.data[[param_TA]]), 0, .data[[param_TA]])),
#'       .cycle_index = dplyr::row_number(),
#'       !!RYR_name := (.data[[".cum_AF"]] * ((.data[["plotWidth"]] - .data[["strip_width"]]) /
#'               .data[["plotWidth"]])) / .data[[".cum_TA"]]
#'     ) |>
#'     dplyr::select(-".cum_AF", -".cum_TA", -".cycle_index") |>
#'     dplyr::ungroup()
#' }
#' 
#' 
#' #' Compute the cumulative tree Relative Yield Ratio (\code{RYR_tree_<p>_cum}).
#' #'
#' #' Uses the cumulative carbon stock at the end of cycle i scaled to per-hectare.
#' #'
#' #' \deqn{RYR\_tree\_cum_i =
#' #'   \frac{carbonStem\_AF_i \times density\_AF}
#' #'        {carbonStem\_TF_i \times density\_TF}}
#' #'
#' #' @param meta       data.frame.
#' #' @param tree_param String.  Base variable name without any suffix,
#' #'   e.g. \code{"carbonStem"}.
#' #' @return \code{meta} with one additional column
#' #'   \code{RYR_tree_<tree_param>_cum}.
#' compute_tree_RYR_cum <- function(meta, tree_param) {
#'   
#'   param_AF <- paste0(tree_param, "_AF")
#'   param_TF <- paste0(tree_param, "_TF")
#'   RYR_name <- paste0("RYR_tree_", tree_param, "_cum")
#'   
#'   required <- c(param_AF, param_TF, "density_AF", "density_TF")
#'   if (!all(required %in% names(meta))) {
#'     message(sprintf(
#'       "  [compute_tree_RYR_cum] Missing column(s): %s — skipped.",
#'       paste(setdiff(required, names(meta)), collapse = ", ")
#'     ))
#'     return(meta)
#'   }
#'   
#'   dplyr::mutate(
#'     meta,
#'     !!RYR_name := dplyr::if_else(
#'       !is.na(.data[[param_TF]])              &
#'         !is.na(.data[["density_TF"]])         &
#'         .data[["density_TF"]]  >  0,
#'       (.data[[param_AF]] * .data[["density_AF"]]) /
#'         (.data[[param_TF]] * .data[["density_TF"]]),
#'       NA_real_
#'     )
#'   )
#' }
#' 
#' 
#' # ==============================================================================
#' # 3. LER HELPER  (variant-agnostic)
#' # ==============================================================================
#' 
#' #' Compute a total Land Equivalent Ratio as the sum of two partial RYR columns.
#' #'
#' #' Generic helper that works for both \code{_cyc} and \code{_cum} variants —
#' #' pass the exact column names already present in \code{meta}.
#' #'
#' #' \deqn{LER = RYR\_crop + RYR\_tree}
#' #'
#' #' The result is \code{NA} when either partial RYR is \code{NA}.  Because
#' #' \code{RYR_tree_cyc} is \code{NA} for early cycles (tree too young), the
#' #' corresponding \code{LER_cyc} rows are also \code{NA}, which is the correct
#' #' behaviour — a LER cannot be computed without a valid tree component.
#' #'
#' #' @param meta         data.frame.
#' #' @param RYR_crop_col String.  Full column name of the crop partial RYR,
#' #'   e.g. \code{"RYR_crop_yield_cyc"}.
#' #' @param RYR_tree_col String.  Full column name of the tree partial RYR,
#' #'   e.g. \code{"RYR_tree_carbonStem_cyc"}.
#' #' @param LER_name     String.  Name for the new LER column.
#' #' @return \code{meta} with one additional column named \code{LER_name}.
#' compute_LER <- function(meta, RYR_crop_col, RYR_tree_col, LER_name) {
#'   
#'   if (!all(c(RYR_crop_col, RYR_tree_col) %in% names(meta))) {
#'     message(sprintf(
#'       "  [compute_LER] Missing RYR column(s): %s — skipped.",
#'       paste(setdiff(c(RYR_crop_col, RYR_tree_col), names(meta)), collapse = ", ")
#'     ))
#'     return(meta)
#'   }
#'   
#'   dplyr::mutate(
#'     meta,
#'     !!LER_name := .data[[RYR_crop_col]] + .data[[RYR_tree_col]]
#'     )
#' }


# ==============================================================================
# 4. MAIN BUILDER
# ==============================================================================

#' Build the Meta-table Training Table.
#'
#' Joins AF / TA / TF cycle aggregates on
#' \code{SimID × Cycle_Nb × Crop_Name}, appends experimental plan features,
#' and computes the full set of RYR and LER indicators in both per-cycle
#' (\code{_cyc}) and cumulative (\code{_cum}) variants.
#'
#'
#' @param path_cycles_AF  String (no extension).  Path to the AF cycles file.
#' @param path_cycles_TA  String (no extension) or \code{NULL}.
#' @param path_cycles_TF  String (no extension) or \code{NULL}.
#' @param plan_df         data.frame.  AF experimental plan.
#' @param plan_df_TF      data.frame or \code{NULL}.  TF plan for density_TF.
#' @param out_dir         String.  Output directory.
#' @param campaign_tag    String.  Campaign identifier.
#' @param plan_features   Character vector or \code{NULL}.
#' @param crop_params     Character vector.  Crop variables for RYR / LER.
#'   Default: \code{c("yield", "biomass", "grainBiomass")}.
#' @param tree_params_cyc Character vector.  Tree carbon variables for _cyc RYR.
#'   Default: \code{c("carbonStem", "carbonBranches", "carbonCoarseRoots")}.
#' @param tree_params_cum Character vector.  Tree carbon variables for _cum RYR.
#'   Default: \code{c("carbonStem", "carbonBranches", "carbonCoarseRoots")}.
#' @param overwrite       Logical.  Default: \code{FALSE}.
#' @return Path (without extension) to the meta-table table file, invisibly.
#'   \code{NULL} if AF cycle data cannot be loaded.
#' @export
build_meta_table <- function(path_cycles_AF,
                             path_cycles_TA  = NULL,
                             path_cycles_TF  = NULL,
                             plan_df,
                             plan_df_TF      = NULL,
                             out_dir,
                             campaign_tag,
                             plan_features   = NULL,
                             # crop_params     = c("yield", "biomass", "grainBiomass"),
                             # tree_params_cyc = c("carbonStem", "carbonBranches",
                             #                     "carbonCoarseRoots"),
                             # tree_params_cum = c("carbonStem", "carbonBranches",
                             #                     "carbonCoarseRoots"),
                             overwrite       = FALSE) {
  
  path_no_ext <- fs::path(out_dir, paste0("meta_table_", campaign_tag))
  
  for (ext in c(".fst", ".parquet", ".csv")) {
    if (!overwrite && fs::is_file(fs::path(paste0(path_no_ext,ext)))) {
      message(sprintf("  [CACHE] meta_table_%s already exists.",
                      campaign_tag))
      return(invisible(path_no_ext))
      break
    }
  }
  
  message("  Building meta-table...")
  
  # --------------------------------------------------------------------------
  # 1. LOAD CYCLE FILES
  # --------------------------------------------------------------------------
  af <- as.data.frame(read_data(path_cycles_AF))
  ta <- if (!is.null(path_cycles_TA)) as.data.frame(read_data(path_cycles_TA)) else NULL
  tf <- if (!is.null(path_cycles_TF)) as.data.frame(read_data(path_cycles_TF)) else NULL
  
  if (is.null(af)) {
    warning("AF cycle data not found at: ", path_cycles_AF)
    return(invisible(NULL))
  }
  
  join_keys <- c("SimID", "Cycle_Nb", "Crop_Name")
  
  # --------------------------------------------------------------------------
  # 2. SUFFIX RENAMING AND JOINS
  # --------------------------------------------------------------------------
  .add_suffix <- function(df, suffix, exclude = join_keys) {
    to_rename <- setdiff(names(df), exclude)
    
    if (length(to_rename) == 0L) {
      return(df)
    }
    
    dplyr::rename_with(df, ~ paste0(.x, suffix), dplyr::all_of(to_rename))
  }
  
  meta <- .add_suffix(af, "_AF")
  
  # TA join — retain crop performance and water balance columns only
  if (!is.null(ta)) {
    ta_cols_keep <- intersect(
      c(join_keys,
        "yield", "grainBiomass", "biomass", "lai",
        "sticsWaterStomatalStress", "sticsNitrogenBiomassStress",
        "sticsWaterSenescenceStress", "sticsNitrogenLaiStress",
        "tempStressLue", "frostStressPlantDensity", "frostStressFoliage",
        "relativeTotalParIncident", "totalCarbonHumusStock",
        "waterUptakeByCrop", "waterDemand", "waterDemandReduced",
        "waterUptakeInSaturationByCrop",
        "soilEvaporation", "mulchEvaporation",
        "runOff", "surfaceRunOff", "drainageBottom", "drainageArtificial",
        "capillaryRise", "irrigation",
        "rainInterceptedByCrop", "rainTransmittedByCrop", "stemFlowByCrop",
        "waterAddedByWaterTable", "waterTakenByDesaturation",
        "nitrogenLeachingBottom",
        "waterStock"
      ),
      names(ta)
    )
    meta <- dplyr::left_join(
      meta,
      .add_suffix(ta[, ta_cols_keep], "_TA"),
      by = join_keys
    )
  }
  
  # TF join — retain tree carbon stocks, structural, and canopy columns only
  if (!is.null(tf)) {
    tf_cols_keep <- intersect(
      c(join_keys,
        grep("^carbon(Stem|Branches|CoarseRoots|FineRoots|Stump|Fruit)",
             names(tf), value = TRUE),
        grep("^tree_(dbh|height)|^totalLeafArea",
             names(tf), value = TRUE)
      ),
      names(tf)
    )
    meta <- dplyr::left_join(
      meta,
      .add_suffix(tf[, tf_cols_keep], "_TF"),
      by = join_keys
    )
  }
  
  # --------------------------------------------------------------------------
  # 3. JOIN PLAN FEATURES
  # --------------------------------------------------------------------------
  if (is.null(plan_features)) {
    plan_features <- intersect(
      c("sim_id",
        "plotWidth", "plotHeight", "strip_width", "density",
        "latitude", "longitude", "climateType",
        "soilDepth", "sand", "clay", "stone", "waterTable",
        "period", "w_type", "w_amp", "w_mean", "w_peak_doy",
        "treeSpecies", "northOrientation",
        "main_crop", "rot_id"
      ),
      names(plan_df)
    )
  }
  
  plan_join <- plan_df |>
    dplyr::select(dplyr::all_of(plan_features)) |>
    dplyr::rename(SimID = sim_id)
  
  meta <- dplyr::left_join(meta, plan_join, by = "SimID")
  
  # Attach density columns with explicit mode suffixes
  if (!is.null(plan_df_TF)) {
    meta <- dplyr::left_join(
      meta,
      plan_df_TF |>
        dplyr::select(sim_id, density) |>
        dplyr::rename(SimID = sim_id, density_TF = density),
      by = "SimID"
    )
    meta <- dplyr::rename(meta, density_AF = density)
  }

  # --------------------------------------------------------------------------
  # 4. SORT
  # --------------------------------------------------------------------------
  # Always sort by harvest year to ensure correct chronological order
  if ("Harvest_Year_AF" %in% names(meta)) {
    meta <- dplyr::arrange(meta, SimID, Harvest_Year_AF, Date_Sowing_AF)
  } else if ("Date_Sowing_AF" %in% names(meta)) {
    meta <- dplyr::arrange(meta, SimID, Date_Sowing_AF)
  }

  # # --------------------------------------------------------------------------
  # # 4b. ANNUAL CARBON DELTAS
  # #
  # # carbonStem_delta_AF/TF = stock at Harvest_Year N  −  stock at Harvest_Year N-1
  # # Computed per SimID, sorted by Harvest_Year_AF.
  # # First year of each simulation gets NA (no prior year available).
  # # --------------------------------------------------------------------------
  # message("  Computing annual carbon deltas...")
  # 
  # carbon_delta_bases <- c("carbonStem", "carbonBranches", "carbonCoarseRoots")
  # 
  # for (base in carbon_delta_bases) {
  #   
  #   af_col   <- paste0(base, "_AF")
  #   tf_col   <- paste0(base, "_TF")
  #   delta_af <- paste0(base, "_delta_AF")
  #   delta_tf <- paste0(base, "_delta_TF")
  #   
  #   # Build a one-row-per-SimID×year table for delta computation
  #   cols_needed <- intersect(c("SimID", "Harvest_Year_AF", af_col, tf_col), 
  #                            names(meta))
  #   
  #   delta_table <- meta |>
  #     dplyr::select(dplyr::all_of(cols_needed)) |>
  #     dplyr::distinct(SimID, Harvest_Year_AF, .keep_all = TRUE) |>
  #     dplyr::arrange(SimID, Harvest_Year_AF) |>
  #     dplyr::group_by(SimID) |>
  #     dplyr::mutate(
  #       !!delta_af := if (af_col %in% names(pick(everything()))) 
  #         .data[[af_col]] - dplyr::lag(.data[[af_col]]) 
  #       else NA_real_,
  #       !!delta_tf := if (tf_col %in% names(pick(everything()))) 
  #         .data[[tf_col]] - dplyr::lag(.data[[tf_col]]) 
  #       else NA_real_
  #     ) |>
  #     dplyr::ungroup() |>
  #     dplyr::select(SimID, Harvest_Year_AF, 
  #                   dplyr::any_of(c(delta_af, delta_tf)))
  #   
  #   # Join back onto the full meta table (broadcasts to all Crop_Name rows)
  #   meta <- dplyr::left_join(meta, delta_table, by = c("SimID", "Harvest_Year_AF"))
  # }
  # 
  # # --------------------------------------------------------------------------
  # # 5. PER-CYCLE RYR  (_cyc)
  # # --------------------------------------------------------------------------
  # message("  Computing per-cycle RYR (_cyc)...")
  # 
  # for (p in crop_params)
  #   meta <- compute_crop_RYR_cyc(meta, p)
  # 
  # for (p in tree_params_cyc)
  #   meta <- compute_tree_RYR_cyc(meta, p)
  # 
  # # --------------------------------------------------------------------------
  # # 6. CUMULATIVE RYR  (_cum)
  # # --------------------------------------------------------------------------
  # message("  Computing cumulative RYR (_cum)...")
  # 
  # for (p in crop_params)     meta <- compute_crop_RYR_cum(meta, p)
  # for (p in tree_params_cum) meta <- compute_tree_RYR_cum(meta, p)
  # 
  # # --------------------------------------------------------------------------
  # # 7. LER  (both variants)
  # # --------------------------------------------------------------------------
  # message("  Computing LER (cyc + cum)...")
  # 
  # tree_ler_cyc <- if (length(tree_params_cyc) > 0L) tree_params_cyc[[1L]] else NULL
  # tree_ler_cum <- if (length(tree_params_cum) > 0L) tree_params_cum[[1L]] else NULL
  # 
  # for (cp in crop_params) {
  #   if (!is.null(tree_ler_cyc))
  #     meta <- compute_LER(
  #       meta,
  #       RYR_crop_col = paste0("RYR_crop_", cp,           "_cyc"),
  #       RYR_tree_col = paste0("RYR_tree_", tree_ler_cyc,  "_cyc"),
  #       LER_name     = paste0("LER_", cp, "_", tree_ler_cyc, "_cyc")
  #     )
  #   if (!is.null(tree_ler_cum))
  #     meta <- compute_LER(
  #       meta,
  #       RYR_crop_col = paste0("RYR_crop_", cp,           "_cum"),
  #       RYR_tree_col = paste0("RYR_tree_", tree_ler_cum,  "_cum"),
  #       LER_name     = paste0("LER_", cp, "_", tree_ler_cum, "_cum")
  #     )
  # }
  # 
  # --------------------------------------------------------------------------
  # 8. SAVE
  # --------------------------------------------------------------------------
  save_data(meta, path_no_ext)
  message(sprintf("  Meta-table : %d rows x %d columns",
                  nrow(meta), ncol(meta)))
  invisible(path_no_ext)
}
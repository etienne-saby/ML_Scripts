# ==============================================================================
# FILE    : utils_cycle_agg.R
# PURPOSE : Aggregate raw daily Hi-sAFe output into per-crop-cycle summaries.
# AUTHOR  : Étienne SABY
# DATE    : 2026
#
# ------------------------------------------------------------------------------
# AGGREGATION STRATEGY
# ------------------------------------------------------------------------------
#
# aggregate_cycles() operates in five steps:
#
#   A. Phenological calendar preparation.
#      The AF pheno calendar is converted to a keyed data.table with
#      Date_Sowing / Date_Harvest interval boundaries.
#
#   B. SimID validity index.
#      For each file type, the set of SimIDs with an existing FST file is
#      pre-computed once.  Each worker checks membership in O(1) rather than
#      scanning a directory.
#
#   C. Per-simulation worker (.agg_one_sim).
#      Each worker:
#        1. Loads its OWN SimID data from individual FST files (~6 MB per type).
#           No shared in-memory table is inherited via fork.
#        2. Runs foverlaps() to assign daily rows to cycles in one vectorised pass.
#        3. Aggregates cell, tree, and climate data by declared rules.
#        4. Returns a small cycle-level data.table (~50 rows × 35 cols).
#
#   D. Parallel dispatch via mclapply.
#      Each of the 96 workers loads ~18 MB total (3 files × ~6 MB).
#      Peak RAM: 96 × 18 MB ≈ 1.7 GB  (vs ~220 GB with the previous design).
#
#   E. Binding and output file writing.
#
# ------------------------------------------------------------------------------
# CELL TWO-PASS AGGREGATION RATIONALE
# ------------------------------------------------------------------------------
# Pass 1 (intra-cell): apply the declared rule per idCell × Cycle_Nb.
# Pass 2 (inter-cell): mean across all cells — scene-level representative value.
# This is correct for all rule types: the per-cell value is computed faithfully
# first, then averaged spatially.
# ==============================================================================


# ==============================================================================
# 0. INTERNAL HELPERS
# ==============================================================================

#' Apply an aggregation rule to a numeric vector.
#'
#' Used as a standalone helper in unit tests.  The main aggregation pipeline
#' uses \code{lapply(.SD, ...)} inside data.table grouping, which is faster.
#'
#' @param values Numeric vector.  NA values are silently excluded.
#' @param rule   String.  One of "max", "min", "sum", "mean", or "last".
#' @return Single numeric value, or NA_real_ if all values are NA or the
#'   rule is unrecognised.
#' @keywords internal
agg_vector <- function(values, rule) {
  v <- values[!is.na(values)]
  if (length(v) == 0L) return(NA_real_)
  switch(rule,
         max  = max(v),
         min  = min(v),
         sum  = sum(v),
         mean = mean(v),
         last = v[length(v)],
         NA_real_
  )
}


#' Load a per-SimID FST file from a raw data directory.
#'
#' Returns NULL (silently) if the file does not exist or is empty.
#' This is the per-worker loading primitive used by .agg_one_sim().
#'
#' @param sim_dir      String. Directory containing per-SimID FST files.
#' @param sid          String. Simulation ID.
#' @param convert_date Logical. Convert "Date" column to Date class. Default TRUE.
#' @return data.table or NULL.
#' @keywords internal
.load_sim_fst <- function(sim_dir, sid, convert_date = TRUE) {
  if (is.null(sim_dir)) return(NULL)
  fpath <- fs::path(sim_dir, paste0(sid, ".fst"))
  if (!file.exists(fpath)) return(NULL)
  dt <- fst::read_fst(fpath, as.data.table = TRUE)
  if (nrow(dt) == 0L) return(NULL)
  if (convert_date && "Date" %in% names(dt))
    dt[, Date := as.Date(Date)]
  dt
}


# ==============================================================================
# 1. CYCLE AGGREGATION
# ==============================================================================

#' Aggregate daily data from per-SimID FST files into per-crop-cycle summaries.
#'
#' Reads raw data from the per-SimID FST directory layout produced by
#' \code{extract_all_raw()}, joins each daily row to its phenological cycle
#' using \code{foverlaps()}, and computes cycle-level summaries according to
#' the rules declared in \code{source_vars}.
#'
#' Unlike the previous implementation, this function does NOT load raw data
#' globally before dispatching workers.  Each mclapply worker loads only its
#' own SimID's FST files on demand, reducing peak RAM from ~220 GB to ~1.7 GB.
#'
#' @param raw_paths       Named list of DIRECTORY paths to raw per-SimID FST
#'   files.  Keys: "cell", "trees", "climate".  Direct return value of
#'   \code{extract_all_raw()}.
#' @param pheno_df        data.frame.  Phenological calendar from
#'   \code{build_pheno_calendars()}.
#' @param plan_df         data.frame.  Experimental plan; must include sim_id
#'   and density columns.
#' @param out_dir         String.  Directory where the cycles file is written.
#' @param campaign_tag    String.  Campaign identifier.
#' @param mode_name       String.  Extraction mode: "AF", "TA", or "TF".
#' @param source_vars     Named list of named character vectors (variable ->
#'   aggregation rule).
#' @param t_base          Numeric.  Base temperature for GDD. Default: 6.
#' @param n_cores         Integer.  Parallel workers. Default: 1L.
#' @param overwrite       Logical.  Re-aggregate if output exists? Default FALSE.
#' @param get_job_details Logical.  Verbose per-simulation messages. Default FALSE.
#' @param progress_every  Integer.  Progress print interval. Default 10L.
#' @return Path (without extension) to written cycles file, invisibly.
#'   NULL if calendar is empty or no cycles were aggregated.
aggregate_cycles <- function(raw_paths,
                             pheno_df,
                             plan_df,
                             out_dir,
                             campaign_tag,
                             mode_name,
                             source_vars     = default_source_vars(),
                             crop_tbase      = default_crop_tbase(),
                             n_cores         = 1L,
                             overwrite       = FALSE,
                             get_job_details = FALSE,
                             progress_every  = 10L) {
  
  path_no_ext <- fs::path(out_dir, paste0("cycles_", campaign_tag, "_", mode_name))
  
  for (ext in c(".fst", ".parquet", ".csv")) {
    if (!overwrite && fs::is_file(fs::path(paste0(path_no_ext,ext)))) {
      message(sprintf("  [CACHE] cycles_%s_%s already exists.",
                      campaign_tag, mode_name))
      return(invisible(path_no_ext))
      break
    }
  }
  
  message(sprintf(
    "\n[aggregate_cycles] Campaign: %s | Mode: %s | %d core(s) | per-SimID load",
    campaign_tag, mode_name, n_cores
  ))
  
  # ==========================================================================
  # STEP A — Phenological calendar preparation
  # ==========================================================================
  pheno <- data.table::as.data.table(pheno_df)
  
  if ("SimID" %in% names(pheno) && !"sim_id" %in% names(pheno))
    data.table::setnames(pheno, "SimID", "sim_id")
  
  for (col in c("Date_Sowing", "Date_Harvest"))
    if (col %in% names(pheno)) pheno[[col]] <- as.Date(pheno[[col]])
  
  pheno <- pheno[!is.na(Date_Harvest)]
  
  if (nrow(pheno) == 0L) {
    warning("Phenological calendar is empty for mode: ", mode_name)
    return(invisible(NULL))
  }
  
  pheno[, Harvest_Year        := as.integer(format(Date_Harvest, "%Y"))]
  pheno[, Cycle_Duration_days := as.integer(Date_Harvest - Date_Sowing) + 1L]
  pheno[, Date_Sowing         := as.Date(Date_Sowing)]
  pheno[, Date_Harvest        := as.Date(Date_Harvest)]
  
  pheno[, t_base := crop_tbase[Crop_Name]]
  pheno[is.na(t_base), t_base := crop_tbase["default"]]
  
  # ==========================================================================
  # STEP B — Pre-compute valid SimID sets (O(1) lookup in workers)
  #
  # File existence is checked once in the parent process.  Each worker then
  # does a simple %in% test — no directory scan inside the hot loop.
  # ==========================================================================
  .valid_sim_ids <- function(sim_dir, sim_ids) {
    if (is.null(sim_dir)) return(character(0L))
    Filter(function(sid) {
      file.exists(fs::path(sim_dir, paste0(sid, ".fst")))
    }, sim_ids)
  }
  
  all_sim_ids     <- unique(pheno$sim_id)
  sim_dir_cell    <- raw_paths[["cell"]]
  sim_dir_tree    <- raw_paths[["trees"]]
  sim_dir_clim    <- raw_paths[["climate"]]
  
  sim_ids_in_cell <- .valid_sim_ids(sim_dir_cell, all_sim_ids)
  sim_ids_in_tree <- .valid_sim_ids(sim_dir_tree, all_sim_ids)
  sim_ids_in_clim <- .valid_sim_ids(sim_dir_clim, all_sim_ids)
  
  message(sprintf(
    "  SimIDs with FST data: cell=%d, trees=%d, climate=%d",
    length(sim_ids_in_cell), length(sim_ids_in_tree), length(sim_ids_in_clim)
  ))
  
  # Local null-coalescing operator — avoids implicit dependency on rlang
  `%||%` <- function(a, b) if (is.null(a)) b else a
  
  # Aggregation rule vectors per source
  cell_agg_rules <- source_vars[["cell"]]    %||% character(0L)
  tree_agg_rules <- source_vars[["trees"]]   %||% character(0L)
  clim_agg_rules <- source_vars[["climate"]] %||% character(0L)
  
  # Meta-columns excluded from numeric aggregation
  cell_meta_cols <- c(
    "SimID", "Date", "Date_end", "zoneName", "cropSpeciesName",
    "cropAge",
    "idCell", "Cycle_Nb", "Crop_Name", "Date_Sowing",
    "Date_Harvest", "Harvest_Year", "Cycle_Duration_days",
    "istart", "iend", "sim_id"
  )
  
  # Tree variable groupings derived from source_vars (kept in sync automatically)
  carbon_pool_vars <- names(tree_agg_rules)[
    tree_agg_rules == "last" & grepl("^carbon", names(tree_agg_rules))
  ]
  leaf_area_vars  <- intersect("totalLeafArea", names(tree_agg_rules))
  tree_other_vars <- setdiff(names(tree_agg_rules),
                             c(carbon_pool_vars, leaf_area_vars, "age"))
  
  # ==========================================================================
  # STEP C — Per-simulation aggregation worker
  #
  # Key property: no raw data table exists in the parent process at this point.
  # Each forked worker therefore inherits ~0 MB of raw data and loads only the
  # ~18 MB it needs for its own SimID.
  # ==========================================================================
  
  .agg_one_sim <- function(sid) {
    
    pheno_sid <- pheno[sim_id == sid]
    if (nrow(pheno_sid) == 0L) return(NULL)
    
    # Cycle interval table for foverlaps
    intervals <- data.table::copy(pheno_sid)
    intervals[, istart := Date_Sowing]
    intervals[, iend   := Date_Harvest]
    data.table::setkey(intervals, istart, iend)
    
    # foverlaps join: assigns every daily row to its cycle in one vectorised pass.
    # copy() prevents in-place modification of the loaded data.table
    # (Date_end is added by reference and must not persist across calls).
    .join_cycles <- function(dt_sid) {
      if (is.null(dt_sid) || nrow(dt_sid) == 0L) return(NULL)
      dt_sid <- data.table::copy(dt_sid)
      dt_sid[, Date_end := Date]
      data.table::setkey(dt_sid, Date, Date_end)
      data.table::setkey(intervals, istart, iend)
      ov <- data.table::foverlaps(
        dt_sid, intervals,
        by.x    = c("Date", "Date_end"),
        by.y    = c("istart", "iend"),
        type    = "within",
        nomatch = NULL
      )
      ov[, c("istart", "iend", "Date_end") := NULL]
      ov
    }
    
    # Base metadata: one row per cycle
    base <- pheno_sid[, .(
      SimID               = sid,
      Mode                = mode_name,
      Zone                = "main",
      Cycle_Nb,
      Crop_Name,
      Date_Sowing,
      Date_Harvest,
      Harvest_Year,
      Cycle_Duration_days
    )]
    
    # ------ Cell aggregation -------------------------------------------------
    # Load this SimID's cell FST file on demand (~6 MB).
    # No cell_raw table exists in the parent — nothing is fork-copied.
    cell_agg <- NULL
    if (sid %in% sim_ids_in_cell) {
      cell_sid <- .load_sim_fst(sim_dir_cell, sid)
      if (!is.null(cell_sid))
        cell_sid <- cell_sid[zoneName == "main"]
      ov <- .join_cycles(cell_sid)
      
      if (!is.null(ov) && nrow(ov) > 0L) {
        data.table::setorder(ov, Cycle_Nb, idCell, Date)
        
        num_vars       <- setdiff(names(ov)[vapply(ov, is.numeric, logical(1L))],
                                  cell_meta_cols)
        vars_with_rule <- intersect(names(cell_agg_rules), num_vars)
        vars_by_rule   <- split(vars_with_rule, cell_agg_rules[vars_with_rule])
        
        # Pass 1: intra-cell aggregation by declared rule
        cell_parts <- lapply(names(vars_by_rule), function(rule) {
          vs <- vars_by_rule[[rule]]
          if (length(vs) == 0L) return(NULL)
          switch(rule,
                 max  = ov[, lapply(.SD, max,  na.rm = TRUE), by = .(idCell, Cycle_Nb, Crop_Name), .SDcols = vs],
                 min  = ov[, lapply(.SD, min,  na.rm = TRUE), by = .(idCell, Cycle_Nb, Crop_Name), .SDcols = vs],
                 sum  = ov[, lapply(.SD, sum,  na.rm = TRUE), by = .(idCell, Cycle_Nb, Crop_Name), .SDcols = vs],
                 mean = ov[, lapply(.SD, mean, na.rm = TRUE), by = .(idCell, Cycle_Nb, Crop_Name), .SDcols = vs],
                 last = ov[, lapply(.SD, function(x) x[.N]),  by = .(idCell, Cycle_Nb, Crop_Name), .SDcols = vs],
                 NULL
          )
        })
        cell_parts <- Filter(function(x) !is.null(x) && nrow(x) > 0L, cell_parts)
        
        if (length(cell_parts) > 0L) {
          per_cell <- Reduce(
            function(a, b) merge(a, b, by = c("idCell", "Cycle_Nb", "Crop_Name"), all = TRUE),
            cell_parts
          )
          # Pass 2: scene-level mean across all cells
          scene_vars <- setdiff(names(per_cell), c("idCell", "Cycle_Nb", "Crop_Name"))
          cell_agg   <- per_cell[,
                                 lapply(.SD, mean, na.rm = TRUE),
                                 by = .(Cycle_Nb, Crop_Name),
                                 .SDcols = scene_vars
          ]
        }
      }
    }
    
    # ------ Tree aggregation -------------------------------------------------
    # Load this SimID's tree FST file on demand (~0.5 MB).
    tree_agg <- NULL
    if (sid %in% sim_ids_in_tree) {
      tree_sid <- .load_sim_fst(sim_dir_tree, sid)
      ov       <- .join_cycles(tree_sid)
      
      if (!is.null(ov) && nrow(ov) > 0L) {
        data.table::setorder(ov, Cycle_Nb, Date)
        
        # Tree age at sowing date (3-day tolerance for output gaps)
        age_at_sow <- ov[Date <= Date_Sowing + lubridate::days(3L),
                         .(Tree_Age = mean(age, na.rm = TRUE)),
                         by = Cycle_Nb]
        
        # Carbon pool end-of-cycle stocks ("last") and peak leaf area ("max")
        stock_present <- intersect(carbon_pool_vars, names(ov))
        leaf_present  <- intersect(leaf_area_vars,   names(ov))
        per_tree_agg  <- NULL
        if (length(stock_present) > 0L || length(leaf_present) > 0L) {
          parts <- list()
          if (length(stock_present) > 0L)
            parts[[1L]] <- ov[, lapply(.SD, function(x) x[.N]),
                              by = Cycle_Nb, .SDcols = stock_present]
          if (length(leaf_present) > 0L)
            parts[[2L]] <- ov[, lapply(.SD, function(x) max(x, na.rm = TRUE)),
                              by = Cycle_Nb, .SDcols = leaf_present]
          parts <- Filter(Negate(is.null), parts)
          per_tree_agg <- if (length(parts) > 1L)
            merge(parts[[1L]], parts[[2L]], by = "Cycle_Nb", all = TRUE) else parts[[1L]]
        }
        
        # # Carbon pool deltas: increment over the cycle (end value − start value)
        # delta_present <- intersect(carbon_pool_vars, names(ov))
        # delta_agg     <- NULL
        # if (length(delta_present) > 0L) {
        #   data.table::setorder(ov, Cycle_Nb, idTree, Date)
        #   delta_agg <- ov[,
        #                   lapply(.SD, function(x) x[.N] - x[1L]),
        #                   by = .(Cycle_Nb, idTree),
        #                   .SDcols = delta_present
        #   ][,
        #     lapply(.SD, mean, na.rm = TRUE),
        #     by = Cycle_Nb,
        #     .SDcols = delta_present
        #   ]
        #   data.table::setnames(delta_agg,
        #                        delta_present,
        #                        paste0(delta_present, "_delta"))
        # }
        
        # Remaining tree variables aggregated by declared rule
        other_present <- intersect(tree_other_vars, names(ov))
        other_agg     <- NULL
        if (length(other_present) > 0L) {
          other_rules  <- tree_agg_rules[other_present]
          vars_by_rule <- split(other_present, other_rules)
          parts <- lapply(names(vars_by_rule), function(rule) {
            vs <- vars_by_rule[[rule]]
            if (length(vs) == 0L) return(NULL)
            p <- switch(rule,
                        last = ov[, lapply(.SD, function(x) x[.N]), by = Cycle_Nb, .SDcols = vs],
                        mean = ov[, lapply(.SD, mean, na.rm = TRUE), by = Cycle_Nb, .SDcols = vs],
                        max  = ov[, lapply(.SD, max,  na.rm = TRUE), by = Cycle_Nb, .SDcols = vs],
                        sum  = ov[, lapply(.SD, sum,  na.rm = TRUE), by = Cycle_Nb, .SDcols = vs],
                        NULL
            )
            if (!is.null(p)) data.table::setnames(p, vs, paste0("tree_", vs))
            p
          })
          parts     <- Filter(Negate(is.null), parts)
          other_agg <- if (length(parts) > 1L)
            Reduce(function(a, b) merge(a, b, by = "Cycle_Nb", all = TRUE), parts) else parts[[1L]]
        }
        
        tree_agg <- Reduce(
          function(a, b) merge(a, b, by = "Cycle_Nb", all = TRUE),
          Filter(function(x) !is.null(x) && nrow(x) > 0L,
                 list(age_at_sow, per_tree_agg, other_agg)) #delta_agg, 
        )
      }
    }
    
    # ------ Climate aggregation ----------------------------------------------
    # Load this SimID's climate FST file on demand (~0.5 MB).
    clim_agg <- NULL
    if (sid %in% sim_ids_in_clim) {
      clim_sid <- .load_sim_fst(sim_dir_clim, sid)
      ov       <- .join_cycles(clim_sid)
      
      if (!is.null(ov) && nrow(ov) > 0L) {
        has_etp <- "etpPenman" %in% names(ov)
        ov[, tmean := (maxTemperature + minTemperature) / 2]
        
        full_agg <- ov[, .(
          GDD_cycle          = sum(pmax(tmean - t_base[1L], 0), na.rm = TRUE),
          airVpd_mean_cycle  = if ("airVpd" %in% names(ov)) mean(airVpd, na.rm = TRUE) else NA_real_,
          frost_events_cycle = sum(minTemperature < 0, na.rm = TRUE),
          ETP_cycle          = if (has_etp) sum(etpPenman, na.rm = TRUE) else NA_real_,
          maxTemperature_extreme = max(maxTemperature, na.rm = TRUE),
          minTemperature_extreme = min(minTemperature, na.rm = TRUE)
        ), by = Cycle_Nb]
        
        # Rule-driven aggregation for remaining climate variables
        clim_num_vars  <- names(ov)[vapply(ov, is.numeric, logical(1L))]
        already_done   <- c("tmean", "maxTemperature", "minTemperature",
                            "airVpd", "etpPenman",
                            "maxTemperature_extreme", "minTemperature_extreme")
        clim_rule_vars <- intersect(
          setdiff(names(clim_agg_rules), already_done), clim_num_vars
        )
        rule_agg <- NULL
        if (length(clim_rule_vars) > 0L) {
          vars_by_rule <- split(clim_rule_vars, clim_agg_rules[clim_rule_vars])
          parts <- lapply(names(vars_by_rule), function(rule) {
            vs <- vars_by_rule[[rule]]
            if (length(vs) == 0L) return(NULL)
            switch(rule,
                   sum  = ov[, lapply(.SD, sum,  na.rm = TRUE), by = Cycle_Nb, .SDcols = vs],
                   mean = ov[, lapply(.SD, mean, na.rm = TRUE), by = Cycle_Nb, .SDcols = vs],
                   max  = ov[, lapply(.SD, max,  na.rm = TRUE), by = Cycle_Nb, .SDcols = vs],
                   min  = ov[, lapply(.SD, min,  na.rm = TRUE), by = Cycle_Nb, .SDcols = vs],
                   last = ov[, lapply(.SD, function(x) x[.N]), by = Cycle_Nb, .SDcols = vs],
                   NULL
            )
          })
          parts    <- Filter(function(x) !is.null(x) && nrow(x) > 0L, parts)
          rule_agg <- if (length(parts) > 0L)
            Reduce(function(a, b) merge(a, b, by = "Cycle_Nb", all = TRUE), parts)
        }
        
        clim_agg <- if (!is.null(rule_agg))
          merge(full_agg, rule_agg, by = "Cycle_Nb", all = TRUE) else full_agg
      }
    }
    
    # ------ Merge all sources by Cycle_Nb (+ Crop_Name where available) ------
    # base always has Crop_Name; cell_agg has it too (two-pass grouping).
    # tree_agg and clim_agg are keyed on Cycle_Nb only — they are broadcast
    # across all Crop_Name values for that cycle via all.x = TRUE.
    .merge_smart <- function(a, b) {
      key <- if ("Crop_Name" %in% names(b)) c("Cycle_Nb", "Crop_Name") else "Cycle_Nb"
      merge(a, b, by = key, all.x = TRUE)
    }
    
    Reduce(
      .merge_smart,
      Filter(function(x) !is.null(x) && nrow(x) > 0L,
             list(base, cell_agg, tree_agg, clim_agg))
    )
  } # end .agg_one_sim
  
  
  # ==========================================================================
  # STEP D — Parallel dispatch over all simulations
  #
  # data.table threads are set to 1 before forking to prevent OpenMP
  # over-subscription (each forked worker would otherwise spawn its own
  # threads, leading to n_cores × dt_threads active threads simultaneously).
  # ==========================================================================
  sim_ids <- unique(pheno$sim_id)
  n_sims  <- length(sim_ids)
  
  message(sprintf("  %d complete cycles across %d simulations | %d core(s)...",
                  nrow(pheno), n_sims, n_cores))
  t_loop <- proc.time()["elapsed"]
  
  dt_threads_outer <- data.table::getDTthreads()
  if (n_cores > 1L) data.table::setDTthreads(1L)
  
  list_results <- parallel::mclapply(
    seq_along(sim_ids),
    function(sim_idx) {
      sid <- sim_ids[[sim_idx]]
      
      # Progress reporting (runs inside the worker — output is asynchronous)
      if (get_job_details) {
        message(sprintf("  [%d/%d] %s", sim_idx, n_sims, sid))
      } else if (progress_every > 0L &&
                 (sim_idx %% progress_every == 0L || sim_idx == n_sims)) {
        elapsed <- round(proc.time()["elapsed"] - t_loop, 1L)
        speed   <- round(sim_idx / max(elapsed, 0.001), 1L)
        eta     <- round((n_sims - sim_idx) / max(speed, 0.001), 0L)
        mem_mb  <- round(gc(reset = FALSE)[2L, 2L], 0L)
        message(sprintf(
          "  Progress: %d/%d (%d%%) | %.1f s | %.1f sim/s | ETA ~%d s | RAM ~%d MB",
          sim_idx, n_sims, round(100L * sim_idx / n_sims),
          elapsed, speed, eta, mem_mb
        ))
      }
      
      tryCatch(
        .agg_one_sim(sid),
        error = function(e) {
          message(sprintf("    WARNING [%s]: %s", sid, conditionMessage(e)))
          NULL
        }
      )
    },
    mc.cores = n_cores
  )
  
  # Restore data.table thread count immediately after mclapply returns
  if (n_cores > 1L) data.table::setDTthreads(dt_threads_outer)
  
  
  # ==========================================================================
  # STEP E — Bind results and write output file
  #
  # rbindlist operates on small per-SimID tables (~50 rows each).
  # Total in-memory size at this point is n_sims × n_cycles × n_vars,
  # which is negligible compared with the raw daily data (now on disk only).
  # ==========================================================================
  message("  Binding all cycle rows...")
  list_results <- Filter(function(x) !is.null(x) && nrow(x) > 0L, list_results)
  
  if (length(list_results) == 0L) {
    warning(sprintf(
      "No cycles aggregated for campaign %s / mode %s.",
      campaign_tag, mode_name
    ))
    return(invisible(NULL))
  }
  
  n_failed <- n_sims - length(list_results)
  if (n_failed > 0L)
    message(sprintf(
      "  %d/%d simulations returned NULL (check WARNING lines above).",
      n_failed, n_sims
    ))
  
  cycles_df     <- data.table::rbindlist(list_results, fill = TRUE, use.names = TRUE)
  total_elapsed <- round(proc.time()["elapsed"] - t_loop, 1L)
  
  save_data(cycles_df, path_no_ext)
  
  # Report output file size (CSV only; silently skips for other backends)
  csv_file  <- paste0(path_no_ext, ".csv")
  fsize_str <- if (file.exists(csv_file))
    sprintf("~%.1f MB", file.size(csv_file) / 1e6) else "?"
  
  message(sprintf(
    "  Saved: %d cycle-rows x %d columns [%s / %s] in %.1f s (%s)\n",
    nrow(cycles_df), ncol(cycles_df),
    campaign_tag, mode_name,
    total_elapsed, fsize_str
  ))
  
  invisible(path_no_ext)
}
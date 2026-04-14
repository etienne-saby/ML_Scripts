# ==============================================================================
# FILE    : utils_data_extract.R
# PURPOSE : Extract raw daily simulation data for all simulations in a campaign.
#           Provides single-simulation and batch extraction functions.
# AUTHOR  : Étienne SABY
# DATE    : 2026
#
# ------------------------------------------------------------------------------
# ARCHITECTURE — PER-SimID FST FILES
# ------------------------------------------------------------------------------
#
# Architecture: one FST file per (SimID × file_type), no merge phase.
#
# Directory layout produced by extract_all_raw():
#
#   <out_dir>/
#     raw_<tag>_cell/
#       Sim_001.fst
#       Sim_002.fst   ...
#     raw_<tag>_trees/
#       Sim_001.fst   ...
#     raw_<tag>_climate/
#       Sim_001.fst   ...
#
# raw_paths returned to aggregate_cycles() is a named list of directory paths.
# Each worker reconstructs its own SimID path as:
#   fs::path(raw_paths[["cell"]], paste0(sid, ".fst"))
#
# ------------------------------------------------------------------------------
# PARALLELISM & OPENMP
# ------------------------------------------------------------------------------
# data.table uses OpenMP threads internally.  When mclapply forks workers each
# worker would also spawn OpenMP threads, causing over-subscription.
# data.table thread count is reduced to 1 before mclapply and restored after.
# ==============================================================================


# ==============================================================================
# 1. SINGLE SIMULATION RAW EXTRACTION
# ==============================================================================

#' Extract raw daily data for one simulation across all source file types.
#'
#' Reads the requested source files, builds a Date column from Day/Month/Year,
#' attaches SimID, and coerces spurious character values (Hi-sAFe "error!"
#' artefacts) to NA_real_.
#'
#' Set \code{n_cores_inner = 1L} (default) when called from an outer mclapply
#' to avoid nested process forking.
#'
#' @param sim_id          String.  Simulation ID, e.g. \code{"Sim_004"}.
#' @param camp_dir        String.  Campaign root directory.
#' @param source_vars     Named list of named character vectors specifying
#'   variables and their aggregation rules.
#' @param get_job_details Logical.  Print per-file progress messages.
#' @param n_cores_inner   Integer.  Cores for inner parallel file reading.
#'   Must be 1L when called from an outer parallel loop. Default: 1L.
#' @return Named list of data.tables, one per file type.  NULL for missing or
#'   empty files.
extract_sim_raw <- function(sim_id, camp_dir,
                            source_vars     = default_source_vars(),
                            get_job_details = FALSE,
                            n_cores_inner   = 1L) {
  
  out_dir <- get_sim_output_dir(camp_dir, sim_id)
  
  # Mandatory identifier columns required regardless of source_vars content
  mandatory <- list(
    cell    = c("Day", "Month", "Year", "zoneName", "idCell",
                "cropSpeciesName", "cropAge"),
    trees   = c("Day", "Month", "Year", "idTree"),
    climate = c("Day", "Month", "Year")
  )
  
  # Columns that must never be coerced to numeric
  char_id_cols <- c("cropSpeciesName", "zoneName", "SimID") #, "idTree")
  
  results <- setNames(
    parallel::mclapply(names(source_vars), function(ftype) {
      
      fpath <- fs::path(out_dir, paste0(sim_id, "_", ftype, ".txt"))
      
      if (!fs::is_file(fpath)) {
        if (get_job_details)
          message(sprintf("    [%s] File not found: %s", sim_id, ftype))
        return(NULL)
      }
      
      cols_needed <- unique(c(mandatory[[ftype]], names(source_vars[[ftype]])))
      dt          <- read_hisafe_file(fpath, cols_needed, nThread = 1L)
      
      if (is.null(dt) || nrow(dt) == 0L) {
        if (get_job_details)
          message(sprintf("    [%s] Empty or unreadable: %s", sim_id, ftype))
        return(NULL)
      }
      
      # Build Date, attach SimID, drop raw date components
      dt[, Date  := lubridate::make_date(Year, Month, Day)]
      dt[, SimID := sim_id]
      dt[, c("Day", "Month", "Year") := NULL]
      
      # Coerce residual character columns (Hi-sAFe "error!" artefacts) to NA
      cols_to_coerce <- names(dt)[
        vapply(dt, is.character, logical(1L)) & !names(dt) %in% char_id_cols
      ]
      
      for (col in cols_to_coerce) {
        suppressWarnings({
          converted <- as.numeric(dt[[col]])
          n_na_before <- sum(is.na(dt[[col]]))
          n_na_after <- sum(is.na(converted))
          if (n_na_after > n_na_before && get_job_details) {
            message(sprintf("    [%s/%s] Coercion '%s': %d NA introduced",
                            sim_id, ftype, col, n_na_after - n_na_before))
          }
          dt[, (col) := converted]
        })
      }
      
      if (get_job_details)
        message(sprintf("    [%s] '%s' OK — %s rows, %d cols",
                        sim_id, ftype,
                        format(nrow(dt), big.mark = ","), ncol(dt)))
      dt
      
    }, mc.cores = n_cores_inner),
    names(source_vars)
  )
  
  results
}


# ==============================================================================
# 2. BATCH RAW EXTRACTION — PER-SimID FST ARCHITECTURE
# ==============================================================================

#' Extract raw daily data for all simulations in a campaign.
#'
#' Writes one FST file per (SimID × file type) into dedicated subdirectories.
#' Eliminates the merge phase that dominated the previous architecture (990s
#' for 886 simulations) and reduces peak RAM from ~220 GB to ~600 MB.
#'
#' \strong{Directory layout}:
#' \preformatted{
#'   <out_dir>/
#'     raw_<campaign_tag>_cell/
#'       Sim_001.fst,  Sim_002.fst, ...
#'     raw_<campaign_tag>_trees/
#'       Sim_001.fst, ...
#'     raw_<campaign_tag>_climate/
#'       Sim_001.fst, ...
#' }
#'
#' \strong{Caching}: a SimID file is skipped if it already exists on disk and
#' \code{overwrite = FALSE}.  Partially-completed campaigns (e.g. after an OOM
#' Kill) resume automatically without re-extracting successful SimIDs.
#'
#' \strong{Return value}: a named list of directory paths (one per file type).
#' Pass directly to \code{aggregate_cycles()} as the \code{raw_paths} argument.
#'
#' @param sim_ids         Character vector.  Simulation IDs to extract.
#' @param camp_dir        String.  Campaign root directory.
#' @param n_cores         Integer.  Parallel workers for mclapply. Default 1L.
#' @param chunk_size      Integer.  Batch size for progress reporting only;
#'   does NOT affect memory (each worker always writes its own file).
#'   Default: 100L.
#' @param out_dir         String.  Parent directory for per-SimID subdirs.
#' @param campaign_tag    String.  Campaign identifier, e.g. "sobol_S11111".
#' @param source_vars     Named list.  Variable declarations per file type.
#' @param overwrite       Logical.  Re-extract even if file exists? Default FALSE.
#' @param get_job_details Logical.  Verbose per-file messages. Default FALSE.
#' @param progress_every  Integer.  Print progress every N sims. Default 100L.
#' @return Named list of directory paths (without SimID or extension), one per
#'   file type.  Invisibly.
#' @export
extract_all_raw <- function(sim_ids,
                            camp_dir,
                            n_cores         = 1L,
                            chunk_size      = 100L,
                            out_dir,
                            campaign_tag,
                            source_vars     = default_source_vars(),
                            overwrite       = FALSE,
                            get_job_details = FALSE,
                            progress_every  = 100L) {
  
  time_start <- proc.time()["elapsed"]
  n_sims     <- length(sim_ids)
  ftypes     <- names(source_vars)
  
  message(sprintf(
    "\n[extract_all_raw] Campaign: %s | %d simulations | %d core(s) | per-SimID FST",
    campaign_tag, n_sims, n_cores
  ))
  message(sprintf("  Extracting %d file type(s) from %d simulations...",
                  length(ftypes), n_sims))
  
  # ---------------------------------------------------------------------------
  # Create per-file-type subdirectories
  # ---------------------------------------------------------------------------
  sim_dirs <- setNames(
    lapply(ftypes, function(ft) {
      d <- fs::path(out_dir, paste0("raw_", campaign_tag, "_", ft))
      fs::dir_create(d)
      d
    }),
    ftypes
  )
  
  # ---------------------------------------------------------------------------
  # Caching: identify which SimIDs still need extraction per file type
  # ---------------------------------------------------------------------------
  todo_by_ftype <- lapply(ftypes, function(ft) {
    if (overwrite) return(sim_ids)
    Filter(function(sid) {
      !file.exists(fs::path(sim_dirs[[ft]], paste0(sid, ".fst")))
    }, sim_ids)
  })
  names(todo_by_ftype) <- ftypes
  
  for (ft in ftypes) {
    n_cached <- n_sims - length(todo_by_ftype[[ft]])
    message(sprintf("  [%s] %d to extract, %d already cached.",
                    ft, length(todo_by_ftype[[ft]]), n_cached))
  }
  
  if (all(vapply(todo_by_ftype, length, integer(1L)) == 0L)) {
    message("  All file types fully cached — skipping extraction.\n")
    return(invisible(sim_dirs))
  }
  
  # ---------------------------------------------------------------------------
  # OpenMP guard
  # ---------------------------------------------------------------------------
  dt_threads_outer <- data.table::getDTthreads()
  if (n_cores > 1L) data.table::setDTthreads(1L)
  
  # ---------------------------------------------------------------------------
  # Per-file-type extraction
  # ---------------------------------------------------------------------------
  for (ft in ftypes) {
    
    sids_todo  <- todo_by_ftype[[ft]]
    if (length(sids_todo) == 0L) next
    
    sim_dir_ft    <- sim_dirs[[ft]]
    source_vars_f <- source_vars[ft]
    n_todo_ft     <- length(sids_todo)
    t0_ft         <- proc.time()["elapsed"]
    n_ok_ft       <- 0L
    n_err_ft      <- 0L
    
    message(sprintf("\n  [%s] %d simulations...", ft, n_todo_ft))
    
    # Split into batches for progress reporting only
    batches   <- split(sids_todo, ceiling(seq_along(sids_todo) / chunk_size))
    n_batches <- length(batches)
    
    for (batch_idx in seq_along(batches)) {
      
      batch   <- batches[[batch_idx]]
      t_batch <- proc.time()["elapsed"]
      
      batch_results <- parallel::mclapply(batch, function(sid) {
        
        out_fst <- fs::path(sim_dir_ft, paste0(sid, ".fst"))
        
        tryCatch({
          dt <- extract_sim_raw(
            sid, camp_dir,
            source_vars     = source_vars_f,
            get_job_details = get_job_details,
            n_cores_inner   = 1L
          )[[ft]]
          
          if (!is.null(dt) && nrow(dt) > 0L) {
            fst::write_fst(dt, out_fst, compress = 50L)
            return(nrow(dt))   # rows extracted
          }
          return(0L)
          
        }, error = function(e) {
          if (get_job_details)
            message(sprintf("    WARNING [%s/%s]: %s", ft, sid, conditionMessage(e)))
          return(NA_integer_)
        })
        
      }, mc.cores = n_cores)
      
      ok_flags  <- !is.na(batch_results) & vapply(batch_results, function(x) !is.na(x) && x > 0L, logical(1L))
      n_ok_ft   <- n_ok_ft  + sum(ok_flags)
      n_err_ft  <- n_err_ft + sum(is.na(unlist(batch_results)))
      
      elapsed_total <- round(proc.time()["elapsed"] - t0_ft,    1L)
      elapsed_batch <- round(proc.time()["elapsed"] - t_batch,  1L)
      speed         <- round(n_ok_ft / max(elapsed_total, 0.001), 1L)
      eta_s         <- round((n_todo_ft - n_ok_ft) / max(speed, 0.001), 0L)
      mem_mb        <- round(gc(reset = FALSE)[2L, 2L], 0L)
      
      message(sprintf(
        "    Batch %d/%d | %d/%d OK | %.1fs (+%.1fs) | %.1f sim/s | ETA ~%ds | RAM ~%d MB",
        batch_idx, n_batches,
        n_ok_ft, n_todo_ft,
        elapsed_total, elapsed_batch,
        speed, eta_s, mem_mb
      ))
      
    } # end batch loop
    
    elapsed_ft <- round(proc.time()["elapsed"] - t0_ft, 1L)
    message(sprintf("  [%s] Done: %d/%d OK, %d failed — %.1fs\n",
                    ft, n_ok_ft, n_todo_ft, n_err_ft, elapsed_ft))
    
  } # end ftype loop
  
  if (n_cores > 1L) data.table::setDTthreads(dt_threads_outer)
  
  total_elapsed <- round(proc.time()["elapsed"] - time_start, 1L)
  message(sprintf(
    "[extract_all_raw] Completed in %.1fs — %d file type(s) extracted.\n",
    total_elapsed, length(ftypes)
  ))
  
  invisible(sim_dirs)
}
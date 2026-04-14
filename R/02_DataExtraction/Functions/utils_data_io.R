# ==============================================================================
# FILE    : utils_data_io.R
# PURPOSE : I/O utilities for the Hi-sAFe extraction pipeline.
#           Covers default variable declarations, raw file reading,
#           multi-backend persistence (CSV / FST / Parquet), and
#           simulation validity checks.
# AUTHOR  : Étienne SABY
# DATE    : 2026
#
# ------------------------------------------------------------------------------
# BACKEND SELECTION
# ------------------------------------------------------------------------------
#
#   "csv"     : data.table::fwrite / fread  — universally readable, no extra
#               packages required.  Supports append mode natively.
#               Recommended for portability and debugging.
#
#   "fst"     : fst::write_fst / read_fst   — ~5–10× faster reads than CSV,
#               column-selective loading.  Best for iterative re-reads.
#               Requires the fst package.
#
#   "parquet" : arrow::write_parquet / read_parquet — language-agnostic,
#               excellent columnar compression.  Best for Python interop.
#               Requires the arrow package.
#
# Append mode (used during chunked extraction) is implemented natively for CSV
# and via read–bind–rewrite for FST and Parquet (which do not support appending
# natively).  Prefer CSV when streaming large campaigns chunk-by-chunk.
# ==============================================================================


# ==============================================================================
# 0. DEFAULT VARIABLE DECLARATIONS
# ==============================================================================

#' Default variables to extract and their cycle-level aggregation rules.
#'
#' Returns a named list with three keys corresponding to Hi-sAFe source files:
#' \code{"cell"}, \code{"trees"}, and \code{"climate"}.  Each element is a
#' named character vector where \strong{names} are Hi-sAFe variable names and
#' \strong{values} are aggregation rules applied over one crop cycle:
#'
#' \describe{
#'   \item{\code{"max"}}{Peak value over the cycle (yield, LAI, biomass, …).}
#'   \item{\code{"min"}}{Minimum value over the cycle.}
#'   \item{\code{"sum"}}{Cumulative flux (precipitation, drainage, N leaching, …).}
#'   \item{\code{"mean"}}{Cycle-average index or rate (stress indices, VPD, …).}
#'   \item{\code{"last"}}{Stock at the end of the cycle (carbon pools, DBH, …).}
#' }
#'
#' Mandatory identifier columns (\code{Day}, \code{Month}, \code{Year},
#' \code{SimID}, \code{idCell}, \code{zoneName}, \code{cropSpeciesName},
#' \code{cropAge}, \code{idTree}) are handled separately
#' in \code{extract_sim_raw()} and are \strong{not} listed here.
#'
#' Tree carbon pool variables (\code{carbonStem}, \code{carbonBranches}, …) use
#' rule \code{"last"} (end-of-cycle stock).  A corresponding \code{_delta}
#' increment column (end − start of cycle) is computed automatically by
#' \code{aggregate_cycles()} from the raw daily series.
#'
#' Pass a modified copy to \code{extract_all_raw()} / \code{aggregate_cycles()}
#' to restrict or extend the variable set without editing this function.
#' All units follow the Hi-sAFe documentation (t/ha, kg C/tree, mm, etc.).
#'
#' @return Named list with keys \code{"cell"}, \code{"trees"}, \code{"climate"}.
#' @export
default_source_vars <- function() {
  list(

    # --------------------------------------------------------------------------
    # CELL — one row per day × cell
    # --------------------------------------------------------------------------
    cell = c(
      # ---- Crop production --------------------------------------------------
      yield                          = "max",   # t ha-1  — peak harvested yield
      grainBiomass                   = "max",   # t ha-1  — peak grain biomass
      biomass                        = "max",   # t ha-1  — peak total above-ground biomass

      # ---- Canopy -----------------------------------------------------------
      lai                            = "max",   # m2 leaf m-2 soil

      # ---- Light ------------------------------------------------------------
      relativeTotalParIncident       = "mean",  # %  — mean fractional PAR reaching crop

      # ---- Stress indices (0–1; 1 = no stress) ------------------------------
      tempStressLue                  = "mean",
      frostStressPlantDensity        = "mean",
      frostStressFoliage             = "mean",
      sticsWaterStomatalStress       = "mean",
      sticsWaterSenescenceStress     = "mean",
      sticsNitrogenLaiStress         = "mean",
      sticsNitrogenBiomassStress     = "mean",

      # ---- Soil carbon ------------------------------------------------------
      totalCarbonHumusStock          = "last",  # kg C ha-1 — end-of-cycle humus stock

      # ---- Nitrogen ---------------------------------------------------------
      nitrogenLeachingBottom         = "sum",   # kg N ha-1 — cumulative leaching

      # ---- Soil water stock -------------------------------------------------
      waterStock                     = "mean",  # mm — mean soil water content

      # ---- Water — crop fluxes (liters per cell) ----------------------------
      waterUptakeByCrop              = "sum",
      waterUptakeInSaturationByCrop  = "sum",
      waterDemand                    = "sum",
      waterDemandReduced             = "sum",

      # ---- Water — tree fluxes (liters per cell) ----------------------------
      waterUptakeByTrees             = "sum",
      waterUptakeInSaturationByTrees = "sum",

      # ---- Water — soil and surface fluxes (mm) ----------------------------
      soilEvaporation                = "sum",
      mulchEvaporation               = "sum",
      runOff                         = "sum",
      surfaceRunOff                  = "sum",
      drainageBottom                 = "sum",
      drainageArtificial             = "sum",
      capillaryRise                  = "sum",
      irrigation                     = "sum",
      waterAddedByWaterTable         = "sum",
      waterTakenByDesaturation       = "sum",

      # ---- Water — interception / stemflow (mm) ----------------------------
      rainInterceptedByCrop          = "sum",
      rainInterceptedByTrees         = "sum",
      rainTransmittedByCrop          = "sum",
      rainTransmittedByTrees         = "sum",
      stemFlowByCrop                 = "sum",
      stemFlowByTrees                = "sum"
    ),

    # --------------------------------------------------------------------------
    # TREES — one row per day × tree
    # --------------------------------------------------------------------------
    trees = c(
      # ---- Dendrometry ------------------------------------------------------
      age                            = "last",  # years — tree age at end of cycle
      dbh                            = "last",  # cm    — diameter at breast height
      height                         = "last",  # m     — total height

      # ---- Carbon pools (kg C per tree) -------------------------------------
      # Rule "last" = end-of-cycle cumulative stock.
      # A "_delta" increment companion is computed automatically by
      # aggregate_cycles() for use in per-cycle RYR calculations.
      carbonStem                     = "last",
      carbonBranches                 = "last",
      carbonStump                    = "last",
      carbonCoarseRoots              = "last",
      carbonFineRoots                = "last",
      carbonFruit                    = "last",

      # ---- Canopy -----------------------------------------------------------
      totalLeafArea                  = "max",   # m2 per tree — peak leaf area

      # ---- Stress indices (0–1; 1 = no stress) ------------------------------
      waterStress                    = "mean",
      nitrogenStress                 = "mean"
    ),

    # --------------------------------------------------------------------------
    # CLIMATE — SafeMacroClimat, one row per day
    # --------------------------------------------------------------------------
    climate = c(
      precipitation                  = "sum",   # mm
      maxTemperature                 = "mean",  # °C
      minTemperature                 = "mean",  # °C
      globalRadiation                = "sum",   # MJ m-2
      airVpd                         = "mean",  # mbar
      etpPenman                      = "sum"    # mm    
      )
  )
}

#' Crop-specific base temperature for GDD calculation. See https://en.wikipedia.org/wiki/Growing_degree-day
#'
#' @return Named numeric vector: crop_name -> t_base (°C)
#' @export
default_crop_tbase <- function() {
  c(
    # 4.5°C group
    "wheat"       = 4.5,
    "durum-wheat" = 4.5,
    "barley"      = 4.5,
    "rye"         = 4.5,
    "oats"        = 4.5,
    "flax"        = 4.5,
    "lettuce"     = 4.5,
    
    # 8°C group
    "sunflower"   = 8.0,
    "potato"      = 8.0,
    
    # 10°C group
    "maize"       = 10.0,
    "corn"        = 10.0,
    "sorghum"     = 10.0,
    "rice"        = 10.0,
    "soybean"     = 10.0,
    "tomato"      = 10.0,
    "grape"       = 10.0,
    "bean"        = 10.0,
    
    # Default fallback
    "default"     = 6.0
  )
}

# ==============================================================================
# 1. RAW FILE READING
# ==============================================================================

#' Read a Hi-sAFe tab-separated output file safely.
#'
#' Uses \code{data.table::fread} for maximum I/O throughput.
#' Replaces \code{"error!"} and blank cells with \code{NA}.
#' Returns \code{NULL} on any read failure.
#'
#' Performance notes:
#' \itemize{
#'   \item \code{nThread = 1L} prevents OpenMP contention when this function
#'         is called from inside a \code{mclapply} worker.
#'   \item \code{colClasses} are pre-computed so that \code{fread} does not
#'         need to guess column types, which is particularly costly for large
#'         files with many columns.
#'   \item \code{integer64 = "double"} ensures all numeric columns share a
#'         consistent type and avoids silent integer overflow surprises.
#' }
#'
#' @param file_path   String.  Full path to the \code{.txt} output file.
#' @param cols_needed Character vector or \code{NULL}.  Subset of columns to
#'   retain.  When \code{NULL}, all columns are returned.
#' @return A \code{data.table}, or \code{NULL} if the file cannot be read.
read_hisafe_file <- function(file_path, cols_needed = NULL, nThread = data.table::getDTthreads()) {
  if (is.na(file_path) || !fs::file_exists(file_path)) return(NULL)

  # Columns that must remain character regardless of their content
  char_id_cols <- c("cropSpeciesName", "zoneName") #, "idTree")

  # Pre-compute colClasses to avoid fread's type-guessing overhead
  col_classes <- if (!is.null(cols_needed)) {
    setNames(
      ifelse(cols_needed %in% char_id_cols, "character", "numeric"),
      cols_needed
    )
  } else {
    setNames(rep("character", length(char_id_cols)), char_id_cols)
  }

  tryCatch({
    dt <- data.table::fread(
      file_path,
      sep        = "\t",
      na.strings = c("", "NA", "error!", "null"),
      data.table = TRUE,
      nThread    = nThread,         # avoids OpenMP contention inside mclapply workers
      integer64  = "double",   # uniform numeric type across all columns
      colClasses = col_classes
    )
    if (!is.null(cols_needed)) {
      cols_found <- intersect(cols_needed, names(dt))
      dt <- dt[, ..cols_found]
    }
    dt
  }, error = function(e) NULL)
}


# ==============================================================================
# 2. MULTI-BACKEND PERSISTENCE
# ==============================================================================

#' Save a data.table or data.frame to disk using the active storage backend.
#'
#' The backend is determined by the \code{STORAGE_BACKEND} global variable
#' (set by \code{03_cluster_extract.R}).  Supported values:
#' \describe{
#'   \item{\code{"csv"}}{data.table::fwrite — supports \code{append} natively.}
#'   \item{\code{"fst"}}{fst::write_fst — append implemented via read–bind–rewrite.}
#'   \item{\code{"parquet"}}{arrow::write_parquet — append via read–bind–rewrite.}
#' }
#'
#' The \code{append} parameter is used during chunked extraction in
#' \code{extract_all_raw()} to stream data to disk without accumulating an
#' entire file type in RAM.  For FST and Parquet backends, append is
#' substantially more expensive than for CSV; prefer \code{backend = "csv"}
#' when running the extraction phase.
#'
#' @param df          data.frame or data.table.  The data to write.
#' @param path_no_ext String.  Output path \strong{without} file extension.
#' @param append      Logical.  If \code{TRUE}, append to an existing file
#'   rather than overwriting it.  Default \code{FALSE}.
#' @return Invisibly, the full path of the written file (with extension).
save_data <- function(df, path_no_ext, append = FALSE) {
  backend <- if (exists("STORAGE_BACKEND", envir = .GlobalEnv))
    get("STORAGE_BACKEND", envir = .GlobalEnv) else "csv"

  if (backend %in% c("arrow", "parquet")) {
    fpath <- paste0(path_no_ext, ".parquet")
    if (append && file.exists(fpath)) {
      existing <- data.table::as.data.table(arrow::read_parquet(fpath))
      df       <- data.table::rbindlist(
        list(existing, data.table::as.data.table(df)), fill = TRUE
      )
      rm(existing)
    }
    arrow::write_parquet(data.table::as.data.table(df), fpath)

  } else if (backend == "fst") {
    fpath <- paste0(path_no_ext, ".fst")
    if (append && file.exists(fpath)) {
      existing <- fst::read_fst(fpath, as.data.table = TRUE)
      df       <- data.table::rbindlist(
        list(existing, data.table::as.data.table(df)), fill = TRUE
      )
      rm(existing)
    }
    fst::write_fst(data.table::as.data.table(df), fpath, compress = 50L)

  } else {
    # CSV — native append support via fwrite
    fpath <- paste0(path_no_ext, ".csv")
    data.table::fwrite(df, fpath, dateTimeAs = "write.csv", append = append)
  }

  invisible(fpath)
}


#' Read a file produced by \code{save_data()}, auto-detecting its extension.
#'
#' Tries \code{.fst}, then \code{.parquet}, then \code{.csv} in that order.
#' Always returns a \code{data.table} (coerces if necessary).
#' Returns \code{NULL} if no matching file is found.
#'
#' @param path_no_ext String.  Path \strong{without} file extension.
#' @return A \code{data.table}, or \code{NULL} if no file is found.
read_data <- function(path_no_ext) {
  fst_path     <- paste0(path_no_ext, ".fst")
  parquet_path <- paste0(path_no_ext, ".parquet")
  csv_path     <- paste0(path_no_ext, ".csv")

  if (file.exists(fst_path)) {
    return(fst::read_fst(fst_path, as.data.table = TRUE))
  } else if (file.exists(parquet_path)) {
    return(data.table::as.data.table(arrow::read_parquet(parquet_path)))
  } else if (file.exists(csv_path)) {
    return(data.table::fread(csv_path, data.table = TRUE, nThread = 1L))
  }
  NULL
}


# ==============================================================================
# 3. SIMULATION VALIDITY HELPERS
# ==============================================================================

#' Identify completed simulations in a campaign directory.
#'
#' Scans \code{simulation.log} files in parallel for an \code{"END"} marker
#' in the last 20 lines, which Hi-sAFe writes upon successful completion.
#'
#' Parallel scanning via \code{mclapply} reduces wall-clock time by ~10–20×
#' compared with a sequential \code{for} loop on large campaigns (2 000+ sims).
#'
#' @param dir_path String.  Campaign root directory containing \code{Sim_*}
#'   sub-directories.
#' @param n_cores  Integer.  Number of cores for parallel log scanning.
#'   Default: \code{4L}.
#' @return Character vector of valid simulation IDs (e.g. \code{"Sim_001"}).
get_valid_sims <- function(dir_path, n_cores = 4L) {
  sim_dirs <- fs::dir_ls(dir_path, type = "directory", regexp = "/Sim_")
  if (length(sim_dirs) == 0L) return(character(0L))

  results <- parallel::mclapply(sim_dirs, function(s_dir) {
    log_file <- fs::path(s_dir, "simulation.log")
    if (!fs::is_file(log_file)) return(NULL)
    tryCatch({
      lines <- tail(readLines(log_file, warn = FALSE), 20L)
      if (any(grepl("END", lines, ignore.case = TRUE)))
        return(fs::path_file(s_dir))
      NULL
    }, error = function(e) NULL)
  }, mc.cores = min(n_cores, length(sim_dirs)))

  unlist(Filter(Negate(is.null), results), use.names = FALSE)
}


#' Locate the output sub-directory for a given simulation.
#'
#' Hi-sAFe writes output files either directly in \code{<sim_id>/} or in
#' \code{<sim_id>/output-<sim_id>/}, depending on the version.  This helper
#' resolves whichever path exists.
#'
#' @param campaign_dir String.  Campaign root directory.
#' @param sim_id       String.  Simulation identifier (e.g. \code{"Sim_004"}).
#' @return String.  Path to the directory that contains the \code{.txt} output
#'   files for this simulation.
get_sim_output_dir <- function(campaign_dir, sim_id) {
  base_path <- fs::path(campaign_dir, sim_id)
  sub_path  <- fs::path(base_path, paste0("output-", sim_id))
  if (fs::is_dir(sub_path)) sub_path else base_path
}

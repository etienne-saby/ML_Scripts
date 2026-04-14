# ==============================================================================
# FILE    : 03_cluster_extract.R
# PURPOSE : Main Hi-sAFe Meta-Model extraction pipeline.
#           Orchestrates four decoupled steps, each independently callable:
#
#             STEP 1  Raw extraction      (modes AF / TA / TF independently)
#             STEP 2  Phenological calendar
#             STEP 3  Cycle aggregation   (modes AF / TA / TF independently)
#             STEP 4  Meta-model table assembly
#
# AUTHOR  : Étienne SABY
# DATE    : 2026
#
# ------------------------------------------------------------------------------
# USAGE
# ------------------------------------------------------------------------------
#
# Minimal (single-node, all steps sequentially, default settings):
#   Rscript --vanilla 03_cluster_extract.R <campaign_name>
#
# Full option set:
#   Rscript --vanilla 03_cluster_extract.R <campaign_name> \
#     --cpus       64        \   # parallel cores (overrides SLURM_CPUS_PER_TASK)
#     --chunk      150       \   # simulations per extraction chunk
#     --mode       AF        \   # see MODE LOGIC below
#     --backend    parquet   \   # csv | fst | parquet
#     --overwrite  FALSE     \   # re-extract even if cache exists?
#     --progress   10        \   # log progress every N simulations
#     --job_details FALSE        # verbose per-file messages
#
# ------------------------------------------------------------------------------
# MODE LOGIC
# ------------------------------------------------------------------------------
#
#   ALL    Run all four steps sequentially on a single node.
#          Suitable for small campaigns (< 500 simulations).
#
#   PHENO  Run STEP 2 only (phenological calendar from plan_df).
#          Very fast (seconds). Must complete before any extraction mode starts.
#          Does NOT read any simulation output files — plan_df is sufficient.
#
#   AF     Run STEP 1 + STEP 3 for the AF mode only.
#   TA     Run STEP 1 + STEP 3 for the TA mode only.
#   TF     Run STEP 1 + STEP 3 for the TF mode only.
#          AF / TA / TF can all run in parallel after PHENO has completed,
#          since they each read the pheno calendar written to disk by PHENO.
#
#   BUILD  Run STEP 4 only (meta-model table assembly from cycle files).
#          Reads cycle files from disk; must run after AF + TA + TF.
#
# ------------------------------------------------------------------------------
# RECOMMENDED PIPELINE (multi-node, optimal parallelism)
# ------------------------------------------------------------------------------
#
#   Stage 1  PHENO       — single lightweight job (~seconds, plan_df only)
#   Stage 2  AF + TA + TF — three parallel jobs (all depend on PHENO)
#   Stage 3  BUILD        — single lightweight job (depends on AF + TA + TF)
#
#   Use submit_extraction.sh to orchestrate the full chain automatically.
#
# ==============================================================================


# ==============================================================================
# 0. PROJECT SETUP
# ==============================================================================
ROOT_DIR <- "/home/sabye/scratch_hisafe/MetAIsAFe"
source(file.path(ROOT_DIR, "01_Scripts/R/00_setup.R"))

detected_backend <- setup_project(load_parts = c("core", "extract"), load_functions = TRUE)
paths <- get_hisafe_paths(ROOT_DIR)


# ==============================================================================
# 1. CLI ARGUMENT PARSING
# ==============================================================================
# Format: <campaign_name> [--key value ...]
# The first positional argument is always the campaign name.
# All subsequent arguments must be --key value pairs.

.parse_cli <- function(args) {
  if (length(args) == 0L)
    stop(
      "ERROR: campaign_name is required.\n",
      "Usage: Rscript --vanilla 03_cluster_extract.R <campaign_name> [--key val ...]"
    )
  
  campaign <- args[1L]
  if (startsWith(campaign, "--"))
    stop("ERROR: The first argument must be campaign_name, not an option flag.")
  
  # Default values for all options
  opts <- list(
    campaign    = campaign,
    cpus        = NA_integer_,  # resolved below from SLURM env if not set
    chunk       = 75L,
    mode        = "ALL",        # ALL | PHENO | AF | TA | TF | BUILD
    backend     = NULL,        # csv | fst | parquet
    overwrite   = FALSE,
    progress    = 10L,
    job_details = FALSE
  )
  
  if (length(args) > 1L) {
    kv <- args[-1L]
    i  <- 1L
    while (i <= length(kv)) {
      if (!startsWith(kv[i], "--") || i + 1L > length(kv))
        stop(sprintf("Invalid option or missing value: '%s'", kv[i]))
      key <- sub("^--", "", kv[i])
      val <- kv[i + 1L]
      opts[[key]] <- switch(key,
                            cpus        = as.integer(val),
                            chunk       = as.integer(val),
                            progress    = as.integer(val),
                            overwrite   = as.logical(val),
                            job_details = as.logical(val),
                            val   # mode, backend remain character
      )
      i <- i + 2L
    }
  }
  opts
}

cfg <- .parse_cli(commandArgs(trailingOnly = TRUE))

# ==============================================================================
# INTERNAL HELPER : .resolve_sim_years()
# ==============================================================================
#' Determine the simulation year range from plan_df alone.
#'
#' Called in PHENO mode before any raw extraction has taken place, so this
#' function must not access any simulation output file.
#'
#' Resolution priority:
#'   1. \code{simDateStart} / \code{simDateEnd} columns (ISO-8601 date strings).
#'   2. \code{start_year}   / \code{end_year}   integer columns.
#'
# If neither pair of columns is present the function stops with an
#' informative error rather than silently returning a wrong range.
#'
#' @param plan_df data.frame. Experimental plan for the AF mode.
#' @return Integer vector of simulation years (e.g. \code{2000:2030}).
#' @keywords internal
.resolve_sim_years <- function(plan_df) {
  
  if (all(c("simDateStart", "simDateEnd") %in% names(plan_df))) {
    plan_df <- dplyr::mutate(plan_df,
                             start_year = as.integer(format(as.Date(simDateStart), "%Y")),
                             end_year   = as.integer(format(as.Date(simDateEnd),   "%Y"))
    )
  }
  
  if (all(c("start_year", "end_year") %in% names(plan_df))) {
    return(seq(
      min(plan_df$start_year, na.rm = TRUE),
      max(plan_df$end_year,   na.rm = TRUE)
    ))
  }
  
  stop(
    "Cannot determine simulation years: plan_df must contain either\n",
    "  'simDateStart' + 'simDateEnd'  (ISO-8601 date strings), or\n",
    "  'start_year'   + 'end_year'    (integer columns)."
  )
}

# ==============================================================================
# 2. CORE CONFIGURATION
# ==============================================================================

# Core count priority: --cpus  >  SLURM_CPUS_PER_TASK  >  1 (fallback)
.slurm_cores <- suppressWarnings(as.integer(Sys.getenv("SLURM_CPUS_PER_TASK")))
N_CORES <- if (!is.na(cfg$cpus) && cfg$cpus > 0L) {
  cfg$cpus
} else if (!is.na(.slurm_cores) && .slurm_cores > 0L) {
  .slurm_cores
} else {
  message("WARNING: neither --cpus nor SLURM_CPUS_PER_TASK is set. Falling back to 1 core.")
  1L
}

CAMPAIGN_NAME   <- cfg$campaign
OVERWRITE       <- cfg$overwrite
GET_JOB_DETAILS <- cfg$job_details
PROGRESS_EVERY  <- cfg$progress
CHUNK_SIZE      <- cfg$chunk
MODE_FILTER     <- toupper(cfg$mode)   # ALL | PHENO | AF | TA | TF | BUILD
STORAGE_BACKEND <- if (!is.null(cfg$backend)) cfg$backend else detected_backend

# Expose backend globally so save_data() / read_data() can detect it.
assign("STORAGE_BACKEND", STORAGE_BACKEND, envir = .GlobalEnv)

# Validate mode early to fail fast before any filesystem work.
.valid_modes      <- c("ALL", "PHENO", "AF", "TA", "TF", "BUILD")
.extraction_modes <- c("ALL", "AF", "TA", "TF")   # modes that run STEPS 1 + 3

if (!MODE_FILTER %in% .valid_modes)
  stop(sprintf(
    "Invalid --mode value: '%s'. Expected one of: %s.",
    MODE_FILTER, paste(.valid_modes, collapse = ", ")
  ))

message("========================================================")
message("Hi-sAFe Meta-Model Extraction")
message(sprintf("Campaign   : %s", CAMPAIGN_NAME))
message(sprintf("Cores      : %d", N_CORES))
message(sprintf("Chunk size : %d", CHUNK_SIZE))
message(sprintf("Mode       : %s", MODE_FILTER))
message(sprintf("Backend    : %s", STORAGE_BACKEND))
message(sprintf("Overwrite  : %s", OVERWRITE))
message(sprintf("Root dir   : %s", fs::path_abs(ROOT_DIR)))
message("========================================================")


# ==============================================================================
# 3. PATH RESOLUTION & VALIDATION
# ==============================================================================

if (!fs::dir_exists(fs::path(ROOT_DIR, "02_Simulations")))
  stop("ROOT_DIR is incorrect — 02_Simulations not found at: ",
       fs::path_abs(fs::path(ROOT_DIR, "02_Simulations")))

DIR_AF <- fs::path(paths$simulations_dir, paste0(CAMPAIGN_NAME, "_AF"))
DIR_TA <- fs::path(paths$simulations_dir, paste0(CAMPAIGN_NAME, "_TA"))
DIR_TF <- fs::path(paths$simulations_dir, paste0(CAMPAIGN_NAME, "_TF"))

for (d in c(DIR_AF, DIR_TA, DIR_TF))
  if (!fs::is_dir(d))
    stop("Campaign directory not found: ", d)

OUT_BASE   <- fs::path(ROOT_DIR, "03_Models", CAMPAIGN_NAME, "Data")
OUT_RAW    <- fs::path(OUT_BASE, "RawData")
OUT_CYCLES <- fs::path(OUT_BASE, "Cycles")
fs::dir_create(OUT_RAW)
fs::dir_create(OUT_CYCLES)

# Experimental plans are needed by all modes.
PLAN_PATH <- fs::path(DIR_AF, paste0(CAMPAIGN_NAME, "_AF_Plan.csv"))
if (!fs::is_file(PLAN_PATH)) stop("Plan file not found: ", PLAN_PATH)
plan_df <- readr::read_csv(PLAN_PATH, show_col_types = FALSE)

if (all(c("simDateStart", "simDateEnd") %in% names(plan_df))) {
  plan_df <- dplyr::mutate(plan_df,
                           start_year = as.integer(format(as.Date(simDateStart), "%Y")),
                           end_year   = as.integer(format(as.Date(simDateEnd),   "%Y"))
  )
}
message(sprintf("Plan loaded: %d simulations", nrow(plan_df)))

PLAN_PATH_TF <- fs::path(DIR_TF, paste0(CAMPAIGN_NAME, "_TF_Plan.csv"))
if (!fs::is_file(PLAN_PATH_TF)) stop("TF plan file not found: ", PLAN_PATH_TF)
plan_df_TF <- readr::read_csv(PLAN_PATH_TF, show_col_types = FALSE)
if (all(c("simDateStart", "simDateEnd") %in% names(plan_df_TF))) {
  plan_df_TF <- dplyr::mutate(plan_df_TF,
                           start_year = as.integer(format(as.Date(simDateStart), "%Y")),
                           end_year   = as.integer(format(as.Date(simDateEnd),   "%Y"))
  )
}


# ==============================================================================
# 4. VALID SIMULATION DETECTION
# ==============================================================================
# Skipped in BUILD mode (cycle files already exist on disk).

if (MODE_FILTER != "BUILD") {
  
  all_plan_sims <- plan_df$sim_id
  plan_df       <- dplyr::filter(plan_df, gen_status == "Success")
  
  valid_AF <- get_valid_sims(DIR_AF)
  valid_TA <- get_valid_sims(DIR_TA)
  valid_TF <- get_valid_sims(DIR_TF)
  
  valid_sims <- Reduce(intersect, list(plan_df$sim_id, valid_AF, valid_TA, valid_TF))
  
  if (length(valid_sims) == 0L)
    stop("No simulations found as complete in AF, TA, and TF simultaneously.")
  
  plan_df    <- dplyr::filter(plan_df,    sim_id %in% valid_sims)
  plan_df_TF <- dplyr::filter(plan_df_TF, sim_id %in% valid_sims)
  
  message(sprintf("Valid simulations (AF ∩ TA ∩ TF): %d", length(valid_sims)))
  
  # Exclusion report
  excluded_sims <- setdiff(all_plan_sims, valid_sims)
  if (length(excluded_sims) > 0L) {
    excl_df <- data.frame(sim_id = excluded_sims) |>
      dplyr::mutate(
        missing_AF = !sim_id %in% valid_AF,
        missing_TA = !sim_id %in% valid_TA,
        missing_TF = !sim_id %in% valid_TF,
        reason = dplyr::case_when(
          missing_AF & missing_TA & missing_TF ~ "missing AF+TA+TF",
          missing_AF & missing_TA              ~ "missing AF+TA",
          missing_AF & missing_TF             ~ "missing AF+TF",
          missing_TA & missing_TF             ~ "missing TA+TF",
          missing_AF                          ~ "missing AF",
          missing_TA                          ~ "missing TA",
          missing_TF                          ~ "missing TF",
          TRUE                                ~ "unknown"
        )
      )
    message("\n  Exclusion summary:")
    print(table(excl_df$reason))
  } else {
    message("  No simulations excluded — all plan entries valid in AF, TA, and TF.")
  }
}


# ==============================================================================
# 5. SOURCE VARIABLE LISTS
# ==============================================================================

base_vars <- default_source_vars()

source_vars_by_mode <- list(
  AF = base_vars,
  TA = base_vars[setdiff(names(base_vars), "trees")],  # no tree data in TA
  TF = base_vars
)


# ==============================================================================
# 6. MODE RESOLUTION — which extraction modes does this job process?
# ==============================================================================

modes_all <- list(AF = DIR_AF, TA = DIR_TA, TF = DIR_TF)

modes_to_run <- switch(MODE_FILTER,
                       ALL   = modes_all,
                       AF    = modes_all["AF"],
                       TA    = modes_all["TA"],
                       TF    = modes_all["TF"],
                       PHENO = list(),   # STEPS 1 + 3 skipped; STEP 2 only
                       BUILD = list()    # STEPS 1–3 skipped; STEP 4 only
)

if (length(modes_to_run) > 0L)
  message(sprintf("\nExtraction modes in this job: %s",
                  paste(names(modes_to_run), collapse = ", ")))


# ==============================================================================
# STEP 1 — RAW EXTRACTION
# ==============================================================================

if (MODE_FILTER %in% .extraction_modes) {
  message("\n--- STEP 1: Raw data extraction ---")
  
  raw_paths <- list()
  
  for (mode_name in names(modes_to_run)) {
    message(sprintf("\n  [%s] Extracting raw data...", mode_name))
    raw_paths[[mode_name]] <- extract_all_raw(
      sim_ids         = valid_sims,
      camp_dir        = modes_to_run[[mode_name]],
      n_cores         = N_CORES,
      chunk_size      = CHUNK_SIZE,
      out_dir         = OUT_RAW,
      campaign_tag    = paste0(CAMPAIGN_NAME, "_", mode_name),
      source_vars     = source_vars_by_mode[[mode_name]],
      overwrite       = OVERWRITE,
      get_job_details = GET_JOB_DETAILS
    )
  }
}


# ==============================================================================
# STEP 2 — PHENOLOGICAL CALENDAR
# ==============================================================================
# Runs in PHENO mode and ALL mode.
# In AF / TA / TF / BUILD modes the calendar is read from the cache written
# to disk by the PHENO job (or the ALL job).
#
# Key design principle: build_pheno_calendars() reads only plan_df — no
# simulation output files are needed.  This is why PHENO can run immediately
# at t=0, before any extraction job has started.

pheno_cache_path <- fs::path(OUT_RAW, paste0("pheno_", CAMPAIGN_NAME, "_AF"))

if (MODE_FILTER %in% c("ALL", "PHENO")) {
  message("\n--- STEP 2: Phenological calendar ---")
  
  sim_years <- .resolve_sim_years(plan_df)
  message(sprintf("  Simulation years: %d – %d", min(sim_years), max(sim_years)))
  
  pheno_AF <- build_pheno_calendars(
    plan_df         = plan_df,
    sim_years       = sim_years,
    out_dir         = OUT_RAW,
    campaign_tag    = paste0(CAMPAIGN_NAME, "_AF"),
    known_crops     = c("wheat", "maize", "rape", "weed", "durum-wheat"),
    overwrite       = OVERWRITE,
    get_job_details = GET_JOB_DETAILS
  )
  
  if (is.null(pheno_AF) || nrow(pheno_AF) == 0L)
    stop("Phenological calendar is empty — cannot continue.")
  
  message(sprintf("  %d cycle-rows across %d simulations.",
                  nrow(pheno_AF), dplyr::n_distinct(pheno_AF$sim_id)))
  
  if (MODE_FILTER == "PHENO") {
    # PHENO job ends here.  AF / TA / TF jobs may now start in parallel.
    message("\n[PHENO] Calendar written to disk. Extraction jobs may now be launched.")
    message(sprintf("  Path: %s", pheno_cache_path))
    quit(save = "no", status = 0L)
  }
  
} else if (MODE_FILTER %in% c("AF", "TA", "TF")) {
  # Single-mode extraction job: read the calendar produced by the PHENO job.
  pheno_AF <- read_data(pheno_cache_path)
  if (is.null(pheno_AF) || nrow(pheno_AF) == 0L)
    stop(
      "Phenological calendar not found on disk. ",
      "The PHENO job must complete before AF / TA / TF jobs are launched.\n",
      "Expected path: ", pheno_cache_path
    )
  message(sprintf("  [CACHE] Phenological calendar loaded: %d cycle-rows.",
                  nrow(pheno_AF)))
}


# ==============================================================================
# STEP 3 — CYCLE AGGREGATION
# ==============================================================================

if (MODE_FILTER %in% .extraction_modes) {
  message("\n--- STEP 3: Cycle aggregation ---")
  
  cycle_paths <- list()
  
  for (mode_name in names(modes_to_run)) {
    cycle_paths[[mode_name]] <- aggregate_cycles(
      raw_paths       = raw_paths[[mode_name]],
      pheno_df        = pheno_AF,
      plan_df         = plan_df,
      out_dir         = OUT_CYCLES,
      campaign_tag    = CAMPAIGN_NAME,
      mode_name       = mode_name,
      source_vars     = source_vars_by_mode[[mode_name]],
      crop_tbase      = default_crop_tbase(),
      n_cores         = N_CORES,
      overwrite       = OVERWRITE,
      get_job_details = GET_JOB_DETAILS,
      progress_every  = PROGRESS_EVERY
    )
  }
}


# ==============================================================================
# STEP 4 — META-MODEL TABLE ASSEMBLY
# ==============================================================================
# Runs in ALL mode (cycle_paths populated in STEP 3) and BUILD mode (cycle
# files resolved directly from disk).

if (MODE_FILTER %in% c("ALL", "BUILD")) {
  message("\n--- STEP 4: Building meta-model table ---")
  
  if (MODE_FILTER == "BUILD") {
    # Resolve and validate cycle file paths from disk.
    cycle_paths <- list(
      AF = fs::path(OUT_CYCLES, paste0("cycles_", CAMPAIGN_NAME, "_AF")),
      TA = fs::path(OUT_CYCLES, paste0("cycles_", CAMPAIGN_NAME, "_TA")),
      TF = fs::path(OUT_CYCLES, paste0("cycles_", CAMPAIGN_NAME, "_TF"))
    )
    for (nm in names(cycle_paths)) {
      if (is.null(read_data(cycle_paths[[nm]])))
        stop(sprintf(
          "Cycle file for mode %s not found at: %s",
          nm, cycle_paths[[nm]]
        ))
    }
  }
  
  meta_path <- build_meta_table(
    path_cycles_AF = cycle_paths[["AF"]],
    path_cycles_TA = cycle_paths[["TA"]],
    path_cycles_TF = cycle_paths[["TF"]],
    plan_df        = plan_df,
    plan_df_TF     = plan_df_TF,
    out_dir        = OUT_BASE,
    campaign_tag   = CAMPAIGN_NAME,
    overwrite      = OVERWRITE
  )
  
} else {
  # Single extraction mode (AF / TA / TF): meta-model table not assembled here.
  meta_path <- NULL
  if (MODE_FILTER %in% c("AF", "TA", "TF"))
    message(sprintf(
      "\n[INFO] Mode %s: meta-model table will be assembled by the BUILD job.",
      MODE_FILTER
    ))
}


# ==============================================================================
# SUMMARY REPORT
# ==============================================================================

message("\n========================================================")
message("STEP(S) COMPLETE")
message(sprintf("Mode             : %s", MODE_FILTER))
message(sprintf("Output directory : %s", fs::path_abs(OUT_BASE)))

if (!is.null(meta_path)) {
  meta_final <- read_data(meta_path)
  if (!is.null(meta_final)) {
    message(sprintf("Meta-model table : %d rows × %d columns",
                    nrow(meta_final), ncol(meta_final)))
    message(sprintf("Simulations      : %d", dplyr::n_distinct(meta_final$SimID)))
    message(sprintf("Unique crops     : %s",
                    paste(sort(unique(meta_final$Crop_Name)), collapse = ", ")))
    yr_col <- grep("^Harvest_Year", names(meta_final), value = TRUE)[1L]
    if (!is.na(yr_col))
      message(sprintf("Year range       : %d – %d",
                      min(meta_final[[yr_col]], na.rm = TRUE),
                      max(meta_final[[yr_col]], na.rm = TRUE)))
  }
}

mem_info <- gc(reset = FALSE)
message(sprintf("Final R memory   : %.0f MB used (peak: %.0f MB)",
                mem_info[2L, 2L], mem_info[2L, 6L]))
message("========================================================")
# ==============================================================================
# SCRIPT : 00_setup.R
# PURPOSE: MASTER SETUP SCRIPT.
#          1. Detects execution environment (local vs cluster).
#          2. Installs/Loads required R packages into the appropriate library.
#          3. Handles complex C++ dependencies (forces binaries on Windows).
#          4. Ensures project folder structure exists.
#          5. Loads all utility functions from the relevant Functions/ subfolder.
#
# USAGE  : Call setup_project() with the desired parts to load.
#          Example: setup_project(load_parts = c("core", "sim", "viz"))
#
# CLUSTER USAGE:
#          On the cluster, load R before sourcing this script:
#            module load R/4.1.x   # adjust version as needed
#            Rscript -e "source('01_Scripts/R/00_setup.R'); setup_project(load_parts=c('core','ml'))"
#          Or inside a SLURM job script (see 01_Scripts/slurm/).
#
# NOTES  : - Arrow requires C++20, unavailable on R 4.1.x / older GCC clusters.
#            Storage backend defaults to CSV via vroom/data.table.
#          - On cluster: viz packages (shiny, plotly, ggridges...) are skipped
#            unless explicitly requested — they are not needed for extraction.
#          - Package library: ~/R/library on cluster, default .libPaths() locally.
#
# AUTHOR : Étienne SABY
# DATE   : 2026
# ==============================================================================

# ==============================================================================
# 0. ENVIRONMENT DETECTION
# ==============================================================================
.CLUSTER_HOSTNAME_PATTERN <- "meso.umontpellier.fr"
.CLUSTER_ROOT             <- "/home/sabye/scratch_hisafe/MetAIsAFe"

#' Detect whether the current session is running on the cluster.
#' Uses hostname pattern AND SLURM_JOB_ID for robustness on compute nodes.
#' @return Logical.
is_cluster <- function() {
  node     <- tryCatch(Sys.info()["nodename"], error = function(e) "")
  on_slurm <- nzchar(Sys.getenv("SLURM_JOB_ID"))
  grepl(.CLUSTER_HOSTNAME_PATTERN, node, ignore.case = TRUE) || on_slurm
}

.ON_CLUSTER <- is_cluster()

if (.ON_CLUSTER) {
  message("  [ENV] Cluster detected (", Sys.info()["nodename"], ")")
} else {
  message("  [ENV] Local environment")
}

# ==============================================================================
# 0.5 PYTHON ENVIRONMENT (PRE-CONFIG)
# ==============================================================================

#' Configure Python/Conda environment for reticulate
#' 
#' Must be called BEFORE any R package that loads Arrow C++ DLLs (e.g. arrow).
#' Pre-loads pyarrow to avoid DLL conflicts with R's arrow package.
#'
#' @return Invisible NULL
.setup_python_env <- function() {
  
  # Guard: only run once per session
  if (isTRUE(.PYTHON_CONFIGURED)) return(invisible(NULL))
  
  if (.ON_CLUSTER) {
    message("[PYTHON] Skipped on cluster (not needed).")
    .PYTHON_CONFIGURED <<- TRUE
    return(invisible(NULL))
  }
  
  # ── Windows DLL path injection ──────────────────────────────────────────
  if (.Platform$OS.type == "windows") {
    conda_info <- tryCatch(reticulate::conda_list(), error = function(e) NULL)
    target     <- conda_info[conda_info$name == "metaisafe", ]
    
    if (!is.null(target) && nrow(target) > 0) {
      env_path <- fs::path_dir(fs::path_dir(target$python[1]))
      
      pyarrow_dir <- fs::path(env_path, "Lib", "site-packages", "pyarrow")
      numpy_core  <- fs::path(env_path, "Lib", "site-packages", "numpy", ".libs")
      
      dll_paths <- c(
        fs::path(env_path, "Library", "bin"),
        fs::path(env_path, "Library", "lib"),
        pyarrow_dir,
        numpy_core,
        fs::path(env_path, "Scripts"),
        env_path
      )
      
      dll_paths <- dll_paths[fs::dir_exists(dll_paths)]
      
      current_path <- Sys.getenv("PATH")
      Sys.setenv(PATH = paste(c(dll_paths, current_path), collapse = ";"))
      Sys.setenv(METAISAFE_DLL_PATHS = paste(dll_paths, collapse = ";"))
      message("[PYTHON] Windows DLL paths injected for 'metaisafe'")
    }
  }
  
  # ── Conda env selection ─────────────────────────────────────────────────
  Sys.setenv(RETICULATE_PYTHON = "")
  tryCatch({
    reticulate::use_condaenv("metaisafe", required = TRUE)
    message("[PYTHON] \u2705 Conda env 'metaisafe' configured (pre-init)")
    
    # ── Pre-load pyarrow BEFORE R's arrow package pollutes the process ────
    tryCatch({
      dll_paths_str <- Sys.getenv("METAISAFE_DLL_PATHS", "")
      if (nzchar(dll_paths_str)) {
        reticulate::py_run_string(sprintf('
import os
for p in r"%s".split(";"):
    p = p.strip()
    if os.path.isdir(p):
        try:
            os.add_dll_directory(p)
        except (OSError, AttributeError):
            pass
', dll_paths_str))
      }
      
      reticulate::py_run_string("import pyarrow")
      message("[PYTHON] \u2705 pyarrow pre-loaded (before R arrow)")
    }, error = function(e) {
      message("[PYTHON WARNING] Could not pre-load pyarrow: ", e$message)
    })
    
  }, error = function(e) {
    message("[PYTHON WARNING] Conda env 'metaisafe' not found. Fallback to system.")
  })
  
  .PYTHON_CONFIGURED <<- TRUE
  invisible(NULL)
}

# Flag: not yet configured
if (!exists(".PYTHON_CONFIGURED")) .PYTHON_CONFIGURED <- FALSE

# ==============================================================================
# CACHE HELPERS (CLUSTER-SAFE & ATOMIC)
# ==============================================================================

# Place the cache in the Conda environment itself if on cluster, or HOME if local
.get_cache_path <- function() {
  if (.ON_CLUSTER && Sys.getenv("CONDA_PREFIX") != "") {
    return(file.path(Sys.getenv("CONDA_PREFIX"), ".hisafe_pkg_cache.rds"))
  }
  return(file.path(Sys.getenv("HOME"), ".hisafe_pkg_cache.rds"))
}

.CACHE_MAX_AGE_HOURS <- 48

.read_pkg_cache <- function() {
  cache_file <- .get_cache_path()
  if (!file.exists(cache_file)) return(NULL)
  
  age_hours <- as.numeric(difftime(Sys.time(), 
                                   file.mtime(cache_file), 
                                   units = "hours"))
  if (age_hours > .CACHE_MAX_AGE_HOURS) {
    message("  [CACHE] Expired (", round(age_hours, 1), "h) — refreshing...")
    return(NULL)
  }
  
  tryCatch(readRDS(cache_file), error = function(e) NULL)
}

.write_pkg_cache <- function(installed_pkgs) {
  cache_file <- .get_cache_path()
  
  # ATOMIC WRITE: Write to a unique temporary file first, then rename.
  # This prevents file corruption when 50 SLURM jobs try to write the cache simultaneously.
  tmp_file <- paste0(cache_file, ".tmp.", Sys.getpid())
  
  tryCatch({
    saveRDS(installed_pkgs, tmp_file)
    # file.rename is an atomic operation on Linux POSIX systems
    file.rename(tmp_file, cache_file)
  }, error = function(e) {
    if (file.exists(tmp_file)) unlink(tmp_file)
    message("  [CACHE] Could not write cache: ", e$message)
  })
}

.get_installed_packages <- function(force_refresh = FALSE) {
  if (!force_refresh) {
    cached <- .read_pkg_cache()
    if (!is.null(cached)) {
      message("  [CACHE] Using cached package list (", length(cached), " packages)")
      return(cached)
    }
  }
  
  message("  [CACHE] Scanning installed packages (I/O intensive)...")
  pkgs <- installed.packages()[, "Package"]
  .write_pkg_cache(pkgs)
  
  return(pkgs)
}

# ==============================================================================
# 1. PACKAGE DEFINITIONS
# ==============================================================================
.pkg_groups <- list(
  
  # Shared by ALL parts — always loaded
  core = c(
    "fs", "digest",
    "dplyr", "tidyr", "readxl", "openxlsx", #"rlang",
    "stringr", "lubridate", "purrr", "readr",
    "vroom", "data.table",
    "parallel", "progressr", "progress",
    "remotes"
  ),
  
  # Part 1 — Simulation generation
  sim = c(
    "lhs", "qrng", "devtools",
    "furrr", "future", "future.apply", "spacefillr"
  ),
  
  # Part 2 — Extraction
  extract = c(
    "fst"
  ),
  
  dan = c(
    "rpart", "rpart.plot"
  ),
  
  # Part 3 - MetaModelling
  ml = c(
    "caret", "recipes", "ranger", "xgboost", "doParallel",
    "gower", "ModelMetrics", "plyr", "tidymodels",
    "vip", "randomForestSRC", "pheatmap", "randomForest",
    "mgcv", "sf", "sensitivity"
  ),
  
  # Part 4 - Interface
  int = c(
    "shiny", "shinydashboard", "shinythemes", "DT",
    "reticulate"
  ),
  
  # Visualization — local only (skipped automatically on cluster)
  viz = c(
    "ggplot2", "plotly", "patchwork",
    "scales", "tibble",
    "ggridges", "ggnewscale", "soiltexture",
    "terra", "tidyterra", "rnaturalearth", "rnaturalearthdata", "rnaturalearthhires",
    "ggspatial", "RColorBrewer",
    "gsl", "GJRM", "gratia"
  )
)

# Packages requiring binary install on Windows
.complex_pkgs <- c("mgcv", "gsl", "sf", "GJRM", "gratia",
                   "terra", "tidyterra", "rnaturalearth", "rnaturalearthdata", "rnaturalearthhires")

# Packages that make no sense on a headless cluster (no display, no browser)
.local_only_pkgs <- c(
  "plotly", "shiny", "shinydashboard", "shinythemes", "DT",
  "readxl", "openxlsx",
  "terra", "tidyterra", "rnaturalearth", "rnaturalearthdata", "rnaturalearthhires"
)


# ==============================================================================
# 2. STORAGE BACKEND DETECTION
# ==============================================================================
.detect_storage_backend <- function() {
  arrow_available <- tryCatch({
    requireNamespace("arrow", quietly = TRUE) &&
      packageVersion("arrow") >= "10.0.0"
  }, error = function(e) FALSE)
  
  fst_available <- tryCatch(
    requireNamespace("fst", quietly = TRUE),
    error = function(e) FALSE
  )
  
  if (arrow_available) {
    message("  Storage backend: Arrow (Parquet)")
    return("arrow")
  } else if (fst_available) {
    message("  Storage backend: fst (binary, fast)")
    return("fst")
  } else {
    message("  Storage backend: CSV (vroom read / data.table::fwrite write)")
    return("csv")
  }
}


# ==============================================================================
# 3. INSTALL HELPER (STRICT CONDA ENFORCEMENT & ATOMIC CACHE)
# ==============================================================================
.install_if_missing <- function(pkgs, force_refresh = FALSE) {
  
  # ----------------------------------------------------------------------------
  # CLUSTER BEHAVIOR: STRICT ENFORCEMENT
  # ----------------------------------------------------------------------------
  if (.ON_CLUSTER) {
    # Skip GUI/viz packages that are useless on headless nodes
    skipped <- intersect(pkgs, .local_only_pkgs)
    if (length(skipped) > 0) {
      message("  [CLUSTER] Skipping GUI/viz packages: ", paste(skipped, collapse = ", "))
    }
    
    pkgs_to_check <- setdiff(pkgs, .local_only_pkgs)
    
    # Read installed packages using the cluster-safe atomic cache
    installed <- .get_installed_packages(force_refresh)
    missing <- setdiff(pkgs_to_check, installed)
    
    # Fatal error if packages are missing (forces the user to use Conda)
    if (length(missing) > 0) {
      stop(sprintf(
        "\n  [FATAL ERROR] Missing packages on cluster: %s\n  Please install them manually using: conda install -c conda-forge r-<pkgname>",
        paste(missing, collapse = ", ")
      ))
    }
    
    message("  [CLUSTER] All required packages are present in the Conda environment.")
    return(invisible(NULL))
  }
  
  # ----------------------------------------------------------------------------
  # LOCAL BEHAVIOR: NATIVE R INSTALLATION (WINDOWS/MAC)
  # ----------------------------------------------------------------------------
  # Ensure local custom library is added to search path if defined
  local_lib <- Sys.getenv("R_LIBS_USER")
  if (local_lib != "" && dir.exists(local_lib)) {
    .libPaths(c(local_lib, .libPaths()))
  }
  
  # Set default CRAN repository
  options(repos = c(CRAN = "https://cloud.r-project.org"))
  
  # Read installed packages using the cache
  installed <- .get_installed_packages(force_refresh)
  
  missing_complex  <- setdiff(intersect(.complex_pkgs, pkgs), installed)
  missing_standard <- setdiff(setdiff(pkgs, .complex_pkgs), installed)
  
  if (length(missing_complex) == 0 && length(missing_standard) == 0) {
    message("  All requested packages are already installed (cache hit).")
    return(invisible(NULL))
  }
  
  # Install missing complex packages (Windows binaries)
  if (.Platform$OS.type == "windows" && length(missing_complex) > 0) {
    message("  Installing complex packages (Windows binaries): ", paste(missing_complex, collapse = ", "))
    install.packages(missing_complex, type = "binary", dependencies = TRUE)
    missing_complex <- character(0) # Clear to prevent double installation below
  }
  
  # Combine remaining missing packages
  all_missing <- c(missing_complex, missing_standard)
  if (length(all_missing) > 0) {
    message("  Installing missing packages: ", paste(all_missing, collapse = ", "))
    install.packages(all_missing, dependencies = TRUE)
  }
  
  # Invalidate and refresh the cache after new installations
  .write_pkg_cache(installed.packages()[, "Package"])
  message("  [CACHE] Package list refreshed after installation.")
}

# ==============================================================================
# 4. LOAD HELPER
# ==============================================================================
.load_packages <- function(pkgs) {
  if (.ON_CLUSTER) pkgs <- setdiff(pkgs, .local_only_pkgs)
  
  # Ne charger que ce qui n'est pas déjà attaché
  already_loaded <- sub("^package:", "", grep("^package:", search(), value = TRUE))
  to_load <- setdiff(pkgs, already_loaded)
  
  if (length(to_load) == 0) {
    message("  All packages already loaded — skipping.")
    return(invisible(NULL))
  }
  
  message(sprintf("  Loading %d/%d packages (rest already attached)...",
                  length(to_load), length(pkgs)))
  
  results <- vapply(to_load, function(pkg) {
    tryCatch({
      suppressWarnings(suppressPackageStartupMessages(library(pkg, character.only = TRUE)))
      TRUE
    }, error = function(e) {
      message(sprintf("  [WARNING] Failed to load: %s — %s", pkg, e$message))
      FALSE
    })
  }, logical(1))
  
  failed <- names(results)[!results]
  if (length(failed) > 0)
    warning("Packages could not be loaded: ", paste(failed, collapse = ", "))
  
  invisible(results)
}


# ==============================================================================
# 5. PATH RESOLVER
# ==============================================================================

#' Resolve project paths for the current environment (local or cluster).
#'
#' On the cluster, \code{root_dir} is overridden by \code{.CLUSTER_ROOT}
#' regardless of the working directory. On a local machine, \code{root_dir}
#' defaults to \code{"."} (project root via here::here() or working dir).
#'
#' @param root_dir String. Local root directory (ignored on cluster).
#' @return Named list of all project paths.
#' @export
get_hisafe_paths <- function(root_dir = ".") {
  
  if (exists(".ON_CLUSTER") && .ON_CLUSTER) root_dir <- .CLUSTER_ROOT
  
  inputs      <- fs::path(root_dir, "00_Inputs")
  scripts     <- fs::path(root_dir, "01_Scripts")
  models      <- fs::path(root_dir, "03_Models")
  templates   <- fs::path(inputs,   "templates")
  gen_params  <- fs::path(templates, "general_base_parameters")
  base_tpl    <- fs::path(templates, "base_template")
  exp_design  <- fs::path(inputs,   "experimental_design")
  weather     <- fs::path(inputs,   "weather")
  
  list(
    root_dir         = root_dir,
    inputs_dir       = inputs,
    
    scripts_dir      = scripts,
    R_scripts_dir    = fs::path(scripts, "R"),
    Py_scripts_dir   = fs::path(scripts, "Python"),
    scripts_simGen   = fs::path(scripts, "R/01_SimGeneration"),
    scripts_DataExt  = fs::path(scripts, "R/02_DataExtraction"),
    scripts_Shiny    = fs::path(scripts, "R/04_Interface"),
    # scripts_DataAn   = fs::path(scripts, "R/03_DataAnalysis"),
    # scripts_ML       = fs::path(scripts, "R/04_MetaModelling"),
    
    simulations_dir  = fs::path(root_dir, "02_Simulations"),
    
    models_dir       = models,
    
    templates_dir         = templates,
    general_params        = gen_params,
    cropInterventions_dir = fs::path(gen_params, "cropInterventions"),
    
    inter_crop_tec        = "weed-restinclieres-btwTrees.tec",
    
    cropSpecies_dir       = fs::path(gen_params, "cropSpecies"),
    treeInterventions_dir = fs::path(gen_params, "treeInterventions"),
    treeSpecies_dir       = fs::path(gen_params, "treeSpecies"),
    
    base_pld    = fs::path(base_tpl, "base_template_A2.pld"),
    base_sim    = fs::path(base_tpl, "base_template_A2.sim"),
    base_export = fs::path(base_tpl, "export.out"),
    
    weather_dir     = weather,
    drias_dir       = fs::path(weather, "raw"),
    default_weather = fs::path(weather, "default_climate.wth"),
    
    climate_sources_csv = fs::path(exp_design, "climate_sources.csv"),
    rotations_csv       = fs::path(exp_design, "rotations_plans.csv"),
    periods_csv         = fs::path(exp_design, "periods_configuration.csv"),
    climate_csv         = fs::path(exp_design, "climate_rules.csv"),
    constraints_csv     = fs::path(exp_design, "constraints_parameters.csv"),
    crop_tec_map_csv    = fs::path(exp_design, "crop_tec_map.csv"),
    tree_config_csv     = fs::path(exp_design, "tree_files_config.csv")
    
  )
}

# ==============================================================================
# 6. MAIN ENTRY POINT
# ==============================================================================

#' Setup Project Environment
#'
#' Installs and loads packages for the requested project parts, initialises
#' the folder structure, loads utility functions, and configures the storage
#' backend.
#'
#' Behaviour differs automatically between local and cluster environments:
#'   - Cluster : uses \code{~/R/library} as personal package library, skips
#'     GUI/viz-only packages, resolves paths from \code{.CLUSTER_ROOT}.
#'   - Local   : uses default \code{.libPaths()}, loads all requested packages.
#'
#' @param load_parts Character vector. Parts to load.
#'   Options: \code{"core"} (always included), \code{"sim"}, \code{"ml"},
#'   \code{"viz"} (silently ignored on cluster).
#' @param root_dir String. Project root (ignored on cluster, overridden by
#'   \code{.CLUSTER_ROOT}).
#' @param load_functions Logical. Source .R files from Functions/ subfolders?
#' @param functions_dir String or NULL. Override auto-detected functions path.
#'
#' @return Invisibly returns \code{USE_ARROW} (logical).
#' @export
setup_project <- function(load_parts     = c("core"),
                          root_dir       = ".",
                          load_functions = TRUE,
                          force_pkg_refresh = FALSE,
                          functions_dir  = NULL) {
  
  # On cluster, viz part is silently dropped
  if (.ON_CLUSTER && "viz" %in% load_parts) {
    message("  [CLUSTER] 'viz' part skipped (headless environment).")
    load_parts <- setdiff(load_parts, "viz")
  }
  
  # ===========================================================================
  # STEP 0.5: PYTHON (only when needed — must run BEFORE arrow loads)
  # ===========================================================================
  if ("int" %in% load_parts) {
    message("\n=== STEP 0.5: CONFIGURING PYTHON ===")
    .setup_python_env()
  }
  
  # ===========================================================================
  # STEP 1: PACKAGES
  # ===========================================================================
  message("\n=== STEP 1: CHECKING PACKAGES ===")
  
  parts_to_load <- unique(c("core", load_parts))
  pkgs_to_load  <- unique(unlist(.pkg_groups[parts_to_load], use.names = FALSE))
  
  .install_if_missing(pkgs_to_load, force_refresh = force_pkg_refresh)
  .load_packages(pkgs_to_load)
  
  # ===========================================================================
  # STEP 2: STORAGE BACKEND
  # ===========================================================================
  message("\n=== STEP 2: CONFIGURING STORAGE BACKEND ===")
  detected_backend <- .detect_storage_backend()
  
  # ===========================================================================
  # STEP 3: PROJECT FOLDER STRUCTURE
  # ===========================================================================
  message("\n=== STEP 3: RESOLVING PATHS ===")
  
  effective_root <- if (.ON_CLUSTER) .CLUSTER_ROOT else root_dir
  paths <- get_hisafe_paths(effective_root)
  
  message("  Root : ", paths$root_dir)
  
  # ===========================================================================
  # STEP 4: LOAD UTILITY FUNCTIONS
  # ===========================================================================
  if (load_functions) {
    message("\n=== STEP 4: LOADING UTILITY FUNCTIONS ===")
    
    # Part sim — loaded via package
    if ("sim" %in% load_parts) {
      
      pkg_path <- fs::path(paths$scripts_simGen, "hisafeGen")
      
      if (fs::is_dir(pkg_path)) {
        
        # Compute hash of source files to detect changes 
        # (readLines strips line endings, preventing Windows/Linux CRLF conflicts)
        r_files  <- list.files(pkg_path, pattern = "\\.(R|DESCRIPTION|NAMESPACE)$", 
                               recursive = TRUE, full.names = TRUE, ignore.case = TRUE)
        src_hash <- digest::digest(lapply(r_files, function(f) {
          tryCatch(readLines(f, warn = FALSE), error = function(e) "")
        }), algo = "md5")
        
        # Save cache file outside the package directory to keep the repo clean
        cache_f     <- fs::path(paths$scripts_simGen, ".hisafeGen_build_hash")
        cached_hash <- tryCatch(readLines(cache_f, warn = FALSE), error = function(e) "")
        
        if (src_hash == cached_hash && requireNamespace("hisafeGen", quietly = TRUE)) {
          suppressPackageStartupMessages(library(hisafeGen))
          message("  [OK] hisafeGen loaded (cache hit, no rebuild needed)")
          
        } else {
          message("  [BUILD] hisafeGen sources changed or missing — compiling...")
          tryCatch({
            devtools::document(pkg_path, quiet = TRUE)
            devtools::install(pkg_path, quiet = TRUE, upgrade = "never")
            suppressPackageStartupMessages(library(hisafeGen))
            
            # Update cache ONLY if installation is successful
            writeLines(src_hash, cache_f) 
            message("  [OK] hisafeGen successfully compiled and loaded")
          }, error = function(e) {
            stop("\n  [FATAL ERROR] Failed to compile hisafeGen: ", e$message)
          })
        }
        
      } else if (requireNamespace("hisafeGen", quietly = TRUE)) {
        suppressPackageStartupMessages(library(hisafeGen))
        message("  [OK] hisafeGen loaded (from Conda library, source dir missing)")
      } else {
        stop("\n  [FATAL ERROR] hisafeGen not found in library and source directory missing at: ", pkg_path)
      }
    }
      
    if ("extract" %in% load_parts) {
      extract_dir <- file.path(paths$scripts_DataExt,"Functions")
      if (dir.exists(extract_dir)) {
        func_files <- list.files(extract_dir, pattern = "\\.R$",
                                 full.names = TRUE, recursive = TRUE)
        results <- vapply(func_files, function(f) {
          tryCatch({ source(f); TRUE },
                   error = function(e) {
                     message("  [ERROR] ", basename(f), ": ", e$message)
                     FALSE
                   })
        }, logical(1))
        message(sprintf("  [extract] %d script(s) loaded, %d failed",
                        sum(results), sum(!results)))
      } else {
        warning("DataExtraction Functions/ not found at: ", extract_dir)
      }
    }
    
    if ("dan" %in% load_parts) {
      dan_dir <- file.path(paths$scripts_DataAn,"Functions")
      if (dir.exists(dan_dir)) {
        func_files <- list.files(dan_dir, pattern = "\\.R$",
                                 full.names = TRUE, recursive = TRUE)
        results <- vapply(func_files, function(f) {
          tryCatch({ source(f); TRUE },
                   error = function(e) {
                     message("  [ERROR] ", basename(f), ": ", e$message)
                     FALSE
                   })
        }, logical(1))
        message(sprintf("  [dan] %d script(s) loaded, %d failed",
                        sum(results), sum(!results)))
      } else {
        warning("DataAnalysis Functions/ not found at: ", dan_dir)
      }
    }
    
    if ("ml" %in% load_parts) {
      ml_dir <- file.path(paths$scripts_ML,"Functions")
      if (dir.exists(ml_dir)) {
        func_files <- list.files(ml_dir, pattern = "\\.R$",
                                 full.names = TRUE, recursive = TRUE)
        results <- vapply(func_files, function(f) {
          tryCatch({ source(f); TRUE },
                   error = function(e) {
                     message("  [ERROR] ", basename(f), ": ", e$message)
                     FALSE
                   })
        }, logical(1))
        message(sprintf("  [ml] %d script(s) loaded, %d failed",
                        sum(results), sum(!results)))
      } else {
        warning("MetaModelling Functions/ not found at: ", ml_dir)
      }
    }
    
    if ("int" %in% load_parts) {
      int_R_dir <- file.path(paths$scripts_Shiny,"R")
      if (dir.exists(int_R_dir)) {
        func_files <- list.files(int_R_dir, pattern = "\\.R$",
                                 full.names = TRUE, recursive = TRUE)
        results <- vapply(func_files, function(f) {
          tryCatch({ source(f); TRUE },
                   error = function(e) {
                     message("  [ERROR] ", basename(f), ": ", e$message)
                     FALSE
                   })
        }, logical(1))
        message(sprintf("  [int] %d script(s) loaded, %d failed",
                        sum(results), sum(!results)))
      } else {
        warning("ShinyInterface R/ not found at: ", int_R_dir)
      }
      int_mod_dir <- file.path(paths$scripts_Shiny,"modules")
      if (dir.exists(int_mod_dir)) {
        func_files <- list.files(int_mod_dir, pattern = "\\.R$",
                                 full.names = TRUE, recursive = TRUE)
        results <- vapply(func_files, function(f) {
          tryCatch({ source(f); TRUE },
                   error = function(e) {
                     message("  [ERROR] ", basename(f), ": ", e$message)
                     FALSE
                   })
        }, logical(1))
        message(sprintf("  [int] %d script(s) loaded, %d failed",
                        sum(results), sum(!results)))
      } else {
        warning("ShinyInterface modules/ not found at: ", int_mod_dir)
      }
    }
  }
  
  # ===========================================================================
  # READY
  # ===========================================================================
  message("\n=================================================")
  message("  ENVIRONMENT  : ", if (.ON_CLUSTER) "CLUSTER" else "LOCAL")
  message("  PROJECT ROOT : ", effective_root)
  message("  PARTS LOADED : ", paste(parts_to_load, collapse = ", "))
  message("  STORAGE      : ",
          detected_backend)
  message("=================================================\n")
  
  invisible(detected_backend)
}
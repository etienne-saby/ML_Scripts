# ==============================================================================
# MetAIsAFe — R/utils_python.R
# ==============================================================================

#' Initialize Python Environment and Import MetAIsAFe Modules
#'
#' Assumes Conda environment AND pyarrow are pre-configured in 00_setup.R.
#' Only imports modules and adds scripts to sys.path.
#'
#' @param Py_scripts_dir Path to Python scripts root
#' @return Invisibly NULL
#' @export
init_python <- function(Py_scripts_dir) {
  
  # ── 1. Vérifier que Python est bien configuré ──────────────────────────
  py_config <- reticulate::py_config()
  message("[PYTHON] Using Python ", py_config$version, " (", py_config$python, ")")
  
  # ── 2. Add MetAIsAFe scripts to sys.path ────────────────────────────────
  scripts_dir <- fs::path(Py_scripts_dir, "03_ML")
  
  if (!fs::dir_exists(scripts_dir)) {
    stop("[PYTHON FATAL] MetAIsAFe scripts not found at: ", scripts_dir, call. = FALSE)
  }
  
  scripts_dir_norm <- fs::path_norm(scripts_dir)
  
  sys <- reticulate::import("sys", convert = FALSE)
  
  if (!scripts_dir_norm %in% reticulate::py_to_r(sys$path)) {
    sys$path$insert(0L, scripts_dir_norm)
    message("[PYTHON] Added to sys.path: ", scripts_dir_norm)
  }
  
  # ── 3. Import MetAIsAFe modules ──────────────────────────────────────────
  tryCatch({
    CONFIG    <- reticulate::import("config", delay_load = FALSE)
    PREDICTOR <- reticulate::import("modeling.predictor", delay_load = FALSE)
    
    assign("CONFIG",    CONFIG,    envir = parent.frame())
    assign("PREDICTOR", PREDICTOR, envir = parent.frame())
    
    message("[PYTHON] \u2705 MetAIsAFe modules imported successfully.")
    
  }, error = function(e) {
    stop(
      "[PYTHON FATAL] Failed to import MetAIsAFe modules.\n",
      "Error: ", conditionMessage(e), "\n\n",
      "Diagnostic steps:\n",
      "  1. Verify Python path: ", py_config$python, "\n",
      "  2. Test in console:\n",
      "       reticulate::use_condaenv('metaisafe', required = TRUE)\n",
      "       reticulate::py_run_string('import modeling.predictor')\n",
      "  3. Check dependencies:\n",
      "       conda activate metaisafe\n",
      "       conda list | grep -E '(sklearn|pyarrow|lightgbm)'",
      call. = FALSE
    )
  })
  
  invisible(NULL)
}

#' Convert a Python Path object to an R character string.
#'
#' reticulate returns Python pathlib.Path objects for CampaignPaths fields.
#' R file functions (file.exists, list.files, etc.) require character strings.
#'
#' @param py_path A Python Path object or R character string.
#' @return Character string.
#' @export
py_path_to_r <- function(py_path) {
  if (is.character(py_path)) return(py_path)
  as.character(py_path)
}
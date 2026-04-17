# ==============================================================================
# MetAIsAFe — app.R
# ==============================================================================
# Entry point for the MetAIsAFe interactive Shiny dashboard.
#
# PURPOSE
#   Interactive exploration of 40-year agroforestry scenario predictions
#   produced by the MetAIsAFe LightGBM meta-modelling pipeline (v5.0).
#
# TABS
#   1. Prediction & Positioning  — scenario inputs + cascade routing output
#   2. Temporal Exploration      — 40-year trajectory curves
#   3. Model Quality             — per-model metrics & CV results
#   4. SHAP Analysis             — global feature importance (beeswarm + bar)
#   5. Documentation             — inline Markdown reference
#
# ARCHITECTURE
#   app.R                — this file: environment setup, UI, server
#   R/utils_python.R     — Python/reticulate initialisation helper
#   modules/mod_*.R      — Shiny modules (one per tab)
#   R/utils_formulas.R   — analytical ratio helpers (RR_*, LER_*)
#   R/utils_plots.R      — shared ggplot2 / plotly helpers
#   www/custom.css       — custom styles
#   www/documentation.md — inline documentation
#
# USAGE
#   shiny::runApp("01_Scripts/R/04_Interface")
#
# PYTHON DEPENDENCY
#   A Python >=3.9 environment with MetAIsAFe dependencies must be
#   reachable by reticulate (venv or Conda).  See R/utils_python.R.
#
# SOBOL REFERENCE FILE  (optional but recommended)
#   ${campaign}/Data/Sobol/sobol_predictions.csv
#   Expected columns: year, target, value, [percentile_*]
#   Format: long — one row per (SimID × year × target).
#   Used for percentile positioning of user scenarios.
#
# AUTHOR  : Étienne SABY
# UPDATED : 2026-05
# ==============================================================================

# ── 0. SETUP ──────────────────────────────────────────────────────────────────
if (!requireNamespace("here", quietly = TRUE)) install.packages("here")
library(here)

source(here::here("01_Scripts/R/00_setup.R"))

paths <- get_hisafe_paths(root_dir = here::here())

setup_project(
  root_dir       = paths$root_dir,
  load_parts     = c("core", "int", "viz"),
  load_functions = TRUE
)

# ── 1. PYTHON ─────────────────────────────────────────────────────────────────
init_python(Py_scripts_dir = paths$Py_scripts_dir)

# ── 2. CAMPAIGNS ──────────────────────────────────────────────────────────────
# Découverte simplifiée avec fs et purrr
.discover_campaigns <- function(models_dir) {
  if (!fs::dir_exists(models_dir)) return(character(0))
  
  fs::dir_ls(models_dir, type = "directory") |>
    purrr::keep(~ {
      mm_dir <- fs::path(.x, "MetaModels")
      fs::dir_exists(mm_dir) && length(fs::dir_ls(mm_dir, glob = "*.joblib")) > 0
    }) |>
    fs::path_file()
}

AVAILABLE_CAMPAIGNS <- .discover_campaigns(paths$models_dir)
DEFAULT_CAMPAIGN    <- if (length(AVAILABLE_CAMPAIGNS) > 0) AVAILABLE_CAMPAIGNS[1] else "sobol_training_1_n2048"

# ── 3. LOADER ─────────────────────────────────────────────────────────────────
.load_climate_surrogates <- function(campaign) {
  joblib <- reticulate::import("joblib")
  
  climate_targets <- c(
    "GDD_cycle_AF", "ETP_cycle_AF", "precipitation_AF",
    "frost_events_cycle_AF", "globalRadiation_AF",
    "maxTemperature_extreme_AF", "minTemperature_extreme_AF"
  )
  
  metamodels_dir <- tryCatch(
    as.character(campaign$metamodels_dir),  # reticulate le convertit
    error = function(e) {
      warning("[CLIMATE] Cannot convert campaign$metamodels_dir: ", e$message)
      NULL
    }
  )
  
  if (is.null(metamodels_dir) || !dir.exists(metamodels_dir)) {
    warning("[CLIMATE] metamodels_dir does not exist: ", metamodels_dir)
    return(list())
  }
  
  message("[CLIMATE] Loading surrogates from: ", metamodels_dir)
  
  models <- list()
  for (target in climate_targets) {
    path <- file.path(metamodels_dir, paste0("climate_surrogate_", target, ".joblib"))

    if (file.exists(path)) {
      models[[target]] <- joblib$load(path)
    } else {
      warning("[CLIMATE] ❌ Not found: ", target)
    }
  }
  
  message("[CLIMATE] Total surrogates loaded: ", length(models), " / ", length(climate_targets))
  models
}
load_campaign_models <- function(campaign_name) {
  message("[MODELS] Loading campaign: ", campaign_name)
  campaign <- CONFIG$get_campaign_paths(campaign_name)
  
  models <- PREDICTOR$load_all_models(campaign)
  clfs   <- PREDICTOR$load_all_classifiers(campaign)
  climate_models <- .load_climate_surrogates(campaign)
  
  # Stunted model avec fs
  stunted_path <- fs::path(campaign$metamodels_dir, "stunted_model.joblib")
  stunted_model <- if (fs::file_exists(stunted_path)) {
    reticulate::import("joblib")$load(stunted_path)
  } else NULL
  
  # Sobol ref avec vroom
  sobol_path <- fs::path(campaign$sobol_data_dir, "sobol_predictions.csv")
  sobol_ref  <- if (fs::file_exists(sobol_path)) {
    vroom::vroom(sobol_path, show_col_types = FALSE)
  } else NULL
  
  list(
    campaign      = campaign,
    campaign_name = campaign_name,
    models        = models,
    clf1          = clfs[[1]],
    clf2          = clfs[[2]],
    stunted_model = stunted_model,
    climate_models= climate_models,
    sobol_ref     = sobol_ref
  )
}

INITIAL_STATE <- load_campaign_models(DEFAULT_CAMPAIGN)

# ── 4. UI ─────────────────────────────────────────────────────────────────────
ui <- shinydashboard::dashboardPage(
  skin = "blue",
  shinydashboard::dashboardHeader(title = "MetAIsAFe Dashboard", titleWidth = 300),
  
  shinydashboard::dashboardSidebar(
    width = 300,
    shinydashboard::sidebarMenu(
      id = "sidebar_menu",
      shinydashboard::menuItem("Predictions", tabName = "tab_prediction", icon = icon("calculator")),
      shinydashboard::menuItem("Trajectories", tabName = "tab_trajectories", icon = icon("chart-line")),
      shinydashboard::menuItem("Quality", tabName = "tab_quality", icon = icon("check-circle")),
      shinydashboard::menuItem("SHAP", tabName = "tab_shap", icon = icon("magnifying-glass"))
    ),
    hr(),
    div(style = "padding: 0 15px;",
        selectInput("selected_campaign", "Training Campaign", choices = AVAILABLE_CAMPAIGNS, selected = DEFAULT_CAMPAIGN))
  ),
  
  shinydashboard::dashboardBody(
    shinydashboard::tabItems(
      shinydashboard::tabItem(tabName = "tab_prediction", predictionUI("mod_prediction")),
      shinydashboard::tabItem(tabName = "tab_trajectories", trajectoriesUI("mod_trajectories")),
      shinydashboard::tabItem(tabName = "tab_quality", qualityUI("mod_quality")),
      shinydashboard::tabItem(tabName = "tab_shap", shapUI("mod_shap"))
    )
  )
)

# ── 5. SERVER ─────────────────────────────────────────────────────────────
server <- function(input, output, session) {
  state <- reactiveValues(
    campaign_name   = INITIAL_STATE$campaign_name,
    campaign        = INITIAL_STATE$campaign,
    models          = INITIAL_STATE$models,
    clf1            = INITIAL_STATE$clf1,
    clf2            = INITIAL_STATE$clf2,
    stunted_model   = INITIAL_STATE$stunted_model,
    climate_models  = INITIAL_STATE$climate_models,
    sobol_ref       = INITIAL_STATE$sobol_ref,
    last_prediction = NULL,
    last_params     = NULL,
    scenario_stack  = list()
  )
  
  observeEvent(input$selected_campaign, {
    req(input$selected_campaign != state$campaign_name)
    id <- showNotification(paste("Loading", input$selected_campaign, "..."), duration = NULL)
    on.exit(removeNotification(id))
    
    res <- tryCatch(load_campaign_models(input$selected_campaign), error = function(e) NULL)
    if (!is.null(res)) {
      state$campaign_name <- res$campaign_name
      state$campaign      <- res$campaign
      state$models        <- res$models
      state$clf1          <- res$clf1
      state$clf2          <- res$clf2
      state$stunted_model <- res$stunted_model
      state$climate_models<- res$climate_models
      state$sobol_ref     <- res$sobol_ref
      state$last_prediction <- NULL
      state$last_params     <- NULL
      state$scenario_stack  <- list()
    }
  })

  predictionServer("mod_prediction", state, PREDICTOR)
  trajectoriesServer("mod_trajectories", state, PREDICTOR)
  qualityServer("mod_quality", state)
  shapServer("mod_shap", state)
}

shinyApp(ui, server)

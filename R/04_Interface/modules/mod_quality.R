# ==============================================================================
# MODULE : Qualité des Modèles
# ==============================================================================

qualityUI <- function(id) {
  ns <- NS(id)
  
  fluidRow(
    column(
      width = 12,
      box(
        title = "Table qualité pipeline", status = "primary", solidHeader = TRUE,
        width = NULL,
        DT::dataTableOutput(ns("quality_table"))
      )
    ),
    column(
      width = 6,
      box(
        title = "CLF1 — Tree Status", status = "info", solidHeader = TRUE,
        width = NULL,
        DT::dataTableOutput(ns("clf1_metrics"))
      )
    ),
    column(
      width = 6,
      box(
        title = "CLF2 — Yield Failure", status = "info", solidHeader = TRUE,
        width = NULL,
        DT::dataTableOutput(ns("clf2_metrics"))
      )
    ),
    column(
      width = 12,
      box(
        title = "Stage 1 — carbonStem (horizons)", status = "success", solidHeader = TRUE,
        width = NULL, collapsible = TRUE,
        DT::dataTableOutput(ns("stage1_metrics"))
      )
    ),
    column(
      width = 12,
      box(
        title = "Stage 2 — yield (row-by-row)", status = "warning", solidHeader = TRUE,
        width = NULL, collapsible = TRUE,
        DT::dataTableOutput(ns("stage2_metrics"))
      )
    ),
    column(
      width = 12,
      box(
        title = "Cross-Validation", status = "info", solidHeader = TRUE,
        width = NULL, collapsible = TRUE, collapsed = TRUE,
        DT::dataTableOutput(ns("cv_metrics"))
      )
    )
  )
}

qualityServer <- function(id, shared_state) {
  moduleServer(id, function(input, output, session) {
    
    # ── Safe path resolver ─────────────────────────────────────────────────
    campaign_dir <- reactive({
      req(shared_state$campaign)
      py_path_to_r(shared_state$campaign$campaign_dir)
    })
    
    metrics_dir <- reactive(file.path(campaign_dir(), "Data", "Metrics"))
    cv_dir      <- reactive(file.path(campaign_dir(), "Data", "CV"))
    metamodels_dir <- reactive(file.path(campaign_dir(), "MetaModels"))
    
    # ── Pipeline summary ───────────────────────────────────────────────────
    pipeline_summary <- reactive({
      path <- file.path(metrics_dir(), "pipeline_summary.json")
      if (file.exists(path)) jsonlite::fromJSON(path) else NULL
    })
    
    output$quality_table <- DT::renderDataTable({
      ps <- pipeline_summary()
      if (is.null(ps)) {
        return(DT::datatable(
          data.frame(Message = "pipeline_summary.json introuvable"),
          options = list(dom = "t"), rownames = FALSE
        ))
      }
      DT::datatable(
        data.frame(Métrique = names(ps), Valeur = as.character(unlist(ps)),
                   stringsAsFactors = FALSE),
        options = list(dom = "t", paging = FALSE), rownames = FALSE
      )
    })
    
    # ── CLF metrics (extracted from joblib metadata) ───────────────────────
    .read_clf_meta <- function(filename) {
      path <- file.path(metamodels_dir(), filename)
      if (!file.exists(path)) return(NULL)
      tryCatch({
        joblib <- reticulate::import("joblib")
        obj    <- joblib$load(path)
        # save_model stores metadata in model.__metaisafe_meta__
        meta <- obj$`__metaisafe_meta__`
        if (is.null(meta)) return(NULL)
        as.list(meta)
      }, error = function(e) {
        # Fallback: try JSON files
        json_path <- file.path(metrics_dir(), 
                               gsub("\\.joblib$", ".json", filename))
        if (file.exists(json_path)) jsonlite::fromJSON(json_path) else NULL
      })
    }
    
    output$clf1_metrics <- DT::renderDataTable({
      meta <- .read_clf_meta("clf1_tree_fail.joblib")
      if (is.null(meta)) {
        return(DT::datatable(
          data.frame(Message = "Métriques CLF1 introuvables"),
          options = list(dom = "t"), rownames = FALSE
        ))
      }
      df <- data.frame(
        Métrique = names(meta),
        Valeur = sapply(meta, function(x) {
          if (is.numeric(x)) round(x, 4) else as.character(x)
        }),
        stringsAsFactors = FALSE
      )
      DT::datatable(df, options = list(dom = "t", paging = FALSE), rownames = FALSE)
    })
    
    output$clf2_metrics <- DT::renderDataTable({
      meta <- .read_clf_meta("clf2_yield_fail.joblib")
      if (is.null(meta)) {
        return(DT::datatable(
          data.frame(Message = "Métriques CLF2 introuvables"),
          options = list(dom = "t"), rownames = FALSE
        ))
      }
      df <- data.frame(
        Métrique = names(meta),
        Valeur = sapply(meta, function(x) {
          if (is.numeric(x)) round(x, 4) else as.character(x)
        }),
        stringsAsFactors = FALSE
      )
      DT::datatable(df, options = list(dom = "t", paging = FALSE), rownames = FALSE)
    })
    
    # ── Stage 1 metrics (from joblib metadata) ─────────────────────────────
    output$stage1_metrics <- DT::renderDataTable({
      mm_dir <- metamodels_dir()
      if (!dir.exists(mm_dir)) {
        return(DT::datatable(
          data.frame(Message = "Dossier MetaModels introuvable"),
          options = list(dom = "t"), rownames = FALSE
        ))
      }
      
      files <- list.files(mm_dir, pattern = "lgbm_carbonStem_.*_h\\d+\\.joblib",
                          full.names = TRUE)
      if (length(files) == 0) {
        # Fallback to JSON metrics
        json_files <- list.files(metrics_dir(), 
                                 pattern = "metrics_carbonStem.*\\.json",
                                 full.names = TRUE)
        if (length(json_files) == 0) {
          return(DT::datatable(
            data.frame(Message = "Aucune métrique Stage 1"),
            options = list(dom = "t"), rownames = FALSE
          ))
        }
        df <- do.call(rbind, lapply(json_files, function(f) {
          m <- jsonlite::fromJSON(f)
          data.frame(
            model     = tools::file_path_sans_ext(basename(f)),
            r2_test   = m$test_r2 %||% NA_real_,
            rmse_test = m$test_rmse %||% NA_real_,
            stringsAsFactors = FALSE
          )
        }))
      } else {
        joblib <- reticulate::import("joblib")
        df <- do.call(rbind, lapply(files, function(f) {
          tryCatch({
            obj  <- joblib$load(f)
            meta <- as.list(obj$`__metaisafe_meta__`)
            data.frame(
              target       = meta$target %||% "?",
              horizon      = meta$horizon %||% NA_integer_,
              r2_test      = meta$r2_test %||% NA_real_,
              spearman_rho = meta$spearman_rho %||% NA_real_,
              n_features   = meta$n_features %||% NA_integer_,
              stringsAsFactors = FALSE
            )
          }, error = function(e) NULL)
        }))
      }
      
      if (is.null(df) || nrow(df) == 0) {
        return(DT::datatable(
          data.frame(Message = "Impossible de lire les métadonnées"),
          options = list(dom = "t"), rownames = FALSE
        ))
      }
      
      DT::datatable(df, options = list(pageLength = 20, dom = "tp"), rownames = FALSE) %>%
        DT::formatRound(
          columns = intersect(c("r2_test", "rmse_test", "spearman_rho"), names(df)),
          digits = 3
        )
    })
    
    # ── Stage 2 metrics ────────────────────────────────────────────────────
    output$stage2_metrics <- DT::renderDataTable({
      dir <- metrics_dir()
      if (!dir.exists(dir)) {
        return(DT::datatable(
          data.frame(Message = "Dossier Metrics introuvable"),
          options = list(dom = "t"), rownames = FALSE
        ))
      }
      
      files <- list.files(dir, pattern = "metrics_yield_(AF|TA)_rowwise\\.json",
                          full.names = TRUE)
      if (length(files) == 0) {
        return(DT::datatable(
          data.frame(Message = "Aucune métrique Stage 2"),
          options = list(dom = "t"), rownames = FALSE
        ))
      }
      
      df <- do.call(rbind, lapply(files, function(f) {
        m <- jsonlite::fromJSON(f)
        data.frame(
          model     = basename(tools::file_path_sans_ext(f)),
          r2_test   = m$test_r2 %||% NA_real_,
          rmse_test = m$test_rmse %||% NA_real_,
          mae_test  = m$test_mae %||% NA_real_,
          stringsAsFactors = FALSE
        )
      }))
      
      DT::datatable(df, options = list(dom = "t", paging = FALSE), rownames = FALSE) %>%
        DT::formatRound(columns = c("r2_test", "rmse_test", "mae_test"), digits = 3)
    })
    
    # ── CV results ─────────────────────────────────────────────────────────
    output$cv_metrics <- DT::renderDataTable({
      dir <- cv_dir()
      if (!dir.exists(dir)) {
        return(DT::datatable(
          data.frame(Message = "Dossier CV introuvable"),
          options = list(dom = "t"), rownames = FALSE
        ))
      }
      
      files <- list.files(dir, pattern = "\\.json$", full.names = TRUE)
      if (length(files) == 0) {
        return(DT::datatable(
          data.frame(Message = "Aucun résultat CV"),
          options = list(dom = "t"), rownames = FALSE
        ))
      }
      
      df <- do.call(rbind, lapply(files, function(f) {
        m <- jsonlite::fromJSON(f)
        data.frame(
          model       = basename(tools::file_path_sans_ext(f)),
          mean_r2_val = m$mean_r2_val %||% NA_real_,
          std_r2_val  = m$std_r2_val %||% NA_real_,
          stringsAsFactors = FALSE
        )
      }))
      
      DT::datatable(df, options = list(dom = "t", paging = FALSE), rownames = FALSE) %>%
        DT::formatRound(columns = c("mean_r2_val", "std_r2_val"), digits = 3)
    })
  })
}

# Null-coalescing operator (avoid conflict with rlang if loaded)
if (!exists("%||%", mode = "function")) {
  `%||%` <- function(a, b) if (is.null(a)) b else a
}
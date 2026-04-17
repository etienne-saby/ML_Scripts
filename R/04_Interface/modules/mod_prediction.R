# ==============================================================================
# MODULE : Prédiction & Positionnement Percentile
# ==============================================================================

predictionUI <- function(id) {
  ns <- NS(id)
  
  fluidRow(
    column(
      width = 4,
      box(
        title = "Scenario parameters", status = "primary", solidHeader = TRUE,
        width = NULL, collapsible = TRUE,
        
        h4("Geography"),
        sliderInput(ns("latitude"), "Latitude (°N)", 
                    min = 41, max = 51, value = 46, step = 0.5),
        sliderInput(ns("longitude"), "Longitude (°E)", 
                    min = -2, max = 8, value = 5, step = 0.5),
        
        hr(),
        h4("Cultural System"),
        selectInput(ns("main_crop"), "Main crop",
                    choices = c("Wheat" = "wheat", "Maize" = "maize"),
                    selected = "wheat"),
        
        hr(),
        h4("Plot design (m)"),
        sliderInput(ns("plotWidth"), "Plot width (m)",
                    min = 4, max = 30, value = 10, step = 1),
        sliderInput(ns("plotHeight"), "Plot height (m)",
                    min = 4, max = 30, value = 10, step = 1),
        
        hr(),
        h4("Pedology"),
        sliderInput(ns("soilDepth"), "Soil depth (m)",
                    min = 0.5, max = 8, value = 1.2, step = 0.1),
        sliderInput(ns("sand"), "Sand (%)",
                    min = 10, max = 80, value = 35, step = 5),
        sliderInput(ns("clay"), "Clay (%)",
                    min = 5, max = 50, value = 25, step = 5),
        sliderInput(ns("stone"), "Stones (%)",
                    min = 0, max = 30, value = 10, step = 5),
        selectInput(ns("waterTable"), "Water Table",
                    choices = c("Absent" = "0", "Present" = "1"),
                    selected = "0"),
        selectInput(ns("w_type"), "WT Type",
                    choices = c("Constant" = "CONST", "Variable" = "VAR"),
                    selected = "CONST"),
        
        hr(),
        actionButton(ns("predict_btn"), "Predict", 
                     class = "btn-primary btn-lg btn-block"),
        actionButton(ns("add_scenario_btn"), "Add to comparator",
                     class = "btn-success btn-block")
      )
    ),
    
    column(
      width = 8,
      box(
        title = "Prediction results", status = "success", solidHeader = TRUE,
        width = NULL,
        uiOutput(ns("prediction_status"))
      ),
      
      box(
        title = "Predicted values at t=40 years", status = "info", solidHeader = TRUE,
        width = NULL, collapsible = TRUE,
        DT::dataTableOutput(ns("stocks_table"))
      ),
      
      box(
        title = "Ratios calculated (analytical formulas)", status = "warning", solidHeader = TRUE,
        width = NULL, collapsible = TRUE,
        uiOutput(ns("ratios_display"))
      ),
      
      box(
        title = "Scenario position within Sobol distribution", status = "primary",
        solidHeader = TRUE, width = NULL, collapsible = TRUE,
        plotlyOutput(ns("percentile_plot"), height = "400px")
      )
    )
  )
}

predictionServer <- function(id, shared_state, predictor) {
  moduleServer(id, function(input, output, session) {
    
    local_state <- reactiveValues(
      prediction_result = NULL,
      prediction_params = NULL,
      ratios            = NULL,
      percentiles       = NULL
    )
    
    # ── Build Python-compatible params from UI inputs ──────────────────────
    .build_params <- function() {
      
      climate <- predict_climate_features(
        lat = input$latitude,
        lon = input$longitude,
        main_crop = input$main_crop,
        climate_models = shared_state$climate_models,
        n_years = 40L
      )
      
      params <- list(
        latitude  = input$latitude,
        plotWidth = input$plotWidth,
        plotHeight = input$plotHeight,
        soilDepth = input$soilDepth,
        sand      = input$sand,
        clay      = input$clay,
        stone     = input$stone,
        waterTable = as.integer(input$waterTable),
        main_crop = input$main_crop,
        w_type    = input$w_type
      )
      
      # Inject climatic arrays
      for (var_name in names(climate)) {
        params[[var_name]] <- as.numeric(climate[[var_name]])
      }
      
      params
    }
    
    # ── Predict ────────────────────────────────────────────────────────────
    observeEvent(input$predict_btn, {
      params <- .build_params()
      
      withProgress(message = "Predicting...", value = 0.5, {
        tryCatch({
          result <- predictor$predict_single_sim(
            params               = params,
            models               = shared_state$models,
            clf1                 = shared_state$clf1,
            clf2                 = shared_state$clf2,
            stunted_model        = shared_state$stunted_model,
            n_years              = 40L,
            return_routing       = TRUE,
            log_transform_stage1 = TRUE
          )
          
          local_state$prediction_result <- result
          local_state$prediction_params <- params
          shared_state$last_prediction  <- result
          shared_state$last_params      <- params
          
          local_state$ratios <- compute_ratios_from_predictions(result, params)
          
          if (!is.null(shared_state$sobol_ref)) {
            local_state$percentiles <- compute_percentiles(
              local_state$ratios, shared_state$sobol_ref
            )
          }
          
          showNotification("Prediction successful !", type = "message", duration = 3)
          
        }, error = function(e) {
          showNotification(
            paste("Error :", e$message), type = "error", duration = 10
          )
        })
      })
    })
    
    # ── Routing status ─────────────────────────────────────────────────────
    output$prediction_status <- renderUI({
      req(local_state$prediction_result)
      res <- local_state$prediction_result
      
      tree_fail  <- as.integer(res$tree_failed)
      yield_fail <- as.integer(res$yield_failed)
      
      status_color <- if (tree_fail == 1L) {
        "danger"
      } else if (yield_fail == 1L) {
        "warning"
      } else {
        "success"
      }
      
      tagList(
        div(
          class = paste0("alert alert-", status_color),
          h4(icon("info-circle"), " Population : ", strong(as.character(res$population))),
          p("Status - tree : ", if (tree_fail == 1L) "Failure" else "OK"),
          p("Status - crop : ", if (yield_fail == 1L) "Failure" else "OK"),
          if (!is.null(res$tree_ok_proba)) {
            p("P(arbre_ok) = ", round(as.numeric(res$tree_ok_proba), 3))
          }
        )
      )
    })
    
    # ── Stocks table ───────────────────────────────────────────────────────
    output$stocks_table <- DT::renderDataTable({
      req(local_state$prediction_result)
      preds <- local_state$prediction_result$predictions
      
      # Build table dynamically — don't hardcode order
      target_names <- names(preds)
      
      units_map <- c(
        carbonStem_AF = "kgC/tree",
        carbonStem_TF = "kgC/tree",
        yield_AF      = "t/ha",
        yield_TA      = "t/ha"
      )
      
      df <- data.frame(
        Variable    = target_names,
        `Valeur t=40` = sapply(target_names, function(tgt) {
          vals <- as.numeric(preds[[tgt]])
          round(vals[length(vals)], 2)
        }),
        Unité = sapply(target_names, function(tgt) {
          if (tgt %in% names(units_map)) units_map[[tgt]] else "—"
        }),
        check.names = FALSE,
        stringsAsFactors = FALSE
      )
      
      DT::datatable(df, options = list(dom = "t", paging = FALSE, searching = FALSE),
                    rownames = FALSE)
    })
    
    # ── Ratios display ─────────────────────────────────────────────────────
    output$ratios_display <- renderUI({
      req(local_state$ratios)
      r <- local_state$ratios
      
      ler_val   <- r$LER
      ler_class <- if (!is.na(ler_val) && ler_val > 1) "text-success" else "text-danger"
      ler_label <- if (!is.na(ler_val) && ler_val > 1) "Over-performance" else "Under-performance"
      
      tagList(
        h5("Relative Ratios (t = 1 → 40)"),
        
        p(strong("RR_crop cumulated : "), format_ratio(r$RR_crop_cumul),
          span(class = "text-muted",
               " = Σ(yield_AF_eff) / Σ(yield_TA)")),
        
        p(strong("RR_crop annual mean : "), format_ratio(r$RR_crop_mean_annual),
          if (r$n_years_yield_excluded > 0) {
            span(class = "text-muted",
                 paste0(" (", r$n_years_yield_excluded,
                        " excluded years — yield_TA < 0.1 t/ha)"))
          }),
        
        p(strong("RR_tree (t=40) : "), format_ratio(r$RR_tree),
          span(class = "text-muted", " = Carbon stocks AF/TF density corrected")),
        
        hr(),
        h5("Land Equivalent Ratio"),
        p(strong("LER = RR_crop_cumul + RR_tree(40) : "),
          format_ratio(ler_val),
          span(class = ler_class, paste0(" ", ler_label))),
        
        p(class = "text-muted", style = "font-size:0.85em;",
          "cult_frac = ", round(r$cult_frac, 3),
          " | LER > 1 → Agroforestry is over-performing compared to monocultures.")
      )
    })    
    # ── Percentile plot ────────────────────────────────────────────────────
    output$percentile_plot <- renderPlotly({
      req(local_state$percentiles, local_state$ratios)
      plot_percentile_position(local_state$ratios, local_state$percentiles,
                               shared_state$sobol_ref)
    })
    
    # ── Add to comparator ──────────────────────────────────────────────────
    observeEvent(input$add_scenario_btn, {
      req(local_state$prediction_result)
      
      n_existing <- length(shared_state$scenario_stack)
      scenario_id <- paste0("scenario_", n_existing + 1)
      
      shared_state$scenario_stack[[scenario_id]] <- list(
        params    = local_state$prediction_params,
        result    = local_state$prediction_result,
        ratios    = local_state$ratios,
        timestamp = Sys.time()
      )
      
      showNotification(paste("Scenario added :", scenario_id), type = "message")
    })
  })
}

# ── Helper for safe ratio formatting ──────────────────────────────────────
format_ratio <- function(x) {
  if (is.null(x) || is.na(x)) return("N/A")
  round(x, 3)
}
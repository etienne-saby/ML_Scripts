# ==============================================================================
# MODULE : Exploration des Trajectoires Temporelles
# ==============================================================================

trajectoriesUI <- function(id) {
  ns <- NS(id)
  
  fluidRow(
    column(
      width = 3,
      box(
        title = "Options d'affichage", status = "primary", solidHeader = TRUE,
        width = NULL,
        
        h4("Variables à afficher"),
        checkboxGroupInput(
          ns("targets"), label = NULL,
          choices = c(
            "Carbone tige AF" = "carbonStem_AF",
            "Carbone tige TF" = "carbonStem_TF",
            "Rendement AF"    = "yield_AF",
            "Rendement TA"    = "yield_TA"
          ),
          selected = c("carbonStem_AF", "yield_AF")
        ),
        
        hr(),
        h4("Analyse de sensibilité"),
        selectInput(
          ns("sensitivity_param"), "Paramètre à faire varier",
          choices = c(
            "Aucun"            = "none",
            "plotWidth"        = "plotWidth",
            "plotHeight"       = "plotHeight",
            "soilDepth"        = "soilDepth",
            "sand"             = "sand",
            "clay"             = "clay",
            "GDD_cycle_AF"     = "GDD_cycle_AF",
            "precipitation_AF" = "precipitation_AF",
            "latitude"         = "latitude"
          ),
          selected = "none"
        ),
        
        conditionalPanel(
          condition = "input.sensitivity_param != 'none'", ns = ns,
          sliderInput(ns("sensitivity_n_steps"), "Nombre de valeurs",
                      min = 3, max = 10, value = 5, step = 1),
          actionButton(ns("run_sensitivity"), "Lancer l'analyse",
                       class = "btn-info btn-block")
        ),
        
        hr(),
        h4("Scénarios enregistrés"),
        uiOutput(ns("scenario_list")),
        actionButton(ns("clear_scenarios"), "Tout effacer",
                     class = "btn-danger btn-block")
      )
    ),
    
    column(
      width = 9,
      box(
        title = "Trajectoire du scénario actuel", status = "success",
        solidHeader = TRUE, width = NULL,
        plotlyOutput(ns("main_trajectory"), height = "450px")
      ),
      box(
        title = "Comparaison multi-scénarios", status = "warning",
        solidHeader = TRUE, width = NULL, collapsible = TRUE,
        plotlyOutput(ns("comparison_plot"), height = "400px")
      ),
      box(
        title = "Sensibilité paramétrique", status = "info",
        solidHeader = TRUE, width = NULL, collapsible = TRUE, collapsed = TRUE,
        plotlyOutput(ns("sensitivity_plot"), height = "400px")
      )
    )
  )
}

trajectoriesServer <- function(id, shared_state, predictor) {
  moduleServer(id, function(input, output, session) {
    
    local_state <- reactiveValues(sensitivity_results = NULL)
    
    # ── Sensitivity parameter ranges (matching SOBOL_BOUNDS_BATCH_2) ──────
    .param_ranges <- list(
      plotWidth        = c(4, 24),
      plotHeight       = c(6, 30),
      soilDepth        = c(0.5, 8),
      sand             = c(10, 80),
      clay             = c(5, 50),
      GDD_cycle_AF     = c(800, 2500),
      precipitation_AF = c(300, 1200),
      latitude         = c(41, 51)
    )
    
    # ── Main trajectory ────────────────────────────────────────────────────
    output$main_trajectory <- renderPlotly({
      req(shared_state$last_prediction)
      plot_trajectory(shared_state$last_prediction, targets = input$targets)
    })
    
    # ── Multi-scenario comparison ──────────────────────────────────────────
    output$comparison_plot <- renderPlotly({
      req(length(shared_state$scenario_stack) > 0)
      target <- if (length(input$targets) > 0) input$targets[1] else "carbonStem_AF"
      plot_scenario_comparison(shared_state$scenario_stack, target = target)
    })
    
    # ── Scenario list ──────────────────────────────────────────────────────
    output$scenario_list <- renderUI({
      stack <- shared_state$scenario_stack
      if (length(stack) == 0) {
        return(p(class = "text-muted", "Aucun scénario enregistré"))
      }
      tagList(
        p(strong(length(stack), " scénario(s) :")),
        lapply(names(stack), function(sc_id) {
          sc <- stack[[sc_id]]
          div(
            p(icon("check-circle", class = "text-success"),
              strong(sc_id),
              br(),
              span(class = "text-muted", format(sc$timestamp, "%H:%M:%S")))
          )
        })
      )
    })
    
    # ── Clear scenarios ────────────────────────────────────────────────────
    observeEvent(input$clear_scenarios, {
      shared_state$scenario_stack <- list()
      showNotification("Scénarios supprimés", type = "warning")
    })
    
    # ── Sensitivity analysis ───────────────────────────────────────────────
    observeEvent(input$run_sensitivity, {
      req(input$sensitivity_param != "none")
      req(shared_state$last_params)
      
      param   <- input$sensitivity_param
      n_steps <- input$sensitivity_n_steps
      
      # Use last prediction params as base (clean Python params, not Shiny IDs)
      base_params <- shared_state$last_params
      
      bounds <- .param_ranges[[param]]
      if (is.null(bounds)) {
        showNotification("Paramètre non reconnu", type = "error")
        return()
      }
      
      param_range <- seq(bounds[1], bounds[2], length.out = n_steps)
      
      withProgress(message = paste("Sensibilité :", param), value = 0, {
        results <- lapply(seq_along(param_range), function(i) {
          setProgress(value = i / length(param_range),
                      detail = paste(i, "/", length(param_range)))
          
          params_i <- base_params
          params_i[[param]] <- param_range[i]
          
          tryCatch({
            predictor$predict_single_sim(
              params               = params_i,
              models               = shared_state$models,
              clf1                 = shared_state$clf1,
              clf2                 = shared_state$clf2,
              stunted_model        = shared_state$stunted_model,
              n_years              = 40L,
              return_routing       = FALSE,
              log_transform_stage1 = TRUE
            )
          }, error = function(e) NULL)
        })
        
        local_state$sensitivity_results <- list(
          param        = param,
          param_values = param_range,
          predictions  = results
        )
        showNotification("Analyse terminée", type = "message")
      })
    })
    
    # ── Sensitivity plot ───────────────────────────────────────────────────
    output$sensitivity_plot <- renderPlotly({
      req(local_state$sensitivity_results)
      
      sens       <- local_state$sensitivity_results
      param_name <- sens$param
      target <- if (length(input$targets) > 0) input$targets[1] else "carbonStem_AF"
      
      df <- data.frame(
        param_value  = sens$param_values,
        target_value = sapply(sens$predictions, function(pred) {
          if (is.null(pred)) return(NA_real_)
          vals <- as.numeric(pred$predictions[[target]])
          vals[length(vals)]
        }),
        stringsAsFactors = FALSE
      ) %>%
        dplyr::filter(!is.na(target_value))
      
      if (nrow(df) == 0) {
        return(plotly::plotly_empty() %>%
                 plotly::layout(title = "Pas de résultats"))
      }
      
      p <- ggplot2::ggplot(df, ggplot2::aes(x = param_value, y = target_value)) +
        ggplot2::geom_line(linewidth = 1.2, colour = "#2E86AB") +
        ggplot2::geom_point(size = 3, colour = "#E63946") +
        ggplot2::labs(
          title = paste("Sensibilité de", target, "à", param_name, "(t=40)"),
          x = param_name,
          y = paste(target, "(t=40)")
        ) +
        metaisafe_theme()
      
      plotly::ggplotly(p, tooltip = c("x", "y"))
    })
  })
}
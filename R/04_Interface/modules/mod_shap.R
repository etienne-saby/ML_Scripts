# ==============================================================================
# MODULE : SHAP Analysis
# ==============================================================================

shapUI <- function(id) {
  ns <- NS(id)
  
  fluidRow(
    column(
      width = 3,
      box(
        title = "Options", status = "primary", solidHeader = TRUE,
        width = NULL,
        selectInput(ns("shap_target"), "Modèle / Cible", choices = NULL),
        sliderInput(ns("n_features"), "Features à afficher",
                    min = 5, max = 30, value = 15, step = 1),
        hr(),
        uiOutput(ns("data_status"))
      )
    ),
    
    column(
      width = 9,
      box(
        title = "Importance moyenne (mean |SHAP|)", status = "success",
        solidHeader = TRUE, width = NULL,
        plotlyOutput(ns("shap_bar"), height = "500px")
      ),
      box(
        title = "Détails par feature", status = "info", solidHeader = TRUE,
        width = NULL, collapsible = TRUE, collapsed = TRUE,
        DT::dataTableOutput(ns("shap_table"))
      )
    )
  )
}

shapServer <- function(id, shared_state) {
  moduleServer(id, function(input, output, session) {
    
    # ── Resolve SHAP data directory ────────────────────────────────────────
    shap_dir <- reactive({
      req(shared_state$campaign)
      file.path(py_path_to_r(shared_state$campaign$campaign_dir),
                "Data", "SHAP")
    })
    
    # ── Load SHAP data ─────────────────────────────────────────────────────
    shap_summary <- reactive({
      dir <- shap_dir()
      path <- file.path(dir, "shap_summary.csv")
      if (file.exists(path)) {
        vroom::vroom(path, show_col_types = FALSE)
      } else NULL
    })
    
    shap_long <- reactive({
      dir <- shap_dir()
      path <- file.path(dir, "shap_long.csv")
      if (file.exists(path)) {
        vroom::vroom(path, show_col_types = FALSE)
      } else NULL
    })
    
    # ── Data status indicator ──────────────────────────────────────────────
    output$data_status <- renderUI({
      summary_ok <- !is.null(shap_summary())
      long_ok    <- !is.null(shap_long())
      
      tagList(
        p(icon(if (summary_ok) "check-circle" else "times-circle",
               class = if (summary_ok) "text-success" else "text-danger"),
          " shap_summary.csv"),
        p(icon(if (long_ok) "check-circle" else "times-circle",
               class = if (long_ok) "text-success" else "text-danger"),
          " shap_long.csv")
      )
    })
    
    # ── Update target choices ──────────────────────────────────────────────
    observe({
      df <- shap_summary()
      if (!is.null(df) && "target" %in% names(df)) {
        targets <- unique(df$target)
        updateSelectInput(session, "shap_target",
                          choices = targets, selected = targets[1])
      }
    })
    
    # ── Filtered data ──────────────────────────────────────────────────────
    shap_filtered <- reactive({
      req(input$shap_target, shap_summary())
      shap_summary() %>%
        dplyr::filter(target == input$shap_target) %>%
        dplyr::arrange(dplyr::desc(mean_abs_shap)) %>%
        utils::head(input$n_features)
    })
    
    # ── Bar chart ──────────────────────────────────────────────────────────
    output$shap_bar <- renderPlotly({
      df <- shap_filtered()
      req(nrow(df) > 0)
      
      p <- ggplot2::ggplot(
        df,
        ggplot2::aes(x = mean_abs_shap,
                     y = stats::reorder(feature, mean_abs_shap))
      ) +
        ggplot2::geom_col(fill = "#2E86AB", alpha = 0.8) +
        ggplot2::labs(
          title = paste("SHAP —", input$shap_target),
          x = "mean |SHAP|",
          y = NULL
        ) +
        metaisafe_theme()
      
      plotly::ggplotly(p, tooltip = c("x", "y"))
    })
    
    # ── Detail table ───────────────────────────────────────────────────────
    output$shap_table <- DT::renderDataTable({
      df <- shap_filtered()
      req(nrow(df) > 0)
      DT::datatable(
        df %>% dplyr::select(feature, mean_abs_shap),
        options = list(pageLength = 20, dom = "tp"),
        rownames = FALSE
      ) %>%
        DT::formatRound(columns = "mean_abs_shap", digits = 4)
    })
  })
}
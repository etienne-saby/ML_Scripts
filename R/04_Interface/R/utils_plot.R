# ==============================================================================
# MetAIsAFe — R/utils_plots.R
# ==============================================================================
# Shared ggplot2 / plotly plotting helpers.
#
# EXPORTS
#   metaisafe_theme()      — ggplot2 theme consistent with the dashboard
#   palette_targets()      — named colour vector for the 4 main targets
#   plot_trajectory()      — single-scenario 40-year trajectory (ggplot2)
#   plotly_trajectory()    — interactive version (plotly)
#
# AUTHOR  : Étienne SABY
# UPDATED : 2026-05
# ==============================================================================

# Stub — implement per project conventions.
# All functions below must be defined before mod_*.R modules are sourced.

#' MetAIsAFe ggplot2 theme
metaisafe_theme <- function(base_size = 12) {
  ggplot2::theme_minimal(base_size = base_size) +
    ggplot2::theme(
      panel.grid.minor  = ggplot2::element_blank(),
      strip.background  = ggplot2::element_rect(fill = "#e2e8f0", colour = NA),
      legend.position   = "bottom"
    )
}

#' Named colour palette for the four primary model targets
palette_targets <- function() {
  c(
    carbonStem_AF = "#2E86AB",
    carbonStem_TF = "#52B788",
    yield_AF      = "#E63946",
    yield_TA      = "#F4A261"
  )
}
# ==============================================================================
# TRAJECTORY PLOTTING
# ==============================================================================

#' Plot 40-year trajectory from a prediction result.
#'
#' @param result List. Output of PREDICTOR$predict_single_sim().
#' @param targets Character vector. Target names to plot.
#' @return plotly object.
#' @export
plot_trajectory <- function(result, targets = c("carbonStem_AF", "yield_AF")) {
  preds <- result$predictions
  years <- as.numeric(result$years)
  
  if (is.null(targets) || length(targets) == 0) {
    targets <- names(preds)
  }
  
  pal <- palette_targets()
  
  # Build long-format data frame
  df_list <- lapply(targets, function(tgt) {
    if (!tgt %in% names(preds)) return(NULL)
    vals <- as.numeric(preds[[tgt]])
    if (length(vals) != length(years)) return(NULL)
    data.frame(
      year   = years,
      value  = vals,
      target = tgt,
      stringsAsFactors = FALSE
    )
  })
  
  df <- do.call(rbind, Filter(Nonnull <- function(x) !is.null(x), df_list))
  
  if (is.null(df) || nrow(df) == 0) {
    return(plotly::plotly_empty() %>%
             plotly::layout(title = "Nothing to plot."))
  }
  
  # Determine units for faceting
  df$facet <- ifelse(
    grepl("^carbonStem", df$target),
    "CarbonStem (kgC/tree)",
    "Yield (t/ha)"
  )
  
  p <- ggplot2::ggplot(df, ggplot2::aes(x = year, y = value, colour = target)) +
    ggplot2::geom_line(linewidth = 1) +
    ggplot2::facet_wrap(~facet, scales = "free_y", ncol = 1) +
    ggplot2::scale_colour_manual(
      values = pal[intersect(names(pal), unique(df$target))],
      labels = function(x) gsub("_", " ", x)
    ) +
    ggplot2::labs(
      title  = paste("Trajectory —", result$population),
      x      = "Year",
      y      = NULL,
      colour = "Feature"
    ) +
    metaisafe_theme()
  
  plotly::ggplotly(p, tooltip = c("x", "y", "colour"))
}


#' Plot multi-scenario comparison for a single target.
#'
#' @param scenario_stack Named list of scenarios (from shared_state).
#' @param target Character. Target to compare.
#' @return plotly object.
#' @export
plot_scenario_comparison <- function(scenario_stack, target = "carbonStem_AF") {
  if (length(scenario_stack) == 0) {
    return(plotly::plotly_empty() %>%
             plotly::layout(title = "No scenario saved."))
  }
  
  df_list <- lapply(names(scenario_stack), function(sc_id) {
    sc     <- scenario_stack[[sc_id]]
    result <- sc$result
    preds  <- result$predictions
    
    if (!target %in% names(preds)) return(NULL)
    
    data.frame(
      year     = as.numeric(result$years),
      value    = as.numeric(preds[[target]]),
      scenario = sc_id,
      stringsAsFactors = FALSE
    )
  })
  
  df <- do.call(rbind, Filter(function(x) !is.null(x), df_list))
  
  if (is.null(df) || nrow(df) == 0) {
    return(plotly::plotly_empty() %>%
             plotly::layout(title = paste("No data found for", target)))
  }
  
  p <- ggplot2::ggplot(df, ggplot2::aes(x = year, y = value, colour = scenario)) +
    ggplot2::geom_line(linewidth = 1) +
    ggplot2::labs(
      title  = paste("Comparison —", gsub("_", " ", target)),
      x      = "Year",
      y      = target,
      colour = "Scenario"
    ) +
    metaisafe_theme()
  
  plotly::ggplotly(p, tooltip = c("x", "y", "colour"))
}


#' Plot percentile positioning of user values within Sobol distribution.
#'
#' @param user_values Named list. Output of compute_ratios_from_predictions().
#' @param percentiles Named list. Output of compute_percentiles().
#' @param sobol_ref Data.frame. Sobol reference (long format).
#' @return plotly object.
#' @export
plot_percentile_position <- function(user_values, percentiles, sobol_ref) {
  if (is.null(percentiles) || length(percentiles) == 0) {
    return(plotly::plotly_empty() %>%
             plotly::layout(title = "Sobol reference unavailable"))
  }
  
  df <- data.frame(
    variable   = names(percentiles),
    percentile = as.numeric(unlist(percentiles)),
    stringsAsFactors = FALSE
  )
  
  # Clean names for display
  df$label <- gsub("_t40$", "", df$variable)
  df$label <- gsub("_", " ", df$label)
  
  p <- ggplot2::ggplot(df, ggplot2::aes(x = percentile,
                                        y = stats::reorder(label, percentile))) +
    ggplot2::geom_col(fill = "#2E86AB", alpha = 0.8) +
    ggplot2::geom_vline(xintercept = 50, linetype = "dashed", colour = "grey40") +
    ggplot2::labs(
      title = "Percentile position (t=40)",
      x     = "Percentile (%)",
      y     = NULL
    ) +
    ggplot2::xlim(0, 100) +
    metaisafe_theme()
  
  plotly::ggplotly(p, tooltip = c("x", "y"))
}
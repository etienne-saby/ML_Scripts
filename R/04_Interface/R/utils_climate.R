# ==============================================================================
# FILE : R/utils_climate.R
# ==============================================================================
predict_climate_features <- function(lat, lon, main_crop,
                                     climate_models, n_years = 40L) {
  
  # ✅ Guard : climate_models vide ou NULL
  if (is.null(climate_models) || length(climate_models) == 0) {
    warning("[CLIMATE] climate_models is NULL or empty — skipping surrogate")
    return(list())
  }
  
  predict_climate_py <- reticulate::import("predict_climate")$predict_climate
  
  py_result <- tryCatch(
    predict_climate_py(lat, lon, main_crop, climate_models, n_years),
    error = function(e) {
      warning("[CLIMATE] Prediction failed: ", e$message)
      NULL
    }
  )
  
  # ✅ Guard : py_result NULL ou vide
  if (is.null(py_result) || length(py_result) == 0) {
    warning("[CLIMATE] py_result is NULL or empty")
    return(list())
  }
  
  # ✅ FIX : setNames pour préserver les noms
  result <- setNames(
    lapply(names(py_result), function(target) {
      .apply_climate_guard(target, as.numeric(py_result[[target]]))
    }),
    names(py_result)
  )
  
  # ✅ Log de validation
  for (var in names(result)) {
    vals <- result[[var]]
    message(sprintf("[CLIMATE] %-35s mean=%.1f  range=[%.1f, %.1f]",
                    var, mean(vals, na.rm=TRUE),
                    min(vals, na.rm=TRUE), max(vals, na.rm=TRUE)))
  }
  
  result
}

#' Apply physical bounds to surrogate climate predictions.
#' @keywords internal
.apply_climate_guard <- function(target, values) {
  switch(target,
         GDD_cycle_AF              = pmax(values, 0),
         ETP_cycle_AF              = pmax(values, 0),
         precipitation_AF          = pmax(values, 0),
         frost_events_cycle_AF     = pmax(round(values), 0L),
         globalRadiation_AF        = pmax(values, 0),
         maxTemperature_extreme_AF = values,
         minTemperature_extreme_AF = values,
         values
  )
}
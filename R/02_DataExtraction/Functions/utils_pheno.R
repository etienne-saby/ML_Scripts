# ==============================================================================
# FILE    : utils_pheno.R
# PURPOSE : Parse crop rotation calendars and generate phenological cycle
#           sequences for all simulations in a Hi-sAFe campaign.
# AUTHOR  : Étienne SABY
# DATE    : 2026
#
# ------------------------------------------------------------------------------
# WORKFLOW
# ------------------------------------------------------------------------------
#
# 1. normalize_crop_name()
#    Strips file extensions and matches raw crop tokens against a list of known
#    crop names using longest-prefix matching (e.g. "durum-wheat" is matched
#    before "wheat").
#
# 2. parse_crop_calendar()
#    Parses a pipe-separated crop_calendar string (from Plan.csv) into a
#    structured data.frame with one row per crop in the rotation sequence.
#    Format: "crop_file.plt:MM-DD>MM-DD|next_crop.plt:MM-DD>MM-DD|..."
#
# 3. parse_md()
#    Internal helper that splits a "MM-DD" or "MM/DD" date string into
#    integer month and day components.
#
# 4. generate_calendar()
#    Expands a parsed sequence data.frame over a simulation period, advancing
#    year automatically and handling winter crops (harvest date < sowing date).
#    Incomplete rotation passes at the end of the period are discarded.
#
# 5. build_pheno_calendars()
#    Master function: iterates over all simulations in plan_df, calls the
#    four helpers above, and writes a consolidated pheno_<tag>.csv file.
#    Rotation structure is inferred from the number of entries in
#    crop_calendar (1 = monoculture, > 1 = rotation) — no separate column
#    is required in plan_df.
# ==============================================================================


# ==============================================================================
# 1. CROP NAME NORMALISATION
# ==============================================================================

#' Normalise a raw crop token from a crop_calendar entry.
#'
#' Processing steps (applied in order):
#' \enumerate{
#'   \item Strip any file extension (e.g. \code{".plt"}, \code{".tec"}).
#'   \item If \code{known_crops} is supplied, return the first element whose
#'         name appears as a case-insensitive prefix of the stripped token.
#'         Candidates are tested longest-first so that \code{"durum-wheat"}
#'         takes precedence over \code{"wheat"}.
#'   \item Fall back to the extension-stripped token if no match is found.
#' }
#'
#' @param raw         String.  Raw crop token, e.g. \code{"wheat_var.plt"}.
#' @param known_crops Character vector or \code{NULL}.  Known crop base names
#'   to match against.
#' @return String.  Normalised crop name.
normalize_crop_name <- function(raw, known_crops = NULL) {
  clean <- tools::file_path_sans_ext(trimws(raw))
  if (!is.null(known_crops) && length(known_crops) > 0L) {
    # Test longest names first to avoid "wheat" matching "durum-wheat"
    known_sorted <- known_crops[order(nchar(known_crops), decreasing = TRUE)]
    for (base in known_sorted) {
      if (grepl(paste0("^", base), clean, ignore.case = TRUE))
        return(base)
    }
  }
  clean
}


# ==============================================================================
# 2. CROP CALENDAR PARSER
# ==============================================================================

#' Parse a crop_calendar string into a structured rotation sequence.
#'
#' Expects a pipe-separated string where each entry follows the format:
#' \code{<crop_file>:<start_MM-DD>><end_MM-DD>}
#'
#' Example:
#' \code{"wheat_var.plt:10-31>09-01|rape.plt:09-02>04-01"}
#'
#' Entries that do not match the expected pattern are skipped with a warning.
#'
#' @param cal_string  String.  A \code{crop_calendar} value from
#'   \code{Plan.csv}.
#' @param known_crops Character vector or \code{NULL}.  Passed to
#'   \code{normalize_crop_name()}.
#' @return A \code{data.frame} with columns \code{sequence_order},
#'   \code{crop_name}, \code{start_md}, \code{end_md}.  Returns a zero-row
#'   \code{data.frame} if the string is empty or all entries fail to parse.
parse_crop_calendar <- function(cal_string, known_crops = NULL) {
  
  empty <- data.frame(
    sequence_order   = integer(0L),
    crop_name        = character(0L),
    start_md         = character(0L),
    end_md           = character(0L),
    stringsAsFactors = FALSE
  )
  
  if (is.na(cal_string) || !nzchar(trimws(cal_string))) return(empty)
  
  entries <- trimws(unlist(strsplit(cal_string, "\\|")))
  entries <- entries[nzchar(entries)]
  if (length(entries) == 0L) return(empty)
  
  pat     <- "^(.+):([0-9]{1,2}[-/][0-9]{1,2})>([0-9]{1,2}[-/][0-9]{1,2})$"
  results <- vector("list", length(entries))
  
  for (i in seq_along(entries)) {
    parts <- regmatches(entries[i],
                        regexec(pat, entries[i], perl = TRUE))[[1L]]
    if (length(parts) != 4L) {
      warning(sprintf(
        "parse_crop_calendar: cannot parse entry %d: '%s' — skipped.",
        i, entries[i]
      ))
      next
    }
    results[[i]] <- data.frame(
      sequence_order   = i,
      crop_name        = normalize_crop_name(parts[2L], known_crops),
      start_md         = gsub("/", "-", parts[3L]),
      end_md           = gsub("/", "-", parts[4L]),
      stringsAsFactors = FALSE
    )
  }
  
  parsed <- Filter(Negate(is.null), results)
  if (length(parsed) == 0L) return(empty)
  dplyr::bind_rows(parsed)
}


#' Parse a \code{"MM-DD"} or \code{"MM/DD"} string into month and day integers.
#'
#' @param md_str String.  Date string in \code{"MM-DD"} or \code{"MM/DD"} format.
#' @return Named list with integer elements \code{$month} and \code{$day}.
#' @keywords internal
parse_md <- function(md_str) {
  parts <- as.integer(unlist(strsplit(as.character(md_str), "[-/]")))
  list(month = parts[1L], day = parts[2L])
}


# ==============================================================================
# 3. CALENDAR GENERATOR
# ==============================================================================

#' Generate a phenological cycle calendar over a simulation period.
#'
#' Works identically for monocultures (\code{nrow(sequence_df) == 1}) and
#' rotations (\code{nrow(sequence_df) > 1}).
#'
#' Algorithm:
#' \enumerate{
#'   \item Iterate over the rotation sequence, one crop at a time.
#'   \item If the computed sowing date falls before the last event date
#'         (i.e. the previous harvest), advance the year by 1.
#'   \item Detect winter crops automatically: if the harvest date is earlier
#'         in the calendar year than the sowing date, push the harvest to
#'         \code{current_year + 1}.
#'   \item A complete pass through the full \code{sequence_df} increments
#'         \code{Cycle_Nb} by 1.
#'   \item Incomplete passes at the end of the simulation period are discarded.
#' }
#'
#' A hard ceiling of \code{max_iterations} guards against infinite loops
#' caused by malformed calendar strings.
#'
#' @param sequence_df A \code{data.frame} produced by
#'   \code{parse_crop_calendar()}.
#' @param start_year  Integer.  First year of the simulation period.
#' @param end_year    Integer.  Last year of the simulation period.
#' @return A \code{data.frame} with columns \code{Cycle_Nb},
#'   \code{Crop_Name}, \code{Date_Sowing}, \code{Date_Harvest},
#'   \code{Harvest_Year}.  Zero rows if no complete cycle fits within the
#'   period.
generate_calendar <- function(sequence_df, start_year, end_year) {
  
  records         <- list()
  current_year    <- start_year
  last_event_date <- lubridate::make_date(current_year, 1L, 1L)
  rotation_nb     <- 1L
  
  # Safety ceiling: prevents infinite loops from malformed calendars
  max_iterations <- (end_year - start_year + 2L) * nrow(sequence_df) * 2L
  iter <- 0L
  
  while (current_year <= end_year) {
    iter <- iter + 1L
    if (iter > max_iterations) {
      warning("generate_calendar: maximum iteration limit reached — possible infinite loop.")
      break
    }
    
    pass_records <- list()
    pass_valid   <- TRUE
    
    for (i in seq_len(nrow(sequence_df))) {
      s <- parse_md(sequence_df$start_md[i])
      e <- parse_md(sequence_df$end_md[i])
      
      if (s$month < 1 || s$month > 12 || s$day < 1 || s$day > 31 ||
          e$month < 1 || e$month > 12 || e$day < 1 || e$day > 31) {
        warning(sprintf("Invalid date in calendar for %s", sid))
        next
      }
      
      d_sow <- lubridate::make_date(current_year, s$month, s$day)
      
      # If sowing date precedes the last harvest, advance one year
      if (d_sow < last_event_date) {
        current_year <- current_year + 1L
        d_sow        <- lubridate::make_date(current_year, s$month, s$day)
      }
      if (current_year > end_year) { pass_valid <- FALSE; break }
      
      d_harvest <- lubridate::make_date(current_year, e$month, e$day)
      
      # Winter crop: harvest date is earlier in year than sowing date
      if (d_harvest < d_sow) {
        d_harvest    <- lubridate::make_date(current_year + 1L, e$month, e$day)
        current_year <- current_year + 1L
      }
      if (lubridate::year(d_harvest) > end_year) { pass_valid <- FALSE; break }
      
      pass_records[[i]] <- data.frame(
        Cycle_Nb     = rotation_nb,
        Crop_Name    = sequence_df$crop_name[i],
        Date_Sowing  = d_sow,
        Date_Harvest = d_harvest,
        Harvest_Year = lubridate::year(d_harvest)
      )
      last_event_date <- d_harvest
    }
    
    if (pass_valid && length(pass_records) > 0L) {
      records     <- c(records, pass_records)
      rotation_nb <- rotation_nb + 1L
      
      year_of_last_harvest <- lubridate::year(last_event_date)
      if (current_year < year_of_last_harvest) {
        current_year <- year_of_last_harvest + 1L
      }
      
    } else {
      break
    }
  }
  
  dplyr::bind_rows(records)
}


# ==============================================================================
# 4. MASTER CALENDAR BUILDER
# ==============================================================================

#' Build phenological calendars for all simulations in a campaign.
#'
#' Reads sowing and harvest dates from the \code{crop_calendar} column of
#' \code{plan_df}.  No direct access to simulation output directories is
#' required.  Rotation structure is inferred from the number of entries in the
#' calendar string: 1 = monoculture, > 1 = rotation.
#'
#' Per-simulation year bounds are taken from \code{plan_df$start_year} and
#' \code{plan_df$end_year} when available, otherwise the global
#' \code{sim_years} range is used.
#'
#' The resulting calendar is written as
#' \code{pheno_<campaign_tag>.csv} in \code{out_dir} and returned invisibly.
#'
#' @param plan_df       data.frame.  Required columns: \code{sim_id},
#'   \code{crop_calendar}.  Optional: \code{start_year}, \code{end_year}.
#' @param sim_years     Integer vector.  Global simulation year range,
#'   used as a fallback when per-sim year bounds are absent from
#'   \code{plan_df}.
#' @param out_dir       String.  Output directory.
#' @param campaign_tag  String.  Campaign identifier, e.g.
#'   \code{"LHS_S1111_AF"}.
#' @param known_crops   Character vector or \code{NULL}.  Crop base names for
#'   normalisation.  Passed to \code{normalize_crop_name()}.
#'   Default: \code{NULL} (raw token used as-is).
#' @param overwrite     Logical.  Re-build even if the output file exists?
#'   Default: \code{FALSE}.
#' @param get_job_details Logical.  Print a message for every simulation
#'   processed.  Default: \code{FALSE}.
#' @return A \code{data.frame} (invisibly) with columns \code{sim_id},
#'   \code{Cycle_Nb}, \code{Crop_Name}, \code{Date_Sowing},
#'   \code{Date_Harvest}, \code{Harvest_Year}.
#'   Returns \code{NULL} invisibly if no calendars could be built.
build_pheno_calendars <- function(plan_df,
                                  sim_years,
                                  out_dir,
                                  campaign_tag,
                                  known_crops     = NULL,
                                  overwrite       = FALSE,
                                  get_job_details = FALSE) {
  
  path_no_ext <- fs::path(out_dir, paste0("pheno_", campaign_tag))
  target_file <- paste0(path_no_ext, ".csv")
  
  if (!overwrite && file.exists(target_file)) {
    message(sprintf("  [CACHE] pheno_%s already exists — skipped.", campaign_tag))
    return(invisible(read_data(path_no_ext)))
  }
  
  # Validate required columns
  required_cols <- c("sim_id", "crop_calendar")
  missing_cols  <- setdiff(required_cols, names(plan_df))
  if (length(missing_cols) > 0L)
    stop("build_pheno_calendars: missing column(s) in plan_df: ",
         paste(missing_cols, collapse = ", "))
  
  message(sprintf(
    "\n[build_pheno_calendars] Campaign: %s | %d simulations",
    campaign_tag, nrow(plan_df)
  ))
  
  all_rows <- list()
  n_skip   <- 0L
  n_mono   <- 0L
  n_rot    <- 0L
  
  for (i in seq_len(nrow(plan_df))) {
    
    sid     <- plan_df$sim_id[i]
    cal_str <- plan_df$crop_calendar[i]
    
    # Parse calendar string, logging any parse warnings before muffling them
    seq_df <- withCallingHandlers(
      parse_crop_calendar(cal_str, known_crops = known_crops),
      warning = function(w) {
        message(sprintf("    [%s] Parse warning: %s", sid, conditionMessage(w)))
        invokeRestart("muffleWarning")
      }
    )
    
    if (is.null(seq_df) || nrow(seq_df) == 0L) {
      if (get_job_details)
        message(sprintf("    [%s] Empty or unparseable crop_calendar — skipped.", sid))
      n_skip <- n_skip + 1L
      next
    }
    
    is_rotation <- nrow(seq_df) > 1L
    if (is_rotation) n_rot <- n_rot + 1L else n_mono <- n_mono + 1L
    
    # Per-simulation year bounds; fall back to global range if absent
    sim_start <- if ("start_year" %in% names(plan_df)) plan_df$start_year[i] else min(sim_years)
    sim_end   <- if ("end_year"   %in% names(plan_df)) plan_df$end_year[i]   else max(sim_years)
    
    cal <- generate_calendar(seq_df, sim_start, sim_end)
    
    if (is.null(cal) || nrow(cal) == 0L) {
      if (get_job_details)
        message(sprintf(
          "    [%s] No complete cycles generated for years %d–%d — skipped.",
          sid, sim_start, sim_end
        ))
      n_skip <- n_skip + 1L
      next
    }
    
    if (get_job_details)
      message(sprintf("    [%s] %s — %d cycle(s)",
                      sid,
                      if (is_rotation) "rotation" else seq_df$crop_name[1L],
                      nrow(cal)))
    
    cal$sim_id <- sid
    all_rows[[length(all_rows) + 1L]] <- cal
  }
  
  message(sprintf("  Monocultures: %d | Rotations: %d | Skipped: %d",
                  n_mono, n_rot, n_skip))
  
  if (length(all_rows) == 0L) {
    warning("No phenological calendars built for campaign: ", campaign_tag)
    return(invisible(NULL))
  }
  
  calendars <- dplyr::bind_rows(all_rows) |>
    dplyr::select(sim_id, Cycle_Nb, Crop_Name,
                  Date_Sowing, Date_Harvest, Harvest_Year) |>
    dplyr::arrange(sim_id, Date_Sowing)
  
  n_sims_ok <- dplyr::n_distinct(calendars$sim_id)
  message(sprintf("  %d cycle-rows across %d simulations.",
                  nrow(calendars), n_sims_ok))
  
  if (get_job_details) {
    message("  Cycle count by crop:")
    print(table(calendars$Crop_Name))
  }
  
  data.table::fwrite(calendars, target_file)
  message(sprintf("  Saved: %s\n", target_file))
  invisible(calendars)
}
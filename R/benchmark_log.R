## benchmark_log.R
## Community benchmark: log-parsing + leaderboard-rendering API.
##
## The raw JSON run-log (one file per harness invocation, produced by the
## harness in the PsychQuantR/DefDiff-benchmark repo) is the SOURCE OF TRUTH.
## This package projects logs to a long-format CSV via `parse_benchmark_logs()`
## and renders a leaderboard table via `bench_render_leaderboard()`; the harness
## itself, its timing/provenance helpers, and the contribution flow live in the
## sibling DefDiff-benchmark repo, not here.
##
## The log JSON shape is the frozen contract; the CSV schema is soft (it can
## gain columns later by re-parsing historical logs), so the column set lives in
## one place — `.bench_csv_columns()`.

## Frozen schema version of the run-log JSON contract.
.BENCH_SCHEMA_VERSION <- 1L

## Marker comments delimiting the regenerated leaderboard region in a README.
.BENCH_LEADERBOARD_BEGIN <- "<!-- BENCHMARK-LEADERBOARD:BEGIN -->"
.BENCH_LEADERBOARD_END   <- "<!-- BENCHMARK-LEADERBOARD:END -->"

## NULL/empty-coalescing helper (kept local to avoid clashing with any other
## operator definition in the package namespace).
.or <- function(x, default) if (is.null(x) || length(x) == 0L) default else x

## Single source of truth for the derived CSV column set AND order.
## `parse_benchmark_logs()` emits exactly these columns in this order, and the
## parser tests assert against this vector.
.bench_csv_columns <- function() {
  c("schema_version", "run_id", "date", "contributor",
    "chip", "cores", "ram_gb", "os_version", "r_version", "blas",
    "harness_version", "system", "system_version", "operation",
    "problem_id", "n", "precision", "threads", "parallel_capable",
    "stage", "median_ms", "iqr_ms", "cv_pct", "reps")
}

# --- Log -> CSV parser ------------------------------------------------------

#' Parse community benchmark run-logs into a long-format data frame
#'
#' Reads every `*.json` run-log in `logs_dir`, broadcasts each log's `meta`
#' provenance block across that log's `measurements`, and returns one row per
#' measurement in the frozen column order. The raw logs are the source of
#' truth; this derived CSV is regenerable, so re-running the parser after a
#' schema change re-projects all historical logs with no data loss.
#'
#' A malformed or empty log file is skipped with a warning naming the file;
#' valid logs in the same directory still parse.
#'
#' @param logs_dir Directory containing run-log JSON files.
#' @param out_csv Optional path. When non-`NULL`, the data frame is written
#'   there as CSV and returned invisibly.
#' @return A data frame with one row per measurement, columns in the order
#'   given by the run-log contract. When `out_csv` is set, returned invisibly.
#' @export
#' @examples
#' \dontrun{
#' # logs accumulate in the PsychQuantR/DefDiff-benchmark repo
#' parse_benchmark_logs("community-logs", out_csv = "community-benchmark.csv")
#' }
parse_benchmark_logs <- function(logs_dir, out_csv = NULL) {
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("parse_benchmark_logs() requires the 'jsonlite' package. ",
         "Install it with install.packages(\"jsonlite\").", call. = FALSE)
  }
  cols  <- .bench_csv_columns()
  files <- list.files(logs_dir, pattern = "\\.json$", full.names = TRUE)
  rows  <- list()

  for (f in files) {
    log <- tryCatch(jsonlite::fromJSON(f, simplifyVector = FALSE),
                    error = function(e) NULL)
    if (is.null(log) || length(.or(log$measurements, list())) == 0L) {
      warning(sprintf("Skipping malformed or empty benchmark log: %s", f),
              call. = FALSE)
      next
    }
    meta <- .or(log$meta, list())
    base <- list(
      schema_version  = .or(log$schema_version, NA_integer_),
      run_id          = .or(log$run_id, NA_character_),
      date            = .or(meta$date, NA_character_),
      contributor     = .or(meta$contributor, NA_character_),
      chip            = .or(meta$hardware$chip, NA_character_),
      cores           = .or(meta$hardware$cores, NA_integer_),
      ram_gb          = .or(meta$hardware$ram_gb, NA_real_),
      os_version      = .or(meta$env$os_version, NA_character_),
      r_version       = .or(meta$env$r_version, NA_character_),
      blas            = .or(meta$env$blas, NA_character_),
      harness_version = .or(log$harness_version, NA_character_)
    )
    for (m in log$measurements) {
      row <- c(base, list(
        system           = .or(m$system, NA_character_),
        system_version   = .or(m$system_version, NA_character_),
        operation        = .or(m$operation, NA_character_),
        problem_id       = .or(m$problem_id, NA_character_),
        n                = .or(m$n, NA_real_),
        precision        = .or(m$precision, NA_character_),
        threads          = as.character(.or(m$threads, NA)),
        parallel_capable = .or(m$parallel_capable, NA),
        stage            = .or(m$stage, NA_character_),
        median_ms        = .or(m$median_ms, NA_real_),
        iqr_ms           = .or(m$iqr_ms, NA_real_),
        cv_pct           = .or(m$cv_pct, NA_real_),
        reps             = .or(m$reps, NA_integer_)
      ))
      rows[[length(rows) + 1L]] <- as.data.frame(row[cols],
                                                 stringsAsFactors = FALSE)
    }
  }

  df <- if (length(rows) == 0L) {
    empty <- as.data.frame(matrix(nrow = 0L, ncol = length(cols)))
    names(empty) <- cols
    empty
  } else {
    do.call(rbind, rows)
  }
  df <- df[, cols, drop = FALSE]

  if (!is.null(out_csv)) {
    utils::write.csv(df, out_csv, row.names = FALSE)
    return(invisible(df))
  }
  df
}

# --- Leaderboard regeneration -----------------------------------------------

## Neutralize markdown-significant characters in contributor-controlled string
## fields before they are interpolated into the README table. Run-logs are
## PR-contributed (untrusted), so a crafted field (table pipes, newlines, code
## spans, an injected heading) must not break out of its cell and poison the
## published README. Numeric fields are coerced separately and are not routed
## through here.
.bench_md_cell <- function(x, max_len = 60L) {
  x <- as.character(x)
  x[is.na(x)] <- ""
  x <- gsub("[|`\r\n]", " ", x)                 # table / code-span / line breaks
  x <- trimws(gsub("[[:space:]]+", " ", x))      # collapse remaining whitespace
  ifelse(nchar(x) > max_len, paste0(substr(x, 1L, max_len - 3L), "..."), x)
}

## Render the steady-state (`eval`) rows of the CSV as a markdown leaderboard
## table. Empty input yields a friendly placeholder line.
.bench_leaderboard_markdown <- function(df) {
  if (nrow(df) == 0L) {
    return("_No community submissions yet. Run the benchmark harness and open a pull request adding your run-log._")
  }
  ev <- df[!is.na(df$stage) & df$stage == "eval", , drop = FALSE]
  if (nrow(ev) == 0L) ev <- df
  ord <- order(ev$operation, ev$problem_id,
               suppressWarnings(as.numeric(ev$n)),
               suppressWarnings(as.numeric(ev$median_ms)))
  ev <- ev[ord, , drop = FALSE]
  c("| Chip | System | Operation | Problem | n | Threads | eval median (ms) |",
    "|---|---|---|---|---|---|---|",
    sprintf("| %s | %s | %s | %s | %s | %s | %s |",
            .bench_md_cell(ev$chip), .bench_md_cell(ev$system),
            .bench_md_cell(ev$operation), .bench_md_cell(ev$problem_id),
            format(as.numeric(ev$n), scientific = TRUE),
            .bench_md_cell(ev$threads),
            formatC(as.numeric(ev$median_ms), format = "f", digits = 3)))
}

#' Regenerate the README leaderboard table from the derived CSV
#'
#' Replaces the content between the leaderboard begin/end marker comments in
#' `readme_path` with a markdown table rendered from `csv_path`. If the markers
#' are absent, the block is appended. Regeneration is idempotent: running it
#' twice against an unchanged CSV produces no diff.
#'
#' @param csv_path Path to the derived `community-benchmark.csv`.
#' @param readme_path Path to the README source containing the markers.
#' @return The `readme_path`, invisibly.
#' @export
bench_render_leaderboard <- function(csv_path, readme_path) {
  df <- if (file.exists(csv_path)) {
    utils::read.csv(csv_path, stringsAsFactors = FALSE, colClasses = "character")
  } else {
    stats::setNames(
      as.data.frame(matrix(nrow = 0L, ncol = length(.bench_csv_columns()))),
      .bench_csv_columns())
  }
  block <- c(.BENCH_LEADERBOARD_BEGIN, "",
             .bench_leaderboard_markdown(df), "",
             .BENCH_LEADERBOARD_END)
  lines <- if (file.exists(readme_path)) readLines(readme_path, warn = FALSE) else character()
  bi <- which(lines == .BENCH_LEADERBOARD_BEGIN)
  ei <- which(lines == .BENCH_LEADERBOARD_END)
  new <- if (length(bi) == 1L && length(ei) == 1L && ei > bi) {
    c(if (bi > 1L) lines[seq_len(bi - 1L)] else character(),
      block,
      if (ei < length(lines)) lines[(ei + 1L):length(lines)] else character())
  } else {
    c(lines, if (length(lines)) "" else character(), block)
  }
  writeLines(new, readme_path)
  invisible(readme_path)
}

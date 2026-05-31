## benchmark_log.R
## Community differentiation benchmark: the log-first data flow.
##
## The raw JSON run-log (one file per harness invocation) is the SOURCE OF
## TRUTH; the long-format CSV is a derived projection produced by
## `parse_benchmark_logs()`, and the README leaderboard is regenerated from the
## CSV by `bench_render_leaderboard()`. See the `community-benchmark` spec.
##
## The log JSON shape is the frozen contract; the CSV schema is soft (it can
## gain columns later by re-parsing historical logs), so the column set lives
## in one place — `.bench_csv_columns()`.

## Frozen schema version of the run-log JSON contract.
.BENCH_SCHEMA_VERSION <- 1L

## Version of the harness that emitted a log; bumped when timing methodology
## changes in a way that makes older logs non-comparable.
.BENCH_HARNESS_VERSION <- "0.1.0"

## Marker comments delimiting the regenerated leaderboard region in the README.
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

# --- Provenance capture -----------------------------------------------------

## Detect the CPU/SoC marketing name (e.g. "Apple M4 Max"). macOS-first,
## best-effort elsewhere. Machine-sourced, never user-typed.
.bench_detect_chip <- function() {
  if (identical(Sys.info()[["sysname"]], "Darwin")) {
    v <- tryCatch(system2("sysctl", c("-n", "machdep.cpu.brand_string"),
                          stdout = TRUE, stderr = NULL),
                  error = function(e) character())
    if (length(v) && nzchar(v[[1L]])) return(v[[1L]])
  }
  unname(Sys.info()[["machine"]])
}

## Detect physical core count (the meaningful "max" thread level on Apple
## Silicon). Falls back to logical-core detection off macOS.
.bench_detect_cores <- function() {
  if (identical(Sys.info()[["sysname"]], "Darwin")) {
    v <- tryCatch(suppressWarnings(as.integer(
           system2("sysctl", c("-n", "hw.physicalcpu"), stdout = TRUE, stderr = NULL))),
         error = function(e) NA_integer_)
    if (length(v) && !is.na(v[[1L]])) return(v[[1L]])
  }
  as.integer(tryCatch(parallel::detectCores(logical = FALSE),
                      error = function(e) NA_integer_))
}

## Detect installed RAM in whole GB (macOS via hw.memsize; NA elsewhere).
.bench_detect_ram_gb <- function() {
  if (identical(Sys.info()[["sysname"]], "Darwin")) {
    b <- tryCatch(suppressWarnings(as.numeric(
           system2("sysctl", c("-n", "hw.memsize"), stdout = TRUE, stderr = NULL))),
         error = function(e) NA_real_)
    if (length(b) && !is.na(b[[1L]])) return(round(b[[1L]] / 1024^3))
  }
  NA_real_
}

## Detect the BLAS/LAPACK backing (collapses Apple Accelerate to "Accelerate").
.bench_detect_blas <- function() {
  lib <- tryCatch(La_library(), error = function(e) "")
  if (!nzchar(lib)) lib <- tryCatch(unname(extSoftVersion()[["BLAS"]]),
                                    error = function(e) "")
  if (grepl("Accelerate", lib, ignore.case = TRUE)) return("Accelerate")
  if (nzchar(lib)) basename(lib) else NA_character_
}

## Detect a human-readable OS version string.
.bench_detect_os <- function() {
  r <- tryCatch(utils::sessionInfo()$running, error = function(e) NULL)
  if (!is.null(r) && nzchar(r)) return(r)
  paste(Sys.info()[["sysname"]], Sys.info()[["release"]])
}

## Build the `meta` provenance block for a run-log. Hardware and environment
## fields are auto-detected from the system, not supplied by the caller; only
## `contributor` and `date` may be overridden (defaulting to the OS user and
## the current time).
bench_capture_provenance <- function(contributor = NULL, date = NULL,
                                     systems = list()) {
  list(
    date        = .or(date, format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z")),
    contributor = .or(contributor, unname(Sys.info()[["user"]])),
    hardware    = list(chip   = .bench_detect_chip(),
                       cores  = .bench_detect_cores(),
                       ram_gb = .bench_detect_ram_gb()),
    env         = list(os_version = .bench_detect_os(),
                       r_version  = as.character(getRversion()),
                       blas       = .bench_detect_blas()),
    systems     = systems,
    sessionInfo = paste(utils::capture.output(utils::sessionInfo()),
                        collapse = "\n")
  )
}

## Construct a sortable, unique run id: timestamp + short random suffix. The
## timestamp is passed in (the runner supplies `Sys.time()`); the suffix uses
## the session RNG so repeated runs on one machine never collide.
bench_new_run_id <- function(now = Sys.time(), suffix = NULL) {
  ts <- format(now, "%Y%m%dT%H%M%OS3%z")
  if (is.null(suffix)) {
    suffix <- paste(sample(c(0:9, letters), 4L, replace = TRUE), collapse = "")
  }
  paste0(ts, "-", suffix)
}

# --- Timing primitives ------------------------------------------------------
#
# Same methodology as inst/benchmarks/comprehensive-ad-comparison.R: bench::mark
# over a pre-allocated input with median / IQR / CV, falling back to a
# replicate-based timer when bench is unavailable. Shared here so the community
# harness reuses the discipline rather than forking a second timing stack.

## Time a steady-state operation over `reps` iterations. Returns median_ms,
## iqr_ms, cv_pct, and the realized rep count.
bench_time_repeated <- function(thunk, reps = 50L) {
  if (requireNamespace("bench", quietly = TRUE)) {
    res <- tryCatch(
      bench::mark(thunk(), iterations = reps, check = FALSE, filter_gc = FALSE),
      error = function(e) NULL)
    if (!is.null(res)) {
      times_ms <- as.numeric(res$time[[1L]]) * 1000
      q <- stats::quantile(times_ms, c(0.25, 0.5, 0.75), na.rm = TRUE)
      return(list(median_ms = unname(q[[2L]]),
                  iqr_ms    = unname(q[[3L]] - q[[1L]]),
                  cv_pct    = 100 * stats::sd(times_ms, na.rm = TRUE) /
                                    mean(times_ms, na.rm = TRUE),
                  reps      = length(times_ms)))
    }
  }
  ts <- replicate(reps, {
    t0 <- proc.time()[["elapsed"]]; thunk(); (proc.time()[["elapsed"]] - t0) * 1000
  })
  list(median_ms = stats::median(ts), iqr_ms = NA_real_, cv_pct = NA_real_,
       reps = length(ts))
}

## Time a one-shot stage (build / import / first_eval / jit_compile) once.
## Startup stages are deliberately measured, not discarded as warm-up.
bench_time_once <- function(thunk) {
  t0 <- proc.time()[["elapsed"]]
  thunk()
  list(median_ms = (proc.time()[["elapsed"]] - t0) * 1000,
       iqr_ms = NA_real_, cv_pct = NA_real_, reps = 1L)
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
#' parse_benchmark_logs(
#'   system.file("benchmarks/community-logs", package = "DefDiff"),
#'   out_csv = "community-benchmark.csv")
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

## Render the steady-state (`eval`) rows of the CSV as a markdown leaderboard
## table. Empty input yields a friendly placeholder line.
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

.bench_leaderboard_markdown <- function(df) {
  if (nrow(df) == 0L) {
    return(paste0("_No community submissions yet. Run ",
                  "`Rscript inst/benchmarks/run-community-benchmark.R --append` ",
                  "and open a PR adding your log file._"))
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

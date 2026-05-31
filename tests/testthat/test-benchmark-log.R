## test-benchmark-log.R
## Parser contract for the community benchmark log-first data flow.
## The raw JSON run-log is the source of truth; parse_benchmark_logs() projects
## logs to the long-format CSV. These tests pin that projection (column set,
## meta broadcast, malformed-skip, CSV write, empty directory).

skip_if_not_installed("jsonlite")

# Exact derived-CSV column contract (frozen order). Pinned here independently of
# the implementation so drift in either direction fails the test.
bench_expected_cols <- c(
  "schema_version", "run_id", "date", "contributor",
  "chip", "cores", "ram_gb", "os_version", "r_version", "blas",
  "harness_version", "system", "system_version", "operation",
  "problem_id", "n", "precision", "threads", "parallel_capable",
  "stage", "median_ms", "iqr_ms", "cv_pct", "reps")

make_log <- function(run_id = "20260531T200000+0800-aaaa", measurements = NULL) {
  if (is.null(measurements)) {
    measurements <- list(
      list(system = "DefDiff", system_version = "0.1.0", operation = "grad",
           problem_id = "sum_v2", n = 1e6, precision = "float64",
           threads = "1", parallel_capable = NA, stage = "build",
           median_ms = 0.4, iqr_ms = NA, cv_pct = NA, reps = 1L),
      list(system = "DefDiff", system_version = "0.1.0", operation = "grad",
           problem_id = "sum_v2", n = 1e6, precision = "float64",
           threads = "1", parallel_capable = NA, stage = "eval",
           median_ms = 12.3, iqr_ms = 0.5, cv_pct = 2.1, reps = 50L))
  }
  list(
    schema_version = 1L, run_id = run_id, harness_version = "0.1.0",
    meta = list(
      date = "2026-05-31T20:00:00+0800", contributor = "tester",
      hardware = list(chip = "Apple M4 Max", cores = 16L, ram_gb = 128),
      env = list(os_version = "macOS 15.5", r_version = "4.4.2", blas = "Accelerate"),
      systems = list(DefDiff = "0.1.0", numDeriv = "2016.8-1.1")),
    measurements = measurements)
}

write_log <- function(dir, log) {
  jsonlite::write_json(log, file.path(dir, paste0(log$run_id, ".json")),
                       auto_unbox = TRUE, pretty = TRUE, na = "null")
}

test_that("a fixture log parses to one row per measurement in contract column order", {
  d <- withr::local_tempdir()
  write_log(d, make_log())
  df <- parse_benchmark_logs(d)
  expect_identical(names(df), bench_expected_cols)
  expect_equal(nrow(df), 2L)
  expect_setequal(df$stage, c("build", "eval"))
  expect_equal(df$median_ms[df$stage == "eval"], 12.3)
  expect_equal(df$reps[df$stage == "build"], 1L)
})

test_that("meta is broadcast across all measurements of a log", {
  d <- withr::local_tempdir()
  write_log(d, make_log())
  df <- parse_benchmark_logs(d)
  expect_true(all(df$chip == "Apple M4 Max"))
  expect_true(all(df$contributor == "tester"))
  expect_true(all(df$run_id == "20260531T200000+0800-aaaa"))
})

test_that("a malformed log is skipped with a warning while valid logs still parse", {
  d <- withr::local_tempdir()
  write_log(d, make_log(run_id = "20260531T200001+0800-bbbb"))
  writeLines("{ this is not valid json", file.path(d, "broken.json"))
  expect_warning(df <- parse_benchmark_logs(d), "malformed|empty")
  expect_equal(nrow(df), 2L)
  expect_true(all(df$run_id == "20260531T200001+0800-bbbb"))
})

test_that("out_csv writes the derived CSV with the contract columns", {
  d <- withr::local_tempdir()
  write_log(d, make_log())
  csv <- file.path(d, "out.csv")
  res <- parse_benchmark_logs(d, out_csv = csv)
  expect_true(file.exists(csv))
  back <- utils::read.csv(csv, stringsAsFactors = FALSE)
  expect_identical(names(back), bench_expected_cols)
  expect_equal(nrow(back), 2L)
})

test_that("an empty logs directory yields a zero-row frame with the contract columns", {
  d <- withr::local_tempdir()
  df <- parse_benchmark_logs(d)
  expect_identical(names(df), bench_expected_cols)
  expect_equal(nrow(df), 0L)
})

test_that("leaderboard render neutralizes markdown injection from contributor fields", {
  d <- withr::local_tempdir()
  csv <- file.path(d, "community-benchmark.csv")
  readme <- file.path(d, "README.md")
  # Run-logs are PR-contributed (untrusted). A malicious chip field with table
  # pipes, a newline, and an injected heading must not break out of its cell.
  evil <- "M| x |\n## PWNED\n[x](http://evil)"
  row <- as.data.frame(as.list(stats::setNames(rep("", length(bench_expected_cols)),
                                               bench_expected_cols)),
                       stringsAsFactors = FALSE)
  row$chip <- evil; row$system <- "DefDiff"; row$operation <- "grad"
  row$problem_id <- "sum_v2"; row$n <- "1000"; row$threads <- "1"
  row$stage <- "eval"; row$median_ms <- "1.0"
  utils::write.csv(row, csv, row.names = FALSE)
  writeLines(c("# Title", "<!-- BENCHMARK-LEADERBOARD:BEGIN -->", "",
               "<!-- BENCHMARK-LEADERBOARD:END -->"), readme)

  bench_render_leaderboard(csv, readme)
  out <- readLines(readme)

  # No injected standalone heading line leaked out of the table cell.
  expect_false(any(grepl("^##\\s*PWNED", out)))
  # The leaderboard region holds exactly header + separator + one data row.
  begin <- which(out == "<!-- BENCHMARK-LEADERBOARD:BEGIN -->")
  end   <- which(out == "<!-- BENCHMARK-LEADERBOARD:END -->")
  region <- out[(begin + 1L):(end - 1L)]
  expect_equal(sum(grepl("^\\|", region)), 3L)
})

## multidim-batch-vs-numpy.R
## Multi-dimensional batch protocol (#4, Reading A): define a formula ONCE,
## apply its gradient across a dimension sweep reusing a single compiled kernel,
## and compare to a hand-coded NumPy gradient.
##
## Run: Rscript inst/benchmarks/multidim-batch-vs-numpy.R
## (maintainer-only; macOS Accelerate + a `python3` with numpy on PATH.)
##
## Two things this demonstrates:
##   1. STORE ONCE, APPLY MANY DIMS — one grad(f) + one compiled kernel serves the
##      whole n sweep (proven via DefDiff:::.dat_jit_state$compile_count delta <= 1,
##      not narrative). The cache key excludes n (see R/fused_jit.R).
##   2. vs NumPy — DD's store-once-compiled-kernel vs NumPy's eager analytic
##      gradient. NumPy has no autodiff; see sidecar_numpy_compare.py header for
##      the fair-comparison framing. Expect NumPy to win small n (low overhead)
##      and DD to win large n (fused + threaded, bandwidth-bound) — a crossover.

suppressMessages({
  library(DefDiff)
  has_bench <- requireNamespace("bench", quietly = TRUE)
})

SIDECAR <- system.file("benchmarks", "sidecar_numpy_compare.py", package = "DefDiff")
if (!nzchar(SIDECAR)) SIDECAR <- "inst/benchmarks/sidecar_numpy_compare.py"  # source-tree run

ns <- c(1e3, 1e4, 1e5, 1e6, 1e7, 1e8)

formulas <- list(
  list(label = "sum(v^3)", expr = "sum_v3", f = function(v) sum(v^3), g = function(v) 3 * v^2),
  list(label = "sum(v^2)", expr = "sum_v2", f = function(v) sum(v^2), g = function(v) 2 * v)
)

numpy_ms <- function(expr_name, n) {
  out <- tryCatch(
    system2("python3", c(SIDECAR, "--expression", expr_name, "--n", format(n, scientific = FALSE)),
            stdout = TRUE, stderr = FALSE),
    error = function(e) NA_character_)
  val <- tryCatch(jsonlite::fromJSON(paste(out, collapse = ""))$wall_ms, error = function(e) NA_real_)
  if (is.null(val)) NA_real_ else as.numeric(val)
}

# Auto-tune ON (default): DD picks the genuinely-fastest path per n per machine
# (base vDSP at small/mid n, fused/threaded at large n) — NOT a forced threshold.
# The warm call before each timing pays the one-time probe; bench measures the
# steady-state learned path. (Earlier versions set jit_threshold=1, which forced
# the threaded path even at tiny n — a benchmark artifact that inflated small-n DD.)
options(DefDiff.jit_threshold = NULL, DefDiff.autotune = TRUE,
        DefDiff.jit_threads = parallel::detectCores(logical = FALSE))

st <- tryCatch(get(".dat_jit_state", asNamespace("DefDiff")), error = function(e) NULL)

cat("# Multi-dimensional batch protocol — DD (store once) vs NumPy (eager)\n\n")

for (fm in formulas) {
  gf <- grad(fm$f)                       # STORE THE FORMULA ONCE
  cat(sprintf("## %s  (stored gradient: %s)\n\n",
              fm$label, paste(deparse(attr(gf, "grad_expr")), collapse = "")))
  cat("| n | DD ms | NumPy ms | winner | DD correct |\n")
  cat("|---|---|---|---|---|\n")

  compiles_before <- if (!is.null(st)) st$compile_count else NA_integer_
  for (n in ns) {
    set.seed(1); v <- rnorm(as.integer(n))
    invisible(gf(v))                     # warm/compile (first n compiles; rest reuse)
    # Inline bench::mark — a wrapper that takes gf(v) as an arg would evaluate it
    # before bench sees it (promise already forced) and time ~0. Keep it inline.
    dd <- if (has_bench) {
      as.numeric(summary(bench::mark(gf(v), iterations = 7, check = FALSE))$median) * 1000
    } else {
      median(replicate(5, system.time(gf(v))[["elapsed"]])) * 1000
    }
    np <- numpy_ms(fm$expr, n)
    ok <- isTRUE(all.equal(as.numeric(gf(v)), as.numeric(fm$g(v)), tolerance = 1e-10))
    winner <- if (is.na(np)) "—" else if (dd <= np) "DD" else "NumPy"
    cat(sprintf("| %.0e | %.3f | %s | %s | %s |\n",
                n, dd, if (is.na(np)) "n/a" else sprintf("%.3f", np), winner, ok))
    invisible(gc(verbose = FALSE))
  }
  if (!is.null(st)) {
    delta <- st$compile_count - compiles_before
    cat(sprintf("\n**Store-once proof**: %d kernel compile(s) across the whole %d-dim sweep ",
                delta, length(ns)))
    cat(if (delta <= 1L) "✓ (one kernel reused across all dimensions)\n\n"
        else "⚠ (expected <= 1)\n\n")
  }
}

cat("---\nBenchmark complete. NumPy = hand-coded analytic gradient (no autodiff); ",
    "see sidecar_numpy_compare.py header.\n", sep = "")

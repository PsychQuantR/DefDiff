# Tests for the fused-JIT gradient backend (add-fused-jit-grad-backend, #2).
# R-side lowering + fusability (tasks 1.1, 1.2) need no compiled kernel.
# Native dispatch / equivalence / fallback tests are gated on macOS + the JIT
# path being available (clang present).

dat_ns <- asNamespace("DefDiff")
fusable   <- get(".is_fusable_elementwise", dat_ns)
should_fire <- get(".fused_should_fire", dat_ns)
lower     <- get(".lower_fused_cpp", dat_ns)

# ---- task 1.1: fusable predicate ----

test_that("arithmetic gradients are fusable, transcendental/matmul are not", {
  expect_true(fusable(quote(3 * v^2), "v"))
  expect_true(fusable(quote(cos(sum(v^2)) * 2 * v), "v"))  # scalar hoisted
  expect_true(fusable(quote(s * v + t * v), "v"))
  expect_true(fusable(quote(4 * sum(v^2) * v), "v"))
  expect_false(fusable(quote(cos(v)), "v"))                 # per-element transcendental
  expect_false(fusable(quote(2^v), "v"))                    # exponent references v
  expect_false(fusable(quote(v %*% w), "v"))                # matrix product
})

test_that("should_fire: any arithmetic-on-var fires (single- and multi-pass); copies/scalars do not", {
  expect_true(should_fire(quote(3 * v^2), "v"))                 # multi-pass
  expect_true(should_fire(quote(s * v + t * v), "v"))           # multi-pass
  expect_true(should_fire(quote(3 * v), "v"))                   # single-pass, now threaded
  expect_true(should_fire(quote(cos(sum(v^2)) * 2 * v), "v"))   # composite single-pass
  expect_false(should_fire(quote(v), "v"))                      # bare copy — excluded
  expect_false(should_fire(quote(-v), "v"))                     # negation copy — excluded
  expect_false(should_fire(quote(2), "v"))                      # pure scalar — excluded
})

# ---- task 1.2: AST -> C++ lowering with hoisted scalar preamble ----

test_that("3*v^2 lowers to the single-pass kernel with no runtime scalars", {
  L <- lower(quote(3 * v^2), "v")
  expect_identical(L$residual, "(3.0 * (v[i] * v[i]))")
  expect_length(L$scalar_exprs, 0L)
  expect_match(L$cpp_src, "out\\[i\\] = \\(3.0 \\* \\(v\\[i\\] \\* v\\[i\\]\\)\\);", fixed = FALSE)
})

test_that("reductions are hoisted to the preamble via fast_sum_sq", {
  L <- lower(quote(4 * sum(v^2) * v), "v")
  scalar_txt <- vapply(L$scalar_exprs, function(e) paste(deparse(e), collapse = ""), character(1))
  expect_true(any(grepl("fast_sum_sq", scalar_txt)))
  # the reduction is computed once, not per element
  expect_false(grepl("sum\\(", L$residual))
})

test_that("cache key is dimension-free (no n, identical across calls)", {
  k1 <- lower(quote(3 * v^2), "v")$cache_key
  k2 <- lower(quote(3 * v^2), "v")$cache_key
  expect_identical(k1, k2)
  expect_false(grepl("\\bn\\b|length", k1))   # key carries no dimension
  expect_match(k1, "^f64:")                   # dtype-tagged
})

# ---- tasks 3.1-3.3: dispatch, equivalence, dimension-independence, fallback ----

jit_ready <- isTRUE(tryCatch(get(".jit_path_available", dat_ns)(), error = function(e) FALSE))

test_that("compound arithmetic gradient routes through the fused backend", {
  skip_if_not(jit_ready, "JIT path not available")
  withr::local_options(DefDiff.jit_threshold = 1000L)  # force fused at small n
  gf <- grad(function(v) sum(v^3))
  expect_true(any(grepl(".dat_fused_try", deparse(body(gf)), fixed = TRUE)))
  set.seed(7); v <- rnorm(5000)
  expect_equal(gf(v), 3 * v^2, tolerance = 1e-10)
})

test_that("fused gradient is dimension-independent (same gf, two lengths)", {
  skip_if_not(jit_ready, "JIT path not available")
  withr::local_options(DefDiff.jit_threshold = 1000L)
  gf <- grad(function(v) sum(v^3))
  set.seed(8); v1 <- rnorm(1500); v2 <- rnorm(2e5)
  expect_equal(gf(v1), 3 * v1^2, tolerance = 1e-10)
  expect_equal(gf(v2), 3 * v2^2, tolerance = 1e-10)
})

test_that("below threshold the result still matches (base path)", {
  skip_if_not(jit_ready, "JIT path not available")
  withr::local_options(DefDiff.jit_threshold = 1e9L)  # never fuse
  gf <- grad(function(v) sum(v^3))
  set.seed(9); v <- rnorm(2000)
  expect_equal(gf(v), 3 * v^2, tolerance = 1e-10)
})

test_that("single-pass fusable shapes are threaded above threshold, not below (extend-fused-jit-single-pass)", {
  skip_if_not(jit_ready, "JIT path not available")
  withr::local_options(DefDiff.jit_threshold = 1000L)
  # 2*v: fused branch IS emitted now, carrying the Metal-window upper bound.
  bd2 <- deparse(body(grad(function(v) sum(v^2))))
  expect_true(any(grepl(".dat_fused_try", bd2, fixed = TRUE)))
  expect_true(any(grepl("metal_threshold", bd2, fixed = TRUE)))   # metal-eligible window
  # cos(v) (per-element transcendental) is not fusable -> never fused.
  expect_false(any(grepl(".dat_fused_try", deparse(body(grad(function(v) sum(sin(v))))), fixed = TRUE)))
  # Below the JIT threshold nothing is compiled (stays on the single-kernel path).
  st <- get(".dat_jit_state", dat_ns); c0 <- st$compile_count
  gf <- grad(function(v) sum(v^2)); set.seed(1); invisible(gf(rnorm(100)))  # n < threshold
  expect_identical(st$compile_count, c0)
})

test_that("Metal-eligible shape yields to Metal above metal_threshold (fused window respected)", {
  skip_if_not(jit_ready, "JIT path not available")
  withr::local_options(DefDiff.jit_threshold = 1000L, DefDiff.metal_threshold = 500L)
  st <- get(".dat_jit_state", dat_ns); c0 <- st$compile_count
  gf <- grad(function(v) sum(v^2))           # canonical 2*v -> Metal-eligible
  set.seed(1); invisible(gf(rnorm(2000)))    # n >= jit(1000) but >= metal(500) -> NOT fused
  expect_identical(st$compile_count, c0)      # fused did not fire; base (Metal/vDSP) took it
})

test_that("single-pass fused path matches closed form (2*v and composite)", {
  skip_if_not(jit_ready, "JIT path not available")
  withr::local_options(DefDiff.jit_threshold = 1000L)   # metal_threshold default 1e9 -> fused fires
  set.seed(3); v <- rnorm(5000)
  expect_equal(grad(function(v) sum(v^2))(v), 2 * v, tolerance = 1e-10)
  expect_equal(grad(function(v) sin(sum(v^2)))(v),
               as.numeric(cos(sum(v^2)) * 2 * v), tolerance = 1e-10)
})

test_that("graceful fallback when JIT disabled still produces correct values", {
  skip_if_not(jit_ready, "JIT path not available")
  withr::local_options(DefDiff.jit_disable = TRUE)
  st <- get(".dat_jit_state", dat_ns)
  st$available <- NULL
  gf <- grad(function(v) sum(v^3))
  set.seed(10); v <- rnorm(3000)
  expect_equal(gf(v), 3 * v^2, tolerance = 1e-10)
  st$available <- NULL
})

test_that("symbolic inspectability survives fused dispatch", {
  skip_if_not(jit_ready, "JIT path not available")
  withr::local_options(DefDiff.jit_threshold = 1000L)
  gf <- grad(function(v) sum(v^3))
  expect_equal(grad_expr(gf), quote(3 * v^2))
})

# ---- auto-tune dispatch (#5) ----

test_that("auto-tune: below floor always base — no probe, no compile", {
  skip_if_not(jit_ready, "JIT path not available")
  st <- get(".dat_jit_state", dat_ns)
  withr::local_options(DefDiff.jit_threshold = NULL, DefDiff.autotune = TRUE,
                       DefDiff.autotune_floor = 1e5)
  st$probe_count <- 0L; c0 <- st$compile_count
  gf <- grad(function(v) sum(v^3))
  set.seed(1); invisible(gf(rnorm(1000)))            # n=1e3 < floor -> base only
  expect_identical(st$probe_count, 0L)
  expect_identical(st$compile_count, c0)
})

test_that("auto-tune: above floor probes once, then reuses the learned winner", {
  skip_if_not(jit_ready, "JIT path not available")
  st <- get(".dat_jit_state", dat_ns)
  withr::local_options(DefDiff.jit_threshold = NULL, DefDiff.autotune = TRUE,
                       DefDiff.autotune_floor = 100)
  rm(list = ls(st$pathchoice), envir = st$pathchoice)  # unlearned shape@bucket
  st$probe_count <- 0L
  gf <- grad(function(v) sum(v^3))
  set.seed(2); v <- rnorm(2000)
  out1 <- gf(v); p1 <- st$probe_count
  out2 <- gf(v); p2 <- st$probe_count
  expect_equal(p1, 1L)                               # first call probed
  expect_equal(p2, 1L)                               # second reused learned choice
  expect_equal(out1, 3 * v^2, tolerance = 1e-10)     # correct regardless of chosen path
  expect_equal(out2, 3 * v^2, tolerance = 1e-10)
})

test_that("auto-tune disabled reverts to the static threshold gate (no probe)", {
  skip_if_not(jit_ready, "JIT path not available")
  st <- get(".dat_jit_state", dat_ns)
  withr::local_options(DefDiff.jit_threshold = NULL, DefDiff.autotune = FALSE)
  st$probe_count <- 0L
  gf <- grad(function(v) sum(v^3))
  set.seed(3); v <- rnorm(2000)                      # < default threshold 1e7 -> base, no probe
  expect_equal(gf(v), 3 * v^2, tolerance = 1e-10)
  expect_identical(st$probe_count, 0L)
})

# ---- auto-tune probe hardening (#6) ----

test_that(".dat_probe_timed resolves a measurable per-iter time and propagates NULL", {
  pt <- get(".dat_probe_timed", dat_ns)
  r <- pt(function() sum(runif(2000)))     # cheap thunk -> repeat-K until measurable
  expect_true(is.finite(r$t) && r$t > 0)
  expect_false(is.null(r$val))
  rn <- pt(function() NULL)                # unavailable path
  expect_null(rn$val)
})

test_that("auto-tune cross-checks base vs fused, fails safe to base on disagreement (#6)", {
  st <- get(".dat_jit_state", dat_ns); ad <- get(".dat_auto_dispatch", dat_ns)
  withr::local_options(DefDiff.jit_threshold = NULL, DefDiff.autotune = TRUE,
                       DefDiff.autotune_floor = 100)
  rm(list = ls(st$pathchoice), envir = st$pathchoice)
  set.seed(1); v <- rnorm(2000)
  expect_warning(
    out <- ad("f64:test_disagree", v,
              fused = function() 2 * v + 1,    # WRONG (disagrees with base)
              base  = function() 2 * v,
              metal_eligible = FALSE),
    "disagree")
  expect_equal(out, 2 * v, tolerance = 1e-10)  # fell safe to base
})

test_that("auto-tune periodically re-probes to escape a noise-frozen choice (#6)", {
  st <- get(".dat_jit_state", dat_ns); ad <- get(".dat_auto_dispatch", dat_ns)
  withr::local_options(DefDiff.jit_threshold = NULL, DefDiff.autotune = TRUE,
                       DefDiff.autotune_floor = 100, DefDiff.autotune_reprobe = 4L)
  rm(list = ls(st$pathchoice), envir = st$pathchoice)
  st$probe_count <- 0L
  set.seed(2); v <- rnorm(2000)
  bt <- function() 2 * v; ft <- function() 2 * v
  for (i in 1:4) invisible(ad("f64:test_reprobe", v, ft, bt, FALSE))
  # call 1 probes; 2,3 cached; call 4 (uses %% 4 == 0) re-probes -> 2 probes total
  expect_equal(st$probe_count, 2L)
})

test_that(".dat_paths_agree ignores attributes — a named input doesn't spuriously disagree (#6 verify)", {
  pa <- get(".dat_paths_agree", dat_ns)
  a <- c(x = 1, y = 2, z = 3); b <- c(1, 2, 3)   # equal values, `a` is named
  expect_true(pa(a, b))                          # was FALSE pre-fix (all.equal attr check)
  expect_false(pa(c(1, 2, 3), c(1, 2, 4)))       # genuine value divergence still caught
})

test_that("auto-tune re-probes by wall-clock age (multi-shape recovery, #7)", {
  st <- get(".dat_jit_state", dat_ns); ad <- get(".dat_auto_dispatch", dat_ns)
  withr::local_options(DefDiff.jit_threshold = NULL, DefDiff.autotune = TRUE,
                       DefDiff.autotune_floor = 100, DefDiff.autotune_reprobe = 1e9,
                       DefDiff.autotune_reprobe_secs = 0.001)
  rm(list = ls(st$pathchoice), envir = st$pathchoice)
  st$probe_count <- 0L
  set.seed(7); v <- rnorm(2000)
  bt <- function() 2 * v; ft <- function() 2 * v
  invisible(ad("f64:test_age", v, ft, bt, FALSE))   # probe 1 (stamps probed_at)
  Sys.sleep(0.05)                                    # exceed reprobe_secs (0.001)
  invisible(ad("f64:test_age", v, ft, bt, FALSE))    # uses=2 (count never fires) but aged -> re-probe
  expect_equal(st$probe_count, 2L)                   # recovered by AGE, not use-count
})

# Tests for dd_freeze (#8): bake the learned dispatch into the emitted body.
# Correctness is timing-independent; the "dispatch removed" claims are asserted
# on the AST (robust under load), not on wall-clock.

dat_ns <- asNamespace("DefDiff")
jit_ready <- isTRUE(tryCatch(get(".jit_path_available", dat_ns)(), error = function(e) FALSE))

test_that("frozen-to-base bodies drop the dispatcher, getOption, and the Metal window", {
  withr::local_options(DefDiff.jit_threshold = NULL, DefDiff.autotune = TRUE,
                       DefDiff.autotune_floor = 1e5)
  set.seed(1); v <- rnorm(1000)
  for (f in list(function(v) sum(v^2), function(v) sum(v^3))) {
    gf <- grad(f)
    gfz <- dd_freeze(gf, n_hint = 1000)           # below floor -> base, no probe
    txt <- paste(deparse(body(gfz)), collapse = " ")
    expect_false(grepl(".dat_auto_dispatch", txt, fixed = TRUE))
    expect_false(grepl("getOption", txt, fixed = TRUE))
    expect_false(grepl(".metal_path_available", txt, fixed = TRUE))  # Metal-if resolved away
    expect_equal(gfz(v), gf(v), tolerance = 1e-12)                   # frozen == unfrozen
  }
})

test_that("frozen gradients match the closed form across shapes and sizes", {
  withr::local_options(DefDiff.jit_threshold = NULL, DefDiff.autotune = TRUE,
                       DefDiff.autotune_floor = 1e5)
  cases <- list(
    list(f = function(v) sum(v^2),      g = function(v) 2 * v),
    list(f = function(v) sum(v^3),      g = function(v) 3 * v^2),
    list(f = function(v) sin(sum(v^2)), g = function(v) cos(sum(v^2)) * 2 * v)
  )
  for (cs in cases) {
    gfz <- dd_freeze(grad(cs$f), n_hint = 1000)
    for (n in c(100L, 5000L)) {
      set.seed(n); v <- rnorm(n)
      expect_equal(as.numeric(gfz(v)), as.numeric(cs$g(v)), tolerance = 1e-10)
    }
  }
})

test_that("freezing at a fused-winning bucket emits the bare fused call with NULL fallback", {
  skip_if_not(jit_ready, "JIT path not available")
  st <- get(".dat_jit_state", dat_ns); lower <- get(".lower_fused_cpp", dat_ns)
  withr::local_options(DefDiff.jit_threshold = NULL, DefDiff.autotune = TRUE,
                       DefDiff.autotune_floor = 100)
  gf <- grad(function(v) sum(v^3))
  n_hint <- 2000
  key <- paste0(lower(quote(3 * v^2), "v")$cache_key, "@", floor(2 * log10(n_hint)))
  st$pathchoice[[key]] <- list(choice = "fused", uses = 1L,
                               probed_at = as.numeric(Sys.time()))  # pre-learned: no probe
  gfz <- dd_freeze(gf, n_hint = n_hint)
  txt <- paste(deparse(body(gfz)), collapse = " ")
  expect_true(grepl(".dat_fused_try", txt, fixed = TRUE))           # bare fused call kept
  expect_false(grepl(".dat_auto_dispatch", txt, fixed = TRUE))      # dispatcher gone
  set.seed(2); v <- rnorm(n_hint)
  expect_equal(gfz(v), 3 * v^2, tolerance = 1e-10)
  rm(list = key, envir = st$pathchoice)
})

test_that("an unlearned bucket is probed once through the existing dispatcher", {
  skip_if_not(jit_ready, "JIT path not available")
  st <- get(".dat_jit_state", dat_ns)
  withr::local_options(DefDiff.jit_threshold = NULL, DefDiff.autotune = TRUE,
                       DefDiff.autotune_floor = 100)
  rm(list = ls(st$pathchoice), envir = st$pathchoice)
  st$probe_count <- 0L
  gfz <- dd_freeze(grad(function(v) sum(v^3)), n_hint = 2000)
  expect_identical(st$probe_count, 1L)            # learned via the EXISTING machinery
  set.seed(3); v <- rnorm(2000)
  expect_equal(gfz(v), 3 * v^2, tolerance = 1e-10)
})

test_that("attributes survive: grad_expr preserved, dd_frozen stamped", {
  gfz <- dd_freeze(grad(function(v) sum(v^3)), n_hint = 1000)
  expect_equal(attr(gfz, "grad_expr"), quote(3 * v^2))
  expect_equal(attr(gfz, "dd_frozen"), 1000)
})

test_that("non-fusable gradients pass through unchanged with a message", {
  gf <- grad(function(v) sum(sin(v)))             # body: bare fast_vv_cos(v)
  expect_message(gfz <- dd_freeze(gf, n_hint = 1000), "no dispatcher")
  expect_identical(body(gfz), body(gf))
  set.seed(4); v <- rnorm(500)
  expect_equal(as.numeric(gfz(v)), cos(v), tolerance = 1e-10)
})

test_that("dd_freeze fails fast on bad input", {
  gf <- grad(function(v) sum(v^2))
  expect_error(dd_freeze(function(v) v, n_hint = 10), class = "DefDiff_not_definable")
  expect_error(dd_freeze(gf), class = "DefDiff_not_definable")            # n_hint required
  expect_error(dd_freeze(gf, n_hint = NA), class = "DefDiff_not_definable")
  expect_error(dd_freeze(gf, n_hint = -5), class = "DefDiff_not_definable")
  expect_error(dd_freeze(grad(function(v, w) sum(v^2) + sum(w^2)), n_hint = 10),
               class = "DefDiff_not_definable")   # single-vector contract
})

# Noise-sensitive ratio test: same skip conventions as test-performance.R.
test_that("[gated] dd_freeze cuts per-call overhead at small n (#8 speed proof)", {
  skip_on_cran(); skip_on_ci()
  skip_if_not_installed("bench")
  withr::local_options(DefDiff.jit_threshold = NULL, DefDiff.autotune = TRUE)
  gf  <- grad(function(v) sum(v^2))
  gfz <- dd_freeze(gf, n_hint = 1000)
  set.seed(1); v <- rnorm(1000)
  invisible(gf(v)); invisible(gfz(v))
  bu <- bench::mark(gf(v),  iterations = 2000, check = FALSE)
  bf <- bench::mark(gfz(v), iterations = 2000, check = FALSE)
  # measured 0.18x under load, 60% is a conservative margin
  expect_lt(as.numeric(summary(bf)$median), as.numeric(summary(bu)$median) * 0.60)
})

# ---- #8 verify in-scope fixes ----

test_that("fractional n_hint is floored so freeze-key and probe agree", {
  skip_if_not(jit_ready, "JIT path not available")
  st <- get(".dat_jit_state", dat_ns); lower <- get(".lower_fused_cpp", dat_ns)
  withr::local_options(DefDiff.jit_threshold = NULL, DefDiff.autotune = TRUE,
                       DefDiff.autotune_floor = 100)
  # Pre-seed fused at the FLOORED bucket; pre-fix the unfloored key missed it.
  key <- paste0(lower(quote(3 * v^2), "v")$cache_key, "@", floor(2 * log10(3162)))
  st$pathchoice[[key]] <- list(choice = "fused", uses = 1L,
                               probed_at = as.numeric(Sys.time()))
  gfz <- dd_freeze(grad(function(v) sum(v^3)), n_hint = 3162.5)
  expect_equal(attr(gfz, "dd_frozen"), 3162)                       # floored
  expect_true(grepl(".dat_fused_try", paste(deparse(body(gfz)), collapse = " "),
                    fixed = TRUE))                                  # learned entry FOUND
  rm(list = key, envir = st$pathchoice)
})

test_that("static-gate override is honored: explicit jit_threshold decides the frozen winner", {
  skip_if_not(jit_ready, "JIT path not available")
  st <- get(".dat_jit_state", dat_ns)
  withr::local_options(DefDiff.jit_threshold = 1000L)               # user's explicit config
  st$probe_count <- 0L
  gf <- grad(function(v) sum(v^3))
  gfz_hi <- dd_freeze(gf, n_hint = 2000)                            # >= thr -> fused
  expect_true(grepl(".dat_fused_try", paste(deparse(body(gfz_hi)), collapse = " "), fixed = TRUE))
  gfz_lo <- dd_freeze(gf, n_hint = 500)                             # < thr -> base
  expect_false(grepl(".dat_fused_try", paste(deparse(body(gfz_lo)), collapse = " "), fixed = TRUE))
  expect_identical(st$probe_count, 0L)                              # no probe under override
  set.seed(5); v <- rnorm(2000)
  expect_equal(gfz_hi(v), 3 * v^2, tolerance = 1e-10)
  expect_equal(gfz_lo(v[1:500]), 3 * v[1:500]^2, tolerance = 1e-10)
})

test_that("a drifted dispatcher call (wrong arity) fails open with a warning", {
  fake <- function(v) NULL
  body(fake) <- quote({ .dat_s <- double(0); .dat_auto_dispatch("k", v) })  # 3 elems, not 6
  attr(fake, "grad_expr") <- quote(2 * v)
  expect_warning(out <- dd_freeze(fake, n_hint = 10), "unrecognized")
  expect_identical(body(out), body(fake))                           # passthrough, not crash
})

test_that("frozen-fused NULL fallback actually falls back to base (codex finding)", {
  skip_if_not(jit_ready, "JIT path not available")
  st <- get(".dat_jit_state", dat_ns); lower <- get(".lower_fused_cpp", dat_ns)
  withr::local_options(DefDiff.jit_threshold = NULL, DefDiff.autotune = TRUE,
                       DefDiff.autotune_floor = 100)
  n_hint <- 2000
  key <- paste0(lower(quote(3 * v^2), "v")$cache_key, "@", floor(2 * log10(n_hint)))
  st$pathchoice[[key]] <- list(choice = "fused", uses = 1L,
                               probed_at = as.numeric(Sys.time()))
  gfz <- dd_freeze(grad(function(v) sum(v^3)), n_hint = n_hint)
  # Shadow .dat_fused_try with an always-NULL stub in a child env -> forces the
  # fallback arm of the frozen body without touching the namespace.
  shadow <- new.env(parent = environment(gfz))
  shadow$.dat_fused_try <- function(...) NULL
  environment(gfz) <- shadow
  set.seed(6); v <- rnorm(n_hint)
  expect_equal(gfz(v), 3 * v^2, tolerance = 1e-10)  # base fallback produced the answer
  rm(list = key, envir = st$pathchoice)
})

test_that("re-freezing a frozen gradient messages distinctly and passes through", {
  gfz <- dd_freeze(grad(function(v) sum(v^2)), n_hint = 1000)
  expect_message(gfz2 <- dd_freeze(gfz, n_hint = 5000), "already frozen")
  expect_identical(body(gfz2), body(gfz))
  expect_equal(attr(gfz2, "dd_frozen"), 1000)        # original stamp retained, not 5000
})

## test-performance.R
##
## SPEED / BENCHMARK tests for the `dat` package (testthat 3e + bench::mark).
##
## ---------------------------------------------------------------------------
## What these tests assert, and what they deliberately do NOT assert
## ---------------------------------------------------------------------------
## * RELATIVE speed only. Every assertion is a one-sided inequality on a
##   RATIO OF MEDIANS (bench::mark median column). There are NO absolute
##   wall-clock-ms thresholds and NO exact-factor assertions -- absolute ms
##   and exact factors are machine/CI-dependent and flake.
## * The HEADLINE, ROBUST claim is: the symbolic DD operators (grad/hessian/
##   jacobian) beat numDeriv's finite differences by ORDERS OF MAGNITUDE.
##   DD does O(1) source-transformation passes producing a closed-form
##   derivative function; numDeriv pays O(n) function evals for grad/jacobian
##   and O(n^2) for the Hessian. Measured margins on the dev machine
##   (R 4.6.0, Apple Silicon, Darwin) range ~1800x-14500x; assertions are set
##   to a SAFE FRACTION of the measured margin (typically /10, sometimes /5
##   or /50) so loaded-machine noise can never flip the inequality.
## * The vDSP/vForce/Metal KERNEL-vs-base-R comparisons are honest about the
##   regime: transcendental vForce kernels (sin/cos/sqrt/tanh/log) genuinely
##   win, so those carry a real (still conservative) speedup assertion; but
##   single-pass arithmetic (fast_scalar_mul vs 2*v) and the two-pass
##   fast_sum_pow are a TIE or a LOSS vs base-R, so those carry only a
##   NON-REGRESSION ceiling -- never a fabricated "fast wins" claim.
## * Scaling tests compare the SAME fast path against ITSELF at a smaller n
##   to detect accidental O(n^2): a linear kernel's decade ratio stays near
##   10, a quadratic one would be ~100.
##
## ---------------------------------------------------------------------------
## Skip discipline (house rules)
## ---------------------------------------------------------------------------
## Every test starts with skip_on_cran() + skip_on_ci() + skip_if_not_installed("bench").
## numDeriv baselines add skip_if_not_installed("numDeriv").
## Any vDSP/vForce/Metal fast-path-dependent test adds
##   skip_on_os(c("windows","linux","solaris"))   (off macOS the fast path
##   falls back to base-R, so there is no speedup to assert) plus, for the
##   universal safety net, skip_if(!DefDiff:::.fast_path_available()).
## Metal tests additionally skip_if(!DefDiff:::.metal_path_available()).
##
## ---------------------------------------------------------------------------
## Methodology notes baked into the helpers (measured pitfalls on this machine)
## ---------------------------------------------------------------------------
## * bench::mark()'s `expression` column is a `bench_expr` (a list), so
##   `bm$median[bm$expression == "label"]` silently returns length 0 (a
##   false pass / NA hazard). We match with `as.character(bm$expression)`.
## * `bm$median` is a `bench_time` S3 vector; coerce with
##   `as.numeric(unclass(.))` before any arithmetic / comparison.
## * Read the MEDIAN, never min/max: at large n a single GC/scheduler hiccup
##   makes per-iteration extremes swing by 100x+ on a sub-ms op; the median
##   is stable, the extremes are not.
## * check = FALSE (numDeriv is an approximation with a different float
##   representation; correctness is asserted in the dedicated numeric-
##   equivalence tests, never here). filter_gc = FALSE keeps the iteration
##   count stable so the median does not collapse on a loaded machine.
## * Warm up EVERY expression once before bench::mark (first call pays
##   allocation / bytecode warm-up).

# --------------------------------------------------------------------------
# Shared helpers (local to this file)
# --------------------------------------------------------------------------

# bench_time (S3) -> plain numeric seconds.
.as_sec <- function(x) as.numeric(unclass(x))

# Median (in seconds) for a labelled bench::mark row. Matches on the DEPARSED
# expression text via as.character(bm$expression) -- the only reliable key,
# because bm$expression is a bench_expr/list and `==` against a label fails.
.bm_median <- function(bm, label) {
  idx <- which(as.character(bm$expression) == label)
  if (length(idx) != 1L) {
    stop(sprintf("bench row '%s' not found uniquely (found %d)", label, length(idx)))
  }
  .as_sec(bm$median[idx])
}

# Median (in seconds) of a single-expression bench::mark by row position.
.bm_median1 <- function(bm) .as_sec(bm$median[1])

# Standardised two-expression measurement: warm up both, then bench::mark
# with the safe option set. Labels are the deparsed call texts the caller
# passes (so .bm_median can find them).
.bench2 <- function(fast_quo, ref_quo, env, min_iterations = 30L) {
  force(env)
  eval(fast_quo, env)   # warm up fast path
  eval(ref_quo,  env)   # warm up baseline
  # Build a bench::mark call whose argument NAMES are the deparsed expression
  # texts, so as.character(bm$expression) matches the labels the tests pass to
  # .bm_median (e.g. "gf(v)", "nd_grad(f, v)"). Pre-naming the quotes
  # fixes the earlier bug where bench labelled the rows "fast"/"ref" and the
  # text lookups found 0 rows.
  exprs <- stats::setNames(
    list(fast_quo, ref_quo),
    c(paste(deparse(fast_quo), collapse = ""),
      paste(deparse(ref_quo),  collapse = ""))
  )
  cl <- as.call(c(quote(bench::mark), exprs,
                  list(check = FALSE, filter_gc = FALSE,
                       min_iterations = min_iterations)))
  eval(cl, env)
}

# Decade self-scaling ratio: median(at big n) / median(at small n) for the
# SAME fast-path function. Used by the scaling tier to detect O(n^2).
.scale_ratio <- function(fn, v_small, v_big,
                         iters_small = 15L, iters_big = 15L) {
  invisible(fn(v_small)); invisible(fn(v_big))   # warm up both sizes
  bm_s <- bench::mark(fn(v_small), check = FALSE, filter_gc = FALSE,
                      min_iterations = iters_small)
  bm_b <- bench::mark(fn(v_big),   check = FALSE, filter_gc = FALSE,
                      min_iterations = iters_big)
  .bm_median1(bm_b) / .bm_median1(bm_s)
}


# ==========================================================================
# TIER: DD operators vs numDeriv (headline robust claim, large margins)
# --------------------------------------------------------------------------
# Symbolic grad/hessian/jacobian (O(1) source-transformation passes) vs the
# matching numDeriv finite-difference routine (O(n) evals for grad/jacobian,
# O(n^2) for the Hessian). n is kept SMALL (50-200) because finite difference
# is unusably slow at large n -- that IS the point. These do NOT route through
# the vDSP/Metal fast path at small n (the win is purely algorithmic, identical
# on every platform), so NO macOS guard is needed here.
# ==========================================================================

test_that("DD grad operator (sum(v^2)) beats numDeriv::grad by orders of magnitude", {
  skip_on_cran(); skip_on_ci()
  skip_if_not_installed("bench"); skip_if_not_installed("numDeriv")

  f  <- function(v) sum(v^2)
  gf <- grad(f)
  set.seed(1); v <- runif(200L)

  bm <- .bench2(quote(gf(v)), quote(nd_grad(f, v)),
                environment(), min_iterations = 50L)
  dd <- .bm_median(bm, "gf(v)")
  nd <- .bm_median(bm, "nd_grad(f, v)")

  # Measured ~2900x-3300x. Assert >10x (leaves ~290x slack); unflakeable.
  expect_lt(dd, nd / 10)
})

test_that("DD grad operator (sin(sum(v^2))) beats numDeriv::grad by orders of magnitude", {
  skip_on_cran(); skip_on_ci()
  skip_if_not_installed("bench"); skip_if_not_installed("numDeriv")

  f  <- function(v) sin(sum(v^2))
  gf <- grad(f)
  set.seed(1); v <- runif(200L)

  bm <- .bench2(quote(gf(v)), quote(nd_grad(f, v)),
                environment(), min_iterations = 50L)
  dd <- .bm_median(bm, "gf(v)")
  nd <- .bm_median(bm, "nd_grad(f, v)")

  # Measured ~3400x-4000x. Assert >10x (~340x slack).
  expect_lt(dd, nd / 10)
})

test_that("DD hessian operator (sum(v^2)) crushes numDeriv::hessian (O(n^2) finite diff)", {
  skip_on_cran(); skip_on_ci()
  skip_if_not_installed("bench"); skip_if_not_installed("numDeriv")

  f  <- function(v) sum(v^2)
  hf <- hessian(f)
  set.seed(1); v <- runif(50L)

  bm <- .bench2(quote(hf(v)), quote(nd_hessian(f, v)),
                environment(), min_iterations = 30L)
  dd <- .bm_median(bm, "hf(v)")
  nd <- .bm_median(bm, "nd_hessian(f, v)")

  # Measured ~10000x-14500x (finite-diff Hessian is quadratic in n).
  # Assert >10x (~1000x slack); /10 kept uniform with the tier.
  expect_lt(dd, nd / 10)
})

test_that("DD hessian operator (sin(sum(v^2))) crushes numDeriv::hessian", {
  skip_on_cran(); skip_on_ci()
  skip_if_not_installed("bench"); skip_if_not_installed("numDeriv")

  f  <- function(v) sin(sum(v^2))
  hf <- hessian(f)
  set.seed(1); v <- runif(50L)

  bm <- .bench2(quote(hf(v)), quote(nd_hessian(f, v)),
                environment(), min_iterations = 30L)
  dd <- .bm_median(bm, "hf(v)")
  nd <- .bm_median(bm, "nd_hessian(f, v)")

  # Measured ~3000x-3300x (recursive Jacobian-of-gradient walker). Assert >10x.
  expect_lt(dd, nd / 10)
})

test_that("DD jacobian operator (c(sum(v),sum(v^2))) beats numDeriv::jacobian by orders of magnitude", {
  skip_on_cran(); skip_on_ci()
  skip_if_not_installed("bench"); skip_if_not_installed("numDeriv")

  f  <- function(v) c(sum(v), sum(v^2))
  jf <- jacobian(f)
  set.seed(1); v <- runif(100L)

  bm <- .bench2(quote(jf(v)), quote(nd_jacobian(f, v)),
                environment(), min_iterations = 50L)
  dd <- .bm_median(bm, "jf(v)")
  nd <- .bm_median(bm, "nd_jacobian(f, v)")

  # Measured ~1800x-2150x (smallest margin in the tier, still huge).
  # Assert >10x (~180x slack).
  expect_lt(dd, nd / 10)
})


# ==========================================================================
# TIER 1: vDSP scalar path (fast_scalar_mul) -- HONEST TIE vs base-R
# --------------------------------------------------------------------------
# grad(sum(v^2)) emits a body dispatching to fast_scalar_mul(2, v) (vDSP) at
# the default high Metal threshold. Both the kernel and base-R `2*v` are
# single-pass, memory-bandwidth-bound; base-R `*` is already optimized C, so
# vDSP does NOT win -- it is a near-tie (measured ~1.0x, occasionally a hair
# slower). Therefore NO speedup is asserted; only a NON-REGRESSION ceiling
# (fast < base * 2) that catches a true regression (e.g. silent fall-through
# to a slow path) while being immune to tie-regime noise. The numDeriv win for
# this exact Tier-1 body lives above; here we document the honest kernel story.
# ==========================================================================

test_that("Tier 1 vDSP gradient path is not a regression vs base-R 2*v (honest tie)", {
  skip_on_cran(); skip_on_ci()
  skip_if_not_installed("bench")
  skip_on_os(c("windows", "linux", "solaris"))
  skip_if(!DefDiff:::.fast_path_available(), "vDSP fast path not available")

  gf <- grad(function(v) sum(v^2))   # -> fast_scalar_mul(2, v) via vDSP
  set.seed(1); v <- runif(1e6L)

  bm <- .bench2(quote(gf(v)), quote(2 * v), environment(), min_iterations = 30L)
  fast <- .bm_median(bm, "gf(v)")
  ref  <- .bm_median(bm, "2 * v")

  # Measured fast/base ~1.0x (TIE). Non-regression ceiling 2x base.
  expect_lt(fast, ref * 2)
})

test_that("fast_scalar_mul vDSP kernel is not a regression vs base-R * (honest tie)", {
  skip_on_cran(); skip_on_ci()
  skip_if_not_installed("bench")
  skip_on_os(c("windows", "linux", "solaris"))
  skip_if(!DefDiff:::.fast_path_available(), "vDSP fast path not available")

  set.seed(1); v <- runif(1e6L)

  bm <- .bench2(quote(fast_scalar_mul(2, v)), quote(2 * v),
                environment(), min_iterations = 30L)
  fast <- .bm_median(bm, "fast_scalar_mul(2, v)")
  ref  <- .bm_median(bm, "2 * v")

  # Measured kernel/base ~1.0x (TIE). Non-regression ceiling 2x base.
  expect_lt(fast, ref * 2)
})


# ==========================================================================
# TIER 2c: vForce elementwise kernels (fast_vv_*) -- REAL SIMD wins
# --------------------------------------------------------------------------
# Apple Accelerate vForce (vvsin/vvcos/vvsqrt/vvtanh/vvlog/vvexp) genuinely
# beats base-R's libm loop for transcendentals. Each threshold is a safe
# fraction (~half or less) of the measured worst-case floor. exp is the
# documented weak case (~1.6x, fragile) so it gets a NON-REGRESSION guard, not
# a speedup claim. The operator-level test confirms grad() carries the win
# end-to-end. v in (0.1, 3) to keep log/sqrt well-defined.
# ==========================================================================

test_that("DD grad operator (sin -> fast_vv_cos) beats numDeriv by a wide margin", {
  skip_on_cran(); skip_on_ci()
  skip_if_not_installed("bench"); skip_if_not_installed("numDeriv")
  skip_on_os(c("windows", "linux", "solaris"))
  skip_if(!DefDiff:::.fast_path_available(), "vForce fast path not available")

  f  <- function(v) sum(sin(v))
  gf <- grad(f)
  set.seed(1); v <- runif(200L, 0.1, 3)

  bm <- .bench2(quote(gf(v)), quote(nd_grad(f, v)),
                environment(), min_iterations = 50L)
  dd <- .bm_median(bm, "gf(v)")
  nd <- .bm_median(bm, "nd_grad(f, v)")

  # Measured ~6900x. Assert >50x (~138x slack).
  expect_lt(dd, nd / 50)
})

test_that("fast_vv_sin kernel is faster than base-R sin at n=1e6", {
  skip_on_cran(); skip_on_ci()
  skip_if_not_installed("bench")
  skip_on_os(c("windows", "linux", "solaris"))
  skip_if(!DefDiff:::.fast_path_available(), "vForce fast path not available")

  set.seed(7); v <- runif(1e6L, 0.1, 3)
  bm <- .bench2(quote(fast_vv_sin(v)), quote(sin(v)),
                environment(), min_iterations = 30L)
  fast <- .bm_median(bm, "fast_vv_sin(v)")
  ref  <- .bm_median(bm, "sin(v)")

  # Measured ~7x-8x. Assert >3x (worst floor ~8x leaves >2x slack).
  expect_lt(fast, ref / 3)
})

test_that("fast_vv_cos kernel is faster than base-R cos at n=1e6", {
  skip_on_cran(); skip_on_ci()
  skip_if_not_installed("bench")
  skip_on_os(c("windows", "linux", "solaris"))
  skip_if(!DefDiff:::.fast_path_available(), "vForce fast path not available")

  set.seed(7); v <- runif(1e6L, 0.1, 3)
  bm <- .bench2(quote(fast_vv_cos(v)), quote(cos(v)),
                environment(), min_iterations = 30L)
  fast <- .bm_median(bm, "fast_vv_cos(v)")
  ref  <- .bm_median(bm, "cos(v)")

  # Measured ~6.7x-7.5x. Assert >3x.
  expect_lt(fast, ref / 3)
})

test_that("fast_vv_sqrt kernel is faster than base-R sqrt at n=1e6", {
  skip_on_cran(); skip_on_ci()
  skip_if_not_installed("bench")
  skip_on_os(c("windows", "linux", "solaris"))
  skip_if(!DefDiff:::.fast_path_available(), "vForce fast path not available")

  set.seed(7); v <- runif(1e6L, 0.1, 3)
  bm <- .bench2(quote(fast_vv_sqrt(v)), quote(sqrt(v)),
                environment(), min_iterations = 30L)
  fast <- .bm_median(bm, "fast_vv_sqrt(v)")
  ref  <- .bm_median(bm, "sqrt(v)")

  # Measured ~8.4x-8.9x (strongest of the six). Assert >3x.
  expect_lt(fast, ref / 3)
})

test_that("fast_vv_tanh kernel is faster than base-R tanh at n=1e6", {
  skip_on_cran(); skip_on_ci()
  skip_if_not_installed("bench")
  skip_on_os(c("windows", "linux", "solaris"))
  skip_if(!DefDiff:::.fast_path_available(), "vForce fast path not available")

  set.seed(7); v <- runif(1e6L, 0.1, 3)
  bm <- .bench2(quote(fast_vv_tanh(v)), quote(tanh(v)),
                environment(), min_iterations = 30L)
  fast <- .bm_median(bm, "fast_vv_tanh(v)")
  ref  <- .bm_median(bm, "tanh(v)")

  # Measured ~4.0x-4.3x (lower margin). Conservative assert >2x.
  expect_lt(fast, ref / 2)
})

test_that("fast_vv_log kernel is faster than base-R log at n=1e6", {
  skip_on_cran(); skip_on_ci()
  skip_if_not_installed("bench")
  skip_on_os(c("windows", "linux", "solaris"))
  skip_if(!DefDiff:::.fast_path_available(), "vForce fast path not available")

  set.seed(7); v <- runif(1e6L, 0.1, 3)
  bm <- .bench2(quote(fast_vv_log(v)), quote(log(v)),
                environment(), min_iterations = 30L)
  fast <- .bm_median(bm, "fast_vv_log(v)")
  ref  <- .bm_median(bm, "log(v)")

  # Measured ~3.1x-3.5x. Conservative assert >2x.
  expect_lt(fast, ref / 2)
})

test_that("fast_vv_exp kernel is not slower than base-R exp at n=1e6 (non-regression, NOT a speedup)", {
  skip_on_cran(); skip_on_ci()
  skip_if_not_installed("bench")
  skip_on_os(c("windows", "linux", "solaris"))
  skip_if(!DefDiff:::.fast_path_available(), "vForce fast path not available")

  set.seed(7); v <- runif(1e6L, 0.1, 3)
  bm <- .bench2(quote(fast_vv_exp(v)), quote(exp(v)),
                environment(), min_iterations = 30L)
  fast <- .bm_median(bm, "fast_vv_exp(v)")
  ref  <- .bm_median(bm, "exp(v)")

  # Documented weak case (~1.6x, fragile). NON-REGRESSION ceiling only:
  # fast < base * 1.1 (10x of slack vs the observed ~1.6x win).
  expect_lt(fast, ref * 1.1)
})

test_that("grad(sum(sin(v))) carries the vForce win end-to-end vs base-R cos(v)", {
  skip_on_cran(); skip_on_ci()
  skip_if_not_installed("bench")
  skip_on_os(c("windows", "linux", "solaris"))
  skip_if(!DefDiff:::.fast_path_available(), "vForce fast path not available")

  g_sin <- grad(function(v) sum(sin(v)))   # derivative routes to fast_vv_cos
  set.seed(7); v <- runif(1e6L, 0.1, 3)

  bm <- .bench2(quote(g_sin(v)), quote(cos(v)),
                environment(), min_iterations = 30L)
  fast <- .bm_median(bm, "g_sin(v)")
  ref  <- .bm_median(bm, "cos(v)")

  # Measured ~6.0x-6.8x (worst 6.05). Assert >2.5x (~2.4x slack).
  expect_lt(fast, ref / 2.5)
})


# ==========================================================================
# TIER: sum-of-powers value substitution (fast_sum_sq / fast_sum_pow)
# --------------------------------------------------------------------------
# fast_sum_sq (vDSP_svesqD, single pass, no v^2 alloc) is a robust ~6x WIN vs
# base-R sum(v^2). fast_sum_pow (vvpow + vDSP_sveD, two pass) currently LOSES
# ~4.5x-5x to base-R's optimized integer powers -- so those are honest
# NON-REGRESSION guards (no speedup claimed), named accordingly.
# ==========================================================================

test_that("fast_sum_sq is much faster than base-R sum(v^2) at large n", {
  skip_on_cran(); skip_on_ci()
  skip_if_not_installed("bench")
  skip_on_os(c("windows", "linux", "solaris"))
  skip_if(!DefDiff:::.fast_path_available(), "vDSP fast path not available")

  set.seed(1); v <- runif(1e6L)
  bm <- .bench2(quote(fast_sum_sq(v)), quote(sum(v^2)),
                environment(), min_iterations = 30L)
  fast <- .bm_median(bm, "fast_sum_sq(v)")
  ref  <- .bm_median(bm, "sum(v^2)")

  # Measured ~6x (worst ~5.98x). Assert >2x faster (~3x slack).
  expect_lt(fast, ref / 2)
})

test_that("fast_sum_pow(v,3) is not catastrophically slower than base-R sum(v^3) (known regression guard)", {
  skip_on_cran(); skip_on_ci()
  skip_if_not_installed("bench")
  skip_on_os(c("windows", "linux", "solaris"))
  skip_if(!DefDiff:::.fast_path_available(), "vDSP fast path not available")

  set.seed(1); v <- runif(1e6L)
  bm <- .bench2(quote(fast_sum_pow(v, 3)), quote(sum(v^3)),
                environment(), min_iterations = 30L)
  fast <- .bm_median(bm, "fast_sum_pow(v, 3)")
  ref  <- .bm_median(bm, "sum(v^3)")

  # Measured ~4.5x-4.6x SLOWER (two-pass vvpow loses to base-R int powers).
  # NO speedup claimed; ceiling base*15 catches only a pathological blowup.
  expect_lt(fast, ref * 15)
})

test_that("fast_sum_pow(v,4) is not catastrophically slower than base-R sum(v^4) (known regression guard)", {
  skip_on_cran(); skip_on_ci()
  skip_if_not_installed("bench")
  skip_on_os(c("windows", "linux", "solaris"))
  skip_if(!DefDiff:::.fast_path_available(), "vDSP fast path not available")

  set.seed(1); v <- runif(1e6L)
  bm <- .bench2(quote(fast_sum_pow(v, 4)), quote(sum(v^4)),
                environment(), min_iterations = 30L)
  fast <- .bm_median(bm, "fast_sum_pow(v, 4)")
  ref  <- .bm_median(bm, "sum(v^4)")

  # Measured ~4.8x-5.0x SLOWER. NO speedup claimed; ceiling base*15.
  expect_lt(fast, ref * 15)
})


# ==========================================================================
# TIER: scaling (self-comparison across n -- anti-O(n^2) guard)
# --------------------------------------------------------------------------
# Compares a fast path against ITSELF at a smaller n. A linear/sub-quadratic
# kernel produces a decade (10x size) time ratio near 10; an accidental
# O(n^2) kernel would produce ~100. Decade probe uses 1e6 -> 1e7 (both far
# past kernel-launch overhead). Asserted bound 40 sits ~2.7x above the worst
# measured (~14.7) and 2.5x below the quadratic threshold (100). A wide-span
# 1e4 -> 1e7 guard (size ratio 1000; quadratic -> ~1e6) asserts < 10000.
# n never exceeds 1e7 (~80 MB doubles).
# ==========================================================================

test_that("Tier 2c grad(sum(exp(v))) scales sub-quadratically (decade ratio 1e6->1e7)", {
  skip_on_cran(); skip_on_ci()
  skip_if_not_installed("bench")
  skip_on_os(c("windows", "linux", "solaris"))
  skip_if(!DefDiff:::.fast_path_available(), "vForce fast path not available")

  gf <- grad(function(v) sum(exp(v)))
  set.seed(1); v6 <- runif(1e6L); set.seed(2); v7 <- runif(1e7L)

  decade_ratio <- .scale_ratio(gf, v6, v7)
  # Measured ~10.6-10.7 (cleanest tier). Linear ~10, quadratic ~100.
  expect_lt(decade_ratio, 40)
})

test_that("Tier 1 grad(sum(v^2)) scales sub-quadratically (decade ratio 1e6->1e7)", {
  skip_on_cran(); skip_on_ci()
  skip_if_not_installed("bench")
  skip_on_os(c("windows", "linux", "solaris"))
  skip_if(!DefDiff:::.fast_path_available(), "vDSP fast path not available")

  gf <- grad(function(v) sum(v^2))
  set.seed(1); v6 <- runif(1e6L); set.seed(2); v7 <- runif(1e7L)

  decade_ratio <- .scale_ratio(gf, v6, v7)
  # Measured ~12.8-14.7 (bandwidth-saturated, slightly noisier). Assert <40.
  expect_lt(decade_ratio, 40)
})

test_that("Tier 3 grad(sum(v^3)) scales sub-quadratically (decade ratio 1e6->1e7)", {
  skip_on_cran(); skip_on_ci()
  skip_if_not_installed("bench")
  skip_on_os(c("windows", "linux", "solaris"))
  skip_if(!DefDiff:::.fast_path_available(), "vDSP fast path not available")

  gf <- grad(function(v) sum(v^3))
  set.seed(1); v6 <- runif(1e6L); set.seed(2); v7 <- runif(1e7L)

  decade_ratio <- .scale_ratio(gf, v6, v7)
  # Measured ~12.7-13.3 (most allocation-heavy fast path). Assert <40.
  expect_lt(decade_ratio, 40)
})

test_that("grad fast path has no catastrophic blowup across the full 1e4->1e7 span", {
  skip_on_cran(); skip_on_ci()
  skip_if_not_installed("bench")
  skip_on_os(c("windows", "linux", "solaris"))
  skip_if(!DefDiff:::.fast_path_available(), "vDSP fast path not available")

  gf <- grad(function(v) sum(v^2))
  set.seed(1); v4 <- runif(1e4L); set.seed(2); v7 <- runif(1e7L)

  span_ratio <- .scale_ratio(gf, v4, v7, iters_small = 20L, iters_big = 10L)
  # Size ratio 1000 -> linear ~1000, quadratic ~1e6. Measured ~694.
  # Assert <10000 (~14x measured, ~100x below quadratic). Catch-all anti-O(n^2).
  expect_lt(span_ratio, 10000)
})


# ==========================================================================
# TIER: Metal threshold guard (add-metal-backend)
# --------------------------------------------------------------------------
# Below the default 1e9 threshold the canonical <scalar>*<var> gradient stays
# on vDSP and pays NO GPU launch / float32 conversion cost. Raw n>=1e9 GPU
# speed is untestable (8GB+), so we assert the right RELATIONSHIPS at moderate
# n (1e6): (1) the guard adds negligible overhead vs the plain vDSP kernel
# (tie); (2) the below-threshold gradient is not a regression vs base-R 2*v
# (tie); (3) the threshold is load-bearing -- forcing Metal at n=1e6 (lowered
# threshold) is ~130x SLOWER, AND lowering the threshold runs without error and
# returns a length-n result. Metal-capability guard required throughout.
# ==========================================================================

test_that("default-threshold canonical gradient adds negligible overhead vs the plain vDSP kernel", {
  skip_on_cran(); skip_on_ci()
  skip_if_not_installed("bench")
  skip_on_os(c("windows", "linux", "solaris"))
  skip_if(!DefDiff:::.metal_path_available(), "Metal backend not available")

  old <- options(DefDiff.metal_threshold = 1e9L); on.exit(options(old), add = TRUE)
  gf <- grad(function(v) sum(v^2))           # default threshold -> vDSP fall-through
  set.seed(1); v <- runif(1e6L)

  bm <- .bench2(quote(gf(v)), quote(DefDiff:::fast_scalar_mul(2, v)),
                environment(), min_iterations = 30L)
  guarded <- .bm_median(bm, "gf(v)")
  plain   <- .bm_median(bm, "DefDiff:::fast_scalar_mul(2, v)")

  # Measured guarded/plain ~1.0x (tie: guard cost is negligible). Ceiling 1.5x.
  expect_lt(guarded, plain * 1.5)
})

test_that("default-threshold gradient is not slower than base-R 2*v on the vDSP path", {
  skip_on_cran(); skip_on_ci()
  skip_if_not_installed("bench")
  skip_on_os(c("windows", "linux", "solaris"))
  skip_if(!DefDiff:::.metal_path_available(), "Metal backend not available")

  old <- options(DefDiff.metal_threshold = 1e9L); on.exit(options(old), add = TRUE)
  gf <- grad(function(v) sum(v^2))
  set.seed(1); v <- runif(1e6L)

  bm <- .bench2(quote(gf(v)), quote(2 * v), environment(), min_iterations = 30L)
  guarded <- .bm_median(bm, "gf(v)")
  ref     <- .bm_median(bm, "2 * v")

  # Measured ~1.03x (tie). NON-REGRESSION ceiling 1.5x base (no win claimed).
  expect_lt(guarded, ref * 1.5)
})

test_that("threshold guard is load-bearing: default vDSP path far faster than forced-Metal at n=1e6, and lowering does not error", {
  skip_on_cran(); skip_on_ci()
  skip_if_not_installed("bench")
  skip_on_os(c("windows", "linux", "solaris"))
  skip_if(!DefDiff:::.metal_path_available(), "Metal backend not available")

  gf <- grad(function(v) sum(v^2))
  set.seed(1); v <- runif(1e6L)

  # Default threshold (vDSP). Warm up + measure under a scoped option.
  default_med <- local({
    old <- options(DefDiff.metal_threshold = 1e9L); on.exit(options(old), add = TRUE)
    invisible(gf(v))
    bm <- bench::mark(gf(v), check = FALSE, filter_gc = FALSE, min_iterations = 20L)
    .bm_median1(bm)
  })

  # Lowered threshold FORCES Metal at n=1e6. Also assert no error + length n.
  metal_med <- local({
    old <- options(DefDiff.metal_threshold = 1L); on.exit(options(old), add = TRUE)
    metal_result <- expect_no_error(gf(v))
    expect_length(metal_result, length(v))
    invisible(gf(v))
    bm <- bench::mark(gf(v), check = FALSE, filter_gc = FALSE, min_iterations = 10L)
    .bm_median1(bm)
  })

  # Measured metal/default ~127x-143x at n=1e6 (per-call float32 conversion +
  # GPU launch dominate). Assert default < metal/10 (~13x slack); unflakeable.
  expect_lt(default_med, metal_med / 10)
})
## test-metal-backend.R
## add-metal-backend: runtime size-thresholded Metal dispatch for the canonical
## Tier 1 `<scalar> * <var>` gradient, with graceful CPU fallback. The Metal
## path (float32 GPU compute) matches the vDSP double path within ~1e-6; below
## the threshold, or when Metal is unavailable, the vDSP / base-R path runs
## unchanged. GPU-dependent assertions skip when Metal is not available.

TOL_METAL <- 1e-6

# .metal_path_available() must never raise — it returns a logical on every
# platform (the fallback contract). This runs everywhere.
test_that(".metal_path_available() returns a logical and never errors", {
  av <- expect_no_error(.metal_path_available())
  expect_true(is.logical(av) && length(av) == 1L && !is.na(av))
})

# Default (high) threshold keeps a small vector on the CPU path — exact vDSP
# result. Runs everywhere (no Metal needed; the else branch is taken).
test_that("default threshold keeps a small vector on the vDSP path", {
  skip_on_os(c("windows", "linux", "solaris"))
  old <- options(DefDiff.metal_threshold = 1e9L); on.exit(options(old), add = TRUE)
  gf <- grad(function(v) sum(v^2))
  expect_equal(gf(c(1, 2, 3)), c(2, 4, 6), tolerance = 4 * .Machine$double.eps)
})

# With the threshold lowered, a canonical gradient routes to Metal and matches
# the vDSP kernel within float32 tolerance.
test_that("lowered threshold routes the canonical gradient to Metal, matching vDSP", {
  skip_on_os(c("windows", "linux", "solaris"))
  skip_if(!.metal_path_available(), "Metal backend not available")
  old <- options(DefDiff.metal_threshold = 1L); on.exit(options(old), add = TRUE)
  v <- c(1.5, -2.0, 3.25, 0.0, 7.5)
  gf <- grad(function(v) sum(v^2))
  metal_result <- gf(v)
  expect_equal(metal_result, DefDiff:::fast_scalar_mul(2, v), tolerance = TOL_METAL)
  expect_equal(metal_result, 2 * v, tolerance = TOL_METAL)
})

# Direct kernel correctness (float32 vs double) when Metal is available.
test_that("metal_scalar_mul matches the double product within 1e-6", {
  skip_on_os(c("windows", "linux", "solaris"))
  skip_if(!.metal_path_available(), "Metal backend not available")
  set.seed(7)
  v <- runif(1000, -10, 10)
  expect_equal(DefDiff:::metal_scalar_mul(2.5, v), 2.5 * v, tolerance = TOL_METAL)
})

# Fallback symmetry: below-threshold result equals the lowered-threshold result
# (the two paths agree within float32 tolerance), so dispatch choice is
# observationally transparent.
test_that("Metal and vDSP paths agree within float32 tolerance", {
  skip_on_os(c("windows", "linux", "solaris"))
  skip_if(!.metal_path_available(), "Metal backend not available")
  v <- c(0.1, 0.2, 0.3, 0.4)
  gf <- grad(function(v) sum(v^2))
  cpu <- local({ old <- options(DefDiff.metal_threshold = 1e9L); on.exit(options(old)); gf(v) })
  gpu <- local({ old <- options(DefDiff.metal_threshold = 1L);  on.exit(options(old)); gf(v) })
  expect_equal(cpu, gpu, tolerance = TOL_METAL)
})

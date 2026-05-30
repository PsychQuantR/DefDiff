## test-real-world-patterns.R
## Phase 4 (add-multi-axis-benchmark-expansion): correctness assertions for
## the 5 real-world ML/stat patterns + closure-thesis OUT-OF-SCOPE guards.
##
## WORKS patterns: assert DD's gradient matches direct-R analytic within 1e-10.
## OUT-OF-SCOPE patterns: assert the specific error condition class fires —
## locks the boundary so a future Tier 4 extension can flip the assertion.

source_path <- system.file("benchmarks", "real_world_patterns.R", package = "DefDiff")
if (!nzchar(source_path) || !file.exists(source_path)) {
  source_path <- file.path("..", "..", "inst", "benchmarks", "real_world_patterns.R")
}
if (file.exists(source_path)) source(source_path)

skip_if_no_real_world <- function() {
  if (!exists("real_world_patterns")) skip("real_world_patterns.R not sourced")
}

# ---- WORKS patterns: numeric equivalence ----

test_that("Phase 4: gaussian_loglik gradient matches analytic", {
  skip_if_no_real_world()
  skip_if_not(.fast_path_available(), "macOS fast-path required")
  f <- real_world_patterns$gaussian_loglik$f_inline
  ag <- real_world_analytic$gaussian_loglik
  gf <- grad(f)
  v <- runif(50L)
  expect_equal(as.numeric(gf(v)), as.numeric(ag(v)), tolerance = 1e-10)
})

test_that("Phase 4: kld_normal gradient matches analytic", {
  skip_if_no_real_world()
  skip_if_not(.fast_path_available(), "macOS fast-path required")
  f <- real_world_patterns$kld_normal$f_inline
  ag <- real_world_analytic$kld_normal
  gf <- grad(f)
  v <- runif(50L)
  expect_equal(as.numeric(gf(v)), as.numeric(ag(v)), tolerance = 1e-10)
})

# ---- OUT-OF-SCOPE patterns: boundary lock ----

test_that("Phase 4: logistic_loss gradient matches analytic (Tier 4 add-rep-generator)", {
  skip_if_no_real_world()
  skip_if_not(.fast_path_available(), "macOS fast-path required")
  # Pre-Tier-4 this raised because rep() was not in the L_3 catalog.
  # add-rep-generator added the L_0 entry + walker recognition; rep with
  # constant first argument now returns 0 gradient, so the full logistic
  # loss expression composes correctly.
  f <- real_world_patterns$logistic_loss$f_inline
  ag <- real_world_analytic$logistic_loss
  gf <- grad(f)
  v <- runif(20L)
  expect_equal(as.numeric(gf(v)), as.numeric(ag(v)), tolerance = 1e-10)
})

test_that("Phase 4: softmax_entropy gradient matches analytic (Tier 4 close-top-level-division-gap)", {
  skip_if_no_real_world()
  skip_if_not(.fast_path_available(), "macOS fast-path required")
  # Pre-Tier-4 this raised DefDiff_not_definable for top-level a/b with v in
  # denominator. close-top-level-division-gap rewrote the L_0 `/` rule to
  # construct the quotient rule inline using .grad_expr for sub-derivatives,
  # so this expression now produces the analytic softmax entropy gradient.
  f <- real_world_patterns$softmax_entropy$f_inline
  ag <- real_world_analytic$softmax_entropy
  gf <- grad(f)
  v <- runif(20L)
  expect_equal(as.numeric(gf(v)), as.numeric(ag(v)), tolerance = 1e-10)
})

test_that("Phase 4: nn_forward gradient matches analytic (Tier 5 add-nn-walker-extension)", {
  skip_if_no_real_world()
  skip_if_not(.fast_path_available(), "macOS fast-path required")
  # Pre-Tier-5-Option-B-lite this raised because %*% wasn't in L_3 catalog
  # AND walker couldn't handle elementwise wrapping of matmul. add-matmul-generator
  # (Tier 4) added linear/quadratic forms, then add-elementwise-matmul-fastpath
  # (Tier 5 Option A) closed single-layer NN, and finally add-nn-walker-extension
  # (Tier 5 Option B-lite) added the recursive chain helper that handles
  # arbitrary-depth elementwise-matmul chains — closing the 2-layer NN forward.
  skip_if_not(exists("real_world_patterns"))
  f_builder <- real_world_patterns$nn_forward$f_inline_builder
  f_n <- f_builder(20L)
  ag <- real_world_analytic$nn_forward
  gf <- grad(f_n)
  v <- runif(20L)
  expect_equal(as.numeric(gf(v)), as.numeric(ag(v)), tolerance = 1e-10)
})

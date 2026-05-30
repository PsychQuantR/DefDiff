## test-matmul-generator.R
## Tier 4 change `add-matmul-generator`: %*% recognized in sum_rule (linear)
## and crossprod_rule (quadratic, constant-weighted) fast paths. L_0 catalog
## entry raises out-of-scope for any other context.

skip_if_no_fast <- function() {
  if (!.fast_path_available()) skip("macOS fast-path required")
}

# ---- 1-4: Three supported patterns + symmetric W sanity ----

test_that("Linear form sum(W %*% v) returns colSums(W)", {
  skip_if_no_fast()
  W <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 2)  # 2x3
  gf <- grad(function(v) sum(W %*% v))
  v <- c(0.5, 1.0, 1.5)
  expected <- as.numeric(colSums(W))
  expect_equal(as.numeric(gf(v)), expected, tolerance = 1e-10)
})

test_that("Quadratic form crossprod(v, W %*% v) for asymmetric W returns (W + t(W)) %*% v", {
  skip_if_no_fast()
  W <- matrix(c(1, 2, 3, 4), nrow = 2)  # asymmetric 2x2
  v <- c(0.5, 1.0)
  gf <- grad(function(v) crossprod(v, W %*% v))
  expected <- as.numeric((W + t(W)) %*% v)
  expect_equal(as.numeric(gf(v)), expected, tolerance = 1e-10)
})

test_that("Quadratic form with symmetric W (sanity — no symmetry detection needed)", {
  skip_if_no_fast()
  # Symmetric 2x2: (W + t(W)) %*% v == 2 * W %*% v, but the rule emits the
  # general form regardless — numeric result is the same.
  W <- matrix(c(2, 1, 1, 3), nrow = 2)
  v <- c(0.5, 1.0)
  gf <- grad(function(v) crossprod(v, W %*% v))
  expected <- as.numeric((W + t(W)) %*% v)  # = 2 * W %*% v for symmetric
  expect_equal(as.numeric(gf(v)), expected, tolerance = 1e-10)
})

test_that("Constant-weighted crossprod(W %*% v, c) returns t(W) %*% c", {
  skip_if_no_fast()
  W <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 2)  # 2x3
  cvec <- c(0.3, 0.7)
  v <- c(0.5, 1.0, 1.5)
  gf <- grad(function(v) crossprod(W %*% v, cvec))
  expected <- as.numeric(t(W) %*% cvec)
  expect_equal(as.numeric(gf(v)), expected, tolerance = 1e-10)
})

# ---- 5: GLM integration ----

test_that("GLM design matrix gradient: sum(X %*% v) returns colSums(X) regardless of v", {
  skip_if_no_fast()
  set.seed(42L)
  X <- matrix(rnorm(5 * 3), 5, 3)
  gf <- grad(function(v) sum(X %*% v))
  expected <- as.numeric(colSums(X))
  # Gradient is constant in v — same result for any input vector
  expect_equal(as.numeric(gf(runif(3))), expected, tolerance = 1e-10)
  expect_equal(as.numeric(gf(rep(0, 3))), expected, tolerance = 1e-10)
})

# ---- 6: Ridge composition (quadratic + Tier 1) ----

test_that("Ridge: crossprod(v, W %*% v) + lambda * crossprod(v, v) composition", {
  skip_if_no_fast()
  W <- matrix(c(1, 2, 3, 4), nrow = 2)
  lambda <- 0.5
  v <- c(0.5, 1.0)
  gf <- grad(function(v) crossprod(v, W %*% v) + lambda * crossprod(v, v))
  expected <- as.numeric((W + t(W)) %*% v) + 2 * lambda * v
  expect_equal(as.numeric(gf(v)), expected, tolerance = 1e-10)
})

# ---- 7-9: Negative tests (out-of-scope rejections) ----

test_that("Standalone W %*% v (no outer reduction) raises", {
  W <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 2)
  # Vector output, outside DD's scalar-output contract. Either the L_0 %*%
  # rule fires or an earlier scalar-check error fires; we accept either.
  expect_error(
    grad(function(v) W %*% v)
  )
})

test_that("Elementwise wrapping sum(tanh(W %*% v)) now WORKS (post Tier 5 Option A)", {
  # Pre-Tier-5-Option-A this raised because the walker couldn't differentiate
  # %*%. `add-elementwise-matmul-fastpath` added a single-layer fast path
  # in .sum_rule that emits gradient t(W) %*% (1 - tanh(W%*%v)^2).
  # Multi-layer composition still raises — see add-elementwise-matmul-fastpath
  # for the 2-layer negative test.
  W <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 2)
  gf <- grad(function(v) sum(tanh(W %*% v)))
  v <- c(0.1, 0.5, -0.2)
  expected <- as.numeric(t(W) %*% (1 - tanh(W %*% v)^2))
  expect_equal(as.numeric(gf(v)), expected, tolerance = 1e-10)
})

test_that("Matmul output in arithmetic context raises", {
  W <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 2)
  # (W %*% v) is a vector; dividing by sum(v) (a scalar) produces a vector;
  # sum() at the outer wraps it. But the inner W%*%v isn't in sum_rule fast
  # paths (`/` is on the outside), so walker fall-through reaches L_0 %*%
  # rule which raises.
  expect_error(
    grad(function(v) sum((W %*% v) / sum(v))),
    class = "DefDiff_not_definable",
    regexp = "%\\*%"
  )
})

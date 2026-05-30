## test-walker-matmul-shape-extension.R
## Phase 5 (`add-walker-shape-extension`): walker now handles `%*%` in
## arbitrary subexpression context — bias terms, parallel paths, compound
## products, multi-layer chains with bias. These tests run end-to-end through
## `grad(...)(v)` and compare to the analytical gradient.

skip_if_no_fast <- function() {
  # No skip: Phase 5 doesn't depend on Accelerate-only fast paths; pullback
  # path is pure R. Helper kept for symmetry with other matmul test files.
  invisible(NULL)
}

# Test matrices: keep small to avoid tanh saturation washing out gradients
W1_test <- matrix(c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6), nrow = 2)  # 2x3
W2_test <- matrix(c(0.05, 0.15, 0.25, 0.35), nrow = 2)         # 2x2
v_test <- c(0.5, 0.7, 0.3)

# ===== 4 positive: bias terms for each f =====

test_that("Phase 5: bias term with sin — sum(sin(W %*% v + b))", {
  skip_if_no_fast()
  W <- W1_test; b <- c(0.1, 0.2)
  gf <- grad(function(v) sum(sin(W %*% v + b)))
  result <- as.numeric(gf(v_test))
  expected <- as.numeric(t(W) %*% cos(W %*% v_test + b))
  expect_equal(result, expected, tolerance = 1e-10)
})

test_that("Phase 5: bias term with cos — sum(cos(W %*% v + b))", {
  skip_if_no_fast()
  W <- W1_test; b <- c(0.1, 0.2)
  gf <- grad(function(v) sum(cos(W %*% v + b)))
  result <- as.numeric(gf(v_test))
  expected <- as.numeric(t(W) %*% (-sin(W %*% v_test + b)))
  expect_equal(result, expected, tolerance = 1e-10)
})

test_that("Phase 5: bias term with exp — sum(exp(W %*% v + b))", {
  skip_if_no_fast()
  W <- W1_test; b <- c(0.1, 0.2)
  gf <- grad(function(v) sum(exp(W %*% v + b)))
  result <- as.numeric(gf(v_test))
  expected <- as.numeric(t(W) %*% exp(W %*% v_test + b))
  expect_equal(result, expected, tolerance = 1e-10)
})

test_that("Phase 5: bias term with tanh — sum(tanh(W %*% v + b))", {
  skip_if_no_fast()
  W <- W1_test; b <- c(0.1, 0.2)
  gf <- grad(function(v) sum(tanh(W %*% v + b)))
  result <- as.numeric(gf(v_test))
  expected <- as.numeric(t(W) %*% (1 - tanh(W %*% v_test + b)^2))
  expect_equal(result, expected, tolerance = 1e-10)
})

# ===== 1 positive: skip connection =====

test_that("Phase 5: skip connection — sum(tanh(W2 %*% v + W1 %*% tanh(W1 %*% v)))", {
  skip_if_no_fast()
  # Skip connection: `W1 %*% tanh(W1 %*% v)` reuses W1, so W1 must be
  # square (n×n). W2 also n×n for additive consistency.
  W1 <- matrix(c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), nrow = 3)
  W2 <- matrix(c(0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85), nrow = 3)
  gf <- grad(function(v) sum(tanh(W2 %*% v + W1 %*% tanh(W1 %*% v))))
  result <- as.numeric(gf(v_test))
  # Analytical: ∇ = t(W2) %*% s2 + t(W1) %*% ((1 - tanh(W1 v)^2) * (t(W1) %*% s2))
  # where s2 = (1 - tanh(W2 v + W1 tanh(W1 v))^2)
  h1 <- as.numeric(W1 %*% v_test)
  th1 <- tanh(h1)
  h2 <- as.numeric(W2 %*% v_test + W1 %*% th1)
  s2 <- 1 - tanh(h2)^2
  expected <- as.numeric(
    t(W2) %*% s2 + t(W1) %*% ((1 - th1^2) * as.numeric(t(W1) %*% s2))
  )
  expect_equal(result, expected, tolerance = 1e-10)
})

# ===== 1 positive: compound product (vector-grain × matrix-mixed) =====

test_that("Phase 5: compound product — sum(sin(v) * tanh(W %*% v))", {
  skip_if_no_fast()
  # Dimensions: sin(v) is n-vec (n=3), tanh(W %*% v) is m-vec (m=2). For the
  # elementwise product they must match; this requires a square W (m=n=3).
  W <- matrix(c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), nrow = 3)
  gf <- grad(function(v) sum(sin(v) * tanh(W %*% v)))
  result <- as.numeric(gf(v_test))
  Wv <- as.numeric(W %*% v_test)
  # ∇ sum(sin(v) * tanh(Wv)) = cos(v) * tanh(Wv) + t(W) %*% (sin(v) * (1 - tanh(Wv)^2))
  expected <- cos(v_test) * tanh(Wv) +
              as.numeric(t(W) %*% (sin(v_test) * (1 - tanh(Wv)^2)))
  expect_equal(result, expected, tolerance = 1e-10)
})

# ===== 1 positive: parallel paths (two matmul branches multiplied) =====

test_that("Phase 5: parallel paths — sum(tanh(W2 %*% v) * tanh(W1 %*% v))", {
  skip_if_no_fast()
  # Both legs must produce same-length vectors for elementwise multiplication.
  # W1, W2 both R^3 → R^2.
  W1 <- W1_test; W2 <- matrix(c(0.7, 0.8, 0.5, 0.6, 0.3, 0.4), nrow = 2)
  gf <- grad(function(v) sum(tanh(W2 %*% v) * tanh(W1 %*% v)))
  result <- as.numeric(gf(v_test))
  h1 <- as.numeric(W1 %*% v_test); th1 <- tanh(h1)
  h2 <- as.numeric(W2 %*% v_test); th2 <- tanh(h2)
  expected <- as.numeric(
    t(W2) %*% ((1 - th2^2) * th1) + t(W1) %*% ((1 - th1^2) * th2)
  )
  expect_equal(result, expected, tolerance = 1e-10)
})

# ===== 1 positive: multi-layer with bias at multiple layers =====

test_that("Phase 5: multi-layer NN with biases — sum(tanh(W2 %*% tanh(W1 %*% v + b1) + b2))", {
  skip_if_no_fast()
  W1 <- W1_test; W2 <- W2_test
  b1 <- c(0.1, 0.2); b2 <- c(0.05, 0.15)
  gf <- grad(function(v) sum(tanh(W2 %*% tanh(W1 %*% v + b1) + b2)))
  result <- as.numeric(gf(v_test))
  h1 <- as.numeric(W1 %*% v_test + b1)
  z1 <- tanh(h1)
  h2 <- as.numeric(W2 %*% z1 + b2)
  s2 <- 1 - tanh(h2)^2
  expected <- as.numeric(t(W1) %*% ((1 - z1^2) * as.numeric(t(W2) %*% s2)))
  expect_equal(result, expected, tolerance = 1e-10)
})

# ===== 4 negative: still out of scope =====

test_that("Phase 5 negative: matmul where W depends on var — sum(tanh(v %*% v))", {
  # `v %*% v` has v as the first matmul operand. Per design's single-variable
  # scope non-goal, matmul-w.r.t.-W is unsupported. Walker's matmul case
  # requires `!.contains_var(W_expr, var)` → falls through to L_0 fallback
  # which has no %*% rule → DefDiff_unknown_generator → wrapped to DefDiff_not_definable.
  expect_error(
    grad(function(v) sum(tanh(v %*% v))),
    class = "DefDiff_not_definable"
  )
})

test_that("Phase 5 negative: non-vForce f in chain — sum(gamma(W %*% v))", {
  # gamma is not in the chain switch {sin,cos,exp,log,tanh,sqrt,atan} and not
  # registered in L_0. Walker chain case returns no outer_deriv → falls to
  # L_0 fallback → raises.
  W <- W1_test
  expect_error(
    grad(function(v) sum(gamma(W %*% v))),
    class = "DefDiff_not_definable"
  )
})

test_that("Phase 5 negative: nested var-dep matmul — sum(tanh(W %*% (v %*% v)))", {
  # Outer matmul W is constant, but inner (v %*% v) is itself a matmul with
  # var-dep first operand. Walker recurses into inner, hits %*% with W=var,
  # falls through to L_0 fallback → raises.
  W <- W1_test
  expect_error(
    grad(function(v) sum(tanh(W %*% (v %*% v)))),
    class = "DefDiff_not_definable"
  )
})

test_that("Phase 5 negative: control flow in body — function with if/else", {
  # Walker has no `if` case; falls to L_0 fallback; no `if` rule there →
  # DefDiff_unknown_generator → wrapped to DefDiff_not_definable. Control flow is
  # explicitly out of scope per design (only differentiable straight-line
  # programs are supported).
  W <- W1_test
  expect_error(
    grad(function(v) sum(if (sum(v) > 0) tanh(W %*% v) else W %*% v)),
    class = "DefDiff_not_definable"
  )
})

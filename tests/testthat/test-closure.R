## test-closure.R
## Tier 3 (add-tier3-catalog-closure-completion):
## Formally exercise the L_3 closure thesis. Every L_3 expression
## (composition of {+, -, *, /, ^, sum, crossprod, cos, sin, exp, log,
## tanh, sqrt, atan} on single variable v) MUST have a working gradient.

TOL_CLOSURE <- 1e-10

# ---- Standard differentiation rules via .sum_rule + .grad_inner ----

test_that("Closure: sum/difference rule sum(v^2 + sin(v))", {
  gf <- grad(function(v) sum(v^2 + sin(v)))
  v <- runif(10L) + 0.1
  expect_equal(gf(v), 2 * v + cos(v), tolerance = TOL_CLOSURE)
})

test_that("Closure: product rule sum(v * sin(v))", {
  gf <- grad(function(v) sum(v * sin(v)))
  v <- runif(10L) + 0.1
  expect_equal(gf(v), sin(v) + v * cos(v), tolerance = TOL_CLOSURE)
})

test_that("Closure: product rule sum(sin(v) * cos(v))", {
  gf <- grad(function(v) sum(sin(v) * cos(v)))
  v <- runif(10L) + 0.1
  expect_equal(gf(v), cos(v) * cos(v) - sin(v) * sin(v), tolerance = TOL_CLOSURE)
})

test_that("Closure: quotient rule sum(v / sin(v))", {
  gf <- grad(function(v) sum(v / sin(v)))
  v <- runif(10L) + 0.5
  expect_equal(gf(v), (sin(v) - v * cos(v)) / sin(v)^2, tolerance = TOL_CLOSURE)
})

test_that("Closure: chain rule through affine sum(sin(v + 1))", {
  gf <- grad(function(v) sum(sin(v + 1)))
  v <- runif(10L) + 0.1
  expect_equal(gf(v), cos(v + 1), tolerance = TOL_CLOSURE)
})

test_that("Closure: chain rule through scalar-mul sum(sin(2 * v))", {
  gf <- grad(function(v) sum(sin(2 * v)))
  v <- runif(10L) + 0.1
  expect_equal(gf(v), 2 * cos(2 * v), tolerance = TOL_CLOSURE)
})

test_that("Closure: crossprod with non-linear inner crossprod(v, sin(v))", {
  gf <- grad(function(v) crossprod(v, sin(v)))
  v <- runif(10L) + 0.1
  expect_equal(gf(v), sin(v) + v * cos(v), tolerance = TOL_CLOSURE)
})

# ---- Nested compositions ----

test_that("Closure: nested chain sum(sin(cos(v)))", {
  gf <- grad(function(v) sum(sin(cos(v))))
  v <- runif(10L) + 0.1
  expect_equal(gf(v), cos(cos(v)) * (-sin(v)), tolerance = TOL_CLOSURE)
})

test_that("Closure: nested chain sum(exp(sin(v)))", {
  gf <- grad(function(v) sum(exp(sin(v))))
  v <- runif(10L) + 0.1
  expect_equal(gf(v), exp(sin(v)) * cos(v), tolerance = TOL_CLOSURE)
})

test_that("Closure: log of polynomial sum(log(1 + v^2))", {
  gf <- grad(function(v) sum(log(1 + v^2)))
  v <- runif(10L) + 0.1
  expect_equal(gf(v), 2 * v / (1 + v^2), tolerance = TOL_CLOSURE)
})

test_that("Closure: explicit product sum(v * v * v)", {
  gf <- grad(function(v) sum(v * v * v))
  v <- runif(10L) + 0.1
  expect_equal(gf(v), 3 * v^2, tolerance = TOL_CLOSURE)
})

# ---- Edge cases for .grad_inner walker ----

test_that(".grad_inner edge: constant — pullback emits zero vector", {
  # Phase 6: walker returns list(value, pullback). For a constant leaf,
  # the pullback is the zero-pullback: pullback(g) = rep(0, length(v)).
  shim <- DefDiff:::.grad_inner(quote(5), "v")
  expect_equal(shim$value, 5)
  expect_identical(shim$pullback(quote(g)), bquote(rep(0, length(v))))
  shim2 <- DefDiff:::.grad_inner(quote(3.14), "v")
  expect_equal(shim2$value, 3.14)
  expect_identical(shim2$pullback(quote(g)), bquote(rep(0, length(v))))
})

test_that(".grad_inner edge: bare variable — pullback is identity", {
  # For the bare variable, pullback(g) = g (upstream gradient passes through).
  shim <- DefDiff:::.grad_inner(quote(v), "v")
  expect_identical(shim$value, quote(v))
  expect_identical(shim$pullback(quote(g)), quote(g))
})

test_that(".grad_inner edge: different symbol — pullback emits zero vector", {
  shim <- DefDiff:::.grad_inner(quote(w), "v")
  expect_identical(shim$value, quote(w))
  expect_identical(shim$pullback(quote(g)), bquote(rep(0, length(v))))
})

test_that(".grad_inner edge: power rule with constant exponent", {
  # Check correctness via grad() integration (avoids direct eval of raw AST)
  gf <- grad(function(v) sum(v^3))
  v <- c(1, 2, 3)
  expect_equal(gf(v), 3 * v^2, tolerance = TOL_CLOSURE)
})

test_that(".grad_inner edge: zero gradient when var not present", {
  # sin(w) + cos(w) doesn't contain v. .sum_rule's `!.contains_var` shortcut
  # returns scalar 0 before reaching walker, so end-to-end gradient is 0.
  w <- c(0.5, 1.0)
  gf <- grad(function(v) sum(sin(w) + cos(w)))
  expect_equal(gf(c(1.1, 2.2, 3.3)), 0)

  # Walker invoked directly: pullback applied to any upstream collapses to
  # rep(0, length(v)) (no var-dependence anywhere in the sub-expression).
  shim <- DefDiff:::.grad_inner(quote(sin(w) + cos(w)), "v")
  expect_true(!is.null(shim$pullback))
  env <- new.env(parent = baseenv())
  env$v <- c(1.1, 2.2, 3.3)
  env$w <- w
  # SAFETY: AST built from trusted internal helpers; env restricted to
  # baseenv() plus test-bound v and w.
  grad_ast <- shim$pullback(quote(rep(1, length(v))))
  expect_equal(as.numeric(eval(grad_ast, envir = env)), c(0, 0, 0))
})

# ---- Additive sums of multiple terms ----

test_that("Closure: additive composition sum(v + v^2 + sin(v))", {
  gf <- grad(function(v) sum(v + v^2 + sin(v)))
  v <- runif(10L) + 0.1
  expect_equal(gf(v), 1 + 2 * v + cos(v), tolerance = TOL_CLOSURE)
})

test_that("Closure: subtractive composition sum(v^2 - sin(v))", {
  gf <- grad(function(v) sum(v^2 - sin(v)))
  v <- runif(10L) + 0.1
  expect_equal(gf(v), 2 * v - cos(v), tolerance = TOL_CLOSURE)
})

# ---- Mixed product + chain ----

test_that("Closure: sum(v * exp(v))", {
  gf <- grad(function(v) sum(v * exp(v)))
  v <- runif(10L) + 0.1
  expect_equal(gf(v), exp(v) + v * exp(v), tolerance = TOL_CLOSURE)
})

test_that("Closure: sum(v^2 * sin(v))", {
  gf <- grad(function(v) sum(v^2 * sin(v)))
  v <- runif(10L) + 0.1
  expect_equal(gf(v), 2 * v * sin(v) + v^2 * cos(v), tolerance = TOL_CLOSURE)
})

# ---- Reciprocal patterns ----

test_that("Closure: sum(1 / v)", {
  gf <- grad(function(v) sum(1 / v))
  v <- runif(10L) + 0.5
  expect_equal(gf(v), -1 / v^2, tolerance = TOL_CLOSURE)
})

test_that("Closure: sum(1 / (1 + v^2))", {
  gf <- grad(function(v) sum(1 / (1 + v^2)))
  v <- runif(10L) + 0.1
  expect_equal(gf(v), -2 * v / (1 + v^2)^2, tolerance = TOL_CLOSURE)
})

# ---- atan ----

test_that("Closure: sum(atan(v))", {
  gf <- grad(function(v) sum(atan(v)))
  v <- runif(10L) + 0.1
  expect_equal(gf(v), 1 / (1 + v^2), tolerance = TOL_CLOSURE)
})

# ---- Polynomial * elementwise ----

test_that("Closure: sum(v^2 * cos(v))", {
  gf <- grad(function(v) sum(v^2 * cos(v)))
  v <- runif(10L) + 0.1
  expect_equal(gf(v), 2 * v * cos(v) - v^2 * sin(v), tolerance = TOL_CLOSURE)
})

# ---- Inverse via sqrt ----

test_that("Closure: sum(sqrt(1 + v^2))", {
  gf <- grad(function(v) sum(sqrt(1 + v^2)))
  v <- runif(10L) + 0.1
  expect_equal(gf(v), v / sqrt(1 + v^2), tolerance = TOL_CLOSURE)
})

# ---- Pre-Tier-3 cases that should still work (regression guard) ----

test_that("Regression: Tier 1 sum(v^2) still works post-Tier-3", {
  gf <- grad(function(v) sum(v^2))
  v <- runif(10L) + 0.1
  expect_equal(gf(v), 2 * v, tolerance = TOL_CLOSURE)
})

test_that("Regression: Tier 2c sum(sin(v)) still works post-Tier-3", {
  gf <- grad(function(v) sum(sin(v)))
  v <- runif(10L) + 0.1
  expect_equal(gf(v), cos(v), tolerance = TOL_CLOSURE)
})

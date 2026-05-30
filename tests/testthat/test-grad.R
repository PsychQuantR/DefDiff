test_that("parse_expr normalises parenthesised calls", {
  expect_equal(DefDiff:::parse_expr(quote((v + w))), quote(v + w))
  expect_equal(DefDiff:::parse_expr(quote(((sum(v^2))))), quote(sum(v^2)))
  expect_equal(DefDiff:::parse_expr(quote(v)), quote(v))
})

test_that("grad_expr handles atomic generators (sum norm-squared)", {
  result <- DefDiff:::.grad_expr(quote(sum(v^2)), "v")
  expect_equal(result, quote(2 * v))
})

test_that("grad on function returns function with correct numeric gradient", {
  gf <- grad(function(v) sum(v^2))
  expect_equal(gf(c(1, 2, 3)), c(2, 4, 6))
  expect_equal(gf(c(-1, 0, 4)), c(-2, 0, 8))
})

test_that("grad on call returns symbolic expression", {
  expect_equal(grad(quote(sum(v^2)), "v"), quote(2 * v))
  expect_equal(grad(quote(crossprod(v, w)), "v"), quote(w))
})

test_that("grad on expression returns expression of length 1", {
  out <- grad(expression(sum(v^2)), "v")
  expect_true(inherits(out, "expression"))
  expect_equal(length(out), 1L)
  expect_equal(out[[1]], quote(2 * v))
})

test_that("chain rule on sin(sum(v^2)) numerically matches 2*cos(sum(v^2))*v", {
  set.seed(42L)
  gf <- grad(function(v) sin(sum(v^2)))
  ref <- function(v) 2 * cos(sum(v^2)) * v
  for (i in seq_len(20L)) {
    v <- stats::runif(4L, -1, 1)
    expect_lt(max(abs(gf(v) - ref(v))), 1e-10)
  }
})

test_that("compositional rules (sum/product/scalar) pass on 5 cases", {
  # 1. norm squared (symbolic)
  expect_equal(grad(quote(sum(v^2)), "v"), quote(2 * v))
  # 2. unary negation (numeric: -∇sum(v^2) = -2v)
  gf <- grad(function(v) -sum(v^2))
  expect_equal(gf(c(1, 2)), c(-2, -4))
  # 3. constant scalar multiply
  gf <- grad(function(v) 3 * sum(v^2))
  expect_equal(gf(c(1, 2)), c(6, 12))
  # 4. log composition: d/dv log(sum(v^2)) = 2v / sum(v^2)
  gf <- grad(function(v) log(sum(v^2)))
  v <- c(1, 2)
  expect_equal(gf(v), 2 * v / sum(v^2))
  # 5. exp composition: d/dv exp(sum(v^2)) = exp(sum(v^2)) * 2v
  gf <- grad(function(v) exp(sum(v^2)))
  v <- c(0.5, 0.5)
  expect_equal(gf(v), exp(sum(v^2)) * (2 * v))
})

test_that("unknown generator raises typed condition", {
  expect_error(
    grad(quote(novel_function(v)), "v"),
    class = "DefDiff_unknown_generator"
  )
})

test_that("if-statement raises DefDiff_not_definable", {
  f <- function(v) if (v[1] > 0) sum(v^2) else 0
  expect_error(grad(f), class = "DefDiff_not_definable")
})

test_that("quadratic form crossprod(v, A %*% v) returns (A + t(A)) %*% v", {
  expect_equal(
    grad(quote(crossprod(v, A %*% v)), "v"),
    quote((A + t(A)) %*% v)
  )
})

test_that("constant-only expression has zero gradient", {
  expect_equal(grad(quote(sum(w^2)), "v"), 0)
})

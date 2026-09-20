# grad_expr() — backend-agnostic symbolic-inspection accessor (#3)

test_that("grad_expr() recovers the symbolic gradient even when fast-pathed", {
  gf <- grad(function(v) sum(v^3))
  expect_equal(grad_expr(gf), quote(3 * v^2))
})

test_that("grad_expr() recovers 2 * v for sum(v^2)", {
  gf <- grad(function(v) sum(v^2))
  expect_equal(grad_expr(gf), quote(2 * v))
})

test_that("grad_expr() on a multi-variable gradient returns a named list", {
  gf <- grad(function(v, w) sum(v^2) + sum(w^3))
  ge <- grad_expr(gf)
  expect_true(is.list(ge))
  expect_named(ge, c("v", "w"))
})

test_that("grad_expr() errors on a non-gradient function", {
  expect_error(grad_expr(function(x) x), class = "DefDiff_not_gradient")
})

## test-grad-hessian-properties.R
## Dimension: properties. Operator-level invariants that hold for all supported
## f: gradient linearity, Hessian symmetry, cross-engine consistency, multi-var
## block symmetry, and normalization value-preservation.

PR_TOL <- 1e-10
PR_TOL_X <- 1e-5

test_that("grad is linear: grad(a*f + b*g) == a*grad(f) + b*grad(g)", {
  v <- c(0.5, 1.2, -0.3)
  gf <- as.numeric(grad(function(v) sum(v^2))(v))
  gg <- as.numeric(grad(function(v) sum(sin(v)))(v))
  gc <- as.numeric(grad(function(v) 2.5 * sum(v^2) + (-1.5) * sum(sin(v)))(v))
  expect_equal(gc, 2.5 * gf - 1.5 * gg, tolerance = PR_TOL)
})

test_that("Hessian is symmetric across fast-path and recursive shapes", {
  v <- c(0.4, 0.9, 1.1)
  Hc <- hessian(function(v) sin(sum(v^2)))(v)
  expect_equal(Hc, t(Hc), tolerance = PR_TOL)
  # Deliberately NON-symmetric A: distinguishes A + t(A) from a wrong 2A.
  A <- matrix(c(2, 0.3, 0.1, 0.5, 3, 1, 0, 1, 4), 3, 3, byrow = TRUE)
  Hq <- hessian(function(v) crossprod(v, A %*% v))(v)
  expect_equal(Hq, t(Hq), tolerance = PR_TOL)
  expect_equal(Hq, A + t(A), tolerance = PR_TOL)
})

test_that("cross-engine: hessian(f) == numerical Jacobian of grad(f)", {
  skip_if_not_installed("numDeriv")
  v <- c(0.4, 0.9, 1.1)
  f <- function(v) sin(sum(v^2))
  gfn <- function(x) as.numeric(grad(f)(x))
  expect_equal(hessian(f)(v), nd_jacobian(gfn, v), tolerance = PR_TOL_X)
})

test_that("multi-variable block Hessian is symmetric across blocks", {
  B <- matrix(c(1, 2, 0, 3, 1, 2), 2, 3, byrow = TRUE)
  H <- hessian(function(v, w) crossprod(v, B %*% w))(c(1, 1), c(1, 1, 1))
  expect_equal(H$v$w, t(H$w$v), tolerance = PR_TOL)
})

test_that("normalize_fast_kernels is value-preserving on a quotient gradient AST", {
  ga <- DefDiff:::.grad_expr(quote(sum(v * exp(v)) / sum(exp(v))), "v")
  na <- DefDiff:::.normalize_fast_kernels(ga)
  vv <- c(0.5, 0.8, 1.2)
  expect_false(any(grepl("fast_", all.names(na))))
  # NOTE: eval() here numerically evaluates gradient ASTs the dat package itself
  # produced (.grad_expr / .normalize_fast_kernels), not any external input —
  # this is the core DAT paradigm (a derivative is a materialized AST).
  expect_equal(eval(na, list(v = vv)), eval(ga, list(v = vv)), tolerance = 1e-12)
})

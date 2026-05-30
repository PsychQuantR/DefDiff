## test-hessian-recursive.R
## add-hessian-recursive-walker: Hessian via the Jacobian of the gradient
## field. Covers the shape/grain classifier, the recursive Jacobian walker,
## the three previously-unsupported families (composite outer-scalar, product
## of forms, mixed quadratic+elementwise), failure modes, and compositions.
## Symbolic output is verified primarily against exact closed-form references
## (TOL_EXACT), with numDeriv::hessian as an independent secondary cross-check
## at a finite-difference-appropriate tolerance (TOL_ND).

TOL_EXACT <- 1e-8
TOL_ND    <- 1e-5

# ===== (1) Shape / grain classifier =====

test_that(".hess_shape classifies scalar / vector_grain / vector_dense", {
  shp <- function(e) DefDiff:::.hess_shape(e, "v")
  # scalars: reductions and length-1 literals
  expect_identical(shp(quote(sum(v^2))), "scalar")
  expect_identical(shp(quote(crossprod(v, v))), "scalar")
  expect_identical(shp(quote(cos(sum(v^2)))), "scalar")  # elem fn of a scalar
  expect_identical(shp(quote(2)), "scalar")
  # vector-grain: bare var, elementwise fns of var, var-free scalar * vector
  expect_identical(shp(quote(v)), "vector_grain")
  expect_identical(shp(quote(cos(v))), "vector_grain")
  expect_identical(shp(quote(2 * v)), "vector_grain")
  expect_identical(shp(quote(v^2)), "vector_grain")
  expect_identical(shp(quote(2 * v + cos(v))), "vector_grain")
  # vector-dense: a var-dependent scalar times a vector introduces off-diagonal
  expect_identical(shp(quote(cos(sum(v^2)) * (2 * v))), "vector_dense")
  expect_identical(shp(quote(crossprod(v, v) * (2 * v))), "vector_dense")
  expect_identical(shp(quote(W %*% v)), "vector_dense")
})

test_that(".hess_shape classifies scalar-denominator quotients, flags vector denominators", {
  shp <- function(e) DefDiff:::.hess_shape(e, "v")
  # A var-dependent scalar denominator makes the Jacobian dense (closed by
  # add-hessian-quotient-walker); previously this was "unknown".
  expect_identical(shp(quote((2 * v) / sum(exp(v)))), "vector_dense")
  # A vector-valued denominator remains outside the rules.
  expect_identical(shp(quote(v / sin(v))), "unknown")
})

# ===== (2) Recursive Jacobian-of-gradient walker =====

# Build a Hessian function directly from the recursive walker (pre-wiring).
build_hess <- function(f, var = "v") {
  body_e <- DefDiff:::.strip_paren(body(f))
  H <- DefDiff:::.hessian_recursive(body_e, var)
  hf <- function() NULL
  formals(hf) <- formals(f)
  body(hf) <- H
  environment(hf) <- environment(f)
  hf
}

test_that("composite outer-scalar sin(sum(v^2)) matches its exact Hessian", {
  f <- function(v) sin(sum(v^2))
  v <- c(1, 2, 3)
  ref <- -4 * sin(sum(v^2)) * outer(v, v) + 2 * cos(sum(v^2)) * diag(length(v))
  H <- build_hess(f)(v)
  expect_equal(H, ref, tolerance = TOL_EXACT)
  expect_equal(max(abs(H - t(H))), 0, tolerance = 1e-12)  # symmetric
})

test_that("product of forms crossprod(v,v)*crossprod(v,v) matches its exact Hessian", {
  f <- function(v) crossprod(v, v) * crossprod(v, v)
  v <- c(1, 2)
  ref <- 8 * outer(v, v) + 4 * as.numeric(crossprod(v, v)) * diag(length(v))
  expect_equal(build_hess(f)(v), ref, tolerance = TOL_EXACT)
})

test_that("mixed crossprod(v) + sum(sin(v)) matches its exact Hessian", {
  f <- function(v) crossprod(v) + sum(sin(v))
  v <- c(0.5, 1.0, 1.5)
  ref <- 2 * diag(length(v)) + diag(-sin(v))
  expect_equal(build_hess(f)(v), ref, tolerance = TOL_EXACT)
})

test_that("recursive Hessians cross-check against numDeriv::hessian", {
  skip_if_not_installed("numDeriv")
  nd <- getFromNamespace("hessian.default", "numDeriv")
  cases <- list(
    list(f = function(v) sin(sum(v^2)),                  v = c(0.4, 0.3, 0.2)),
    list(f = function(v) crossprod(v, v) * crossprod(v, v), v = c(1, 2)),
    list(f = function(v) crossprod(v) + sum(sin(v)),     v = c(0.5, 1.0, 1.5))
  )
  for (cc in cases) {
    H <- build_hess(cc$f)(cc$v)
    expect_lt(max(abs(H - nd(cc$f, cc$v))), TOL_ND)
  }
})

# ===== (2c) Failure modes =====

test_that("unsupported generator propagates the grad engine condition", {
  # gamma is outside the differentiation catalog: grad raises, hessian inherits.
  expect_error(
    DefDiff:::.hessian_recursive(quote(sum(gamma(v))), "v"),
    class = "DefDiff_not_definable"
  )
})

test_that("scalar-denominator quotient gradient is constructed (was a boundary)", {
  # Previously hessian_not_supported; closed by add-hessian-quotient-walker.
  skip_if_not_installed("numDeriv")
  f <- function(v) sum(v * exp(v)) / sum(exp(v))
  v <- c(0.5, 1.0, 1.5)
  expect_equal(build_hess(f)(v), nd_hessian(f, v), tolerance = TOL_ND)
})

test_that("vector-denominator quotient gradient raises hessian_not_supported", {
  # vector / vector remains the out-of-scope quotient shape.
  expect_error(
    DefDiff:::.jacobian_inner(quote(v / sin(v)), "v", "v"),
    class = "hessian_not_supported"
  )
})

# ===== (5) Compositions over the supported catalog =====

test_that("compositions over the catalog match numDeriv", {
  skip_if_not_installed("numDeriv")
  nd <- getFromNamespace("hessian.default", "numDeriv")
  A <- matrix(c(2, 1, 0, 1, 3, 1, 0, 1, 2), 3, 3)
  cases <- list(
    list(f = function(v) sum(tanh(v)) + crossprod(v, A %*% v),       v = c(0.3, 0.2, 0.4)),
    list(f = function(v) sin(sum(v^2)) + crossprod(v, v) * crossprod(v, v), v = c(0.4, 0.3, 0.2))
  )
  for (cc in cases) {
    H <- build_hess(cc$f)(cc$v)
    expect_lt(max(abs(H - nd(cc$f, cc$v))), TOL_ND)
  }
})

## test-multi-variable-hessian.R
## add-multi-variable-hessian: block Hessian for scalar-output functions of
## k >= 2 vector variables. hessian(f) returns a named list of named lists;
## H[[a]][[b]] is the n_a x n_b block of second partials d^2 f / da db, each
## computed as the Jacobian of the per-variable gradient grad_a(f) w.r.t. b
## via the walker generalized to (a) know all vector variables and (b) emit
## rectangular blocks. Blocks are verified against numDeriv::hessian slices on
## the flattened argument, plus cross-block symmetry.

TOL_EXACT <- 1e-8
TOL_ND    <- 1e-5

# NOTE: eval() below evaluates the walker's OWN generated R AST (quoted
# expressions built by the helpers under test) in this test's local scope with
# fixed v/w values. This is intended symbolic-AST verification, not evaluation
# of external/untrusted input.

# ===== (1) Generalized walker helpers (all_vars-aware) =====

test_that(".hess_shape with all_vars classifies other vector variables as vector, not scalar", {
  shp <- function(e) DefDiff:::.hess_shape(e, "v", c("v", "w"))
  expect_identical(shp(quote(w)), "vector_grain")        # other vector var, not scalar
  expect_identical(shp(quote(v)), "vector_grain")        # diff var
  expect_identical(shp(quote(v * w)), "vector_grain")    # vector*vector Hadamard
  expect_identical(shp(quote(rep(1, length(v)))), "vector_grain")  # constant vector
  # var-free genuine scalar constant stays scalar
  expect_identical(DefDiff:::.hess_shape(quote(k), "v", c("v", "w")), "scalar")
})

test_that(".jacobian_inner emits a rectangular zero block for a constant w.r.t. the diff var", {
  # grad-shaped expr `w` (other var) differentiated w.r.t. v -> zero matrix
  # n_v rows x n_v cols (length of the expr x length of diff var).
  J <- DefDiff:::.jacobian_inner(quote(w), "v", c("v", "w"))
  v <- c(1, 2, 3); w <- c(4, 5, 6)
  M <- eval(J)
  expect_equal(M, matrix(0, 3, 3), tolerance = TOL_EXACT)
})

test_that(".hess_diag handles a vector-grain Hadamard product via the product rule", {
  # d/dv (v * w) per coordinate = w (w constant w.r.t. v)
  d <- DefDiff:::.hess_diag(quote(v * w), "v", c("v", "w"))
  v <- c(1, 2); w <- c(3, 4)
  expect_equal(eval(d), c(3, 4), tolerance = TOL_EXACT)
})

# ===== (2) Block assembly via the public hessian() =====

# Flatten f(v, w, ...) into g(x) and slice numDeriv's Hessian per block.
nd_block_maxerr <- function(f, vlist) {
  nd <- getFromNamespace("hessian.default", "numDeriv")
  vars <- names(formals(f)); lens <- vapply(vlist, length, integer(1)); off <- c(0, cumsum(lens))
  flat <- unlist(vlist, use.names = FALSE)
  g <- function(x) {
    args <- lapply(seq_along(vars), function(i) x[(off[i] + 1):off[i + 1]])
    do.call(f, args)
  }
  ND <- nd(g, flat)
  H <- do.call(hessian(f), vlist)
  err <- 0
  for (i in seq_along(vars)) for (j in seq_along(vars)) {
    blk <- H[[vars[i]]][[vars[j]]]
    ref <- ND[(off[i] + 1):off[i + 1], (off[j] + 1):off[j + 1], drop = FALSE]
    err <- max(err, max(abs(blk - ref)))
  }
  err
}

test_that("hessian(sum(v*w)) returns the correct named-list block structure", {
  H <- do.call(hessian(function(v, w) sum(v * w)), list(v = c(1, 2), w = c(3, 4)))
  expect_type(H, "list")
  expect_identical(names(H), c("v", "w"))
  expect_identical(names(H$v), c("v", "w"))
  expect_equal(H$v$v, matrix(0, 2, 2), tolerance = TOL_EXACT)
  expect_equal(H$w$w, matrix(0, 2, 2), tolerance = TOL_EXACT)
  expect_equal(H$v$w, diag(2), tolerance = TOL_EXACT)
  expect_equal(H$w$v, diag(2), tolerance = TOL_EXACT)
})

# ===== (3) numDeriv block-slice equivalence + symmetry =====

test_that("each block equals its numDeriv Hessian slice", {
  skip_if_not_installed("numDeriv")
  A <- matrix(c(1, 2, 3, 4, 5, 6), 2, 3)
  expect_lt(nd_block_maxerr(function(v, w) sum(v * w),          list(v = c(1, 2), w = c(3, 4))), TOL_ND)
  expect_lt(nd_block_maxerr(function(v, w) sum(v^2 * w),        list(v = c(1, 2, 3), w = c(2, 1, 4))), TOL_ND)
  expect_lt(nd_block_maxerr(function(v, w) crossprod(v, A %*% w), list(v = c(1, 2), w = c(1, 2, 3))), TOL_ND)
  expect_lt(nd_block_maxerr(function(v, w) sum(v^2) + sum(w^3), list(v = c(1, 2), w = c(2, 3))), TOL_ND)
})

test_that("mixed matmul block is rectangular and equals A / t(A)", {
  A <- matrix(c(1, 2, 3, 4, 5, 6), 2, 3)
  H <- do.call(hessian(function(v, w) crossprod(v, A %*% w)), list(v = c(1, 2), w = c(1, 2, 3)))
  expect_equal(dim(H$v$w), c(2L, 3L))
  expect_equal(H$v$w, A, tolerance = TOL_EXACT)
  expect_equal(H$w$v, t(A), tolerance = TOL_EXACT)
})

test_that("assembled Hessian is symmetric across blocks", {
  H <- do.call(hessian(function(v, w) sum(v^2 * w)), list(v = c(1, 2, 3), w = c(2, 1, 4)))
  for (a in c("v", "w")) for (b in c("v", "w")) {
    expect_equal(H[[a]][[b]], t(H[[b]][[a]]), tolerance = 1e-10)
  }
})

test_that("three-variable function returns a 3x3 block grid matching numDeriv", {
  skip_if_not_installed("numDeriv")
  f <- function(a, b, c) sum(a * b) + sum(b * c)
  expect_lt(nd_block_maxerr(f, list(a = c(1, 2), b = c(3, 4), c = c(5, 6))), TOL_ND)
})

# ===== (3c) Failure modes =====

test_that("unsupported generator in a per-variable gradient propagates the grad condition", {
  expect_error(
    hessian(function(v, w) sum(gamma(v)) + sum(w)),
    class = "DefDiff_not_definable"
  )
})

test_that("scalar-denominator quotient multi-variable gradient is constructed (was a boundary)", {
  # Previously hessian_not_supported; closed by add-hessian-quotient-walker.
  skip_if_not_installed("numDeriv")
  f <- function(v, w) sum(v * w) / sum(w)
  v <- c(1, 2, 3); w <- c(4, 5, 6)
  H <- hessian(f)(v, w)
  flat <- function(x) f(x[1:3], x[4:6])
  Hnd <- nd_hessian(flat, c(v, w))
  expect_equal(H$v$v, Hnd[1:3, 1:3], tolerance = TOL_ND)
  expect_equal(H$v$w, Hnd[1:3, 4:6], tolerance = TOL_ND)
  expect_equal(H$w$v, Hnd[4:6, 1:3], tolerance = TOL_ND)
  expect_equal(H$w$w, Hnd[4:6, 4:6], tolerance = TOL_ND)
})

## test-hessian.R
## Coverage:
##   - Recognized patterns (elementwise sum(v^k), sum(<atom>(v)), quadratic form)
##   - hessian_not_supported condition for unrecognized inputs
##   - DefDiff_not_definable for multi-variable inputs
##   - Numeric equivalence vs numDeriv::hessian (skipped if absent)

TOL_HESS <- 1e-8

test_that("hessian length-1 (n=1) input returns a correctly-sized 1x1 matrix", {
  # Witness for the n=1 diag(scalar) bug: the fast-path emits diag(<length-1
  # vector>), which R reinterprets as a *dimension*. The correct result is a
  # 1x1 matrix holding the scalar second derivative.
  H2 <- hessian(function(v) sum(v^2))(c(5))
  expect_equal(dim(H2), c(1L, 1L))
  expect_equal(as.numeric(H2), 2, tolerance = TOL_HESS)

  H3 <- hessian(function(v) sum(v^3))(c(2))
  expect_equal(dim(H3), c(1L, 1L))
  expect_equal(as.numeric(H3), 12, tolerance = TOL_HESS)

  Hs <- hessian(function(v) sum(sin(v)))(c(0.7))
  expect_equal(dim(Hs), c(1L, 1L))
  expect_equal(as.numeric(Hs), -sin(0.7), tolerance = TOL_HESS)

  He <- hessian(function(v) sum(exp(v)))(c(0.5))
  expect_equal(dim(He), c(1L, 1L))
  expect_equal(as.numeric(He), exp(0.5), tolerance = TOL_HESS)

  Hsq <- hessian(function(v) 3 * sum(v^2))(c(4))
  expect_equal(dim(Hsq), c(1L, 1L))
  expect_equal(as.numeric(Hsq), 6, tolerance = TOL_HESS)
})

test_that("hessian(sum(v^k)) produces diagonal Hessian for k=2,3,4", {
  build_f <- function(k) {
    body_expr <- bquote(sum(v ^ .(k)))
    f <- function(v) NULL
    body(f) <- body_expr
    f
  }
  for (k in 2:4) {
    f <- build_f(k)
    hf <- hessian(f)
    expect_true(is.call(body(hf)))
    expect_identical(body(hf)[[1L]], as.name("diag"))
    v <- c(1, 2, 3)
    expected <- diag(k * (k - 1) * v^(k - 2))
    expect_equal(hf(v), expected, tolerance = TOL_HESS,
                 info = paste("k =", k))
  }
})

test_that("hessian(sum(<atom>(v))) for sin/cos/exp/log produces correct diagonal", {
  cases <- list(
    list(f = function(v) sum(sin(v)), expected_fn = function(v) diag(-sin(v))),
    list(f = function(v) sum(cos(v)), expected_fn = function(v) diag(-cos(v))),
    list(f = function(v) sum(exp(v)), expected_fn = function(v) diag(exp(v))),
    list(f = function(v) sum(log(v)), expected_fn = function(v) diag(-1 / v^2))
  )
  v <- c(0.5, 1.0, 1.5)
  for (c in cases) {
    hf <- hessian(c$f)
    expect_true(is.call(body(hf)))
    expect_identical(body(hf)[[1L]], as.name("diag"))
    expect_equal(hf(v), c$expected_fn(v), tolerance = TOL_HESS)
  }
})

test_that("hessian(crossprod(v, A %*% v)) produces A + t(A)", {
  A <- matrix(c(1, 2, 3, 4), 2, 2)
  hf <- hessian(function(v) crossprod(v, A %*% v))
  v <- c(1, 2)
  expected <- A + t(A)  # symmetric 2x2 with diag(2, 8) and off-diag 5, 5
  expect_equal(hf(v), expected, tolerance = TOL_HESS)
})

test_that("hessian(c * sum(v^2)) produces scaled diagonal", {
  hf <- hessian(function(v) 3 * sum(v^2))
  v <- c(1, 2, 3)
  expect_equal(hf(v), diag(rep(6, 3)), tolerance = TOL_HESS)
})

test_that("hessian on composite outer-scalar is supported via the recursive walker", {
  # Previously raised hessian_not_supported; closed by add-hessian-recursive-walker.
  hf <- hessian(function(v) sin(sum(v^2)))
  expect_true(is.function(hf))
  v <- c(1, 2, 3)
  ref <- -4 * sin(sum(v^2)) * outer(v, v) + 2 * cos(sum(v^2)) * diag(length(v))
  expect_equal(hf(v), ref, tolerance = TOL_HESS)
})

test_that("hessian constructs scalar-denominator quotients via the recursive walker", {
  # Previously raised hessian_not_supported; closed by add-hessian-quotient-walker.
  skip_if_not_installed("numDeriv")
  f <- function(v) sum(v * exp(v)) / sum(exp(v))
  hf <- hessian(f)
  expect_true(is.function(hf))
  v <- c(0.5, 1.0, 1.5)
  expect_equal(hf(v), nd_hessian(f, v), tolerance = 1e-5)
})

test_that("hessian still raises hessian_not_supported for a vector-denominator quotient", {
  # The remaining quotient boundary: a vector-valued denominator is outside the rules.
  expect_error(
    DefDiff:::.jacobian_inner(quote(v / sin(v)), "v", "v"),
    class = "hessian_not_supported"
  )
})

test_that("hessian on multi-variable input returns a named list of blocks", {
  # Previously raised DefDiff_not_definable; closed by add-multi-variable-hessian.
  H <- do.call(hessian(function(v, w) sum(v^2) + sum(w^2)),
               list(v = c(1, 2, 3), w = c(4, 5)))
  expect_type(H, "list")
  expect_identical(names(H), c("v", "w"))
  expect_equal(H$v$v, diag(rep(2, 3)), tolerance = TOL_HESS)
  expect_equal(H$w$w, diag(rep(2, 2)), tolerance = TOL_HESS)
  expect_equal(H$v$w, matrix(0, 3, 2), tolerance = TOL_HESS)
  expect_equal(H$w$v, matrix(0, 2, 3), tolerance = TOL_HESS)
})

test_that("hessian numeric output equals numDeriv::hessian within tolerance", {
  skip_if_not_installed("numDeriv")
  cases <- list(
    function(v) sum(v^2),
    function(v) sum(v^3),
    function(v) sum(sin(v)),
    function(v) sum(exp(v))
  )
  v <- c(0.5, 1.0, 1.5)
  # numDeriv::hessian is a UseMethod generic; DefDiff::hessian.function shadows
  # numDeriv::hessian.function via search path. Call numDeriv's method
  # explicitly via getFromNamespace.
  nd_hessian_function <- getFromNamespace("hessian.default", "numDeriv")
  for (f in cases) {
    hf <- hessian(f)
    nd <- nd_hessian_function(f, v)
    expect_lt(max(abs(hf(v) - nd)), TOL_HESS,
              label = paste("hessian vs numDeriv for", deparse(body(f))))
  }
})

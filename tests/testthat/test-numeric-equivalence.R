## test-numeric-equivalence.R
## Dimension: numeric-equivalence. Cross-validates DD grad/hessian/jacobian
## against TWO independent ground truths (numDeriv + an internal central
## finite-difference) AND closed forms, and cross-checks the two ground truths
## against each other (GT-vs-GT). Numeric output of grad()/hessian()/jacobian()
## is platform-independent (the macOS fast path is a performance rewrite only),
## so none of these are skip-gated on OS.

TOL_DD_ND <- 1e-8    # DD vs numDeriv
TOL_FD    <- 1e-5    # DD vs our own central finite-difference (h-limited)
TOL_EXACT <- 1e-10   # DD vs closed form
TOL_GTGT  <- 1e-5    # numDeriv vs FD cross-agreement

# Independent central finite-difference oracles (NOT numDeriv).
cfd_grad <- function(f, x, h = 1e-6) {
  vapply(seq_along(x), function(i) {
    e <- numeric(length(x)); e[i] <- h
    (f(x + e) - f(x - e)) / (2 * h)
  }, numeric(1))
}
cfd_hess <- function(f, x, h = 1e-4) {
  n <- length(x); H <- matrix(0, n, n)
  for (i in seq_len(n)) for (j in seq_len(n)) {
    ei <- numeric(n); ei[i] <- h; ej <- numeric(n); ej[j] <- h
    H[i, j] <- (f(x + ei + ej) - f(x + ei - ej) -
                f(x - ei + ej) + f(x - ei - ej)) / (4 * h * h)
  }
  H
}

test_that("grad: DD == numDeriv == central-FD across one expr per catalog family", {
  skip_if_not_installed("numDeriv")
  v <- c(0.7, -0.4, 1.3)
  fns <- list(
    sumsq       = function(v) sum(v^2),
    sin_sumsq   = function(v) sin(sum(v^2)),
    exp_sin     = function(v) sum(exp(sin(v))),
    log_quad    = function(v) log(sum(v^2)),
    sqrt_quad   = function(v) sqrt(sum(v^2)),
    cos_sum     = function(v) sum(cos(v)),
    tanh_sum    = function(v) sum(tanh(v)),
    atan_sum    = function(v) sum(atan(v)),
    prod_vforce = function(v) sum(v * sin(v)),
    quot_top    = function(v) sum(v^2) / sum(v),
    pow3        = function(v) sum(v^3),
    deep_nest   = function(v) exp(sin(log(sum(v^2))))
  )
  for (nm in names(fns)) {
    f  <- fns[[nm]]
    dd <- as.numeric(grad(f)(v))
    nd <- nd_grad(f, v)
    fd <- cfd_grad(f, v)
    expect_equal(dd, nd, tolerance = TOL_DD_ND, info = nm)
    expect_equal(dd, fd, tolerance = TOL_FD,    info = nm)
    expect_equal(nd, fd, tolerance = TOL_GTGT,  info = nm)   # GT-vs-GT
  }
})

test_that("grad: seeded random-input fuzz agrees with numDeriv over catalog", {
  skip_if_not_installed("numDeriv")
  set.seed(2024)
  fns <- list(
    function(v) sum(v^2),      function(v) sin(sum(v^2)),
    function(v) sum(exp(v)),   function(v) sum(v * cos(v)),
    function(v) log(sum(v^2)), function(v) sum(tanh(v))
  )
  for (trial in 1:25) {
    x <- runif(4, 0.2, 1.5)   # strictly positive: safe for log
    for (f in fns) {
      expect_equal(as.numeric(grad(f)(x)), nd_grad(f, x),
                   tolerance = TOL_DD_ND, info = paste("trial", trial))
    }
  }
})

test_that("grad chain-rule consistency: grad(exp(g)) == exp(g)*grad(g)", {
  skip_if_not_installed("numDeriv")
  v <- c(0.4, 0.9, 1.1)
  g <- function(v) sum(v^2)
  dd_outer <- as.numeric(grad(function(v) exp(sum(v^2)))(v))
  manual   <- exp(g(v)) * as.numeric(grad(function(v) sum(v^2))(v))
  nd       <- nd_grad(function(v) exp(sum(v^2)), v)
  expect_equal(dd_outer, manual, tolerance = TOL_EXACT)
  expect_equal(dd_outer, nd,     tolerance = TOL_DD_ND)
})

test_that("grad: 2-layer matmul-tanh chain matches numDeriv", {
  skip_if_not_installed("numDeriv")
  W1 <- matrix(c(0.2, 0.4, -0.1, 0.3, 0.5, -0.2), 3, 2)
  W2 <- matrix(c(0.6, -0.3, 0.4), 1, 3)
  fnn <- function(v) sum(tanh(W2 %*% tanh(W1 %*% v)))
  x <- c(0.3, -0.5)
  expect_equal(as.numeric(grad(fnn)(x)), nd_grad(fnn, x),
               tolerance = TOL_DD_ND)
})

test_that("grad: logistic-regression loss gradient matches numDeriv", {
  skip_if_not_installed("numDeriv")
  X <- matrix(c(1, 0.5, -0.3, 1, 0.2, 0.8), 2, 3, byrow = TRUE)
  y <- c(1, 0)
  loss <- function(b) sum(log(1 + exp(X %*% b)) - y * (X %*% b))
  b0 <- c(0.1, -0.2, 0.3)
  expect_equal(as.numeric(grad(loss)(b0)), nd_grad(loss, b0),
               tolerance = TOL_DD_ND)
})

test_that("hessian: separable elementwise diagonals match numDeriv, FD, closed form", {
  skip_if_not_installed("numDeriv")
  v <- c(0.6, 0.3, 1.1)
  cases <- list(
    sin  = list(f = function(v) sum(sin(v)),  d2 = -sin(v)),
    cos  = list(f = function(v) sum(cos(v)),  d2 = -cos(v)),
    exp  = list(f = function(v) sum(exp(v)),  d2 =  exp(v)),
    log  = list(f = function(v) sum(log(v)),  d2 = -1 / v^2),
    tanh = list(f = function(v) sum(tanh(v)), d2 = -2 * tanh(v) * (1 - tanh(v)^2)),
    pow3 = list(f = function(v) sum(v^3),     d2 =  6 * v)
  )
  for (nm in names(cases)) {
    f <- cases[[nm]]$f
    H <- hessian(f)(v)
    expect_equal(diag(H), cases[[nm]]$d2, tolerance = TOL_EXACT, info = nm)
    expect_equal(H, nd_hessian(f, v), tolerance = 1e-5, info = nm)
    expect_equal(H, cfd_hess(f, v),          tolerance = 1e-4, info = nm)
    expect_true(all(abs(H[upper.tri(H)]) < TOL_EXACT), info = nm)
  }
})

test_that("hessian: crossprod(v, A %*% v) equals A + t(A), confirmed by numDeriv & FD", {
  skip_if_not_installed("numDeriv")
  A <- matrix(c(2, 0.3, 0.1, 0.5, 3, 1, 0, 1, 4), 3, 3, byrow = TRUE)  # non-symmetric
  v <- c(0.6, -0.3, 1.1)
  f_scalar <- function(v) as.numeric(crossprod(v, A %*% v))
  H <- hessian(function(v) crossprod(v, A %*% v))(v)
  expect_equal(H, A + t(A), tolerance = TOL_EXACT)
  expect_equal(H, nd_hessian(f_scalar, v), tolerance = 1e-5)
  expect_equal(H, cfd_hess(f_scalar, v),          tolerance = 1e-4)
})

test_that("hessian: scalar-denominator quotient matches numDeriv and FD", {
  skip_if_not_installed("numDeriv")
  v <- c(0.6, -0.3, 1.1)
  f <- function(v) sum(v * exp(v)) / sum(exp(v))   # softmax-mean style
  H <- hessian(f)(v)
  expect_equal(H, nd_hessian(f, v), tolerance = 1e-5)
  expect_equal(H, cfd_hess(f, v),          tolerance = 1e-4)
  expect_equal(H, t(H), tolerance = TOL_EXACT)
})

test_that("hessian: multi-variable blocks assemble to the full numDeriv Hessian", {
  skip_if_not_installed("numDeriv")
  B <- matrix(c(1, 2, 0, 3, 1, 2), 2, 3, byrow = TRUE)
  Hb <- hessian(function(v, w) crossprod(v, B %*% w))
  H  <- do.call(Hb, list(v = c(1, 1), w = c(1, 1, 1)))
  expect_equal(H$v$w, B,    tolerance = TOL_EXACT)
  expect_equal(H$w$v, t(B), tolerance = TOL_EXACT)
  expect_true(all(abs(H$v$v) < TOL_EXACT))
  expect_true(all(abs(H$w$w) < TOL_EXACT))
  full <- function(x) { v <- x[1:2]; w <- x[3:5]; as.numeric(crossprod(v, B %*% w)) }
  assembled <- rbind(cbind(H$v$v, H$v$w), cbind(H$w$v, H$w$w))
  expect_equal(assembled, nd_hessian(full, c(1, 1, 1, 1, 1)), tolerance = 1e-5)
})

test_that("jacobian: vector-of-reductions Jacobian matches numDeriv", {
  skip_if_not_installed("numDeriv")
  v <- c(0.5, 1.2, -0.3)
  F <- function(v) c(sum(v^2), sum(sin(v)), sum(exp(v)))
  J  <- jacobian(F)(v)
  # numDeriv::jacobian is an S3 generic; DefDiff::jacobian.function shadows
  # numDeriv's jacobian.function via the global S3 table when dat is loaded.
  # Call numDeriv's method explicitly (mirrors the hessian.default idiom).
  nd <- getFromNamespace("jacobian.default", "numDeriv")(F, v)
  expect_equal(dim(J), c(3L, 3L))
  expect_equal(J, nd, tolerance = TOL_DD_ND)
  expect_equal(J[1, ], 2 * v,  tolerance = TOL_EXACT)
  expect_equal(J[2, ], cos(v), tolerance = TOL_EXACT)
  expect_equal(J[3, ], exp(v), tolerance = TOL_EXACT)
})

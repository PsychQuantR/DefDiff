## test-hessian-quotient.R
## add-hessian-quotient-walker: extend the recursive Hessian walker to construct
## the Hessian for gradient ASTs of quotient shape `a / b` where `b` is a SCALAR
## denominator (possibly var-dependent) and `a` is vector-grain or scalar.
## Vector-denominator quotients remain `hessian_not_supported`. Verified against
## numDeriv, mirroring the multi-variable-hessian template.

TOL_EXACT <- 1e-8
TOL_ND    <- 1e-5

# ===== Task 1.1 (RED) — softmax-normalizer ============================
# f(v) = sum(v * exp(v)) / sum(exp(v)): vector numerator over a scalar
# (squared) denominator that depends on v. The grad engine produces
# grad f = (grad N * D - N * grad D) / D^2; the walker must differentiate that
# quotient one more time.
test_that("softmax-normalizer quotient Hessian is constructed and matches numDeriv", {
  skip_if_not_installed("numDeriv")
  f <- function(v) sum(v * exp(v)) / sum(exp(v))
  hf <- hessian(f)
  expect_true(is.function(hf))
  v <- c(0.5, 1.0, 1.5)
  expect_equal(hf(v), nd_hessian(f, v), tolerance = TOL_ND)
})

# ===== Task 1.2 (RED) — constant-denominator scaled Hessian ===========
# A var-free scalar denominator: the Hessian reduces to H(numerator) / cc.
test_that("constant-denominator quotient Hessian reduces to a scaled Hessian", {
  cc <- 4
  f <- function(v) sum(v^2) / cc
  v <- c(1, 2, 3)
  expect_equal(hessian(f)(v), diag(rep(2, length(v))) / cc, tolerance = TOL_EXACT)
})

# ===== Task 1.3 (boundary guard) — vector denominator stays unsupported =====
# Must pass BEFORE and AFTER the change: a quotient whose denominator is
# vector-valued is outside the scalar-output contract.
test_that("vector-denominator quotient still raises hessian_not_supported", {
  expect_error(
    DefDiff:::.jacobian_inner(quote(v / sin(v)), "v", "v"),
    class = "hessian_not_supported"
  )
})

# ===== Task 4.1 (RED) — multi-variable quotient blocks ================
# A scalar-output function of two vector variables whose per-variable gradients
# are scalar-denominator quotients. The off-diagonal block H[[v]][[w]] arises
# from differentiating a var-DEPENDENT scalar denominator -> dense (rank-one),
# not diagonal. Each block must equal its numDeriv slice; cross-blocks must be
# transposes.
test_that("multi-variable quotient Hessian blocks match numDeriv and are symmetric", {
  skip_if_not_installed("numDeriv")
  f <- function(v, w) sum(v^2) / sum(w^2)
  v <- c(1, 2, 3); w <- c(2, 4)
  H <- hessian(f)(v, w)
  flat <- function(x) f(x[1:3], x[4:5])
  Hnd <- nd_hessian(flat, c(v, w))
  expect_equal(H$v$v, Hnd[1:3, 1:3], tolerance = TOL_ND)
  expect_equal(H$w$w, Hnd[4:5, 4:5], tolerance = TOL_ND)
  expect_equal(H$v$w, Hnd[1:3, 4:5], tolerance = TOL_ND)
  expect_equal(H$w$v, Hnd[4:5, 1:3], tolerance = TOL_ND)
  expect_equal(H$v$w, t(H$w$v), tolerance = 1e-10)
})

# ===== Task 4.2 — failure modes =====================================
# A quotient whose numerator gradient is uncomputable propagates the grad
# engine's condition (DefDiff_not_definable), NOT hessian_not_supported.
test_that("quotient with an uncomputable numerator propagates the grad condition", {
  expect_error(
    hessian(function(v) sum(gamma(v)) / sum(exp(v))),
    class = "DefDiff_not_definable"
  )
})

# ===== Normalize prerequisite (design acceptance criterion) ===========
# .normalize_fast_kernels must strip every vDSP fast kernel from a
# fast-dispatched gradient AST and preserve its value, so the walker downstream
# sees a uniform canonical surface.
test_that(".normalize_fast_kernels strips fast kernels and preserves value", {
  g <- DefDiff:::.grad_expr(quote(sum(v * exp(v)) / sum(exp(v))), "v")
  n <- DefDiff:::.normalize_fast_kernels(g)
  expect_false(grepl("fast_", paste(deparse(n), collapse = "")))
  v <- c(0.5, 1.0, 1.5)
  # eval() runs the grad engine's OWN generated AST (n and g), not external
  # input — symbolic-AST value verification, safe by construction.
  expect_equal(eval(n), eval(g), tolerance = 1e-12)
})

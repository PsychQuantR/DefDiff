## test-jacobian.R
## add-vector-output-jacobian (Option A): symbolic per-output-component Jacobian.
## jacobian(function(v) c(g_1, ..., g_m)) stacks m grad rows into an m x n
## matrix AST via .grad_expr + rbind — no runtime tape, closure-thesis-preserving.
## Verified against numDeriv::jacobian; explicit boundary for non-assembly shapes.

TOL_EXACT <- 1e-8
TOL_ND    <- 1e-5

# eval() below runs the walker's OWN generated AST (quoted matrix expression
# built by the jacobian helpers) in a local scope — symbolic-AST verification,
# not evaluation of external/untrusted input.

# ===== Task 1.1 (RED) — c(...) vector output ==========================
test_that("c(...) vector output returns the m x n Jacobian", {
  f <- function(v) c(sum(v), sum(v^2))
  J <- jacobian(f)(c(1, 2, 3))
  expect_equal(J, rbind(c(1, 1, 1), c(2, 4, 6)), tolerance = TOL_EXACT)
})

# ===== Task 2.1 (RED) — scalar degrade + chain-catalog components =====
test_that("scalar-output body degrades to a 1 x n row Jacobian", {
  f <- function(v) sum(v^2)
  v <- c(1, 2, 3)
  J <- jacobian(f)(v)
  expect_equal(dim(J), c(1L, 3L))
  expect_equal(J, matrix(2 * v, nrow = 1), tolerance = TOL_EXACT)
})

test_that("components drawn from the chain catalog stack correctly", {
  f <- function(v) c(sum(sin(v)), sum(exp(v)))
  v <- c(0.5, 1.0, 1.5)
  J <- jacobian(f)(v)
  expect_equal(J, rbind(cos(v), exp(v)), tolerance = TOL_EXACT)
})

# ===== Task 3.1 (RED) — boundary + failure modes =====================
test_that("out-of-catalog component propagates the grad engine condition", {
  f <- function(v) c(sum(v), sum(gamma(v)))
  expect_error(jacobian(f), class = "DefDiff_not_definable")
})

test_that("implicit (non-c(...)) vector body raises jacobian_not_supported", {
  w <- c(10, 20, 30)
  f <- function(v) v * w
  expect_error(jacobian(f), class = "jacobian_not_supported")
})

test_that("control flow in the body raises DefDiff_not_definable", {
  f <- function(v) if (v[1] > 0) c(sum(v), sum(v^2)) else c(0, 0)
  expect_error(jacobian(f), class = "DefDiff_not_definable")
})

# ===== Task 4.1 (RED) — numDeriv equivalence =========================
test_that("symbolic Jacobian equals numDeriv::jacobian", {
  skip_if_not_installed("numDeriv")
  cases <- list(
    list(f = function(v) c(sum(v), sum(v^2)),          v = c(1, 2, 3)),
    list(f = function(v) c(sum(sin(v)), sum(exp(v))),  v = c(0.5, 1.0, 1.5)),
    list(f = function(v) c(sum(v^2), sum(v^3), sum(v)), v = c(0.7, 1.3, 2.1, 0.4))
  )
  for (cc in cases) {
    expect_equal(jacobian(cc$f)(cc$v), nd_jacobian(cc$f, cc$v),
                 tolerance = TOL_ND)
  }
})

# ===== Task 4.2 (RED) — jacobian.call programmatic entry =============
test_that("jacobian.call returns a matrix-valued call with no tape constructs", {
  J_ast <- jacobian(quote(c(sum(v), sum(v^2))), "v")
  expect_true(is.call(J_ast))
  v <- c(1, 2, 3)
  # eval() runs the jacobian engine's OWN generated AST (J_ast), symbolic
  # verification, not external input.
  expect_equal(eval(J_ast), rbind(c(1, 1, 1), c(2, 4, 6)), tolerance = TOL_EXACT)
  expect_false(grepl("tape|backward", paste(deparse(J_ast), collapse = "")))
})

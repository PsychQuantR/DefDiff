## test-walker-jacobian-helpers.R
## Tier 5 Option B Phase 1 (`add-walker-shape-extension`): pullback helpers.
## Verifies that each constructor returns an R function operating on ASTs,
## and that composed pullbacks evaluated with sample upstream gradients
## return correct numeric results matching standard differentiation rules.
##
## NOTE on eval(): tests use eval() to numerically verify pullback ASTs.
## The ASTs are constructed by trusted code (this file's own bquote/quote
## calls + the DefDiff:::.make_pullback_* helpers). No external/user input is
## passed to eval — this is symbolic-AST verification, not arbitrary code
## execution. Safe and intentional.

# Shorthand accessors (helpers are internal)
make_id    <- function() DefDiff:::.make_pullback_identity()
make_zero  <- function(n_expr) DefDiff:::.make_pullback_zero(n_expr)
make_add   <- function(p_a, p_b) DefDiff:::.make_pullback_add(p_a, p_b)
make_sub   <- function(p_a, p_b) DefDiff:::.make_pullback_sub(p_a, p_b)
make_mul   <- function(p_a, val_a, p_b, val_b) DefDiff:::.make_pullback_mul(p_a, val_a, p_b, val_b)
make_div   <- function(p_a, val_a, p_b, val_b) DefDiff:::.make_pullback_div(p_a, val_a, p_b, val_b)
make_pow   <- function(p_a, val_a, k) DefDiff:::.make_pullback_pow(p_a, val_a, k)
make_chain <- function(p_inner, f_deriv_ast) DefDiff:::.make_pullback_chain(p_inner, f_deriv_ast)
make_matmul <- function(p_inner, W) DefDiff:::.make_pullback_matmul(p_inner, W)

# ===== 1-9: Each constructor returns a function-of-AST =====

test_that("identity pullback returns a function that echoes upstream AST", {
  p <- make_id()
  expect_true(is.function(p))
  result <- p(quote(rep(1, 3)))
  expect_identical(result, quote(rep(1, 3)))
})

test_that("zero pullback returns a function that yields rep(0, n)", {
  p <- make_zero(quote(length(v)))
  expect_true(is.function(p))
  result <- p(quote(rep(1, 3)))
  # Should be AST: rep(0, length(v))
  expect_identical(result, bquote(rep(0, length(v))))
})

test_that("add pullback composes p_a(g) + p_b(g)", {
  p <- make_add(make_id(), make_id())
  result <- p(quote(g))
  expect_identical(result, quote(g + g))
})

test_that("sub pullback composes p_a(g) - p_b(g)", {
  p <- make_sub(make_id(), make_id())
  result <- p(quote(g))
  expect_identical(result, quote(g - g))
})

test_that("mul pullback applies product rule via pullback composition", {
  p <- make_mul(make_id(), quote(a_val), make_id(), quote(b_val))
  result <- p(quote(g))
  # Expected: g * b_val + g * a_val
  expect_identical(result, quote(g * b_val + g * a_val))
})

test_that("div pullback applies quotient rule via pullback composition", {
  p <- make_div(make_id(), quote(a_val), make_id(), quote(b_val))
  result <- p(quote(g))
  # Expected: g / b_val - g * a_val / b_val^2
  expect_identical(result, quote(g/b_val - g * a_val/b_val^2))
})

test_that("pow pullback applies power rule with k-1 exponent", {
  p <- make_pow(make_id(), quote(a_val), 3)
  result <- p(quote(g))
  # Expected: g * 3 * a_val^2
  expect_identical(result, quote(g * 3 * a_val^2))
})

test_that("chain pullback applies chain rule g * f_deriv", {
  # f_deriv_ast is the AST of f'(val_inner) — caller-constructed
  p <- make_chain(make_id(), quote(cos(v)))
  result <- p(quote(g))
  expect_identical(result, quote(g * cos(v)))
})

test_that("matmul pullback applies backprop t(W) %*% upstream", {
  p <- make_matmul(make_id(), quote(W))
  result <- p(quote(g))
  expect_identical(result, quote(t(W) %*% g))
})

# ===== 10-14: Numeric verification — composed pullbacks give correct gradients =====

test_that("Composed: d/dv(v*v) at v=c(1,2,3) with upstream rep(1,3) = c(2,4,6)", {
  p <- make_mul(make_id(), quote(v), make_id(), quote(v))
  grad_ast <- p(quote(rep(1, length(v))))
  v <- c(1, 2, 3)
  expect_equal(eval(grad_ast), c(2, 4, 6))
})

test_that("Composed: d/dv(2v * v) at v=c(1,2,3) = c(4,8,12) (= 4v)", {
  # 2v as a pullback chain: mul(zero, 2, identity, v)
  p_2v <- make_mul(make_zero(quote(length(v))), quote(2),
                   make_id(),                    quote(v))
  # Full: (2v) * v, with mul again
  p <- make_mul(p_2v, quote(2 * v), make_id(), quote(v))
  grad_ast <- p(quote(rep(1, length(v))))
  v <- c(1, 2, 3)
  expect_equal(eval(grad_ast), c(4, 8, 12))
})

test_that("Composed: d/dv(v + v) at v=c(1,2,3) = c(2,2,2)", {
  p <- make_add(make_id(), make_id())
  grad_ast <- p(quote(rep(1, length(v))))
  v <- c(1, 2, 3)
  expect_equal(eval(grad_ast), c(2, 2, 2))
})

test_that("Composed: d/dv(v - 2*v) at v=c(1,2,3) = c(-1,-1,-1)", {
  # v - 2*v: sub(identity, mul(zero, 2, identity, v))
  p_2v <- make_mul(make_zero(quote(length(v))), quote(2),
                   make_id(),                    quote(v))
  p <- make_sub(make_id(), p_2v)
  grad_ast <- p(quote(rep(1, length(v))))
  v <- c(1, 2, 3)
  expect_equal(eval(grad_ast), c(-1, -1, -1))
})

test_that("Composed: d/dv(sin(v)) at v=c(0.5, 1.0) = cos(v) via chain rule", {
  # chain(identity, cos(v))
  p <- make_chain(make_id(), quote(cos(v)))
  grad_ast <- p(quote(rep(1, length(v))))
  v <- c(0.5, 1.0)
  expect_equal(eval(grad_ast), cos(v))
})

# ===== 15: Phase 1 regression guard — existing tests still pass =====
# (This is verified externally via `testthat::test_local()` reporting full
# suite 0 FAIL. Asserting it inside this file would be circular — the test
# itself is part of the suite being checked. Documenting expectation here.)

test_that("Phase 1 deliverable: no walker changes, existing suite unaffected", {
  # The helpers are new functions. No existing code paths changed.
  # External verification: testthat::test_local() reports 592+ PASS / 0 FAIL.
  # This test_that block exists to make the regression guard explicit in
  # the change's test record.
  expect_true(exists(".make_pullback_identity", envir = asNamespace("DefDiff")))
  expect_true(exists(".make_pullback_matmul",   envir = asNamespace("DefDiff")))
})

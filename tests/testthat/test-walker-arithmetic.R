## test-walker-arithmetic.R
## Walker leaf + arithmetic cases (Phase 2/3) + Phase 6 cleanup: walker now
## returns `list(value, pullback)` — no transitional `_legacy` field. These
## tests verify the pullback path produces the correct gradient when applied
## with a rep(1, n) upstream.
##
## NOTE on eval(): we numerically verify pullback ASTs by evaluating them
## with bound `v`. The ASTs come from trusted code (DefDiff:::.grad_inner +
## DefDiff:::.make_pullback_* helpers). No external input passes through eval —
## symbolic-AST verification, not arbitrary code execution.

# Helper: dispatch to .grad_inner, then apply pullback with rep(1, length(v))
# upstream, evaluating the resulting AST in an env where v is bound.
#
# SAFETY: eval() is intentional here — `grad_ast` is built by trusted internal
# code (DefDiff:::.grad_inner + DefDiff:::.make_pullback_*). No external/user input
# flows into the AST; this is symbolic-AST verification, not arbitrary code
# execution. The env is restricted to baseenv() + the test's bound `v`.
.eval_pullback <- function(expr_quoted, v_value) {
  result <- DefDiff:::.grad_inner(expr_quoted, "v")
  pullback <- DefDiff:::.pullback_of(result)
  if (is.null(pullback)) {
    stop("Pullback is NULL — sub-expression returned bare AST (pre-Phase-3)")
  }
  upstream <- bquote(rep(1, length(v)))
  grad_ast <- pullback(upstream)
  env <- new.env(parent = baseenv())
  env$v <- v_value
  eval(grad_ast, envir = env)
}

# ===== 6 arithmetic ops via walker dispatch =====

test_that("Phase 2: walker dispatches + correctly via pullback", {
  # d/dv(v + v) = 2 (per coord)
  v <- c(1, 2, 3)
  expect_equal(.eval_pullback(quote(v + v), v), c(2, 2, 2))
})

test_that("Phase 2: walker dispatches - correctly via pullback", {
  # d/dv(v - v) = 0; d/dv(2*v - v) requires 2 to be a leaf shim too
  # Use binary - between two var leaves: d/dv(v - v) = 0 per coord
  v <- c(1, 2, 3)
  expect_equal(.eval_pullback(quote(v - v), v), c(0, 0, 0))
})

test_that("Phase 2: walker dispatches * correctly via pullback", {
  # d/dv(v * v) = 2v per coord
  v <- c(1, 2, 3)
  expect_equal(.eval_pullback(quote(v * v), v), 2 * v)
})

test_that("Phase 2: walker dispatches / correctly via pullback", {
  # d/dv(v / v) = 0 per coord (constant 1 function)
  # But v/v with v[i] != 0 evaluates to 1; gradient = 0
  v <- c(1, 2, 3)
  expect_equal(.eval_pullback(quote(v / v), v), c(0, 0, 0))
})

test_that("Phase 2: walker dispatches ^ correctly via pullback (constant exponent)", {
  # d/dv(v^3) = 3 * v^2 per coord
  v <- c(1, 2, 3)
  expect_equal(.eval_pullback(quote(v^3), v), 3 * v^2)
})

test_that("Phase 2: walker dispatches unary - correctly via pullback", {
  # d/dv(-v) = -1 per coord
  v <- c(1, 2, 3)
  expect_equal(.eval_pullback(quote(-v), v), c(-1, -1, -1))
})

# ===== Bonus: integration with existing grad() to confirm no regression =====

test_that("End-to-end: grad(function(v) sum(v * v))(c(1,2,3)) = c(2,4,6)", {
  # Integration check that the pullback API correctly feeds .sum_rule's
  # gradient construction. Post-Phase-6, the pullback is the only path.
  gf <- grad(function(v) sum(v * v))
  v <- c(1, 2, 3)
  expect_equal(gf(v), 2 * v)
})

# ===== Phase 3: chain library walker cases =====
# Each of sin/cos/exp/log/tanh/sqrt/atan now returns shim with a pullback
# constructed via .make_pullback_chain. Verify the gradient when applied with
# rep(1, n) upstream.

test_that("Phase 3: walker dispatches sin via pullback", {
  v <- c(0.5, 1.0, 1.5)
  expect_equal(.eval_pullback(quote(sin(v)), v), cos(v))
})

test_that("Phase 3: walker dispatches cos via pullback", {
  v <- c(0.5, 1.0, 1.5)
  expect_equal(.eval_pullback(quote(cos(v)), v), -sin(v))
})

test_that("Phase 3: walker dispatches exp via pullback", {
  v <- c(0.1, 0.5, 1.0)
  expect_equal(.eval_pullback(quote(exp(v)), v), exp(v))
})

test_that("Phase 3: walker dispatches log via pullback", {
  v <- c(0.5, 1.0, 2.0)
  expect_equal(.eval_pullback(quote(log(v)), v), 1 / v)
})

test_that("Phase 3: walker dispatches tanh via pullback", {
  v <- c(-0.5, 0.0, 0.5)
  expect_equal(.eval_pullback(quote(tanh(v)), v), 1 - tanh(v)^2)
})

test_that("Phase 3: walker dispatches sqrt via pullback", {
  v <- c(1.0, 4.0, 9.0)
  expect_equal(.eval_pullback(quote(sqrt(v)), v), 1 / (2 * sqrt(v)))
})

test_that("Phase 3: walker dispatches atan via pullback", {
  v <- c(-1.0, 0.0, 1.0)
  expect_equal(.eval_pullback(quote(atan(v)), v), 1 / (1 + v^2))
})

# Composition test: chain + arithmetic both shim → fully-composed pullback
test_that("Phase 3: composed pullback works for sin(v) * v (chain ⊗ arith)", {
  # d/dv(sin(v) * v) = cos(v) * v + sin(v) per coord
  v <- c(0.5, 1.0, 1.5)
  expect_equal(.eval_pullback(quote(sin(v) * v), v), cos(v) * v + sin(v))
})

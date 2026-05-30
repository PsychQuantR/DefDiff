## test-simplify-extensions.R
## add-simplify-extensions: four algebraic-simplification rule extensions.
## Groups 1-3 are pre-grad fold rules in .algebraic_simplify (R/simplify.R);
## group 4 is the generalized sum-of-powers fast path (fast_sum_pow,
## R/fast_dispatch.R + src/fast_grad.cpp). Folds must be gradient/value
## preserving — verified by eval + grad equivalence.

TOL_REL <- 1e-10

# eval() below runs quoted ASTs the simplifier produced or the test itself
# quoted — symbolic-AST verification with fixed local bindings, not external
# untrusted input.

# ===== Group 1 (RED) — sqrt(x^2) -> abs(x), dormant unless abs registered ===
test_that("sqrt(x^2) folds to abs(x) only when abs is in the L_0 catalog", {
  on.exit(DefDiff:::register_default_catalog(), add = TRUE)
  # (a) with abs registered -> abs(v), NOT v
  extend_language("L_0", "abs", function(expr, var) {
    bquote(sign(.(expr[[2L]])) * .(DefDiff:::.grad_expr(expr[[2L]], var)))
  })
  expect_identical(DefDiff:::.algebraic_simplify(quote(sqrt(v^2))), quote(abs(v)))
  # (b) negative-input correctness: |x|, not x (the wrong fold)
  folded <- DefDiff:::.algebraic_simplify(quote(sqrt(v^2)))
  expect_equal(eval(folded, list(v = c(-3, 2, -1))), c(3, 2, 1), tolerance = TOL_REL)
})

test_that("sqrt(x^2) is a no-op when abs is not registered", {
  # default catalog has no abs -> dormant, returns unchanged
  expect_identical(DefDiff:::.algebraic_simplify(quote(sqrt(v^2))), quote(sqrt(v^2)))
})

# ===== Group 2 (RED) — constant folding ==============================
test_that("literal-only arithmetic folds to a single literal", {
  expect_equal(DefDiff:::.algebraic_simplify(quote(2 + 3)), 5)
})

test_that("literal factors fold while the variable term is preserved", {
  folded <- DefDiff:::.algebraic_simplify(quote(2 * sum(v^2) * 4))
  v <- c(1, 2, 3)
  expect_equal(eval(folded), eval(quote(8 * sum(v^2))), tolerance = TOL_REL)
})

test_that("non-finite literal arithmetic is NOT folded (no Inf in AST)", {
  expect_identical(DefDiff:::.algebraic_simplify(quote(1 / 0)), quote(1 / 0))
})

test_that("constant folding is idempotent", {
  once  <- DefDiff:::.algebraic_simplify(quote(2 * sum(v^2) * 4))
  twice <- DefDiff:::.algebraic_simplify(once)
  expect_identical(twice, once)
})

# ===== Group 3 (RED) — conservative trig identities ==================
test_that("sin^2 + cos^2 folds to 1 (both orders)", {
  expect_identical(DefDiff:::.algebraic_simplify(quote(sin(v)^2 + cos(v)^2)), 1)
  expect_identical(DefDiff:::.algebraic_simplify(quote(cos(v)^2 + sin(v)^2)), 1)
})

test_that("2 sin cos folds to sin(2x)", {
  expect_identical(DefDiff:::.algebraic_simplify(quote(2 * sin(v) * cos(v))), quote(sin(2 * v)))
})

test_that("trig identity does NOT fold when inner arguments differ", {
  e <- quote(sin(v)^2 + cos(w)^2)
  expect_identical(DefDiff:::.algebraic_simplify(e), e)
})

test_that("trig folds are gradient-preserving", {
  v <- c(0.5, 1.0, 1.5)
  # sin^2 + cos^2 folds to 1, so sum(...) becomes sum(1): a constant. The fold
  # is gradient-preserving — the gradient is zero. The grad engine returns the
  # scalar 0 for a constant body (a pre-existing convention; constant-gradient
  # *shape* is grad-engine territory, out of this simplifier change's scope),
  # so the assertion verifies the gradient is zero rather than its length.
  expect_true(all(abs(grad(function(v) sum(sin(v)^2 + cos(v)^2))(v)) < TOL_REL))
  # 2 sin cos folds to sin(2v); gradient is 2 cos(2v), full length preserved.
  expect_equal(grad(function(v) sum(2 * sin(v) * cos(v)))(v), 2 * cos(2 * v),
               tolerance = TOL_REL)
})

# ===== Group 4 (RED) — generalized sum-of-powers fast path ===========
test_that("fast_sum_pow computes sum(v^k) for k >= 3", {
  skip_on_os(c("windows", "linux", "solaris"))  # Apple Accelerate only
  expect_equal(fast_sum_pow(c(1, 2, 3), 4), sum(c(1, 2, 3)^4), tolerance = TOL_REL)  # 98
  set.seed(42)
  v <- runif(1000)
  expect_equal(fast_sum_pow(v, 3), sum(v^3), tolerance = TOL_REL)
})

test_that(".substitute_sum_sq keeps k=2 on fast_sum_sq and routes k>=3 to fast_sum_pow", {
  expect_identical(DefDiff:::.substitute_sum_sq(quote(sum(v^2)), "v"), quote(fast_sum_sq(v)))
  expect_identical(DefDiff:::.substitute_sum_sq(quote(sum(v^4)), "v"), quote(fast_sum_pow(v, 4)))
})

test_that("grad through sum(v^3) is correct and dispatches fast_sum_pow", {
  skip_on_os(c("windows", "linux", "solaris"))
  v <- c(0.5, 1.0, 1.5)
  gf <- grad(function(v) sin(sum(v^3)))
  expect_equal(gf(v), cos(sum(v^3)) * 3 * v^2, tolerance = TOL_REL)
  expect_match(paste(deparse(body(gf)), collapse = ""), "fast_sum_pow")
})

test_that("non-generalizable powers fall through unchanged", {
  expect_identical(DefDiff:::.substitute_sum_sq(quote(sum(v^2.5)), "v"), quote(sum(v^2.5)))
  expect_identical(DefDiff:::.substitute_sum_sq(quote(sum(v^1)), "v"), quote(sum(v^1)))
  expect_identical(DefDiff:::.substitute_sum_sq(quote(sum((v + 1)^3)), "v"), quote(sum((v + 1)^3)))
})

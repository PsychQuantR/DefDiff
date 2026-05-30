## test-rep-generator.R
## Tier 4 change `add-rep-generator`: rep() registered as L_0 data primitive.
## When first arg doesn't contain v, rep produces a constant vector (gradient
## = 0). When first arg contains v, raise DefDiff_not_definable.

skip_if_no_fast <- function() {
  if (!.fast_path_available()) skip("macOS fast-path required")
}

# ---- Constant first-argument cases (WORKS) ----

test_that("rep(scalar, length(v)) constant gradient evaluates correctly", {
  skip_if_no_fast()
  gf <- grad(function(v) sum(rep(5, length(v)) * v))
  expect_equal(as.numeric(gf(c(1, 2, 3))), c(5, 5, 5), tolerance = 1e-10)
})

test_that("rep(c(1,-1), length.out=length(v)) constant pattern", {
  skip_if_no_fast()
  gf <- grad(function(v) sum(rep(c(1, -1), length.out = length(v)) * v))
  out <- gf(c(0.1, 0.2, 0.3, 0.4))
  expect_equal(as.numeric(out), c(1, -1, 1, -1), tolerance = 1e-10)
})

test_that("logistic_loss with inlined labels matches analytic gradient", {
  skip_if_no_fast()
  f <- function(v) sum(log(1 + exp(-rep(c(1, -1), length.out = length(v)) * v)))
  gf <- grad(f)
  v <- c(0.1, 0.5, -0.3, 0.2)
  y <- rep(c(1, -1), length.out = length(v))
  expected <- -y * exp(-y * v) / (1 + exp(-y * v))
  expect_equal(as.numeric(gf(v)), as.numeric(expected), tolerance = 1e-10)
})

# ---- Variable first-argument cases (rejected) ----

test_that("rep(v, 3) raises DefDiff_not_definable naming rep", {
  expect_error(
    grad(function(v) sum(rep(v, 3))),
    class = "DefDiff_not_definable",
    regexp = "rep"
  )
})

test_that("rep(v + 1, 3) — variable inside first-arg expression — raises", {
  # The check is on .contains_var(first_arg, var), not on whether first_arg
  # is literally the variable symbol. v+1 contains v, so this rejects.
  expect_error(
    grad(function(v) sum(rep(v + 1, 3))),
    class = "DefDiff_not_definable",
    regexp = "rep"
  )
})

# ---- Edge case: zero-constant ----

test_that("rep(0, length(v)) edge case returns zero gradient (no special-case bug)", {
  skip_if_no_fast()
  # The L_0 rule returns 0 for constant values; combined with product rule
  # and the algebraic_simplify's 0*x→0 fold, the gradient of
  # sum(rep(0, length(v)) * v) should be c(0, 0, 0).
  gf <- grad(function(v) sum(rep(0, length(v)) * v))
  expect_equal(as.numeric(gf(c(1, 2, 3))), c(0, 0, 0), tolerance = 1e-10)
})

# ---- Top-level fire (not inside sum) ----

test_that("rep rule fires for top-level call too (catalog lookup, not sum-only)", {
  # Verify the rule is reachable from .grad_expr directly, not only via
  # .sum_rule's walker fallback. We can't grad(rep(...)) standalone because
  # the output is a vector and DD requires scalar output, but the catalog
  # lookup itself returns a non-error rule.
  expect_true(is.function(DefDiff:::.dat_env$catalog$L_0[["rep"]]))
  # And the rule returns 0 when called directly with a constant first arg:
  rule <- DefDiff:::.dat_env$catalog$L_0[["rep"]]
  expect_identical(rule(quote(rep(5, 3)), "v"), 0)
})

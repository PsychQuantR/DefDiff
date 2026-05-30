## test-algebraic-simplify.R
## Tier 4 change `add-algebraic-simplifier`: pre-grad AST simplifier.
## 15 unit tests cover all 9 fold rules + nested + negative guard +
## idempotence + non-call passthrough. 3 integration tests verify the
## hook into .grad_expr restores Tier 1 fast-path dispatch.

simplify <- DefDiff:::.algebraic_simplify

# ---- Rules 1 & 2: exp(log(x)) / log(exp(x)) inverse pairs ----

test_that("Rule 1: exp(log(v)) folds to v", {
  expect_identical(simplify(quote(exp(log(v)))), quote(v))
})

test_that("Rule 1 with complex inner: exp(log(sum(v^2))) folds to sum(v^2)", {
  expect_identical(simplify(quote(exp(log(sum(v^2))))), quote(sum(v^2)))
})

test_that("Rule 2: log(exp(v)) folds to v", {
  expect_identical(simplify(quote(log(exp(v)))), quote(v))
})

# ---- Rules 3 & 4: power identities ----

test_that("Rule 3: v^1 folds to v", {
  expect_identical(simplify(quote(v^1)), quote(v))
})

test_that("Rule 4: v^0 folds to 1", {
  expect_identical(simplify(quote(v^0)), 1)
})

test_that("Nested fold: 2 * v^1 folds to 2 * v", {
  expect_identical(simplify(quote(2 * v^1)), quote(2 * v))
})

# ---- Rules 5-7: additive identities ----

test_that("Rule 5: 0 + v folds to v", {
  expect_identical(simplify(quote(0 + v)), quote(v))
})

test_that("Rule 6: v + 0 folds to v", {
  expect_identical(simplify(quote(v + 0)), quote(v))
})

test_that("Rule 7: v - 0 folds to v", {
  expect_identical(simplify(quote(v - 0)), quote(v))
})

# ---- Rules 8 & 9: multiplicative identity + annihilator ----

test_that("Rule 8: 1 * v and v * 1 fold to v", {
  expect_identical(simplify(quote(1 * v)), quote(v))
  expect_identical(simplify(quote(v * 1)), quote(v))
})

test_that("Rule 9: 0 * v and v * 0 fold to 0", {
  expect_identical(simplify(quote(0 * v)), 0)
  expect_identical(simplify(quote(v * 0)), 0)
})

# ---- Bottom-up traversal ----

test_that("Bottom-up: exp(log(exp(log(v)))) folds to v", {
  expect_identical(simplify(quote(exp(log(exp(log(v)))))), quote(v))
})

# ---- Negative guard ----

test_that("Negative guard: exp(log(v) + 1) does NOT fold", {
  # log is inside `+`, not the direct argument of exp; pattern must not match.
  expect_identical(simplify(quote(exp(log(v) + 1))), quote(exp(log(v) + 1)))
})

# ---- Idempotence ----

test_that("Idempotence: simplify twice equals simplify once", {
  # Representative compound expression touching multiple rules.
  e <- quote(0 + (1 * exp(log(sum(v^2)))) - 0)
  once <- simplify(e)
  twice <- simplify(once)
  expect_identical(twice, once)
  # And the resulting form is the fully-folded sum(v^2):
  expect_identical(once, quote(sum(v^2)))
})

# ---- Non-call passthrough ----

test_that("Non-call inputs pass through unchanged", {
  expect_identical(simplify(quote(v)), quote(v))         # symbol
  expect_identical(simplify(5), 5)                       # numeric literal
  expect_identical(simplify(5L), 5L)                     # integer literal
  expect_identical(simplify("abc"), "abc")               # string (atomic)
})

# ===== Integration with grad() =====

skip_if_no_fast <- function() {
  if (!.fast_path_available()) skip("macOS fast-path required")
}

test_that("Integration: grad(exp(log(sum(v^2)))) body identical to grad(sum(v^2)) body", {
  skip_if_no_fast()
  gf_simpl <- grad(function(v) exp(log(sum(v^2))))
  gf_plain <- grad(function(v) sum(v^2))
  expect_identical(body(gf_simpl), body(gf_plain))
})

test_that("Integration: both functions produce identical numeric output", {
  skip_if_no_fast()
  gf_simpl <- grad(function(v) exp(log(sum(v^2))))
  v <- c(1, 2, 3)
  expect_equal(as.numeric(gf_simpl(v)), c(2, 4, 6), tolerance = 1e-10)
})

test_that("Integration: log(exp(sum(v))) gradient equals sum(v) gradient", {
  skip_if_no_fast()
  gf_simpl <- grad(function(v) log(exp(sum(v))))
  gf_plain <- grad(function(v) sum(v))
  v <- c(1, 2, 3)
  expect_equal(as.numeric(gf_simpl(v)), as.numeric(gf_plain(v)), tolerance = 1e-10)
})

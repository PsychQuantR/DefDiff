## test-multi-variable-grad.R
## add-multi-variable-gradient: grad() supports scalar-output functions of
## multiple vector arguments. Return-shape rule: length(vars) == 1 → bare
## result (unchanged); length(vars) >= 2 → named list keyed by variable.
## Covers all four S3 methods (function, call, formula, expression) plus the
## absent-variable and control-flow boundaries.

# ===== (a) two-variable product: named list + values =====

test_that("grad.function: 2-var sum(v*w) returns named list of gradients", {
  gf <- grad(function(v, w) sum(v * w))
  r <- gf(c(1, 2, 3), c(10, 20, 30))
  expect_type(r, "list")
  expect_identical(names(r), c("v", "w"))
  expect_equal(as.numeric(r$v), c(10, 20, 30))  # d/dv = w
  expect_equal(as.numeric(r$w), c(1, 2, 3))      # d/dw = v
})

# ===== (b) variables of different lengths =====

test_that("grad.function: variables of different lengths (sum(v)*sum(w))", {
  gf <- grad(function(v, w) sum(v) * sum(w))
  r <- gf(c(1, 2, 3), c(4, 5))
  # d/dv sum(v)*sum(w) = sum(w) broadcast over v; d/dw = sum(v) over w
  expect_equal(as.numeric(r$v), rep(9, 3))   # sum(c(4,5)) = 9
  expect_equal(as.numeric(r$w), rep(6, 2))   # sum(c(1,2,3)) = 6
})

# ===== (c) three-variable function =====

test_that("grad.function: 3-variable function returns 3-element named list", {
  gf <- grad(function(a, b, c) sum(a * b + c))
  r <- gf(c(1, 2), c(3, 4), c(5, 6))
  expect_identical(names(r), c("a", "b", "c"))
  expect_equal(as.numeric(r$a), c(3, 4))         # d/da = b
  expect_equal(as.numeric(r$b), c(1, 2))         # d/db = a
  expect_equal(as.numeric(r$c), rep(1, 2))       # d/dc sum(... + c) = 1
})

# ===== (d) single explicit var returns bare vector, not a list =====

test_that("grad.function: vars='v' single-explicit returns bare vector", {
  gf <- grad(function(v, w) sum(v * w), vars = "v")
  r <- gf(c(1, 2, 3), c(10, 20, 30))
  expect_false(is.list(r))
  expect_equal(as.numeric(r), c(10, 20, 30))
})

# ===== (e) default vars = all formals =====

test_that("grad.function: default vars equals all formal arguments", {
  # No explicit vars → grad w.r.t. every formal (the full gradient).
  gf <- grad(function(v, w) sum(v * w))
  r <- gf(c(1, 1, 1), c(2, 2, 2))
  expect_identical(names(r), c("v", "w"))
})

# ===== (f) grad.call multi-var: named list of ASTs =====

test_that("grad.call: multi-var returns named list of gradient ASTs", {
  r <- grad(quote(sum(v * w)), c("v", "w"))
  expect_type(r, "list")
  expect_identical(names(r), c("v", "w"))
  expect_identical(r$v, quote(w))
  expect_identical(r$w, quote(v))
})

test_that("grad.call: single var still returns a bare AST", {
  r <- grad(quote(sum(v^2)), "v")
  expect_false(is.list(r))
  expect_identical(r, quote(2 * v))
})

# ===== (g) grad.formula multi-var: named list of formulas, LHS preserved =====

test_that("grad.formula: multi-var one-sided returns named list of formulas", {
  r <- grad(~ sum(v * w), c("v", "w"))
  expect_type(r, "list")
  expect_identical(names(r), c("v", "w"))
  expect_identical(r$v, ~ w)
  expect_identical(r$w, ~ v)
})

test_that("grad.formula: multi-var two-sided preserves LHS in each element", {
  r <- grad(y ~ sum(v * w), c("v", "w"))
  expect_identical(r$v, y ~ w)
  expect_identical(r$w, y ~ v)
})

# ===== (h) grad.expression multi-var: named list of expression objects =====

test_that("grad.expression: multi-var returns named list of expression objects", {
  r <- grad(expression(sum(v * w)), c("v", "w"))
  expect_type(r, "list")
  expect_identical(names(r), c("v", "w"))
  expect_true(is.expression(r$v) && is.expression(r$w))
  expect_identical(r$v, expression(w))
  expect_identical(r$w, expression(v))
})

# ===== (i) absent variable yields a zero gradient, not an error =====

test_that("grad.function: absent variable yields a zero gradient", {
  # w does not appear in the body; its partial derivative is zero everywhere.
  gf <- grad(function(v, w) sum(v^2), vars = c("v", "w"))
  r <- gf(c(1, 2, 3), c(0, 0))
  expect_equal(as.numeric(r$v), c(2, 4, 6))   # d/dv sum(v^2) = 2v
  expect_true(all(r$w == 0))                   # d/dw = 0 (absent → zero)
})

# ===== (j) control flow still raises DefDiff_not_definable =====

test_that("grad.function: control flow in multi-var body still raises", {
  expect_error(
    grad(function(v, w) if (sum(v) > 0) sum(v * w) else 0, vars = c("v", "w")),
    class = "DefDiff_not_definable"
  )
})

## test-top-level-div.R
## Tier 4 change `close-top-level-division-gap`: top-level `/` (both
## operands scalar reductions, both contain v) now produces a working
## gradient via the inline quotient-rule construction in the L_0 `/` rule.
##
## All test cases are scalar/scalar at top level — vector/vector forms like
## v/sin(v) remain out of DD's single-var scalar-output contract and are
## covered (inside-sum variant only) by test-closure.R.

skip_if_no_fast <- function() {
  if (!.fast_path_available()) skip("macOS fast-path required")
}

test_that("Quotient rule: sum(v*exp(v)) / sum(exp(v))", {
  skip_if_no_fast()
  gf <- grad(function(v) sum(v * exp(v)) / sum(exp(v)))
  v <- runif(20L)
  S <- sum(exp(v)); T_val <- sum(v * exp(v))
  expected <- exp(v) * ((1 + v) * S - T_val) / S^2
  expect_equal(as.numeric(gf(v)), as.numeric(expected), tolerance = 1e-10)
})

test_that("Quotient rule: sum(v^2) / sum(v)", {
  skip_if_no_fast()
  gf <- grad(function(v) sum(v^2) / sum(v))
  v <- runif(20L)
  # d/dv_i of T/S where T=sum(v^2), S=sum(v):
  # dT/dv_i = 2*v_i, dS/dv_i = 1, so df/dv_i = (2*v_i * S - T * 1) / S^2
  S <- sum(v); T_val <- sum(v^2)
  expected <- (2 * v * S - T_val) / S^2
  expect_equal(as.numeric(gf(v)), as.numeric(expected), tolerance = 1e-10)
})

test_that("Quotient rule: crossprod(v,v) / sum(v^2) — degenerate identity", {
  skip_if_no_fast()
  # crossprod(v,v) and sum(v^2) are mathematically equal; ratio = 1 constant;
  # gradient should be exactly 0 (within float tolerance).
  gf <- grad(function(v) crossprod(v, v) / sum(v^2))
  v <- runif(20L)
  out <- as.numeric(gf(v))
  expect_equal(out, rep(0, length(v)), tolerance = 1e-10)
})

test_that("Constant numerator: 1 / sum(v^2)", {
  skip_if_no_fast()
  gf <- grad(function(v) 1 / sum(v^2))
  v <- runif(20L)
  # d/dv_i (1 / sum(v^2)) = -2*v_i / sum(v^2)^2
  expected <- -2 * v / sum(v^2)^2
  expect_equal(as.numeric(gf(v)), as.numeric(expected), tolerance = 1e-10)
})

test_that("Full softmax entropy: log(sum(exp(v))) - sum(v*exp(v))/sum(exp(v))", {
  skip_if_no_fast()
  gf <- grad(function(v) log(sum(exp(v))) - sum(v * exp(v)) / sum(exp(v)))
  v <- runif(20L)
  p <- exp(v) / sum(exp(v))
  expected <- -p * (v - sum(v * p))
  expect_equal(as.numeric(gf(v)), as.numeric(expected), tolerance = 1e-10)
})

test_that("Regression: sum(v^2) / 2 still uses constant-denominator fast path", {
  skip_if_no_fast()
  # Constant denominator → existing branch returns grad(numerator) / const,
  # NOT the quotient-rule construction. Body should have shape `<grad> / 2`,
  # not `(<da>*<b> - <a>*<db>) / <b>^2`.
  gf <- grad(function(v) sum(v^2) / 2)
  b <- body(gf)
  # Single-pass fusable shapes (`(2*v)/2`) now carry the auto-tune dispatch wrapper
  # (auto-tune dispatch #5): { .dat_s <- ...; .dat_auto_dispatch(key, v,
  # fused_thunk, function() <base>, metal) }. Peel to the base thunk's body `<grad>/2`.
  if (is.call(b) && identical(b[[1L]], as.name("{"))) {
    for (s in as.list(b)[-1L]) {
      if (is.call(s) && is.symbol(s[[1L]]) &&
          identical(as.character(s[[1L]]), ".dat_auto_dispatch")) {
        bt <- s[[5L]]                                   # 4th arg = base = function() <base>
        if (is.call(bt) && identical(bt[[1L]], as.name("function"))) b <- bt[[3L]]
        break
      }
      if (is.call(s) && identical(s[[1L]], as.name("if")) &&
          is.call(s[[2L]]) && identical(s[[2L]][[1L]], as.name("is.null"))) {
        b <- s[[3L]][[3L]]; break                       # legacy wrapper
      }
    }
  }
  # The base body is the AST `<grad(sum v^2)> / 2`; top-level should be `/` with
  # `2` literal as the second operand.
  expect_true(is.call(b))
  expect_identical(b[[1L]], quote(`/`))
  expect_identical(b[[3L]], 2)
  # Numeric correctness:
  v <- runif(20L)
  expect_equal(as.numeric(gf(v)), as.numeric(v), tolerance = 1e-10)
})

test_that("Catalog gap propagation: sum(v^2) / sum(gamma(v)) raises with gamma named", {
  skip_if_no_fast()
  # gamma is not in declared L_3 catalog. The .grad_expr sub-derivative call
  # for sum(gamma(v)) propagates DefDiff_unknown_generator through the L_0 `/`
  # rule's tryCatch, re-raising with top-level-division context that names gamma.
  expect_error(
    grad(function(v) sum(v^2) / sum(gamma(v))),
    regexp = "gamma"
  )
})

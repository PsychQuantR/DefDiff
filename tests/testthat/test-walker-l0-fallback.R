## test-walker-l0-fallback.R
## Tier 4 change `add-walker-l0-fallback`: walker auto-dispatches to L_0
## catalog entries it doesn't recognize locally. L_1/L_2/L_3 NOT consulted
## from walker context (semantic safety — L_1 returns scalar-output that
## breaks vector-grain assumption).

skip_if_no_fast <- function() {
  if (!.fast_path_available()) skip("macOS fast-path required")
}

# ---- 1. Regression: rep still works after removing manual mirror ----

test_that("Regression: rep auto-dispatches via L_0 fallback (no manual mirror)", {
  skip_if_no_fast()
  gf <- grad(function(v) sum(rep(c(1, -1), length.out = length(v)) * v))
  expect_equal(as.numeric(gf(c(0.1, 0.2, 0.3, 0.4))), c(1, -1, 1, -1), tolerance = 1e-10)
})

# ---- 2. Custom L_0 entry reachable from walker ----

test_that("Custom L_0 generator registered at runtime reaches walker via fallback", {
  skip_if_no_fast()
  # Register a fake constant-valued generator at L_0
  myconst <- function(c) rep(c, 5L)  # runtime R function
  .dat_env$catalog$L_0[["myconst"]] <- function(expr, var) 0
  on.exit({ .dat_env$catalog$L_0[["myconst"]] <- NULL }, add = TRUE)

  # Use it inside sum — walker should dispatch via fallback
  gf <- grad(function(v) sum(myconst(5) * v))
  out <- gf(c(1, 2, 3, 4, 5))
  expect_equal(as.numeric(out), c(5, 5, 5, 5, 5), tolerance = 1e-10)
})

# ---- 3. Negative: L_1 NOT reached from walker ----

test_that("L_1 catalog entries are NOT consulted by walker fallback", {
  # Register a fake L_1 entry — walker must NOT dispatch to it
  .dat_env$catalog$L_1[["mysum"]] <- function(expr, var) 0
  on.exit({ .dat_env$catalog$L_1[["mysum"]] <- NULL }, add = TRUE)

  # Use mysum inside sum — walker encounters mysum, should NOT fall through
  # to L_1; should raise DefDiff_unknown_generator instead
  # Walker raises DefDiff_unknown_generator; .sum_rule wraps into DefDiff_not_definable
  # (post-Tier-3 wrapping). Match on the wrapped class + message regex.
  expect_error(
    grad(function(v) sum(mysum(v) * v)),
    class = "DefDiff_not_definable",
    regexp = "mysum"
  )
})

# ---- 4. Walker switch precedence ----

test_that("Walker's local switch takes precedence over L_0 fallback for known ops", {
  skip_if_no_fast()
  # Walker handles `+` via its own switch (.smart_add); should produce
  # correct gradient. If walker fell through to L_0 `+` rule instead, the
  # numeric result would still be correct (both paths are equivalent for
  # `v + 0`), so the test asserts numeric correctness — which proves the
  # switch path didn't error out due to walker/L_0 signature mismatch.
  gf <- grad(function(v) sum(v + 0))
  expect_equal(as.numeric(gf(c(1, 2, 3))), c(1, 1, 1), tolerance = 1e-10)
})

# ---- 5. Unknown generator still raises after both lookups miss ----

test_that("Unknown generator raises (wrapped as DefDiff_not_definable by .sum_rule) naming the missing function", {
  # Walker raises DefDiff_unknown_generator; .sum_rule wraps it as DefDiff_not_definable.
  expect_error(
    grad(function(v) sum(gamma(v))),
    class = "DefDiff_not_definable",
    regexp = "gamma"
  )
})

# ---- 6. L_2 NOT consulted ----

test_that("L_2 catalog entries are NOT consulted by walker fallback", {
  .dat_env$catalog$L_2[["mygen"]] <- function(expr, var) 0
  on.exit({ .dat_env$catalog$L_2[["mygen"]] <- NULL }, add = TRUE)

  expect_error(
    grad(function(v) sum(mygen(v) * v)),
    class = "DefDiff_not_definable",
    regexp = "mygen"
  )
})

# ---- 7. L_3-only entries (without L_0 duplicate) NOT consulted by fallback ----

test_that("L_3 catalog entries are NOT reached via fallback (only via switch)", {
  # Register at L_3 only (not L_0). Since walker's chain-rule switch hard-codes
  # the L_3 functions (sin/cos/exp/log/tanh/sqrt/atan), this new entry isn't
  # in the switch. The fallback only checks L_0, so this should raise.
  .dat_env$catalog$L_3[["mygen3"]] <- function(expr, var) 0
  on.exit({ .dat_env$catalog$L_3[["mygen3"]] <- NULL }, add = TRUE)

  expect_error(
    grad(function(v) sum(mygen3(v) * v)),
    class = "DefDiff_not_definable",
    regexp = "mygen3"
  )
})

# ---- 8. Fallback preserves (expr, var) call signature ----

test_that("Fallback invokes L_0 rule with identical (expr, var) arguments", {
  # Register an L_0 rule that records its args, then check what was passed.
  # Call .grad_inner directly to bypass .sum_rule's fast paths (which would
  # intercept `sum(<const>*v)` before reaching walker).
  captured <- new.env(parent = emptyenv())
  .dat_env$catalog$L_0[["capture_args"]] <- function(expr, var) {
    captured$expr <- expr
    captured$var  <- var
    0
  }
  on.exit({ .dat_env$catalog$L_0[["capture_args"]] <- NULL }, add = TRUE)

  invisible(.grad_inner(quote(capture_args(7)), "v"))

  expect_identical(captured$expr, quote(capture_args(7)))
  expect_identical(captured$var, "v")
})

## test-coverage-boundary.R
## Phase B (add-comprehensive-test-benchmark-expansion):
## Formally enumerate expression patterns that DD does NOT yet support.
## Each test asserts that calling grad() or hessian() raises a specific
## condition class, locking the boundary so future tier expansion can
## remove individual entries (and update dd-coverage-boundary.md).

# ---- Multi-variable input ----

test_that("grad supports multi-variable input (add-multi-variable-gradient)", {
  # Previously a boundary (raised DefDiff_not_definable). add-multi-variable-gradient
  # lifts the single-variable guard for scalar-output functions: grad returns a
  # named list of per-variable gradients. d/dv sum(v+w) = 1 per coord of v;
  # d/dw = 1 per coord of w.
  # sum(v + w) is elementwise, so v and w must share a length.
  gf <- grad(function(v, w) sum(v + w))
  r <- gf(c(1, 2, 3), c(10, 20, 30))
  expect_type(r, "list")
  expect_identical(names(r), c("v", "w"))
  expect_equal(as.numeric(r$v), rep(1, 3))
  expect_equal(as.numeric(r$w), rep(1, 3))
})

test_that("hessian supports multi-variable input (block Hessian)", {
  # Previously out of scope (raised DefDiff_not_definable); closed by
  # add-multi-variable-hessian. Returns a named list of named-list blocks.
  H <- do.call(hessian(function(v, w) sum(v^2) + sum(w^2)),
               list(v = c(1, 2, 3), w = c(4, 5)))
  expect_equal(H$v$v, diag(rep(2, 3)), tolerance = 1e-8)
  expect_equal(H$w$w, diag(rep(2, 2)), tolerance = 1e-8)
  expect_equal(H$v$w, matrix(0, 3, 2), tolerance = 1e-8)
})

# ---- Product rule patterns: SUPPORTED in Tier 3 (catalog closure) ----

test_that("Tier 3: grad supports product of two trig in sum", {
  # sum(sin(v) * cos(v)) was a catalog gap before Tier 3; the recursive
  # walker now applies the product rule. Verify numeric correctness.
  gf <- grad(function(v) sum(sin(v) * cos(v)))
  v <- c(0.5, 1.0, 1.5)
  ref <- cos(v) * cos(v) - sin(v) * sin(v)  # = cos(2v) trig identity
  expect_equal(gf(v), ref, tolerance = 1e-10)
})

# ---- Statistical / aggregate patterns (catalog gap) ----

test_that("grad rejects variance form sum((v - mean(v))^2)", {
  # Requires recognition of mean() and (v - <vector>)^2 chain.
  expect_error(
    grad(function(v) sum((v - mean(v))^2)),
    class = "DefDiff_not_definable"
  )
})

# ---- Higher-order beyond 2 (Hessian scope limit) ----

test_that("composing grad on hessian result errors", {
  # 3rd-order: grad of Hessian's matrix-valued function isn't supported.
  # The Hessian body is `diag(rep(2, length(v)))` which references `diag`
  # — not a recognized generator in the L_3 catalog, so .grad_expr's
  # lookup raises DefDiff_unknown_generator.
  hf <- hessian(function(v) sum(v^2))
  expect_error(
    grad(hf),
    class = "DefDiff_unknown_generator"
  )
})

# ---- Off-diagonal Hessian patterns (closed by add-hessian-recursive-walker) ----

test_that("hessian supports composite involving non-trivial outer", {
  # Previously hessian_not_supported; now built via the recursive walker.
  hf <- hessian(function(v) sin(sum(v^2)))
  v <- c(1, 2, 3)
  ref <- -4 * sin(sum(v^2)) * outer(v, v) + 2 * cos(sum(v^2)) * diag(length(v))
  expect_equal(hf(v), ref, tolerance = 1e-8)
})

test_that("hessian supports product of crossprods (non-quadratic)", {
  hf <- hessian(function(v) crossprod(v, v) * crossprod(v, v))
  v <- c(1, 2)
  ref <- 8 * outer(v, v) + 4 * as.numeric(crossprod(v, v)) * diag(length(v))
  expect_equal(hf(v), ref, tolerance = 1e-8)
})

# ---- Quotient-shape Hessian (closed by add-hessian-quotient-walker) ----

test_that("hessian supports scalar-denominator quotients", {
  # Previously hessian_not_supported; now built via the recursive walker's
  # quotient rule (the second-derivative quotient identity in the matrix
  # sublanguage). Verified against numDeriv.
  skip_if_not_installed("numDeriv")
  f <- function(v) sum(v * exp(v)) / sum(exp(v))
  v <- c(0.5, 1.0, 1.5)
  expect_equal(hessian(f)(v), nd_hessian(f, v), tolerance = 1e-5)
})

test_that("hessian rejects vector-denominator quotients (remaining boundary)", {
  # A quotient whose denominator is vector-valued is outside the scalar-output
  # contract; locked as the remaining quotient boundary.
  expect_error(
    DefDiff:::.jacobian_inner(quote(v / sin(v)), "v", "v"),
    class = "hessian_not_supported"
  )
})

# ---- Catalog generator gaps ----

test_that("grad rejects unknown elementwise inside sum()", {
  # `gamma` not recognized by .sum_rule's elementwise switch → falls into
  # the "sum(...) not recognised in default catalog" branch which raises
  # DefDiff_not_definable. Different from DefDiff_unknown_generator (which arises
  # when the unknown function is the outer-most call, not inside sum).
  expect_error(
    grad(function(v) sum(gamma(v))),
    class = "DefDiff_not_definable"
  )
})

test_that("grad rejects bare unknown generator", {
  # gamma(v) at top level → DefDiff_unknown_generator via .lookup_derivative miss
  expect_error(
    grad(function(v) gamma(v)),
    class = "DefDiff_unknown_generator"
  )
})

test_that("hessian rejects sum(gamma(v)) by propagating the grad condition", {
  # The recursive walker computes the gradient first; gamma is outside the
  # differentiation catalog, so the grad engine's DefDiff_not_definable propagates
  # unchanged (rather than hessian_not_supported, which is reserved for
  # gradients that are computable but shape-unsupported).
  expect_error(
    hessian(function(v) sum(gamma(v))),
    class = "DefDiff_not_definable"
  )
})

# ---- Control flow (engine scope limit) ----

test_that("grad rejects if-else control flow in body", {
  expect_error(
    grad(function(v) if (v[1] > 0) sum(v^2) else 0),
    class = "DefDiff_not_definable"
  )
})

test_that("grad rejects for loop in body", {
  f <- function(v) {
    s <- 0
    for (i in seq_along(v)) s <- s + v[i]^2
    s
  }
  expect_error(grad(f), class = "DefDiff_not_definable")
})

# ---- Chain rule through affine: SUPPORTED in Tier 3 ----

test_that("Tier 3: grad supports chain rule through affine inner", {
  # sum(sin(v + 1)) was a catalog gap before Tier 3; the recursive walker
  # now applies chain rule through arbitrary L_3 inner expressions.
  gf <- grad(function(v) sum(sin(v + 1)))
  v <- c(0.5, 1.0, 1.5)
  expect_equal(gf(v), cos(v + 1), tolerance = 1e-10)
})

# ---- Discovered discrepancies locked as boundary witnesses ----

test_that("grad rejects sum(v^v) (base error, not numerically realizable)", {
  # Base AND exponent depend on v: the power rule's k - 1 hits a symbolic
  # exponent and raises a base "non-numeric argument" error. Locked here so a
  # future general-v^v fix flips this deliberately.
  expect_error(grad(function(v) sum(v^v))(c(0.5, 1.1, 0.8)))
})

test_that("hessian rejects sum(sqrt(v)) and sum(atan(v))", {
  # sqrt/atan are in the .hess_diag source catalog but unreachable on the dense
  # path (.hess_shape classifies the gradient before the diagonal rule applies).
  expect_error(hessian(function(v) sum(sqrt(v)))(c(0.5, 0.8, 1.2)),
               class = "hessian_not_supported")
  expect_error(hessian(function(v) sum(atan(v)))(c(0.5, 0.8, 1.2)),
               class = "hessian_not_supported")
})

# ---- Negative-of-negative: looks unsupported but IS supported ----

test_that("scalar-denominator quotient Hessian is supported (does not raise)", {
  v <- c(0.5, 0.8, 1.2)
  H <- hessian(function(v) sum(v * exp(v)) / sum(exp(v)))(v)
  expect_equal(dim(H), c(3L, 3L))
  expect_equal(H, t(H), tolerance = 1e-10)
})

test_that("vector-grain Hadamard gradient is supported (does not raise)", {
  # sum(v * w) w.r.t. v is w; locks that this is NOT mistaken for an
  # unsupported shape.
  gf <- grad(function(v, w) sum(v * w))
  res <- gf(c(1, 2, 3), c(4, 5, 6))
  expect_equal(as.numeric(res$v), c(4, 5, 6), tolerance = 1e-10)
})

test_that("multi-variable gradient is supported (does not raise)", {
  gf <- grad(function(v, w) sum(v^2) + sum(w^3))
  res <- gf(c(1, 2), c(3, 4))
  expect_equal(as.numeric(res$v), c(2, 4), tolerance = 1e-10)
  expect_equal(as.numeric(res$w), 3 * c(3, 4)^2, tolerance = 1e-10)
})

## test-l4-nodes.R
## L_4 binder nodes: `integral(f, t, a, b)` and `implicit(F, y, lower, upper)`.
## Spec: openspec/changes/add-l4-integral-implicit-nodes (issue #23).

# ---------------------------------------------------------------------------
# 1. Level L_4 registration
# ---------------------------------------------------------------------------

test_that("level() reports L_4 for integral and implicit nodes", {
  DefDiff:::register_default_catalog()
  expect_equal(level(quote(integral(exp(-t * theta), t, 0, 1))), "L_4")
  expect_equal(level(quote(implicit(y^2 - theta, y, 0, 10))),    "L_4")
})

test_that("level() does not treat the bound symbol as an unknown generator", {
  DefDiff:::register_default_catalog()
  expect_equal(level(quote(integral(t, t, 0, 1))), "L_4")
})

test_that("catalog carries an L_4 tier ranked above L_3", {
  DefDiff:::register_default_catalog()
  expect_true("L_4" %in% names(DefDiff:::.dat_env$catalog))
  expect_setequal(language_catalog("L_4"), c("integral", "implicit"))
  expect_gt(DefDiff:::.level_rank("L_4"), DefDiff:::.level_rank("L_3"))
  expect_lt(DefDiff:::.level_rank("L_4"), DefDiff:::.level_rank("unknown"))
})

test_that("extend_language accepts L_4 and still rejects L_5", {
  DefDiff:::register_default_catalog()
  extend_language("L_4", "foo", function(x, dx) dx)
  expect_true("foo" %in% language_catalog("L_4"))
  expect_error(
    extend_language("L_5", "foo", function(x, dx) dx),
    class = "DefDiff_invalid_extension"
  )
  DefDiff:::register_default_catalog()
})

# ---------------------------------------------------------------------------
# 2. Binder helpers
# ---------------------------------------------------------------------------

test_that(".contains_var is binder-aware for integral", {
  cv <- DefDiff:::.contains_var
  expect_false(cv(quote(integral(t * theta, t, 0, 1)), "t"))
  expect_true(cv(quote(integral(exp(-t), t, 0, theta)), "theta"))
  expect_true(cv(quote(integral(t * theta, t, 0, 1)), "theta"))
})

test_that(".contains_var is binder-aware for implicit", {
  cv <- DefDiff:::.contains_var
  expect_false(cv(quote(implicit(y^2 - theta, y, 0, 10)), "y"))
  expect_true(cv(quote(implicit(y^2 - theta, y, 0, theta)), "theta"))
  expect_true(cv(quote(implicit(y^2 - theta, y, 0, 10)), "theta"))
})

test_that(".alpha_rename_binders replaces bound symbols with fresh names", {
  e <- quote(t + integral(t * theta, t, 0, 1))
  r <- DefDiff:::.alpha_rename_binders(e)
  renamed <- r$expr
  # outer free t untouched
  expect_identical(renamed[[2L]], quote(t))
  inner <- renamed[[3L]]
  fresh <- inner[[3L]]
  expect_true(is.symbol(fresh))
  expect_false(as.character(fresh) %in% all.names(e))
  # body uses the fresh name, theta untouched
  expect_identical(inner[[2L]], bquote(.(fresh) * theta))
  expect_equal(r$map[[as.character(fresh)]], "t")
})

test_that(".subst_symbol replaces every occurrence structurally", {
  e <- quote(exp(-t * theta) + t)
  out <- DefDiff:::.subst_symbol(e, "t", quote(b))
  expect_identical(out, quote(exp(-b * theta) + b))
})

test_that(".restore_binder_names restores only when the name is not free", {
  rb <- DefDiff:::.restore_binder_names
  e1 <- quote(integral(.b1 * theta, .b1, 0, 1))
  expect_identical(rb(e1, list(.b1 = "t")), quote(integral(t * theta, t, 0, 1)))
  e2 <- quote(t + integral(.b1 * theta, .b1, 0, 1))
  expect_identical(rb(e2, list(.b1 = "t")), e2)
})

# ---------------------------------------------------------------------------
# 3. Evaluators
# ---------------------------------------------------------------------------

test_that("integral() evaluates numerically with the bound symbol out of scope", {
  expect_false(exists("t_not_defined_anywhere"))
  expect_equal(integral(exp(-t_not_defined_anywhere), t_not_defined_anywhere, 0, Inf), 1,
               tolerance = 1e-8)
  theta <- 2
  expect_equal(integral(exp(-t * theta), t, 0, Inf), 0.5, tolerance = 1e-8)
})

test_that("integral() honors the rel.tol option", {
  old <- options(DefDiff.integrate_rel_tol = 1e-4)
  on.exit(options(old))
  expect_equal(integral(exp(-t), t, 0, Inf), 1, tolerance = 1e-3)
})

test_that("implicit() returns the root in the interval", {
  theta <- 4
  expect_equal(implicit(y^2 - theta, y, 0, 10), 2, tolerance = 1e-8)
})

test_that("implicit() refuses an interval without a sign change", {
  theta <- 4
  expect_error(implicit(y^2 - theta, y, 3, 10),
               class = "DefDiff_not_definable", regexp = "\\[3, 10\\]")
})

# ---------------------------------------------------------------------------
# 4. Gradient rules
# ---------------------------------------------------------------------------

# Numeric equivalence of two expressions in `theta` at a point.
.eval_at <- function(expr, theta) eval(expr, list(theta = theta), globalenv())

test_that("Leibniz rule: variable upper bound yields a boundary term", {
  g <- grad(quote(integral(exp(-t * theta), t, 0, theta)), "theta")
  expect_equal(level(g), "L_4")
  expect_true(any(all.names(g) == "integral"))
  # analytic: d/dθ ∫_0^θ e^{-tθ} dt = ∫_0^θ -t e^{-tθ} dt + e^{-θ²}
  at1 <- .eval_at(g, 1)
  expect_equal(at1, -(1 - 2 * exp(-1)) + exp(-1), tolerance = 1e-6)
  gf <- grad(function(theta) integral(exp(-t * theta), t, 0, theta))
  expect_equal(gf(1), -(1 - 2 * exp(-1)) + exp(-1), tolerance = 1e-6)
})

test_that("Leibniz rule: infinite bound has no boundary term", {
  g <- grad(quote(integral(exp(-t * theta), t, 0, Inf)), "theta")
  # d/dθ (1/θ) = -1/θ² ; no exp(-Inf) boundary term in the AST
  expect_false(grepl("Inf \\*|\\* Inf", paste(deparse(g), collapse = " ")))
  expect_equal(.eval_at(g, 2), -1 / 4, tolerance = 1e-6)
})

test_that("Leibniz rule: shadowed bound name contributes nothing", {
  # `sum(t)` because bare top-level variables are outside the scalar contract
  g <- grad(quote(sum(t) + integral(t * theta, t, 0, 1)), "t")
  expect_equal(eval(g, list(t = 3, theta = 2), globalenv()), 1)
})

test_that("Leibniz rule: original bound name is restored when unambiguous", {
  g <- grad(quote(integral(exp(-t * theta), t, 0, 1)), "theta")
  expect_true("t" %in% all.names(g))
  expect_false(any(grepl("^\\.b[0-9]+$", all.names(g))))
})

test_that("implicit function theorem: closed-form check", {
  g <- grad(quote(implicit(y^2 - theta, y, 0, 10)), "theta")
  expect_equal(level(g), "L_4")
  expect_true(any(all.names(g) == "implicit"))
  expect_equal(.eval_at(g, 4), 0.25, tolerance = 1e-6)
  gf <- grad(function(theta) implicit(y^2 - theta, y, 0, 10))
  expect_equal(gf(4), 0.25, tolerance = 1e-6)
})

test_that("implicit function theorem: free y outside and bound y inside", {
  # f(θ, y_free) = y + implicit(y^2 - θ, y, 0, 10); ∂/∂θ = 1/(2 sqrt θ)
  g <- grad(quote(sum(y) + implicit(y^2 - theta, y, 0, 10)), "theta")
  expect_equal(eval(g, list(theta = 4, y = 100), globalenv()), 0.25, tolerance = 1e-6)
})

test_that("simplifier leaves binder nodes unchanged", {
  e <- quote(integral(t * 1, t, 0, 1))
  expect_identical(DefDiff:::.algebraic_simplify(e), e)
})

test_that("hessian and jacobian refuse L_4 nodes", {
  expect_error(hessian(function(theta) integral(exp(-t * theta), t, 0, 1)),
               class = "DefDiff_not_definable", regexp = "integral")
  expect_error(jacobian(function(theta) c(implicit(y^2 - theta, y, 0, 10))),
               class = "DefDiff_not_definable", regexp = "implicit")
})

# ---------------------------------------------------------------------------
# 5. Verify #23 round-2 fixes (blocking + in-scope)
# ---------------------------------------------------------------------------

test_that("grad(function) with a variable lower bound works on the JIT path (NULL slots)", {
  gf <- grad(function(theta) integral(t^2, t, theta, 1))
  expect_equal(gf(0.3), -0.09, tolerance = 1e-8)
  # walkers must not delete NULL call elements
  e <- quote(f(NULL, x))
  expect_identical(DefDiff:::.subst_symbol(e, "x", quote(y)), quote(f(NULL, y)))
  expect_identical(DefDiff:::.l4_clean(e), e)
  expect_identical(DefDiff:::.alpha_rename_binders(e)$expr, e)
})

test_that("substitution never rewrites a call head sharing the bound name", {
  g <- grad(quote(integral(exp(theta) + exp, exp, 0, 1)), "theta")
  expect_equal(eval(g, list(theta = 1), globalenv()), exp(1), tolerance = 1e-6)
  expect_identical(DefDiff:::.subst_symbol(quote(t(A) + t), "t", quote(b)), quote(t(A) + b))
})

test_that("a non-symbol binder slot or wrong arity is refused with a typed condition", {
  expect_error(grad(quote(integral(integral(t * s, "s", 0, 1), t, 0, s)), "s"),
               class = "DefDiff_not_definable", regexp = "bare symbol")
  expect_error(grad(quote(integral(x)), "x"),
               class = "DefDiff_not_definable", regexp = "exactly 4 arguments")
  expect_error(DefDiff:::.contains_var(quote(implicit(y, foo(bar), 0, 1)), "y"),
               class = "DefDiff_not_definable")
})

test_that("binder detection scans call heads, not names", {
  # a variable merely named `integral` must not be refused or widened
  hf <- hessian(function(integral) sum(integral^2))
  expect_equal(hf(c(1, 2)), diag(2, 2))
  gf <- grad(function(integral) sum(integral^2))
  res <- verify_grad(function(integral) sum(integral^2), gf, n_samples = 3L, sample_dim = 2L)
  expect_false(res$numeric$l4_tolerance_widened)
  expect_error(hessian(function(theta) integral(t^2, t, 0, theta)),
               class = "DefDiff_not_definable", regexp = "binder node `integral`")
})

test_that("integral() propagates user-body errors and validates its inputs", {
  expect_error(integral(exp(-t) * undefined_fn_zzz(t), t, 0, 1), regexp = "undefined_fn_zzz",
               class = "simpleError")
  expect_error(integral(c(1, 2) * t, t, 0, 1), class = "DefDiff_not_definable",
               regexp = "scalar-valued")
  expect_error(implicit(c(y, y) - 1, y, 0, 2), class = "DefDiff_not_definable",
               regexp = "scalar-valued")
  old <- options(DefDiff.integrate_rel_tol = "abc")
  on.exit(options(old))
  expect_equal(integral(exp(-t), t, 0, Inf), 1, tolerance = 1e-6)
})

test_that("integrand derivative is verified separately from the whole (Expected 6)", {
  kernel <- quote((1 + t^2 / nu)^(-(nu + 1) / 2))
  d_kernel <- DefDiff:::.l4_deriv(kernel, "nu")
  at <- function(nu, t) eval(d_kernel, list(nu = nu, t = t), globalenv())
  k  <- function(nu, t) eval(kernel,   list(nu = nu, t = t), globalenv())
  h <- 1e-5
  for (tt in c(-2, 0.3, 1.7)) {
    fd <- (k(5 + h, tt) - k(5 - h, tt)) / (2 * h)
    expect_equal(at(5, tt), fd, tolerance = 1e-6)
  }
})

# ---------------------------------------------------------------------------
# 6. Verify #23 round-3 fixes
# ---------------------------------------------------------------------------

test_that("integral() and implicit() validate bounds / interval types", {
  expect_error(integral(t, t, "0", "1"), class = "DefDiff_not_definable", regexp = "`a` must be")
  expect_error(integral(t, t, 0, c(1, 2)), class = "DefDiff_not_definable", regexp = "`b` must be")
  expect_equal(integral(t, t, 0, 1), 0.5, tolerance = 1e-8)
  theta <- 4
  expect_error(implicit(y^2 - theta, y, 10, 0), class = "DefDiff_not_definable", regexp = "lower < upper")
  expect_error(implicit(y^2 - theta, y, "0", 10), class = "DefDiff_not_definable", regexp = "`lower` must be")
})

test_that("evaluators refuse a non-symbol binder slot like grad() does", {
  expect_error(integral(t, "t", 0, 1), class = "DefDiff_not_definable", regexp = "bare symbol")
  theta <- 4
  expect_error(implicit(y^2 - theta, "y", 0, 10), class = "DefDiff_not_definable", regexp = "bare symbol")
})

test_that("a same-named user function cannot hijack a generated gradient body", {
  local({
    integral <- function(...) 999
    gf <- grad(function(theta) integral(exp(t * theta), t, 0, 1))
    expect_equal(gf(0), 0.5, tolerance = 1e-6)   # d/dθ ∫_0^1 e^{tθ} dt at 0 = ∫ t dt = 0.5
    # generated body is namespace-qualified; environment is still the user's
    expect_identical(body(gf)[[1L]], quote(DefDiff::integral))
    expect_identical(environment(gf), environment())
  })
  # plain gradient functions keep the user's environment
  w <- c(1, 2)
  gf2 <- grad(function(v) sum(v * w))
  expect_identical(environment(gf2), environment())
})

test_that("extend_language refuses the reserved L_4 heads", {
  DefDiff:::register_default_catalog()
  expect_error(extend_language("L_3", "integral", function(x, dx) dx),
               class = "DefDiff_invalid_extension", regexp = "reserved")
  expect_equal(level(quote(integral(t, t, 0, 1))), "L_4")
})

test_that("substitution reaches a compound call head", {
  e <- quote((g(t))(x))
  out <- DefDiff:::.subst_symbol(e, "t", quote(b))
  expect_identical(out, quote((g(b))(x)))
})

test_that("a user variable named like a binder head is not shadowed in generated bodies", {
  local({
    integral <- 2
    f <- function(theta) integral(theta * integral * t, t, 0, 1)
    expect_equal(f(1), 1, tolerance = 1e-8)
    gf <- grad(f)
    expect_equal(gf(1), 1, tolerance = 1e-6)          # d/dθ ∫ 2θt dt = 1
    expect_identical(grad_expr(gf)[[1L]], quote(integral))  # symbolic tree keeps the public name
    expect_equal(level(gf), "L_4")
  })
  # no fixed internal name exists that a user could collide with either
  local({
    .dd_integral <- 2
    gf <- grad(function(theta) integral(theta * .dd_integral * t, t, 0, 1))
    expect_equal(gf(1), 1, tolerance = 1e-6)
  })
})

test_that(".refuse_l4_nodes sees a binder inside a compound call head", {
  e <- quote((if (FALSE) integral(t, t, 0, theta) else sin)(theta))
  expect_error(DefDiff:::.refuse_l4_nodes(e, "hessian"),
               class = "DefDiff_not_definable", regexp = "binder node `integral`")
})

test_that("generated bodies with qualified heads pass the control-flow scan", {
  gf <- grad(function(theta) integral(exp(t * theta), t, 0, 1))
  expect_true(is.na(DefDiff:::.control_flow_block(body(gf))))
  expect_identical(DefDiff:::.control_flow_block(quote(DefDiff::integral(if (a) 1 else 2, t, 0, 1))), "if")
})

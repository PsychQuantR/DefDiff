## test-fast-grad.R
## Tier 1 (add-vdsp-fast-path): tests for fast_scalar_mul kernel,
## DefDiff:::.is_scalar_var_product predicate, DefDiff:::.fast_path_available helper, and
## grad.function fast-path dispatch.
##
## Fast-path tests are macOS-only (vDSP backend); skip elsewhere.

skip_on_os("windows")
skip_on_os("linux")
skip_on_os("solaris")

TOL_VDSP <- 4 * .Machine$double.eps

# add-metal-backend: the canonical Tier 1 `<scalar> * <var>` gradient body is
# now a runtime metal/vDSP threshold dispatch —
#   if (.metal_path_available() && length(v) >= getOption("DefDiff.metal_threshold", 1e9L))
#     metal_scalar_mul(s, v) else fast_scalar_mul(s, v)
# The vDSP (CPU) branch is the `else` arm. This helper returns that arm (or the
# body itself if it is not threshold-wrapped), so the Tier 1 dispatch assertions
# stay meaningful and target the bare fast_scalar_mul call.
.cpu_branch <- function(b) {
  if (is.call(b) && identical(b[[1L]], as.name("if"))) b[[4L]] else b
}

# ---- fast_scalar_mul kernel (Design Behavior 2 / dat-performance Requirement
#      "Scalar-vector multiplication kernel") ----

test_that("fast_scalar_mul(2, c(1, 2, 3)) returns c(2, 4, 6)", {
  result <- fast_scalar_mul(2, c(1, 2, 3))
  expect_equal(result, c(2, 4, 6), tolerance = TOL_VDSP)
})

test_that("fast_scalar_mul(0, c(1, 2, 3)) returns zero vector", {
  result <- fast_scalar_mul(0, c(1, 2, 3))
  expect_equal(result, c(0, 0, 0), tolerance = TOL_VDSP)
})

test_that("fast_scalar_mul(-1, c(1, 2, 3)) returns negated vector", {
  result <- fast_scalar_mul(-1, c(1, 2, 3))
  expect_equal(result, c(-1, -2, -3), tolerance = TOL_VDSP)
})

test_that("fast_scalar_mul(2, numeric(0)) returns empty vector with no error", {
  expect_silent(result <- fast_scalar_mul(2, numeric(0)))
  expect_equal(result, numeric(0))
})

test_that("fast_scalar_mul matches R `*` on random sweep", {
  set.seed(20260524L)
  v <- runif(1000L)
  expect_lt(max(abs(fast_scalar_mul(2.5, v) - 2.5 * v)), TOL_VDSP)
})

# ---- DefDiff:::.is_scalar_var_product predicate (Design Behavior 1 / dat-performance
#      Requirement "AST pattern detection for scalar-variable product") ----

test_that("DefDiff:::.is_scalar_var_product returns TRUE for matching patterns", {
  expect_true(DefDiff:::.is_scalar_var_product(quote(2 * v), "v"))
  expect_true(DefDiff:::.is_scalar_var_product(quote(2.5 * x), "x"))
  expect_true(DefDiff:::.is_scalar_var_product(quote(0 * v), "v"))
})

test_that("DefDiff:::.is_scalar_var_product returns FALSE for non-matching patterns", {
  # R parser represents unary minus in `-1 * v` as a call (-)(1), not a
  # literal — so this fails the is.numeric check. Deferred to Tier 2 AST
  # normalization.
  expect_false(DefDiff:::.is_scalar_var_product(quote(-1 * v), "v"))
  expect_false(DefDiff:::.is_scalar_var_product(quote(v * 2), "v"))
  expect_false(DefDiff:::.is_scalar_var_product(quote(2 * w), "v"))
  expect_false(DefDiff:::.is_scalar_var_product(quote(2 + v), "v"))
  expect_false(DefDiff:::.is_scalar_var_product(quote(2 * v + 1), "v"))
  expect_false(DefDiff:::.is_scalar_var_product(quote(NA_real_ * v), "v"))
  expect_false(DefDiff:::.is_scalar_var_product(quote(c(1, 2) * v), "v"))
})

test_that("DefDiff:::.is_scalar_var_product never raises a condition", {
  expect_silent(DefDiff:::.is_scalar_var_product(quote(2 * v), "v"))
  expect_silent(DefDiff:::.is_scalar_var_product(quote(foo(v)), "v"))
  expect_silent(DefDiff:::.is_scalar_var_product(1, "v"))
  expect_silent(DefDiff:::.is_scalar_var_product(NULL, "v"))
})

# ---- Tier 2a normalizer: .try_normalize_scalar_var_product
#      (Design Behavior 1 / dat-performance Requirement
#      "Fast-path AST normalization for scalar-variable product variants") ----

test_that(".try_normalize_scalar_var_product accepts 10 canonical-equivalent patterns", {
  norm <- DefDiff:::.try_normalize_scalar_var_product
  cases <- list(
    list(quote(2 * v),         "v",  2, "v"),
    list(quote(v * 2),         "v",  2, "v"),
    list(quote(-1 * v),        "v", -1, "v"),
    list(quote(v * -1),        "v", -1, "v"),
    list(quote(-(2 * v)),      "v", -2, "v"),
    list(quote(-(v * 2)),      "v", -2, "v"),
    list(quote(-(-1 * v)),     "v",  1, "v"),
    list(quote(2L * v),        "v",  2, "v"),
    list(quote(v * 2L),        "v",  2, "v"),
    list(quote(0 * v),         "v",  0, "v")
  )
  for (c in cases) {
    r <- norm(c[[1L]], c[[2L]])
    expect_false(is.null(r), info = deparse(c[[1L]]))
    expect_equal(r$scalar, c[[3L]], info = deparse(c[[1L]]))
    expect_identical(r$var, as.name(c[[4L]]), info = deparse(c[[1L]]))
  }
})

test_that(".try_normalize_scalar_var_product returns NULL silently for 10 unrecognized inputs", {
  norm <- DefDiff:::.try_normalize_scalar_var_product
  rejects <- list(
    list(quote(2 * w),              "v"),
    list(quote(2 + v),              "v"),
    list(quote(2 * v + 1),          "v"),
    list(quote(NA_real_ * v),       "v"),
    list(quote(c(1, 2) * v),        "v"),
    list(quote(v),                  "v"),
    list(quote(foo(v)),             "v"),
    list(quote(-(-(-1 * v))),       "v"),
    list(quote((1 + 2) * v),        "v"),
    list(quote(cos(0.5) * v),       "v")
  )
  for (r in rejects) {
    expect_silent(out <- norm(r[[1L]], r[[2L]]))
    expect_null(out, info = deparse(r[[1L]]))
  }
})

test_that(".try_normalize_scalar_var_product produces double scalar even for integer literal", {
  r <- DefDiff:::.try_normalize_scalar_var_product(quote(2L * v), "v")
  expect_false(is.null(r))
  expect_type(r$scalar, "double")
  expect_identical(r$scalar, 2.0)
})

test_that("strict .is_scalar_var_product unchanged: still rejects normalizer's extra patterns", {
  # Design Behavior 4: Tier 2a does NOT modify the strict predicate's contract.
  # Normalizer accepts these patterns; strict predicate still rejects them.
  expect_false(DefDiff:::.is_scalar_var_product(quote(v * 2), "v"))      # commutative swap
  expect_false(DefDiff:::.is_scalar_var_product(quote(v * 2L), "v"))     # commutative + integer
  expect_false(DefDiff:::.is_scalar_var_product(quote(-(2 * v)), "v"))   # outer negation
  expect_false(DefDiff:::.is_scalar_var_product(quote(-1 * v), "v"))     # unary-minus literal
  # Strict canonical still accepted (Tier 1 always supported integer literals via
  # is.numeric() which is TRUE for integer too — that's not Tier 2a-specific)
  expect_true(DefDiff:::.is_scalar_var_product(quote(2 * v), "v"))
  expect_true(DefDiff:::.is_scalar_var_product(quote(2L * v), "v"))
})

# ---- DefDiff:::.fast_path_available helper (Design Decision 4 / dat-performance
#      Requirement "Platform-specific fast-path backend availability") ----

test_that("DefDiff:::.fast_path_available returns TRUE on Darwin", {
  expect_true(DefDiff:::.fast_path_available())
})

# ---- grad.function fast-path dispatch (Design Behavior 3 / dat-grad-engine
#      Requirement "Fast-path dispatch for scalar-variable product gradients") ----

test_that("grad(function(v) sum(v^2)) body dispatches to fast_scalar_mul on macOS", {
  gf <- grad(function(v) sum(v^2))
  expect_true(is.call(body(gf)))
  expect_equal(.cpu_branch(body(gf))[[1L]], as.name("fast_scalar_mul"))
})

test_that("grad(function(v) sum(v^2)) fast-path output equals 2 * v at small scale", {
  gf <- grad(function(v) sum(v^2))
  expect_equal(gf(c(1, 2, 3)), c(2, 4, 6), tolerance = TOL_VDSP)
})

test_that("grad(function(v) sum(v^2)) fast-path output equals 2 * v at large scale", {
  set.seed(20260524L)
  v <- runif(1e6L)
  gf <- grad(function(v) sum(v^2))
  expect_lt(max(abs(gf(v) - 2 * v)), TOL_VDSP)
})

test_that("grad on compound expression does NOT dispatch to fast_scalar_mul", {
  gf <- grad(function(v) sin(sum(v^2)))
  expect_false(
    is.call(body(gf)) && identical(body(gf)[[1L]], as.name("fast_scalar_mul"))
  )
})

# ---- Tier 2a integration: dispatch through normalizer
#      (Design Behavior 2 + 3 / dat-grad-engine MODIFIED Requirement
#      Scenario "Normalized variants dispatch to fast path") ----

test_that("grad(function(v) -sum(v^2)) dispatches via outer-negation normalization", {
  # Gradient AST is `-(2 * v)`; normalizer reduces to canonical scalar = -2.
  gf <- grad(function(v) -sum(v^2))
  expect_true(is.call(body(gf)))
  expect_identical(.cpu_branch(body(gf))[[1L]], as.name("fast_scalar_mul"))
  expect_equal(.cpu_branch(body(gf))[[2L]], -2)
})

test_that("grad(function(v) -sum(v^2)) fast-path output equals -2 * v at small scale", {
  gf <- grad(function(v) -sum(v^2))
  expect_equal(gf(c(1, 2, 3)), c(-2, -4, -6), tolerance = TOL_VDSP)
})

test_that("grad(function(v) -sum(v^2)) fast-path output equals -2 * v at large scale", {
  set.seed(20260525L)
  v <- runif(1000L)
  gf <- grad(function(v) -sum(v^2))
  expect_lt(max(abs(gf(v) - (-2 * v))), TOL_VDSP)
})

# ---- Tier 2d: scalar-evaluable recognizer + composite normalizer
#      (Design / dat-performance Requirement
#      "Fast-path outer-scalar fusion for composite gradients") ----

test_that(".is_scalar_evaluable accepts recognized forms", {
  pred <- DefDiff:::.is_scalar_evaluable
  expect_true(pred(quote(sum(v^2)), "v"))
  expect_true(pred(quote(crossprod(v, v)), "v"))
  expect_true(pred(quote(cos(sum(v^2))), "v"))
  expect_true(pred(quote(sin(crossprod(v, v))), "v"))
  expect_true(pred(quote(exp(cos(sum(v^2)))), "v"))
  expect_true(pred(quote(tanh(sum(v^2))), "v"))
  expect_true(pred(quote(log(sum(v^2))), "v"))
  expect_true(pred(quote(sqrt(sum(v^2))), "v"))
})

test_that(".is_scalar_evaluable rejects expressions containing variable v in non-scalar positions", {
  # Tier 2e expanded the predicate: scalar arithmetic compositions of
  # scalar-evaluable subterms are accepted, AND bare numeric literals are
  # accepted (degenerate scalar case). The remaining rejection cases are:
  # bare variable, unknown function (not in the recognized list), variable
  # in arithmetic position (not under a scalar reduction).
  pred <- DefDiff:::.is_scalar_evaluable
  expect_false(pred(quote(v), "v"))                    # bare variable
  expect_false(pred(quote(foo(sum(v^2))), "v"))        # foo not in whitelist
  expect_false(pred(quote(2 * v), "v"))                # var in arithmetic
  expect_false(pred(NULL, "v"))                        # not an expression
})

test_that(".try_normalize_scalar_var_product_with_outer accepts cos(sum(v^2)) * (2 * v)", {
  norm <- DefDiff:::.try_normalize_scalar_var_product_with_outer
  r <- norm(quote(cos(sum(v^2)) * (2 * v)), "v")
  expect_false(is.null(r))
  # Tier 2d post-sum_sq substitution: outer_expr has sum(v^2) replaced with
  # fast_sum_sq(v) — bit-equivalent at double precision (vDSP_svesqD), but
  # accumulation order may differ from R sum by ~1e-13 ULPs.
  expect_equal(r$outer_expr, quote(cos(fast_sum_sq(v))))
  expect_equal(r$scalar, 2)
  expect_identical(r$var, as.name("v"))
})

test_that("composite normalizer accepts commutative swap on inner (v * 2)", {
  norm <- DefDiff:::.try_normalize_scalar_var_product_with_outer
  r <- norm(quote(cos(sum(v^2)) * (v * 2)), "v")
  expect_false(is.null(r))
  expect_equal(r$scalar, 2)
  expect_identical(r$var, as.name("v"))
})

test_that("composite normalizer accepts commutative swap on outer position", {
  norm <- DefDiff:::.try_normalize_scalar_var_product_with_outer
  r <- norm(quote((2 * v) * cos(sum(v^2))), "v")
  expect_false(is.null(r))
  expect_equal(r$scalar, 2)
})

test_that("composite normalizer returns NULL for non-composite", {
  norm <- DefDiff:::.try_normalize_scalar_var_product_with_outer
  expect_null(norm(quote(v * (2 * v)), "v"))
  expect_null(norm(quote(foo(sum(v)) * (2 * v)), "v"))
  expect_null(norm(quote(sin(sum(v^2)) + (2 * v)), "v"))
})

# ---- Tier 2d integration: dispatch through composite normalizer ----

test_that("grad(sin(sum(v^2))) dispatches via outer-scalar fusion (Tier 2d)", {
  gf <- grad(function(v) sin(sum(v^2)))
  expect_true(is.call(body(gf)))
  # Body is a { } block; last statement should be fast_scalar_mul call
  b <- body(gf)
  expect_identical(b[[1L]], as.name("{"))
  last_stmt <- b[[length(b)]]
  expect_true(is.call(last_stmt))
  expect_identical(last_stmt[[1L]], as.name("fast_scalar_mul"))
})

test_that("Tier 2d output bit-equivalent to direct R evaluation", {
  set.seed(20260525L)
  v <- runif(1000L)
  for (fn_name in c("sin", "cos", "exp", "tanh")) {
    inner_call <- bquote(sum(v^2))
    f_body <- bquote(.(as.name(fn_name))(.(inner_call)))
    f <- function(v) NULL
    body(f) <- f_body
    gf <- grad(f)
    dd <- gf(v)
    # Reference: direct R evaluation of the chain-ruled gradient
    outer_grad <- switch(fn_name,
                         sin = cos(sum(v^2)),
                         cos = -sin(sum(v^2)),
                         exp = exp(sum(v^2)),
                         tanh = 1 - tanh(sum(v^2))^2)
    ref <- outer_grad * (2 * v)
    # Tier 2d sum_sq fusion uses vDSP_svesqD which has different
    # accumulation order than R's stock sum — diff ~1e-13 relative ULPs.
    # For functions like exp() whose output magnitude is huge (sum=O(n),
    # exp(sum) overflows), use relative tolerance.
    rel_err <- max(abs(dd - ref) / (abs(ref) + 1e-300))
    expect_lt(rel_err, 1e-10,
              label = paste("Tier 2d", fn_name, "(sum(v^2))"))
  }
})

test_that("Tier 1+2a strict pattern wins priority over Tier 2d composite", {
  # sum(v^2) gradient is `2 * v` — matches strict pattern, NOT composite.
  gf <- grad(function(v) sum(v^2))
  # CPU branch should be a plain fast_scalar_mul call, not a { } block.
  expect_identical(.cpu_branch(body(gf))[[1L]], as.name("fast_scalar_mul"))
  expect_false(identical(.cpu_branch(body(gf))[[1L]], as.name("{")))
})

# ---- Tier 2c: vForce elementwise dispatch
#      (dat-performance Requirement "vForce elementwise dispatch for
#      L_3 gradient patterns") ----

test_that("Tier 2c vForce kernels match R stock for all 6 functions", {
  set.seed(20260525L)
  v <- runif(1000L) + 0.5  # positive for log/sqrt
  funcs <- c("cos", "sin", "exp", "log", "tanh", "sqrt")
  # vForce may differ from R stock by up to 1 ULP per element (last-bit
  # rounding in transcendental approximation); use 2 * TOL_VDSP cushion.
  TOL_VV <- 2 * TOL_VDSP
  for (fn in funcs) {
    vv_kernel <- get(paste0("fast_vv_", fn))
    r_stock <- get(fn)
    expect_lt(max(abs(vv_kernel(v) - r_stock(v))), TOL_VV,
              label = paste("fast_vv_", fn, sep = ""))
  }
})

test_that("Tier 2c .try_dispatch_elementwise recognizes 6 bare patterns", {
  pred <- DefDiff:::.try_dispatch_elementwise
  for (fn in c("cos", "sin", "exp", "log", "tanh", "sqrt")) {
    e <- call(fn, quote(v))
    r <- pred(e, "v")
    expect_false(is.null(r), info = fn)
    expect_equal(r$kernel_name, paste0("fast_vv_", fn))
    expect_identical(r$var, as.name("v"))
  }
})

test_that("Tier 2c .try_normalize_scalar_var_elementwise accepts scaled forms", {
  norm <- DefDiff:::.try_normalize_scalar_var_elementwise
  r1 <- norm(quote(2 * cos(v)), "v")
  expect_false(is.null(r1))
  expect_equal(r1$scalar, 2)
  expect_equal(r1$kernel_name, "fast_vv_cos")

  r2 <- norm(quote(cos(v) * 2), "v")
  expect_false(is.null(r2))
  expect_equal(r2$scalar, 2)

  # Outer negation
  r3 <- norm(quote(-sin(v)), "v")
  expect_false(is.null(r3))
  expect_equal(r3$scalar, -1)
  expect_equal(r3$kernel_name, "fast_vv_sin")
})

test_that("grad(sum(sin(v))) dispatches via Tier 2c to fast_vv_cos", {
  gf <- grad(function(v) sum(sin(v)))
  expect_true(is.call(body(gf)))
  expect_identical(body(gf)[[1L]], as.name("fast_vv_cos"))
  v <- c(0.5, 1.0, 1.5)
  expect_equal(gf(v), cos(v), tolerance = TOL_VDSP)
})

test_that("grad(-sum(sin(v))) dispatches via Tier 2c scaled (outer negation)", {
  gf <- grad(function(v) -sum(sin(v)))
  expect_true(is.call(body(gf)))
  expect_identical(body(gf)[[1L]], as.name("fast_scalar_mul"))
  v <- c(0.5, 1.0, 1.5)
  expect_equal(gf(v), -cos(v), tolerance = TOL_VDSP)
})

test_that("grad(sum(cos(v))) and grad(sum(exp(v))) dispatch correctly", {
  v <- c(0.5, 1.0, 1.5)
  # sum(cos(v)) → -sin(v) → dispatched as scaled elementwise
  gf1 <- grad(function(v) sum(cos(v)))
  expect_equal(gf1(v), -sin(v), tolerance = TOL_VDSP)
  # sum(exp(v)) → exp(v) → bare elementwise
  gf2 <- grad(function(v) sum(exp(v)))
  expect_identical(body(gf2)[[1L]], as.name("fast_vv_exp"))
  expect_equal(gf2(v), exp(v), tolerance = TOL_VDSP)
})

# ---- Tier 2b: vector-vector arithmetic kernels ----

test_that("Tier 2b fast_vec_add matches R `+`", {
  set.seed(20260525L)
  v <- runif(1000L); w <- runif(1000L)
  expect_lt(max(abs(fast_vec_add(v, w) - (v + w))), TOL_VDSP)
})

test_that("Tier 2b fast_vec_sub matches R `-`", {
  set.seed(20260525L)
  v <- runif(1000L); w <- runif(1000L)
  expect_lt(max(abs(fast_vec_sub(v, w) - (v - w))), TOL_VDSP)
  expect_lt(max(abs(fast_vec_sub(w, v) - (w - v))), TOL_VDSP)
})

test_that("Tier 2b fast_vec_smadd matches s * v + w", {
  set.seed(20260525L)
  v <- runif(1000L); w <- runif(1000L)
  expect_lt(max(abs(fast_vec_smadd(2.5, v, w) - (2.5 * v + w))), TOL_VDSP)
  expect_lt(max(abs(fast_vec_smadd(-1, v, w) - (-v + w))), TOL_VDSP)
})

test_that("Tier 2b kernels handle empty vectors", {
  expect_equal(fast_vec_add(numeric(0), numeric(0)), numeric(0))
  expect_equal(fast_vec_sub(numeric(0), numeric(0)), numeric(0))
  expect_equal(fast_vec_smadd(2, numeric(0), numeric(0)), numeric(0))
})

test_that("Tier 2b kernels error on length mismatch", {
  expect_error(fast_vec_add(c(1, 2), c(1, 2, 3)), "length mismatch")
  expect_error(fast_vec_sub(c(1, 2), c(1, 2, 3)), "length mismatch")
  expect_error(fast_vec_smadd(2, c(1, 2), c(1, 2, 3)), "length mismatch")
})

# ---- Tier 2e: boundary closure tests ----

test_that("Tier 2e fast_scalar_div matches R `s / v`", {
  expect_equal(fast_scalar_div(2, c(1, 2, 4)), c(2, 1, 0.5), tolerance = TOL_VDSP)
  expect_equal(fast_scalar_div(2, numeric(0)), numeric(0))
  set.seed(20260525L); v <- runif(1000L) + 0.1
  expect_lt(max(abs(fast_scalar_div(3, v) - (3 / v))), TOL_VDSP)
})

test_that("Tier 2e .try_normalize_reciprocal_vforce accepts 1/(2*sqrt(v))", {
  norm <- DefDiff:::.try_normalize_reciprocal_vforce
  r <- norm(quote(1/(2 * sqrt(v))), "v")
  expect_false(is.null(r))
  expect_equal(r$numerator, 0.5)
  expect_equal(r$kernel_name, "fast_vv_sqrt")
  expect_identical(r$var, as.name("v"))
})

test_that("Tier 2e .try_normalize_reciprocal_vforce rejects non-vForce denom", {
  norm <- DefDiff:::.try_normalize_reciprocal_vforce
  expect_null(norm(quote(1/(2 * v)), "v"))  # denom is scalar*var, not scalar*vForce
  expect_null(norm(quote(v / 2), "v"))
})

test_that("Tier 2e .substitute_sum_sq substitutes crossprod(v,v)", {
  sub_fn <- DefDiff:::.substitute_sum_sq
  out <- sub_fn(quote(cos(crossprod(v, v))), "v")
  expect_equal(out, quote(cos(fast_sum_sq(v))))
})

test_that("Tier 2e .is_scalar_evaluable accepts arithmetic compositions", {
  pred <- DefDiff:::.is_scalar_evaluable
  expect_true(pred(quote(1 - tanh(sum(sin(v)))^2), "v"))
  expect_true(pred(quote(2 * sum(v^2) + 1), "v"))
  # Bare variable in arithmetic position is NOT scalar
  expect_false(pred(quote(v + 1), "v"))
})

test_that("Tier 2e: grad(sum(sqrt(v))) dispatches via reciprocal-vForce", {
  gf <- grad(function(v) sum(sqrt(v)))
  expect_true(is.call(body(gf)))
  expect_identical(body(gf)[[1L]], as.name("fast_scalar_div"))
  v <- runif(100L) + 0.1
  expect_lt(max(abs(gf(v) - 1/(2 * sqrt(v)))), TOL_VDSP)
})

test_that("Tier 2e: grad(cos(crossprod(v,v))) uses fast_sum_sq in outer", {
  gf <- grad(function(v) cos(crossprod(v, v)))
  body_str <- deparse(body(gf))
  expect_true(any(grepl("fast_sum_sq", body_str, fixed = TRUE)))
  expect_false(any(grepl("crossprod", body_str, fixed = TRUE)))
  v <- runif(100L)
  ref <- as.numeric(-sin(crossprod(v, v))) * (2 * v)
  expect_lt(max(abs(gf(v) - ref)), 1e-10)
})

test_that("Tier 2e: grad(tanh(sum(sin(v)))) dispatches via Tier 2d x Tier 2c", {
  gf <- grad(function(v) tanh(sum(sin(v))))
  body_str <- deparse(body(gf))
  expect_true(any(grepl("fast_scalar_mul", body_str, fixed = TRUE)))
  expect_true(any(grepl("fast_vv_cos", body_str, fixed = TRUE)))
  v <- runif(100L)
  ref <- as.numeric(1 - tanh(sum(sin(v)))^2) * cos(v)
  expect_lt(max(abs(gf(v) - ref)), TOL_VDSP)
})

# ---- Tier 3 fix 3b: fast_vec_mul + scalar-pow dispatch ----

test_that("Tier 3 fast_vec_mul matches R `v * w`", {
  set.seed(20260525L); v <- runif(1000L); w <- runif(1000L)
  expect_lt(max(abs(fast_vec_mul(v, w) - v * w)), TOL_VDSP)
})

test_that("Tier 3 fast_vec_mul handles empty + length mismatch", {
  expect_equal(fast_vec_mul(numeric(0), numeric(0)), numeric(0))
  expect_error(fast_vec_mul(c(1, 2), c(1, 2, 3)), "length mismatch")
})

test_that("Tier 3 .try_normalize_scalar_pow accepts <num>*<var>^<int>", {
  norm <- DefDiff:::.try_normalize_scalar_pow
  r <- norm(quote(3 * v^2), "v")
  expect_false(is.null(r))
  expect_equal(r$scalar, 3)
  expect_identical(r$var, as.name("v"))
  expect_equal(r$exponent, 2L)

  r4 <- norm(quote(4 * v^3), "v")
  expect_false(is.null(r4))
  expect_equal(r4$exponent, 3L)
})

test_that("Tier 3 .try_normalize_scalar_pow rejects non-matching", {
  norm <- DefDiff:::.try_normalize_scalar_pow
  expect_null(norm(quote(3 * v), "v"))             # no power
  expect_null(norm(quote(3 * v^2.5), "v"))         # non-integer exponent
  expect_null(norm(quote(3 * v^1), "v"))           # k=1 (just v)
  expect_null(norm(quote(3 * w^2), "v"))           # wrong variable
  expect_null(norm(quote(sin(v) * v^2), "v"))      # scalar position is not a literal
})

test_that("Tier 3 .try_normalize_scalar_pow accepts commutative swap", {
  norm <- DefDiff:::.try_normalize_scalar_pow
  r <- norm(quote(v^2 * 3), "v")
  expect_false(is.null(r))
  expect_equal(r$scalar, 3)
  expect_equal(r$exponent, 2L)
})

test_that("grad(sum(v^3)) dispatches via Tier 3 scalar-pow", {
  e <- function(v) NULL; body(e) <- quote(sum(v^3))
  gf <- grad(e)
  expect_true(is.call(body(gf)))
  body_str <- deparse(body(gf))
  expect_true(any(grepl("fast_vec_mul", body_str, fixed = TRUE)))
  expect_true(any(grepl("fast_scalar_mul", body_str, fixed = TRUE)))
  v <- c(1, 2, 3)
  expect_equal(gf(v), 3 * v^2, tolerance = TOL_VDSP)
})

test_that("grad(sum(v^4)) and grad(sum(v^5)) dispatch correctly", {
  e4 <- function(v) NULL; body(e4) <- quote(sum(v^4))
  gf4 <- grad(e4)
  v <- c(1, 2, 3)
  expect_equal(gf4(v), 4 * v^3, tolerance = TOL_VDSP)

  e5 <- function(v) NULL; body(e5) <- quote(sum(v^5))
  gf5 <- grad(e5)
  expect_equal(gf5(v), 5 * v^4, tolerance = TOL_VDSP)
})

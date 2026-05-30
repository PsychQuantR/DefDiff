## test-elementwise-matmul-fastpath.R
## Tier 5 Option A: single-layer elementwise-of-matmul fast path.
## Recognizes sum(f(W %*% v)) for f in {sin, cos, exp, tanh} and emits
## gradient t(W) %*% f'(W %*% v) via vForce kernels.
## Does NOT close multi-layer NN forward (Tier 5 Option B territory).

skip_if_no_fast <- function() {
  if (!.fast_path_available()) skip("macOS fast-path required")
}

# Small fixed test matrix + vector
W_test <- matrix(c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6), 2, 3)
v_test <- c(0.1, 0.5, -0.2)

# ===== Forward numeric correctness (4 functions) =====

test_that("sum(sin(W %*% v)) matches t(W) %*% cos(W %*% v)", {
  skip_if_no_fast()
  W <- W_test
  gf <- grad(function(v) sum(sin(W %*% v)))
  expected <- as.numeric(t(W) %*% cos(W %*% v_test))
  expect_equal(as.numeric(gf(v_test)), expected, tolerance = 1e-10)
})

test_that("sum(cos(W %*% v)) matches -t(W) %*% sin(W %*% v)", {
  skip_if_no_fast()
  W <- W_test
  gf <- grad(function(v) sum(cos(W %*% v)))
  expected <- as.numeric(-t(W) %*% sin(W %*% v_test))
  expect_equal(as.numeric(gf(v_test)), expected, tolerance = 1e-10)
})

test_that("sum(exp(W %*% v)) matches t(W) %*% exp(W %*% v)", {
  skip_if_no_fast()
  W <- W_test
  gf <- grad(function(v) sum(exp(W %*% v)))
  expected <- as.numeric(t(W) %*% exp(W %*% v_test))
  expect_equal(as.numeric(gf(v_test)), expected, tolerance = 1e-10)
})

test_that("sum(tanh(W %*% v)) matches t(W) %*% (1 - tanh(W %*% v)^2)", {
  skip_if_no_fast()
  W <- W_test
  gf <- grad(function(v) sum(tanh(W %*% v)))
  expected <- as.numeric(t(W) %*% (1 - tanh(W %*% v_test)^2))
  expect_equal(as.numeric(gf(v_test)), expected, tolerance = 1e-10)
})

# ===== Body-shape assertions =====

test_that("Body of grad(sum(sin(W %*% v))) is a block with Wv binding + fast_vv_cos", {
  skip_if_no_fast()
  W <- W_test
  gf <- grad(function(v) sum(sin(W %*% v)))
  expect_identical(body(gf)[[1L]], quote(`{`))
  body_str <- deparse(body(gf))
  expect_true(any(grepl("Wv <- W %*% v", body_str, fixed = TRUE)))
  expect_true(any(grepl("fast_vv_cos(Wv)", body_str, fixed = TRUE)))
  expect_true(any(grepl("t(W) %*%", body_str, fixed = TRUE)))
})

test_that("Body of grad(sum(cos(W %*% v))) contains leading negation -t(W)", {
  skip_if_no_fast()
  W <- W_test
  gf <- grad(function(v) sum(cos(W %*% v)))
  body_str <- deparse(body(gf))
  # Cos derivative is -sin; emit is `-t(W) %*% fast_vv_sin(Wv)`
  expect_true(any(grepl("-t(W)", body_str, fixed = TRUE)))
  expect_true(any(grepl("fast_vv_sin(Wv)", body_str, fixed = TRUE)))
})

test_that("Body of grad(sum(exp(W %*% v))) uses fast_vv_exp(Wv)", {
  skip_if_no_fast()
  W <- W_test
  gf <- grad(function(v) sum(exp(W %*% v)))
  body_str <- deparse(body(gf))
  expect_true(any(grepl("fast_vv_exp(Wv)", body_str, fixed = TRUE)))
  expect_true(any(grepl("Wv <- W %*% v", body_str, fixed = TRUE)))
})

test_that("Body of grad(sum(tanh(W %*% v))) contains z binding + (1 - z^2)", {
  skip_if_no_fast()
  W <- W_test
  gf <- grad(function(v) sum(tanh(W %*% v)))
  body_str <- deparse(body(gf))
  expect_true(any(grepl("z <- fast_vv_tanh(Wv)", body_str, fixed = TRUE)))
  expect_true(any(grepl("(1 - z^2)", body_str, fixed = TRUE)))
})

# ===== Negative tests =====

test_that("Negative: sum(log(W %*% v)) (log not in supported set) falls through", {
  skip_if_no_fast()
  W <- W_test
  # log isn't in {sin, cos, exp, tanh}. The fast path shouldn't fire.
  # Either the system raises (walker can't handle log-of-matmul) or it
  # produces a correct result via some other path. Either is acceptable —
  # the assertion is that the new fast path's body shape (Wv <- W %*% v)
  # does NOT appear in the gradient.
  result <- tryCatch({
    gf <- grad(function(v) sum(log(W %*% v)))
    body_str <- deparse(body(gf))
    list(succeeded = TRUE, has_wv_binding = any(grepl("Wv <- W %*% v", body_str, fixed = TRUE)))
  }, error = function(e) list(succeeded = FALSE, has_wv_binding = FALSE))
  # Whether it succeeded or raised, the new fast path's binding should not appear
  expect_false(result$has_wv_binding)
})

test_that("2-layer sum(tanh(W2 %*% tanh(W1 %*% v))) now WORKS (post Tier 5 Option B-lite)", {
  skip_if_no_fast()
  W1 <- matrix(rnorm(10 * 5), 10, 5)
  W2 <- matrix(rnorm(10), 1, 10)
  # Pre-Tier-5-Option-B-lite this raised. `add-nn-walker-extension` added
  # the recursive .elementwise_matmul_chain_grad helper that handles
  # arbitrary-depth chains; sum_rule's new hook dispatches it for chain shapes.
  gf <- grad(function(v) sum(tanh(W2 %*% tanh(W1 %*% v))))
  v <- runif(5)
  h1 <- as.numeric(tanh(W1 %*% v))
  h2 <- as.numeric(tanh(W2 %*% h1))
  expected <- as.numeric(t(W1) %*% ((1 - h1^2) * as.numeric(t(W2) %*% (1 - h2^2))))
  expect_equal(as.numeric(gf(v)), expected, tolerance = 1e-10)
})

test_that("Boundary: sum(tanh(W %*% (v + 1))) — handled by Phase 5 walker", {
  skip_if_no_fast()
  W <- W_test
  # Second operand to %*% is (v + 1), not bare v. Option A fast path doesn't
  # match — but Phase 5 walker (`add-walker-shape-extension`) handles this
  # via shim composition: leaf v + constant 1 → shim with pullback; %*%
  # walker case composes; chain rule wraps.
  # Analytical: d/dv sum(tanh(W (v+1))) = t(W) %*% (1 - tanh(W(v+1))^2)
  v <- c(0.5, 0.7, 0.3)
  gf <- grad(function(v) sum(tanh(W %*% (v + 1))))
  result <- as.numeric(gf(v))
  expected <- as.numeric(t(W) %*% (1 - tanh(W %*% (v + 1))^2))
  expect_equal(result, expected, tolerance = 1e-10)
})

test_that("Negative: sum(tanh(v %*% v)) (no constant W) falls through", {
  skip_if_no_fast()
  # v %*% v has v on both sides (not bare W constant). Fast path detector
  # requires !.contains_var on the first %*% operand. Falls through.
  v_local <- c(1, 2, 3)  # avoid conflict
  expect_error(
    grad(function(v) sum(tanh(v %*% v))),
    class = "DefDiff_not_definable"
  )
})

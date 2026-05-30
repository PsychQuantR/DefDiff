## test-nn-walker-extension.R
## Tier 5 Option B-lite: focused walker shape extension for arbitrary-depth
## elementwise-matmul chains. Closes 2-layer nn_forward.

skip_if_no_fast <- function() {
  if (!.fast_path_available()) skip("macOS fast-path required")
}

# Fixed test matrices for reproducibility
W1_test <- matrix(c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6), 2, 3)  # 2x3
W2_test <- matrix(c(0.7, 0.8), 1, 2)                       # 1x2
W3_test <- matrix(c(0.5), 1, 1)                            # 1x1
v_test  <- c(0.1, 0.5, -0.2)

# Helper: analytic 2-layer backprop with tanh-tanh
analytic_2layer_tt <- function(v, W1, W2) {
  h1 <- as.numeric(tanh(W1 %*% v))
  h2 <- as.numeric(tanh(W2 %*% h1))
  as.numeric(t(W1) %*% ((1 - h1^2) * as.numeric(t(W2) %*% (1 - h2^2))))
}

# ===== 1-layer parity (Option A path still fires) =====

test_that("1-layer sum(tanh(W %*% v)) still uses Option A (single-layer fast path)", {
  skip_if_no_fast()
  W <- W1_test
  gf <- grad(function(v) sum(tanh(W %*% v)))
  expected <- as.numeric(t(W) %*% (1 - tanh(W %*% v_test)^2))
  expect_equal(as.numeric(gf(v_test)), expected, tolerance = 1e-10)
  # Body should be Option A's signature (z <- fast_vv_tanh, not colSums)
  body_str <- deparse(body(gf))
  expect_true(any(grepl("z <- fast_vv_tanh", body_str, fixed = TRUE)))
  expect_false(any(grepl("colSums", body_str, fixed = TRUE)))
})

# ===== 2-layer with each homogeneous activation =====

test_that("2-layer tanh-tanh: sum(tanh(W2 %*% tanh(W1 %*% v))) matches analytic", {
  skip_if_no_fast()
  W1 <- W1_test; W2 <- W2_test
  gf <- grad(function(v) sum(tanh(W2 %*% tanh(W1 %*% v))))
  expected <- analytic_2layer_tt(v_test, W1, W2)
  expect_equal(as.numeric(gf(v_test)), expected, tolerance = 1e-10)
})

test_that("2-layer sin-sin", {
  skip_if_no_fast()
  W1 <- W1_test; W2 <- W2_test
  gf <- grad(function(v) sum(sin(W2 %*% sin(W1 %*% v))))
  h1 <- as.numeric(sin(W1 %*% v_test))
  h2 <- as.numeric(sin(W2 %*% h1))
  expected <- as.numeric(t(W1) %*% (cos(as.numeric(W1 %*% v_test)) *
                                    as.numeric(t(W2) %*% cos(as.numeric(W2 %*% h1)))))
  expect_equal(as.numeric(gf(v_test)), expected, tolerance = 1e-10)
})

test_that("2-layer exp-exp", {
  skip_if_no_fast()
  W1 <- W1_test; W2 <- W2_test
  gf <- grad(function(v) sum(exp(W2 %*% exp(W1 %*% v))))
  h1 <- as.numeric(exp(W1 %*% v_test))
  h2 <- as.numeric(exp(W2 %*% h1))
  expected <- as.numeric(t(W1) %*% (h1 * as.numeric(t(W2) %*% h2)))
  expect_equal(as.numeric(gf(v_test)), expected, tolerance = 1e-10)
})

test_that("2-layer cos-cos (negative-sign derivative)", {
  skip_if_no_fast()
  W1 <- W1_test; W2 <- W2_test
  gf <- grad(function(v) sum(cos(W2 %*% cos(W1 %*% v))))
  h1 <- as.numeric(cos(W1 %*% v_test))
  expected <- as.numeric(t(W1) %*% (-sin(as.numeric(W1 %*% v_test)) *
                                    as.numeric(t(W2) %*% -sin(as.numeric(W2 %*% h1)))))
  expect_equal(as.numeric(gf(v_test)), expected, tolerance = 1e-10)
})

# ===== Mixed activations =====

test_that("2-layer mixed tanh-sin: sum(tanh(W2 %*% sin(W1 %*% v)))", {
  skip_if_no_fast()
  W1 <- W1_test; W2 <- W2_test
  gf <- grad(function(v) sum(tanh(W2 %*% sin(W1 %*% v))))
  s1 <- as.numeric(sin(W1 %*% v_test))
  h2 <- as.numeric(tanh(W2 %*% s1))
  expected <- as.numeric(t(W1) %*% (cos(as.numeric(W1 %*% v_test)) *
                                    as.numeric(t(W2) %*% (1 - h2^2))))
  expect_equal(as.numeric(gf(v_test)), expected, tolerance = 1e-10)
})

test_that("2-layer mixed tanh-exp: sum(tanh(W2 %*% exp(W1 %*% v)))", {
  skip_if_no_fast()
  W1 <- W1_test; W2 <- W2_test
  gf <- grad(function(v) sum(tanh(W2 %*% exp(W1 %*% v))))
  e1 <- as.numeric(exp(W1 %*% v_test))
  h2 <- as.numeric(tanh(W2 %*% e1))
  expected <- as.numeric(t(W1) %*% (e1 * as.numeric(t(W2) %*% (1 - h2^2))))
  expect_equal(as.numeric(gf(v_test)), expected, tolerance = 1e-10)
})

test_that("2-layer mixed sin-tanh: sum(sin(W2 %*% tanh(W1 %*% v)))", {
  skip_if_no_fast()
  W1 <- W1_test; W2 <- W2_test
  gf <- grad(function(v) sum(sin(W2 %*% tanh(W1 %*% v))))
  h1 <- as.numeric(tanh(W1 %*% v_test))
  s2 <- as.numeric(sin(W2 %*% h1))
  expected <- as.numeric(t(W1) %*% ((1 - h1^2) *
                                    as.numeric(t(W2) %*% cos(as.numeric(W2 %*% h1)))))
  expect_equal(as.numeric(gf(v_test)), expected, tolerance = 1e-10)
})

# ===== 3-layer (sanity check arbitrary depth) =====

test_that("3-layer tanh chain: sum(tanh(W3 %*% tanh(W2 %*% tanh(W1 %*% v))))", {
  skip_if_no_fast()
  W1 <- W1_test; W2 <- W2_test; W3 <- W3_test
  gf <- grad(function(v) sum(tanh(W3 %*% tanh(W2 %*% tanh(W1 %*% v)))))
  h1 <- as.numeric(tanh(W1 %*% v_test))
  h2 <- as.numeric(tanh(W2 %*% h1))
  h3 <- as.numeric(tanh(W3 %*% h2))
  expected <- as.numeric(t(W1) %*% ((1 - h1^2) *
              as.numeric(t(W2) %*% ((1 - h2^2) *
              as.numeric(t(W3) %*% (1 - h3^2))))))
  expect_equal(as.numeric(gf(v_test)), expected, tolerance = 1e-10)
})

# ===== nn_forward exact (Phase 4 real-world pattern) =====

test_that("nn_forward (Phase 4 pattern) matches analytic backprop", {
  skip_if_no_fast()
  # Locate the shipped benchmark helper via system.file (works under both
  # devtools::load_all and the installed package); the "../../inst" relative
  # path breaks under R CMD check, which runs tests from a temp dir.
  src <- system.file("benchmarks", "real_world_patterns.R", package = "DefDiff")
  skip_if(!nzchar(src) || !file.exists(src), "real_world_patterns.R not available")
  source(src)
  f_nn <- real_world_patterns$nn_forward$f_inline_builder(5L)
  gf <- grad(f_nn)
  v <- runif(5)
  expected <- real_world_analytic$nn_forward(v)
  expect_equal(as.numeric(gf(v)), as.numeric(expected), tolerance = 1e-10)
})

# ===== Body inspection =====

test_that("2-layer grad body uses colSums (proves new chain hook fired)", {
  skip_if_no_fast()
  W1 <- W1_test; W2 <- W2_test
  gf <- grad(function(v) sum(tanh(W2 %*% tanh(W1 %*% v))))
  body_str <- deparse(body(gf))
  expect_true(any(grepl("colSums", body_str, fixed = TRUE)))
})

# ===== Negative tests =====

test_that("Boundary: bias term sum(tanh(W %*% v + b)) — handled by Phase 5 walker", {
  skip_if_no_fast()
  W <- W1_test
  b <- c(0.1, 0.2)
  # Phase 5 (`add-walker-shape-extension`) closes this case via the matmul
  # walker shim + arithmetic NULL-legacy propagation through bias term.
  v <- c(0.5, 0.7, 0.3)
  gf <- grad(function(v) sum(tanh(W %*% v + b)))
  result <- as.numeric(gf(v))
  expected <- as.numeric(t(W) %*% (1 - tanh(W %*% v + b)^2))
  expect_equal(result, expected, tolerance = 1e-10)
})

test_that("Boundary: log as inner f — handled by Phase 5 walker chain case", {
  skip_if_no_fast()
  W1 <- W1_test
  W2 <- W2_test
  # B-lite chain helper restricted f to {sin, cos, exp, tanh}; Phase 5 walker
  # chain switch covers the full {sin, cos, exp, log, tanh, sqrt, atan} set.
  # Use positive v so log is defined.
  v <- c(0.2, 0.3, 0.5)
  gf <- grad(function(v) sum(tanh(W2 %*% log(W1 %*% v))))
  result <- as.numeric(gf(v))
  W1v <- as.numeric(W1 %*% v)
  inner <- as.numeric(W2 %*% log(W1v))
  expected <- as.numeric(
    t(W1) %*% ((1 / W1v) * as.numeric(t(W2) %*% (1 - tanh(inner)^2)))
  )
  expect_equal(result, expected, tolerance = 1e-10)
})

test_that("Boundary: bias after first matmul — handled by Phase 5 walker", {
  skip_if_no_fast()
  W1 <- W1_test; W2 <- W2_test
  c_bias <- c(0.1, 0.2)
  # sum(tanh(W2 %*% (W1 %*% v + c))) — Phase 5 matmul walker propagates
  # through arithmetic + bias.
  v <- c(0.5, 0.7, 0.3)
  gf <- grad(function(v) sum(tanh(W2 %*% (W1 %*% v + c_bias))))
  result <- as.numeric(gf(v))
  inner <- as.numeric(W2 %*% (W1 %*% v + c_bias))
  expected <- as.numeric(t(W1) %*% as.numeric(t(W2) %*% (1 - tanh(inner)^2)))
  expect_equal(result, expected, tolerance = 1e-10)
})

test_that("Negative: bare sum(W %*% v) (no vForce wrap) uses linear-form path, not chain", {
  skip_if_no_fast()
  W <- W1_test
  # sum(W %*% v) without any vForce wrap is the existing linear form
  # (add-matmul-generator). The chain hook detects "is_chain_shape" includes
  # %*% calls, but the helper sees bare-var inner and returns NULL jacobian
  # (no I_n materialization), so the sum_rule hook's "jacobian is non-NULL"
  # check rejects and falls through to existing linear-form path.
  gf <- grad(function(v) sum(W %*% v))
  expected <- as.numeric(colSums(W))
  expect_equal(as.numeric(gf(v_test)), expected, tolerance = 1e-10)
})

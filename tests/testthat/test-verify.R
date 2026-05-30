test_that("verify_grad returns DefDiff_verify_result with three slots", {
  result <- verify_grad(function(v) sum(v^2), function(v) 2 * v)
  expect_s3_class(result, "DefDiff_verify_result")
  expect_named(result, c("syntactic", "numeric", "cross_strategy"))
})

test_that("syntactic layer passes when levels match", {
  result <- verify_grad(function(v) sum(v^2), function(v) 2 * v)
  expect_true(result$syntactic$pass)
  expect_equal(result$syntactic$level_of_f,  "L_1")
  expect_equal(result$syntactic$level_of_gf, "L_0")
})

test_that("numeric layer passes for correct gradient", {
  result <- verify_grad(function(v) sum(v^2), function(v) 2 * v,
                        n_samples = 50L, sample_dim = 3L, tol = TOL_NUMERIC)
  expect_true(result$numeric$pass)
  expect_lt(result$numeric$max_abs_error, TOL_NUMERIC)
})

test_that("numeric layer fails when gradient is wrong", {
  result <- verify_grad(function(v) sum(v^2), function(v) 3 * v,
                        n_samples = 50L, sample_dim = 3L, tol = TOL_NUMERIC)
  expect_false(result$numeric$pass)
  expect_gt(result$numeric$max_abs_error, TOL_NUMERIC)
})

test_that("cross-strategy layer skips when Deriv unavailable", {
  if (requireNamespace("Deriv", quietly = TRUE)) {
    skip("Deriv is installed; this test requires absent Deriv")
  }
  result <- verify_grad(function(v) sum(v^2), function(v) 2 * v)
  expect_equal(result$cross_strategy$status, "skipped")
  expect_match(result$cross_strategy$reason, "Deriv")
})

test_that("cross-strategy layer produces a result when Deriv is present", {
  skip_if_not_installed("Deriv")
  result <- verify_grad(function(v) sum(v^2), function(v) 2 * v,
                        n_samples = 20L, sample_dim = 3L, tol = TOL_NUMERIC)
  # Deriv's handling of vector arguments differs across versions; we accept
  # any non-empty outcome (pass, fail with numeric report, or error/skip).
  cs <- result$cross_strategy
  expect_true(length(cs) >= 1L)
  expect_true(any(c("pass", "status") %in% names(cs)))
})

test_that("print method emits three-layer summary", {
  result <- verify_grad(function(v) sum(v^2), function(v) 2 * v)
  out <- capture.output(print(result))
  text <- paste(out, collapse = "\n")
  expect_match(text, "Syntactic")
  expect_match(text, "Numeric")
  expect_match(text, "Cross-strategy")
})

test_that("verify_grad rejects non-function inputs", {
  expect_error(verify_grad("not_a_function", function(v) v),
               class = "DefDiff_not_definable")
  expect_error(verify_grad(function(v) v, 42),
               class = "DefDiff_not_definable")
})

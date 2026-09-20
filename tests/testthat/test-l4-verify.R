## test-l4-verify.R
## verify_grad() tolerance tier for gradients containing L_4 nodes.

test_that("verify_grad widens tolerance for L_4 gradients and flags it", {
  # ∫_0^1 exp(t * theta) dt is well-posed on theta ∈ [-1, 1]
  f  <- function(theta) integral(exp(t * theta), t, 0, 1)
  gf <- grad(f)
  res <- verify_grad(f, gf, n_samples = 5L, sample_dim = 1L, tol = 1e-6)
  expect_true(res$numeric$l4_tolerance_widened)
  expect_equal(res$numeric$tol_effective, 1e-4)
  expect_true(res$numeric$pass)
  out <- capture.output(print(res))
  expect_true(any(grepl("tol widened to 1e-04 (L_4 nodes present)", out, fixed = TRUE)))
})

test_that("verify_grad does not widen tolerance without L_4 nodes", {
  f  <- function(v) sum(v^2)
  gf <- grad(f)
  res <- verify_grad(f, gf, n_samples = 5L, sample_dim = 3L, tol = 1e-6)
  expect_false(res$numeric$l4_tolerance_widened)
  expect_equal(res$numeric$tol_effective, 1e-6)
})

test_that("widening applies to the numeric layer only and is shown on FAIL", {
  f  <- function(theta) integral(exp(t * theta), t, 0, 1)
  gf_wrong <- grad(f)
  body(gf_wrong) <- quote(1)   # wrong on purpose
  attr(gf_wrong, "grad_expr") <- quote(integral(exp(t * theta) * t, t, 0, 1))
  res <- verify_grad(f, gf_wrong, n_samples = 3L, sample_dim = 1L, tol = 1e-6)
  expect_false(res$numeric$pass)
  out <- capture.output(print(res))
  expect_true(any(grepl("tol widened to 1e-04", out, fixed = TRUE)))
})

test_that("verify_grad reports FAIL instead of aborting when f rejects the sample dimension", {
  f  <- function(theta) integral(exp(t * theta), t, 0, 1)
  res <- verify_grad(f, grad(f), n_samples = 2L, sample_dim = 3L)
  expect_false(res$numeric$pass)
  expect_match(res$numeric$reason, "sample_dim = 1L")
})

test_that("verify_grad distinguishes f errors from non-finite values", {
  f_nan <- function(theta) sqrt(theta) * 0 / 0
  res <- verify_grad(f_nan, function(theta) 0, n_samples = 2L, sample_dim = 1L)
  expect_false(res$numeric$pass)
  expect_match(res$numeric$reason, "non-finite")
  f_err <- function(theta) stop("boom")
  res2 <- verify_grad(f_err, function(theta) 0, n_samples = 2L, sample_dim = 1L)
  expect_match(res2$numeric$reason, "errored.*boom")
})

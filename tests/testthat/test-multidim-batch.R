# Tests for the multi-dimensional batch protocol (add-multidim-batch / #4, Reading A).
# Correctness + dimension-independence; small-n, platform-independent (no perf).

test_that("the same stored gradient is correct across differing dimensions", {
  gf <- grad(function(v) sum(v^2))          # store once
  for (n in c(100L, 5000L, 50000L)) {
    set.seed(n); v <- rnorm(n)
    expect_equal(gf(v), 2 * v, tolerance = 1e-10)  # one gf, many dims
  }
})

test_that("dd_batch maps one stored gradient across a list of mixed-dim inputs", {
  b <- dd_batch(function(v) sum(v^3))
  set.seed(1)
  inputs <- list(rnorm(100), rnorm(5000), rnorm(50000))
  out <- b(inputs)
  expect_type(out, "list")
  expect_length(out, 3L)
  for (i in seq_along(inputs)) {
    expect_equal(out[[i]], 3 * inputs[[i]]^2, tolerance = 1e-10)
  }
})

test_that("dd_batch accepts variadic vectors and preserves the inspectable formula", {
  b <- dd_batch(function(v) sum(v^3))
  set.seed(2); v1 <- rnorm(10); v2 <- rnorm(20)
  out <- b(v1, v2)
  expect_length(out, 2L)
  expect_equal(out[[1L]], 3 * v1^2, tolerance = 1e-10)
  expect_equal(out[[2L]], 3 * v2^2, tolerance = 1e-10)
  # the stored gradient formula is recoverable, dimension-free
  expect_equal(attr(b, "grad_expr"), quote(3 * v^2))
})

test_that("dd_batch errors on empty input", {
  b <- dd_batch(function(v) sum(v^2))
  expect_error(b(), class = "DefDiff_not_definable")
})

test_that("dd_batch rejects a multi-variable f (single-vector contract, #4 verify HIGH)", {
  expect_error(dd_batch(function(v, w) sum(v^2) + sum(w^3)),
               class = "DefDiff_not_definable")
})

test_that("dd_batch still unwraps a plain list of vectors (data.frame guard intact)", {
  b <- dd_batch(function(v) sum(v^2))
  out <- b(list(1:3, c(0.5, 1.5)))     # single plain list -> 2 inputs
  expect_length(out, 2L)
  expect_equal(out[[1L]], 2 * (1:3), tolerance = 1e-10)
})

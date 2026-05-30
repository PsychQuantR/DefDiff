## test-acceptance.R
## Manual smoke tests from design.md acceptance criteria.

test_that("acceptance 1: grad(function(v) sum(v^2))(c(1,2,3)) returns c(2,4,6)", {
  gf <- grad(function(v) sum(v^2))
  expect_equal(gf(c(1, 2, 3)), c(2, 4, 6))
})

test_that("acceptance 2: grad(quote(sum(v^2)), 'v') is symbolically 2*v", {
  expect_equal(grad(quote(sum(v^2)), "v"), quote(2 * v))
})

test_that("acceptance 3: level(quote(sin(sum(v^2)))) returns 'L_3'", {
  expect_equal(level(quote(sin(sum(v^2)))), "L_3")
})

test_that("acceptance 4: extend_language registers erf and level recognises", {
  DefDiff:::register_default_catalog()
  extend_language(
    "L_3", "erf",
    function(x, dx) bquote(2 / sqrt(pi) * exp(-(.(x))^2) * .(dx))
  )
  expect_equal(level(quote(erf(sum(v^2)))), "L_3")
})

test_that("acceptance 5: verify_grad numeric$max_abs_error < 1e-6", {
  result <- verify_grad(function(v) sum(v^2), function(v) 2 * v)
  expect_lt(result$numeric$max_abs_error, 1e-6)
})

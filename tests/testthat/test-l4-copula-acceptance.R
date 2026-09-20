## test-l4-copula-acceptance.R
## Acceptance case for L_4 (issue #22 / #23): the Student-t quantile,
## defined through its CDF, differentiated with respect to the degrees of
## freedom nu. The Gamma prefactor of the density is replaced by a normalizing
## integral so the whole object is an L_4 term over the L_3 kernel
## (1 + t^2/nu)^(-(nu+1)/2); Gamma support via extend_language() is #17.

test_that("Student-t quantile is differentiable w.r.t. nu (nested binders)", {
  skip_on_cran()
  kernel <- quote((1 + t^2 / nu)^(-(nu + 1) / 2))
  q_expr <- bquote(implicit(
    integral(.(kernel), t, -Inf, y) - u * integral(.(kernel), t, -Inf, Inf),
    y, -50, 50))
  g <- grad(q_expr, "nu")
  expect_equal(level(g), "L_4")
  expect_true(all(c("integral", "implicit") %in% all.names(g)))

  q_fn <- function(nu, u) NULL
  body(q_fn) <- q_expr
  at <- function(nu, u = 0.9) eval(g, list(nu = nu, u = u), globalenv())

  # central difference on the quantile itself, step chosen against quadrature noise
  h <- 1e-3
  fd <- (q_fn(5 + h, 0.9) - q_fn(5 - h, 0.9)) / (2 * h)
  expect_equal(at(5), fd, tolerance = 1e-4)
  # sanity: the quantile itself matches qt() at nu = 5
  expect_equal(q_fn(5, 0.9), stats::qt(0.9, df = 5), tolerance = 1e-6)
})

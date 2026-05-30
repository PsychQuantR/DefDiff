test_that(".dat_env$catalog structure matches spec", {
  DefDiff:::register_default_catalog()
  cat <- DefDiff:::.dat_env$catalog
  expect_named(cat, c("L_0", "L_1", "L_2", "L_3"))
  expect_true(is.list(cat$L_0))
  expect_true(is.list(cat$L_3))
})

test_that("extend_language registers a new L_3 generator", {
  DefDiff:::register_default_catalog()
  expect_false("erf" %in% language_catalog("L_3"))
  extend_language(
    "L_3", "erf",
    function(x, dx) bquote(2 / sqrt(pi) * exp(-(.(x))^2) * .(dx))
  )
  expect_true("erf" %in% language_catalog("L_3"))
  expect_equal(level(quote(erf(v))), "L_3")
  expect_equal(level(quote(erf(sum(v^2)))), "L_3")
})

test_that("extend_language rejects invalid arguments", {
  DefDiff:::register_default_catalog()
  expect_error(
    extend_language("L_5", "foo", function(x, dx) dx),
    class = "DefDiff_invalid_extension"
  )
  expect_error(
    extend_language("L_3", c("a", "b"), function(x, dx) dx),
    class = "DefDiff_invalid_extension"
  )
  expect_error(
    extend_language("L_3", "foo", "not_a_function"),
    class = "DefDiff_invalid_extension"
  )
})

test_that("register_default_catalog resets user-added generators", {
  DefDiff:::register_default_catalog()
  extend_language("L_3", "my_custom", function(x, dx) dx)
  expect_true("my_custom" %in% language_catalog("L_3"))
  DefDiff:::register_default_catalog()
  expect_false("my_custom" %in% language_catalog("L_3"))
})

test_that("language_catalog returns full list or filtered vector", {
  DefDiff:::register_default_catalog()
  full <- language_catalog()
  expect_named(full, c("L_0", "L_1", "L_2", "L_3"))

  l3 <- language_catalog("L_3")
  expect_type(l3, "character")
  for (fn in c("sin", "cos", "exp", "log", "sqrt", "tanh", "atan")) {
    expect_true(fn %in% l3, info = paste("missing default L_3 generator:", fn))
  }
})

test_that("language_catalog rejects unknown level", {
  expect_error(language_catalog("L_99"), class = "DefDiff_invalid_extension")
})

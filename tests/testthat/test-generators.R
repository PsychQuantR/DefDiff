test_that("L_0 defaults registered", {
  DefDiff:::register_default_catalog()
  l0 <- language_catalog("L_0")
  for (op in c("+", "-", "*", "/")) {
    expect_true(op %in% l0, info = paste("missing L_0 op:", op))
  }
})

test_that("L_1 defaults registered", {
  DefDiff:::register_default_catalog()
  l1 <- language_catalog("L_1")
  for (op in c("sum", "crossprod", "^")) {
    expect_true(op %in% l1, info = paste("missing L_1 op:", op))
  }
})

test_that("L_2 defaults registered (level tagging)", {
  DefDiff:::register_default_catalog()
  l2 <- language_catalog("L_2")
  for (op in c("%*%", "t")) {
    expect_true(op %in% l2, info = paste("missing L_2 op:", op))
  }
})

test_that("L_3 defaults registered (>= 7 analytic functions)", {
  DefDiff:::register_default_catalog()
  l3 <- language_catalog("L_3")
  for (fn in c("sin", "cos", "exp", "log", "sqrt", "tanh", "atan")) {
    expect_true(fn %in% l3, info = paste("missing L_3 generator:", fn))
  }
  expect_gte(length(l3), 7L)
})

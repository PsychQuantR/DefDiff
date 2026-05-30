## test-walker-finalizer.R
## Phase 6 (`add-walker-shape-extension`): the `_legacy` field is removed.
## `.finalize_reduction_grad` now has one job — for a shim, invoke
## `pullback(rep(1, length(value)))`; for a bare AST (defensive — walker no
## longer emits one for recognized cases), pass through.
##
## SAFETY note on eval(): the compound-shim test evaluates a synthetically-
## built gradient AST with v bound. The AST is constructed from trusted
## internal helpers; no external input flows through eval().

test_that("Phase 6: finalizer invokes pullback for a real walker shim", {
  walker_result <- DefDiff:::.grad_inner(quote(v * v), "v")
  # Post-Phase-6: no _legacy field, pullback is the only path.
  expect_false("_legacy" %in% names(walker_result))
  expect_true(!is.null(walker_result$pullback))
  out <- DefDiff:::.finalize_reduction_grad(walker_result, quote(v * v))
  expect_true(is.call(out))  # Returns an AST, not a value
})

test_that("Phase 6: finalized pullback evaluates to the correct gradient", {
  walker_result <- DefDiff:::.grad_inner(quote(v * v), "v")
  grad_ast <- DefDiff:::.finalize_reduction_grad(walker_result, quote(v * v))
  # SAFETY: eval is intentional — grad_ast comes from DefDiff:::.grad_inner +
  # DefDiff:::.finalize_reduction_grad (trusted internal helpers). No external
  # input passes through. env is restricted to baseenv() + test-bound v.
  env <- new.env(parent = baseenv())
  env$v <- c(1, 2, 3)
  expect_equal(eval(grad_ast, envir = env), c(2, 4, 6))
})

test_that("Phase 6: finalizer errors when shim has NULL pullback", {
  walker_result <- list(value = quote(v), pullback = NULL)
  expect_error(
    DefDiff:::.finalize_reduction_grad(walker_result, quote(v)),
    class = "DefDiff_not_definable"
  )
})

test_that("Phase 6: finalizer passes through bare AST unchanged (defensive)", {
  # Walker no longer emits bare ASTs for recognized cases, but the finalizer
  # remains defensive so custom `extend_language()`-registered generators
  # returning bare ASTs continue to work.
  bare <- quote(cos(v))
  expect_identical(DefDiff:::.finalize_reduction_grad(bare, quote(sin(v))), bare)
  expect_identical(DefDiff:::.finalize_reduction_grad(0, quote(v)), 0)
})

test_that("Phase 6: identity-pullback shim recovers rep(1, length(value))", {
  # Construct a synthetic identity-pullback shim (as if the walker hit only
  # the leaf-var case) and verify the finalizer produces rep(1, length(v)) —
  # the correct gradient of sum(v).
  shim <- list(
    value = quote(v),
    pullback = DefDiff:::.make_pullback_identity()
  )
  out <- DefDiff:::.finalize_reduction_grad(shim, quote(v))
  expect_identical(out, bquote(rep(1, length(v))))
})

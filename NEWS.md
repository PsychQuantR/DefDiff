# dat 0.1.0

First tagged minor release. The package is a complete, installable R package
(`R CMD build` + `R CMD INSTALL` succeed; `library(DefDiff)` works) implementing
Definable Differentiation (DD): closed-form symbolic differentiation that stays
within a declared generator catalog (the closure thesis), with macOS Accelerate
(vDSP/vForce) and Metal fast paths.

## Derivative operators

* `grad()` — scalar-output gradient, single- and multi-variable (a named list of
  per-variable gradients for `k >= 2` vector arguments). S3 methods for function,
  call, expression, and formula input.
* `hessian()` — scalar-output Hessian via a fast-path table plus a recursive
  Jacobian-of-gradient walker. Supports composite outer-scalar forms, quadratic
  forms, multi-variable block Hessians (a named list of `n_a x n_b` blocks), and
  scalar-denominator quotients (e.g. the softmax normalizer). n = 1 returns a
  correct 1x1 matrix.
* `jacobian()` — vector-output Jacobian (Option A: symbolic per-output-component,
  closure-thesis-preserving) for explicit `c(...)` assemblies of catalog
  components; scalar-output bodies degrade to a 1xn row. `jacobian.call` provides
  the programmatic entry point.

All three are closed-form (no runtime tape) and verified against `numDeriv`.

## Performance

* Apple Accelerate fast paths: vDSP `fast_scalar_mul` (Tier 1), vForce
  elementwise kernels (`fast_vv_sin/cos/exp/log/tanh/sqrt`), single-pass
  `fast_sum_sq`, and generalized `fast_sum_pow(v, k)` for `sum(v^k)`, k >= 3.
* Metal GPU backend, integrated and threshold-gated: the canonical
  `<scalar> * <var>` gradient routes to a Metal compute kernel when Metal is
  available and `length(v) >= getOption("dat.metal_threshold", 1e9L)`, otherwise
  to vDSP. macOS-only; float32 result matches the double path within ~1e-6.
* Pre-grad algebraic simplifier: constant folding, conservative trigonometric
  identities, and a dormant `sqrt(x^2) -> abs(x)` rule.
* Reverse-mode pullback walker (Tier 5 Option B) closes matmul-composed
  expressions such as multi-layer `sum(f(W2 %*% f(W1 %*% v)))`.

## Testing

* Speed-regression suite (`tests/testthat/test-performance.R`): relative
  ratio-of-medians assertions (DD operators beat `numDeriv` by orders of
  magnitude; genuine vForce/`fast_sum_sq` wins; honest non-regression ceilings
  for ties and known losses), governed by the `dat-benchmark-suite` spec.
* Full suite: 1137 tests passing, 0 failures, 0 warnings on macOS.

## Fixes

* Silenced the array-vector recycling deprecation warning emitted by the L_0
  division rule for `crossprod(v, v) / sum(v^2)` (1x1 array coerced with
  `as.numeric()`).
* `R CMD check` hardening (0 errors). Fixed an S3 generic-name collision: dat's
  `grad.function` / `hessian.function` / `jacobian.function` methods hijacked
  numDeriv's same-named generics for function input whenever dat was loaded, so
  `numDeriv::grad/hessian/jacobian(f, x)` wrongly dispatched to dat in the
  installed package (197 test errors under `R CMD check`, invisible under
  `devtools::load_all`). Tests now call numDeriv's methods directly via a
  `helper-numderiv.R` accessor. Also: dropped an unintended `export()` of the
  internal `.grad_body_for_var`, fixed a fragile relative `source()` path in a
  test (now `system.file()`), and use `CXX_STD = CXX17` instead of a literal
  `-std=c++17` flag. Two `R CMD check` WARNINGs remain by design — the shipped
  Metal `.metallib` binary and the Objective-C++ compile flags — both inherent
  to the macOS/Metal backend and harmless for `remotes::install_github`.

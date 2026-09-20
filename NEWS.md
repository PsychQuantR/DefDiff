# DefDiff 0.2.0

* New language level **L_4** with two binder nodes, `integral(f, t, a, b)` and
  `implicit(F, y, lower, upper)`. `grad()` differentiates them with the Leibniz
  rule and the implicit function theorem and returns an L_4 tree; the nodes
  evaluate directly via `stats::integrate()` / `stats::uniroot()`. `level()`
  reports `"L_4"`. Bound variables are alpha-renamed at entry so substitution is
  capture-safe. `verify_grad()` widens its tolerance to `1e-4` for L_4 gradients
  and reports it. `hessian()` / `jacobian()` refuse L_4 nodes (second order: #24).

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

* `dd_batch()` — multi-dimensional batch protocol (#4): stores `grad(f)` once and
  maps it across inputs of differing dimension, reusing the single compiled kernel
  (cache key excludes `n`); the stored formula stays inspectable via
  `attr(b, "grad_expr")`. vs NumPy (with auto-tune dispatch, #5): DD wins from
  ~1e6 up; NumPy wins n <= 1e5 (the R closure + `.Call` FFI floor that NumPy's
  in-process C does not pay).

* `grad_expr()` — backend-agnostic accessor that recovers the symbolic gradient
  AST from a gradient function. `body(gf)` is often a fast-path kernel call
  (e.g. `fast_scalar_mul(2, v)`) rather than the readable AST; `grad_expr(gf)`
  returns the preserved symbolic form (`2 * v`, or a named list for a
  multi-variable gradient) regardless of evaluation backend.

## Performance

* Apple Accelerate fast paths: vDSP `fast_scalar_mul` (Tier 1), vForce
  elementwise kernels (`fast_vv_sin/cos/exp/log/tanh/sqrt`), single-pass
  `fast_sum_sq`, and generalized `fast_sum_pow(v, k)` for `sum(v^k)`, k >= 3.
  Above `DefDiff.reduce_threshold` (default 1e7) `fast_sum_sq`/`fast_sum_pow` run a
  parallel partial-reduction across `DefDiff.jit_threads` workers (7.1x on
  `fast_sum_sq@1e8`); below the threshold they are byte-for-byte single-threaded.
* Metal GPU backend, integrated and threshold-gated: the canonical
  `<scalar> * <var>` gradient routes to a Metal compute kernel when Metal is
  available and `length(v) >= getOption("dat.metal_threshold", 1e9L)`, otherwise
  to vDSP. macOS-only; float32 result matches the double path within ~1e-6.
* Pre-grad algebraic simplifier: constant folding, conservative trigonometric
  identities, and a dormant `sqrt(x^2) -> abs(x)` rule.
* Reverse-mode pullback walker (Tier 5 Option B) closes matmul-composed
  expressions such as multi-layer `sum(f(W2 %*% f(W1 %*% v)))`.
* Auto-tune dispatch (#5): instead of a hardcoded base-vs-fused size threshold,
  `.dat_auto_dispatch` measures both paths end-to-end on the first call for each
  (gradient shape, `log10(n)` bucket) and caches the faster, self-calibrating per
  machine. `DefDiff.autotune` (default on); `DefDiff.jit_threshold` set or
  `DefDiff.autotune=FALSE` reverts to the static gate; below
  `DefDiff.autotune_floor` (1e5) always base. Closes the n<=1e7 gap vs NumPy: DD
  now wins from ~1e6 up (the earlier "NumPy wins <=1e7" reading was a
  `jit_threshold=1` benchmark artifact that forced threading at tiny n).
* Auto-tune probe hardening (#6): a repeat-K-until-measurable timer (`proc.time`
  was too coarse and defaulted at mid-n), periodic re-probe every
  `DefDiff.autotune_reprobe` (64) uses to escape a noise-frozen choice, half-decade
  buckets, a base-vs-fused agreement check that fail-safes to base on divergence, and
  option coercion. New options `DefDiff.autotune_min_probe_s` / `autotune_reprobe`.
* `dd_freeze()` (#8): bake the auto-tune-learned dispatch decision into the
  gradient body — `dd_freeze(gf, n_hint)` returns a closure whose body is the
  bare winning call (no dispatcher / option reads / Metal-window check), cutting
  the per-call overhead at small n toward the R-call + `.Call` floor (frozen
  ~0.8–1.3 µs measured; ~60–67% cut on a quiet machine, larger under load).
  Opt-in and additive; forfeits runtime adaptivity, session-scoped; wrong
  `n_hint` costs performance, never correctness.
* Auto-tune wall-time re-probe (#7): the #6 re-probe was use-count only, so multi-shape
  workloads (never reaching `autotune_reprobe` uses on one bucket) couldn't recover a
  noise-poisoned choice. A bucket now re-probes on use-count OR wall-clock age
  (`DefDiff.autotune_reprobe_secs`, default 60) — shape-independent recovery. Plus a 2s
  probe wall-cap, `metal_threshold` coercion, and a `make bench-guard` target that runs
  the gated 1e8 learned-fused guard (human-run; the repo has no CI to wire it into). A
  re-probe re-times both paths (~tens of ms, one-time per bucket per `reprobe_secs`
  window), so re-probes stay rare at the default 60s.
* Fused-JIT backend (Backend B): a fusable arithmetic gradient is lowered to a
  single-pass C++ kernel, runtime-compiled with the system `clang`
  (`-O3 -mcpu=native`), cached by per-element AST + dtype (never `n`), and run
  across worker threads. Threshold-gated (`DefDiff.jit_threshold`, default 1e7)
  with graceful fallback to vDSP. Covers both multi-pass shapes (`3*v^2`,
  measured 4x over the vDSP 2-pass path at n=1e7) and single-pass shapes (`2*v`,
  the `cos(sum(v^2))*2*v` composite), the latter threaded above the threshold and
  below `metal_threshold` so the Metal GPU lane is preserved. macOS-only.

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

# Changelog

All notable changes to the `DefDiff` package are documented here.
Format loosely follows Keep a Changelog; entries reference their Spectra change.

## [Unreleased]

### Fixed

- **L_4 binder nodes — verify #23 round-3** (`add-l4-integral-implicit-nodes`).
  Generated gradient bodies call `DefDiff::integral` / `DefDiff::implicit`
  (a `::` call performs no lookup of those names in the user's environment),
  so no user binding of `integral` / `implicit` — function or variable — can
  capture the generated code; base `::` is trusted like every other base
  operator a generated body uses. The gradient function keeps the user's
  environment. `.control_flow_block()` handles qualified / compound heads. `grad_expr()` keeps the
  public names and `level()` recognizes the qualified heads.
  `.first_binder_head()` scans compound call heads.
  `integral()` / `implicit()` type-check their bounds / interval (a string bound
  used to make `stats::integrate` silently switch to an infinite-range
  transform and return 0) and refuse a non-symbol binder slot like `grad()`
  does. Gradient functions whose body contains L_4 calls now evaluate in a
  child environment that binds DefDiff's own `integral` / `implicit`, so a
  same-named user function cannot hijack the generated body; `extend_language()`
  refuses the two reserved heads. `.refuse_l4_nodes()` names the offending head
  again; `.subst_symbol()` also scans compound call heads. `verify_grad()`
  distinguishes "f errored" / "wrong length" / "non-finite" in its FAIL reason.
  `DESCRIPTION` declares `Depends: R (>= 4.0.0)` for `deparse1()`.

- **L_4 binder nodes — verify #23 round-2 hardening** (`add-l4-integral-implicit-nodes`,
  commit `bb06090`). The three AST walkers in `l4_nodes.R` now iterate
  `seq_along(expr)[-1L]` and skip `NULL` slots: `call[[i]] <- NULL` deletes the
  element in R, which crashed the fused-JIT gradient body (`function()` nodes
  carry `NULL` formals / srcref) for `grad(function(theta) integral(t^2, t, theta, 1))`.
  Substitution no longer rewrites a call head that shares the bound name
  (`t(A)`, `exp(x)`). `.check_binder_node()` refuses wrong arity or a non-symbol
  binder slot with `DefDiff_not_definable` (a string-literal binder used to give
  a silently wrong derivative). `.has_binder_call()` scans call heads, so a
  variable merely named `integral` is neither refused by `hessian()` nor widens
  `verify_grad()`'s tolerance. `verify_grad()` widening is numeric-layer only and
  is printed on the FAIL line too; an `f()` that errors on a sample reports FAIL
  with a reason instead of aborting. `integral()` validates scalar integrands,
  reads `rel.tol` through `.dat_opt_pos_num()`, propagates user-body errors
  unchanged and only re-labels quadrature failures; `implicit()` validates the
  shape of `F`. DESCRIPTION now says five-tier.

### Added

- **Language level L_4: `integral()` and `implicit()` binder nodes**
  (`add-l4-integral-implicit-nodes`, `definable_algebra#23`). `integral(f, t, a, b)`
  and `implicit(F, y, lower, upper)` are plain R calls whose second element is an
  unevaluated body and whose third element is the bound symbol. `grad()` applies
  the Leibniz rule (variable bounds and `±Inf` supported; boundary term at `±Inf`
  taken as zero under a documented decay precondition) and the implicit function
  theorem (root bracketed by a user-given interval with a sign-change check); the
  result stays an L_4 tree, evaluated at the leaves by `stats::integrate()` and
  `stats::uniroot()`. `level()` reports `"L_4"`; `extend_language()` accepts it.
  Binder semantics: `.contains_var()` is binder-aware and bound variables are
  alpha-renamed once at `grad()` entry so substitutions never capture. The
  simplifier treats binder nodes as opaque; `hessian()` / `jacobian()` refuse them
  explicitly (second order is `#24`). `verify_grad()` widens its numeric tolerance
  to `1e-4` for L_4 gradients and reports it. Acceptance case: the Student-t
  quantile defined through its CDF, differentiated w.r.t. the degrees of freedom.

- **`dd_batch()` — multi-dimensional batch protocol** (#4, Reading A). Stores a
  formula's gradient once (`grad(f)`, the dimension-free symbolic form) and returns
  a function that maps it across inputs of differing length, reusing the single
  compiled kernel (the fused-JIT cache key excludes `n`). The stored formula stays
  inspectable via `attr(b, "grad_expr")`. Sugar over `grad()`, no native code. A
  benchmark harness (`inst/benchmarks/multidim-batch-vs-numpy.R` + `sidecar_numpy_compare.py`)
  proves store-once (1 kernel compile across a 6-dim sweep) and measures vs hand-coded
  NumPy gradients. With auto-tune dispatch (below), DD wins from ~1e6 up; NumPy wins
  n ≤ 1e5 (R closure + `.Call` FFI floor). See `docs/dd-multidim-batch.md`.
- **Auto-tune base-vs-fused dispatch** (#5). `.dat_auto_dispatch` replaces the
  hardcoded size threshold: on the first call for each (gradient shape, `log10(n)`
  bucket) it times the base vDSP and fused-threaded paths end-to-end and caches the
  faster, self-calibrating per machine (the crossover depends on `n` and hardware, so
  no single path is always fastest). New options `DefDiff.autotune` (default `TRUE`),
  `DefDiff.autotune_floor` (1e5; below it always base — the tiny-n thread-spawn trap),
  `DefDiff.autotune_reps` (3; min-of-N filters probe noise). `DefDiff.jit_threshold`
  set or `DefDiff.autotune=FALSE` reverts to the static gate (reproducible benchmarks +
  path-forcing tests). Fixes the n≤1e7 gap vs NumPy and the root cause behind it: #2's
  threshold (1e7) was calibrated on isolated-kernel timing, but end-to-end `gf(v)` is
  overhead-dominated at mid-n — auto-tune measures the real end-to-end winner instead.
- **Auto-tune probe hardening** (#6, from the #5 verify). Makes the probe *measure*
  instead of *default*, and recover from noise: (a) **repeat-K-until-measurable timer**
  — `proc.time` elapsed is ~1–16ms, so a single mid-n run quantized to `0.0s` and the
  tie rule defaulted to base; now the path runs in a doubling loop until elapsed clears
  a resolution floor (`DefDiff.autotune_min_probe_s`, 5ms), per-iter = total/iters;
  (b) **periodic re-probe** every `DefDiff.autotune_reprobe` (64) uses, so a noise-frozen
  wrong choice self-corrects within a session; (c) **half-decade buckets** (`floor(2*log10(n))`)
  for intra-decade crossovers; (d) a **base ≡ fused agreement check** (bounded sample) that
  fail-safes to base + warns on a lowering divergence; (e) option coercion (NA/garbage →
  default). Gated large-n test (`DAT_LARGE_BENCH=1`) guards "auto-tune learns fused at 1e8".
- **`dd_freeze()` — bake the learned dispatch into the body** (#8). The dual of
  auto-tune: `dd_freeze(gf, n_hint)` resolves the base-vs-fused winner for
  `n_hint`'s size bucket (reading the learned `pathchoice`, or probing once
  through the existing dispatcher) and regenerates the closure body as the bare
  winning call — no `getOption`, no cache lookup, no dispatcher; canonical
  shapes' Metal-window check is resolved at freeze time too. Measured @1e3 under
  background load (same-state comparison): `sum(v^2)` 7.09 → 1.27 µs, `sum(v^3)`
  6.68 → 2.21 µs; the frozen absolute lands at the structural floor (independent
  re-measure 0.82 µs vs bare-`.Call` 0.74 µs / NumPy ~1 µs). Quiet-machine cut is
  arithmetically capped at ~60–67% (unfrozen 2.38 µs → frozen ~0.8–1.0 µs) — the
  loaded −82% inflates the dispatcher-heavy baseline more than the frozen path. Trade-offs by design: forfeits runtime adaptivity, solidifies
  current options, session-scoped (do not serialize across versions); a wrong
  `n_hint` misplaces performance, never correctness. Non-fusable shapes (already
  a bare kernel call) pass through unchanged; unrecognized body shapes return
  unfrozen with a warning (never wrong). Fail-fast on missing/invalid `n_hint`.
- **Auto-tune wall-time re-probe + speed-guard enforcement** (#7, from the #6 verify).
  The #6 re-probe triggered only on per-bucket use-count, so a multi-shape workload
  (which never accumulates `autotune_reprobe` uses on one bucket) never recovered a
  noise-poisoned choice. Added a **wall-clock age axis**: the `pathchoice` entry now
  carries `probed_at`, and a bucket re-probes when its use-count OR its age exceeds
  `DefDiff.autotune_reprobe_secs` (default 60) — wall-time is shape-independent, closing
  the multi-shape blind spot. Also: `.dat_probe_timed` gains a 2s wall-cap (bounds a
  pathological `autotune_min_probe_s`); `metal_threshold` reads now go through the option
  coercion; and `make bench-guard` runs the gated 1e8 learned-fused guard so the
  speed-claim guard is actually runnable (the repo has no CI to wire it into).
- **Fused-JIT gradient backend** (Spectra change `add-fused-jit-grad-backend`, #2).
  A genuinely multi-pass fusable *arithmetic* gradient (e.g. `grad(sum(v^3))` →
  `3 * v^2`) is now lowered to a single-pass C++ kernel, compiled at runtime with
  the system `clang` (`-O3 -mcpu=native`), `dlopen`'d, cached, and applied across
  worker threads. Above `getOption("DefDiff.jit_threshold", 1e7)` this replaces the
  chained vDSP multi-pass emission; below threshold, without `clang`, or on any
  compile failure it falls back to the existing path (correctness never depends on
  the JIT path). Measured **4× over the vDSP 2-pass path** on `sum(v^3)` at n=1e7.
  Two senses of dimension-independence are preserved: `grad_expr()` still returns
  the n-free `3 * v^2`, and the warm-up cache key is the per-element AST + dtype,
  **never `n`** (one compiled kernel serves every vector length). New options:
  `DefDiff.jit_threshold`, `DefDiff.jit_threads`, `DefDiff.jit_disable`. macOS-first,
  non-CRAN (consistent with the Metal posture). Phase 1′: arithmetic element-wise
  only; per-element transcendental fusion deferred. Exports `.jit_path_available()`
  and `.dat_fused_try()` for the emitted dispatch bodies.
- **`grad_expr()` — backend-agnostic symbolic-gradient accessor** (Spectra change
  `add-grad-expr-accessor`, #3). `grad(<function>)` returns a gradient *function*
  whose `body()` is often a fast-path kernel call (e.g. `fast_scalar_mul(2, v)`)
  rather than the readable AST, so `body(gf)` is not a reliable way to inspect the
  gradient. The symbolic gradient is now preserved as a `grad_expr` attribute
  regardless of evaluation backend (plain AST, vDSP/vForce, Metal, or a future
  fused evaluator); `grad_expr(gf)` returns it (`2 * v`, or a named list of calls
  for a multi-variable gradient). Errors with condition `DefDiff_not_gradient`
  when `gf` lacks the attribute. This keeps the closure-thesis inspectability
  claim independent of how the body is compiled.

### Changed

- **Threaded reduction kernels** (Spectra change `add-threaded-reduction-kernels`, #2).
  `fast_sum_sq` / `fast_sum_pow` now run a parallel partial-reduction above
  `getOption("DefDiff.reduce_threshold", 1e7)` (workers from `DefDiff.jit_threads`):
  disjoint slices reduced per-thread via `vDSP_svesqD` / `vvpow`+`vDSP_sveD`, partials
  summed on the main thread. Below the threshold the single-threaded vDSP path is
  byte-for-byte unchanged. Public signatures unchanged (the kernels read the options
  internally). Measured `fast_sum_sq@1e8` 24 ms → 3.4 ms (7.1×); this threads the
  fused composite preamble so `grad(sin(sum(v^2)))@1e8` drops ~28 ms → ~8 ms, now
  beating JAX (~15 ms). Threaded partial sums match base-R `sum` within 1e-10 relative.
- **Fused-JIT backend now also threads single-pass shapes** (Spectra change
  `extend-fused-jit-single-pass`, #2). The dispatch gate widened from multi-pass-only
  to any fusable arithmetic gradient with an arithmetic op on the variable, so
  `2*v` (`sum(v^2)`) and the composite `cos(sum(v^2))*2*v` (`sin(sum(v^2))`) now
  route to the threaded fused kernel above `jit_threshold`. The Metal GPU lane is
  preserved: a Metal-eligible canonical `<scalar>*<var>` fuses only in
  `[jit_threshold, metal_threshold)`, so `n >= metal_threshold` still routes to
  Metal. Closes the single-pass residual vs the single-threaded vDSP path
  (`sum(v^2)@1e8` ~13 ms -> ~5 ms); against JAX this is now a bandwidth-ceiling
  photo-finish rather than the prior ~6x loss. Reduction preambles
  (`fast_sum_sq`/`fast_sum_pow`) remain single-threaded, so reduction-dominated
  composites do not fully close. Behavior change: single-pass shapes that
  previously stayed on `fast_scalar_mul` now carry the fused wrapper above
  threshold (the kernel remains the below-threshold/fallback path).
- **Renamed the package `dat` -> `DefDiff`** (Definable Differentiation) and the
  public mirror repo `definable-calculus-r` -> `DefDiff`. The old name `dat` came
  from Definable Algebra **Theory** but the package only implements the
  **differentiation** slice, so the name now matches what it does (and reads as a
  searchable word, unlike a bare acronym). User-facing surface renamed —
  `library(DefDiff)`, the `DefDiff.metal_threshold` option, and the
  `DefDiff_not_definable` / `DefDiff_verify_result` condition/result classes;
  internal dot-prefixed helpers keep their `.dat_*` names. GitHub keeps a redirect
  so old `install_github("PsychQuantR/definable-calculus-r")` URLs still work.
- **Package hygiene.** Removed 4 stale `inst/benchmarks/*.bak` backup files
  (they were git-tracked and shipped to users via `inst/`). Trimmed the
  installable package: only `inst/benchmarks/real_world_patterns.R` (a test
  dependency) ships now; the maintainer-only microbenchmarks and AD-comparison
  scripts stay in the repo but are `.Rbuildignore`d. Documented the local-only
  `references/` (PyTorch/JAX AD baselines, gitignored) in `CLAUDE.md`.
- **Version 0.1.0** (first minor release) + first `NEWS.md`. Package builds and
  installs cleanly (`R CMD build` + `R CMD INSTALL`; `library(DefDiff)` works).
  `src/Makevars` uses `CXX_STD = CXX17` instead of a literal `-std=c++17` flag.

### Fixed

- **R CMD check hardening (1 ERROR + 3 WARNINGs -> 0 ERROR + 2 by-design).**
  Fixed an S3 generic-name collision: dat's `grad.function` / `hessian.function`
  / `jacobian.function` hijacked numDeriv's same-named generics for function
  input whenever dat was loaded, so `numDeriv::grad/hessian/jacobian(f, x)`
  wrongly dispatched to dat in the **installed** package (197 test errors under
  `R CMD check`, invisible under `devtools::load_all`). Tests now call numDeriv's
  methods directly via a `helper-numderiv.R` accessor. Also dropped an
  unintended `export()` of internal `.grad_body_for_var`, and replaced a fragile
  relative `source("../../inst/...")` in a test with `system.file()`. The two
  remaining WARNINGs (the shipped `.metallib` binary and the Objective-C++
  compile flags) are inherent to the macOS/Metal backend and do not block
  `remotes::install_github`.
- **Array-vector recycling deprecation** in the L_0 division rule: the gradient
  of `crossprod(v, v) / sum(v^2)` emitted `crossprod(v, v) * (2 * v)` (1x1 array
  x vector); the numerator/denominator are now coerced with `as.numeric()`.
- **Hessian for length-1 (n=1) input.** The Hessian fast-path emitted
  `diag(<length-1 vector>)`, which R reinterprets as a matrix *dimension*, so
  `hessian(function(v) sum(v^2))(c(5))` returned a 2x2 identity instead of the
  1x1 `[[2]]` (and `sum(v^3))(c(2))` returned 12x12, `sum(sin(v))` returned
  0x0). All five fast-path diagonals are now sized via
  `diag(d, nrow = length(v))`; output for `n >= 2` is unchanged. `grad()` was
  never affected. (add-comprehensive-grad-hessian-tests)

### Added

- **Vector-output `jacobian()`** (Option A: symbolic per-output-component,
  closure-thesis-preserving) for `c(...)` assemblies of catalog components.
  (add-vector-output-jacobian)
- **Scalar-denominator quotient Hessian** — fifth recursive-walker shape rule
  (e.g. the softmax normalizer `sum(v*exp(v)) / sum(exp(v))`).
  (add-hessian-quotient-walker)
- **Generalized sum-of-powers fast path** `fast_sum_pow(v, k)` plus pre-grad
  simplifier rules (constant folding, conservative trig identities, dormant
  `sqrt(x^2) -> abs`). (add-simplify-extensions)
- **Metal GPU backend, integrated and threshold-gated** — the canonical
  `<scalar> * <var>` gradient routes to a Metal compute kernel above
  `getOption("dat.metal_threshold", 1e9L)`, else vDSP. (enable-metal-backend)
- **Automated speed-regression test suite** (`tests/testthat/test-performance.R`)
  with relative ratio-of-medians assertions, governed by a new
  `dat-benchmark-suite` requirement. (add-speed-regression-tests)
- **Public install mirror** `PsychQuant/definable-calculus-r` (package-only, no
  theory IP) for `remotes::install_github`; the source repo is now mounted as an
  `Academic` submodule.
- **`make publish` mirror-sync tooling** (`tools/publish-mirror.sh` + `Makefile`,
  both `.Rbuildignore`d): `R CMD build` -> IP-leak guard (fails closed on any
  excluded dir) -> clone the public mirror -> overwrite with the curated tarball
  + source-controlled README -> confirm -> push. `--dry-run` / `--yes` modes.
- **Comprehensive cross-validated test suite** for `grad` / `hessian` /
  `jacobian` across eight dimensions: numeric-equivalence (triple ground truth
  — numDeriv + an independent central finite-difference + closed form, with a
  ground-truth-vs-ground-truth cross-check), catalog coverage, composition /
  nesting, multi-variable block Hessian, edge / numerical (n=1, moderate n,
  zeros, degenerate denominators), property-based invariants (gradient
  linearity, Hessian symmetry, cross-engine `Hessian == numerical-Jacobian(grad)`),
  boundary must-raise plus negative-of-negative, and fast-path-vs-recursive
  equivalence. Full suite at 1110 passing. (add-comprehensive-grad-hessian-tests)

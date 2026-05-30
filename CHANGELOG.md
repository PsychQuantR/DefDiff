# Changelog

All notable changes to the `DefDiff` package are documented here.
Format loosely follows Keep a Changelog; entries reference their Spectra change.

## [Unreleased]

### Changed

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

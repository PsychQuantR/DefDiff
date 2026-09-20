# DefDiff — Definable Differentiation for R

> **`DefDiff`** = **Def**inable **Diff**erentiation — the differentiation reference implementation of the Definable Algebra Theory (DAT) framework.

Closed-form symbolic differentiation for vector calculus in R. The `grad()`,
`hessian()`, and `jacobian()` operators source-transform an R function into a
new R function that returns the exact derivative — no finite differences and no
runtime tape. Since 0.2.0, `grad()` also differentiates through definite
integrals and implicitly defined quantities (`integral()` / `implicit()` binder
nodes — Leibniz rule and the implicit function theorem), so normalizing
constants, CDFs and quantiles are in scope. On macOS the gradient fast paths dispatch to Apple Accelerate
(vDSP / vForce) and, above a size threshold, to a Metal GPU kernel.

## Requirements

macOS (the fast paths link the Accelerate and Metal frameworks). The package
builds and installs with the standard R + Xcode command-line toolchain.

## Install

```r
# install.packages("remotes")
remotes::install_github("PsychQuantR/DefDiff")
```

## Usage

```r
library(DefDiff)

gf <- grad(function(v) sum(v^2))
gf(c(1, 2, 3))                       # 2 4 6

hf <- hessian(function(v) sin(sum(v^2)))
hf(c(1, 2, 3))                       # 3x3 Hessian

jf <- jacobian(function(v) c(sum(v), sum(v^2)))
jf(c(1, 2, 3))                       # 2x3 Jacobian
```

See `NEWS.md` for the capability set. Correctness is verified against
`numDeriv`; the test suite includes relative-speed benchmarks.

## Community speed benchmark

Run the cross-system differentiation benchmark (DefDiff vs numDeriv vs, when installed, PyTorch and JAX) on your own machine and add your result to the leaderboard at **[PsychQuantR/DefDiff-benchmark](https://github.com/PsychQuantR/DefDiff-benchmark)** — install DefDiff, run the harness there, and open a one-file pull request with your run-log.

---

This repository is a **published mirror** of the installable package, generated
from a private source-of-truth repo. **Found a bug or want a feature? Open an
issue here** — this is the right place to report. Pull requests are not
accepted on the mirror (the source lives elsewhere); please file an issue.

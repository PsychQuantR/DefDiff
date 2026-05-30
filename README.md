# dat — Definable Differentiation for R

> The name **`dat`** comes from **D**efinable **A**lgebra **T**heory (DAT); this package is its differentiation reference implementation.

Closed-form symbolic differentiation for vector calculus in R. The `grad()`,
`hessian()`, and `jacobian()` operators source-transform an R function into a
new R function that returns the exact derivative — no finite differences and no
runtime tape. On macOS the gradient fast paths dispatch to Apple Accelerate
(vDSP / vForce) and, above a size threshold, to a Metal GPU kernel.

## Requirements

macOS (the fast paths link the Accelerate and Metal frameworks). The package
builds and installs with the standard R + Xcode command-line toolchain.

## Install

```r
# install.packages("remotes")
remotes::install_github("PsychQuant/definable-calculus-r")
```

## Usage

```r
library(dat)

gf <- grad(function(v) sum(v^2))
gf(c(1, 2, 3))                       # 2 4 6

hf <- hessian(function(v) sin(sum(v^2)))
hf(c(1, 2, 3))                       # 3x3 Hessian

jf <- jacobian(function(v) c(sum(v), sum(v^2)))
jf(c(1, 2, 3))                       # 2x3 Jacobian
```

See `NEWS.md` for the capability set. Correctness is verified against
`numDeriv`; the test suite includes relative-speed benchmarks.

---

This repository is a **published mirror** of the installable package, generated
from a private source-of-truth repo. Do not open PRs here; file issues instead.

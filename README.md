# DefDiff — Definable Differentiation for R

> **`DefDiff`** = **Def**inable **Diff**erentiation — the differentiation reference implementation of the Definable Algebra Theory (DAT) framework.

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

Compare differentiation systems (DefDiff, numDeriv, and — when installed —
PyTorch and JAX) on your own machine and add your result to the leaderboard.
Timing is reported per stage (cold-start `import` / `build` / `jit_compile`
through steady-state `eval`) and across single- vs multi-thread settings. The
data is observational — each contributor is one machine — so it is summarized
with a mixed-effects model (machine as a random effect), never as a balanced
factorial.

### Contributing a run

1. `Rscript inst/benchmarks/run-community-benchmark.R --append`
2. Commit the new `inst/benchmarks/community-logs/<run_id>.json` and open a PR
   adding just that one file. One log file per run means PRs never collide.
3. A maintainer regenerates the CSV and the table below with `make leaderboard`.

The raw run-logs in `inst/benchmarks/community-logs/` are the source of truth;
`inst/benchmarks/community-benchmark.csv` is a regenerable projection of them.

<!-- BENCHMARK-LEADERBOARD:BEGIN -->

| Chip | System | Operation | Problem | n | Threads | eval median (ms) |
|---|---|---|---|---|---|---|
| Apple M5 Max | DefDiff | grad | sum_sin_v | 1e+03 | 1 | 0.002 |
| Apple M5 Max | DefDiff | grad | sum_sin_v | 1e+03 | max | 0.002 |
| Apple M5 Max | PyTorch | grad | sum_sin_v | 1e+03 | max | 0.031 |
| Apple M5 Max | PyTorch | grad | sum_sin_v | 1e+03 | 1 | 0.032 |
| Apple M5 Max | numDeriv | grad | sum_sin_v | 1e+03 | 1 | 66.465 |
| Apple M5 Max | numDeriv | grad | sum_sin_v | 1e+03 | max | 68.350 |
| Apple M5 Max | DefDiff | grad | sum_v2 | 1e+03 | 1 | 0.002 |
| Apple M5 Max | DefDiff | grad | sum_v2 | 1e+03 | max | 0.002 |
| Apple M5 Max | PyTorch | grad | sum_v2 | 1e+03 | max | 0.030 |
| Apple M5 Max | PyTorch | grad | sum_v2 | 1e+03 | 1 | 0.031 |
| Apple M5 Max | numDeriv | grad | sum_v2 | 1e+03 | 1 | 48.428 |
| Apple M5 Max | numDeriv | grad | sum_v2 | 1e+03 | max | 52.568 |
| Apple M5 Max | DefDiff | grad | sum_v3 | 1e+03 | max | 0.002 |
| Apple M5 Max | DefDiff | grad | sum_v3 | 1e+03 | 1 | 0.002 |
| Apple M5 Max | PyTorch | grad | sum_v3 | 1e+03 | 1 | 0.029 |
| Apple M5 Max | PyTorch | grad | sum_v3 | 1e+03 | max | 0.030 |
| Apple M5 Max | numDeriv | grad | sum_v3 | 1e+03 | 1 | 58.362 |
| Apple M5 Max | numDeriv | grad | sum_v3 | 1e+03 | max | 59.554 |
| Apple M5 Max | DefDiff | hessian | sum_v2 | 2e+01 | 1 | 0.001 |
| Apple M5 Max | DefDiff | hessian | sum_v2 | 2e+01 | max | 0.001 |
| Apple M5 Max | numDeriv | hessian | sum_v2 | 2e+01 | max | 1.992 |
| Apple M5 Max | numDeriv | hessian | sum_v2 | 2e+01 | 1 | 2.249 |

<!-- BENCHMARK-LEADERBOARD:END -->

---

This repository is a **published mirror** of the installable package, generated
from a private source-of-truth repo. **Found a bug or want a feature? Open an
issue here** — this is the right place to report. Pull requests are not
accepted on the mirror (the source lives elsewhere); please file an issue.

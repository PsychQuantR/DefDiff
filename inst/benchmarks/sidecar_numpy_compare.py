"""NumPy hand-coded-gradient sidecar for the multi-dimensional batch benchmark (#4).

Mirrors sidecar_jax_compare.py's CLI/JSON contract so multidim-batch-vs-numpy.R
can shell out per (expression, n) and read `{"wall_ms": ...}` from stdout.

FAIR-COMPARISON FRAMING (read this before interpreting the numbers):
NumPy has no autodiff. So "NumPy's gradient" here is the *hand-written analytic
gradient* of each formula, evaluated eagerly as a NumPy array op. The benchmark
therefore measures:

    store-once-compiled-kernel (DD: grad(f) symbolic -> cached fused kernel,
                                reused across dims)
        vs
    eager-array-op (NumPy: the analytic gradient expression recomputed per call)

for the SAME gradient. It is NOT DD-autodiff vs NumPy-autodiff (NumPy has none).
float64 throughout to match DD.
"""

import argparse
import json
import statistics
import sys
import time


def numpy_grad(name: str, v):
    """Hand-coded analytic gradient of each formula, eager NumPy."""
    import numpy as np

    table = {
        "sum_v2":     lambda v: 2.0 * v,                                   # grad sum(v^2)
        "sum_v3":     lambda v: 3.0 * v ** 2,                              # grad sum(v^3)
        "sum_sin_v":  lambda v: np.cos(v),                                 # grad sum(sin(v))
        "sin_sum_v2": lambda v: np.cos(np.sum(v ** 2)) * 2.0 * v,          # grad sin(sum(v^2))
    }
    if name not in table:
        raise ValueError(f"unknown expression: {name}")
    return table[name](v)


def run_numpy_grad(name: str, n: int, reps: int = 7, warmup: int = 2) -> float:
    import numpy as np

    rng = np.random.default_rng(1)
    v = np.asarray(rng.standard_normal(n), dtype=np.float64)
    for _ in range(warmup):       # warm caches / page-in
        numpy_grad(name, v)
    times: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = numpy_grad(name, v)
        # touch the result so a lazy view can't elide the work (NumPy is eager,
        # but this matches the JAX block_until_ready discipline)
        _ = out[0]
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--expression", required=True,
                   choices=["sum_v2", "sum_v3", "sum_sin_v", "sin_sum_v2"])
    p.add_argument("--n", type=int, required=True)
    args = p.parse_args()

    try:
        wall = run_numpy_grad(args.expression, args.n)
        print(json.dumps({"wall_ms": wall}))
        sys.exit(0)
    except ImportError as e:
        print(json.dumps({"error": f"numpy not installed: {e}"}))
        sys.exit(2)  # distinct exit code for missing-dep
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()

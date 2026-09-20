// fused_jit_runtime.cpp
// Backend B (add-fused-jit-grad-backend, #2): the threaded driver that applies
// a runtime-compiled fused gradient kernel across worker threads.
//
// The kernel itself is generated as C++ source by R/fused_jit.R, compiled at
// runtime with the system clang into a standalone .so, dyn.load'd, and its
// symbol address passed here as an R externalptr (NativeSymbol). This shim
// holds the only R-aware + threading logic; the runtime kernel is pure C math
// with the signature:
//
//   extern "C" void dat_fused_kernel(const double* v, double* out,
//                                     long n, const double* s);
//
// `n` is a RUNTIME argument, so one compiled kernel serves every vector length
// (dimension-independence). Threads partition [0, n); scalars `s` (hoisted
// reductions / constants) are shared read-only across all chunks.

#include <Rcpp.h>
#include <thread>
#include <vector>
#include <algorithm>

typedef void (*dat_kernel_t)(const double*, double*, long, const double*);

//' Apply a runtime-compiled fused gradient kernel across worker threads
//'
//' Internal. Splits the element range across `nthreads` workers, each calling
//' the compiled kernel on its disjoint slice. The kernel pointer comes from
//' \code{getNativeSymbolInfo(...)$address} of a dyn.load'd runtime .so.
//'
//' @param kernel_ptr An externalptr (NativeSymbol) to the compiled kernel.
//' @param v Numeric input vector.
//' @param scalars Numeric vector of hoisted scalars (length 0 allowed).
//' @param nthreads Worker thread count (>= 1).
//' @return Numeric vector \code{out} of the same length as \code{v}.
// [[Rcpp::export]]
Rcpp::NumericVector dat_fused_apply(SEXP kernel_ptr, Rcpp::NumericVector v,
                                    Rcpp::NumericVector scalars, int nthreads) {
  R_xlen_t n = v.size();
  if (n == 0) return Rcpp::NumericVector(0);
  // no_init: skip the zero-fill pass — the kernel writes every element, so
  // value-initialization would be a wasted full memory pass (matters at 1e8).
  Rcpp::NumericVector out(Rcpp::no_init(n));

  dat_kernel_t fn = reinterpret_cast<dat_kernel_t>(R_ExternalPtrAddr(kernel_ptr));
  if (fn == nullptr) Rcpp::stop("dat_fused_apply: null kernel pointer");

  const double* vp = REAL(v);
  double* op = REAL(out);
  const double* sp = (scalars.size() > 0) ? REAL(scalars) : nullptr;

  if (nthreads < 1) nthreads = 1;
  if (static_cast<R_xlen_t>(nthreads) > n) nthreads = static_cast<int>(n);

  if (nthreads == 1) {
    fn(vp, op, static_cast<long>(n), sp);
    return out;
  }

  // Disjoint contiguous chunks; no shared writes, no R API calls inside the
  // worker (the kernel is pure C math on raw pointers), so this is thread-safe.
  std::vector<std::thread> workers;
  workers.reserve(nthreads);
  R_xlen_t chunk = (n + nthreads - 1) / nthreads;
  for (int t = 0; t < nthreads; ++t) {
    R_xlen_t lo = static_cast<R_xlen_t>(t) * chunk;
    R_xlen_t hi = std::min(n, lo + chunk);
    if (lo >= hi) break;
    workers.emplace_back([=]() {
      fn(vp + lo, op + lo, static_cast<long>(hi - lo), sp);
    });
  }
  for (auto& w : workers) w.join();
  return out;
}

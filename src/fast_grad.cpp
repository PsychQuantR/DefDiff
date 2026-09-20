// fast_grad.cpp
// Rcpp bridge to Apple Accelerate framework's vDSP_vsmulD.
// Implements Design Behavior 2 (Scalar-vector multiplication kernel) of
// add-vdsp-fast-path: hand-tuned NEON SIMD scalar-vector multiply that
// bypasses R's per-element arithmetic overhead.

#include <Rcpp.h>
#include <vector>
#include <thread>
#include <algorithm>

// Forward-declare vDSP_vsmulD instead of including <Accelerate/Accelerate.h>:
// the full header defines `COMPLEX` as a typedef which collides with R's
// `Rcomplex *(COMPLEX)(SEXP)` function prototype in Rinternals.h. Linking
// against -framework Accelerate (via src/Makevars) supplies the symbol.
#ifdef __APPLE__
extern "C" {
  typedef long vDSP_Stride;
  typedef unsigned long vDSP_Length;
  void vDSP_vsmulD(const double *__A, vDSP_Stride __IA,
                   const double *__B,
                   double *__C, vDSP_Stride __IC,
                   vDSP_Length __N);
  // vDSP_svesqD: sum-of-squares in single pass. No intermediate v^2
  // allocation; reads v once, accumulates sum of v[i]^2 directly.
  void vDSP_svesqD(const double *__A, vDSP_Stride __IA,
                   double *__C,
                   vDSP_Length __N);
  // vForce elementwise transcendental kernels (Tier 2c).
  // Signature convention: vv<fn>(double *out, const double *in, const int *n)
  void vvcos (double *__y, const double *__x, const int *__n);
  void vvsin (double *__y, const double *__x, const int *__n);
  void vvexp (double *__y, const double *__x, const int *__n);
  void vvlog (double *__y, const double *__x, const int *__n);
  void vvtanh(double *__y, const double *__x, const int *__n);
  void vvsqrt(double *__y, const double *__x, const int *__n);
  // vForce elementwise power: z[i] = x[i] ^ y[i]  (y = exponents, x = bases).
  void vvpow (double *__z, const double *__y, const double *__x, const int *__n);
  // vDSP_sveD: sum of all elements in a single pass.
  void vDSP_sveD(const double *__A, vDSP_Stride __IA,
                 double *__C, vDSP_Length __N);
}
#endif

// Internal helper for vForce kernel dispatch (called from each wrapper).
#ifdef __APPLE__
static inline Rcpp::NumericVector dat_vv_dispatch(
    Rcpp::NumericVector v,
    void (*kernel)(double *, const double *, const int *)) {
  R_xlen_t n_xl = v.size();
  Rcpp::NumericVector out(Rcpp::no_init(n_xl));
  if (n_xl > 0) {
    int n_int = static_cast<int>(n_xl);
    kernel(REAL(out), REAL(v), &n_int);
  }
  return out;
}
#endif

//' Fast scalar-vector multiplication via Apple Accelerate vDSP
//'
//' Elementwise product of scalar `s` and numeric vector `v`, computed
//' through `vDSP_vsmulD` from Apple Accelerate framework (hand-tuned NEON
//' SIMD). On non-macOS platforms an error is raised. Used internally by
//' \code{grad.function} when the symbolic gradient AST matches the pattern
//' `<numeric_literal_scalar> * <variable_symbol>`.
//'
//' @param s Single finite numeric scalar.
//' @param v Numeric vector (any length, including zero).
//' @return Numeric vector of the same length as `v`, with each element equal
//'   to `s * v[i]` within `4 * .Machine$double.eps` of the R-native product.
//' @export
// [[Rcpp::export]]
Rcpp::NumericVector fast_scalar_mul(double s, Rcpp::NumericVector v) {
#ifdef __APPLE__
  R_xlen_t n = v.size();
  Rcpp::NumericVector out(Rcpp::no_init(n));
  if (n > 0) {
    vDSP_vsmulD(REAL(v), 1, &s, REAL(out), 1, static_cast<vDSP_Length>(n));
  }
  return out;
#else
  Rcpp::stop("fast_scalar_mul: Apple Accelerate backend not available on this platform (input length %td, scalar %g)",
             static_cast<std::ptrdiff_t>(v.size()), s);
#endif
}

//' Fast sum-of-squares via Apple Accelerate vDSP
//'
//' Computes sum(v^2) in single pass without intermediate v^2 allocation
//' via vDSP_svesqD. Used internally by grad.function when the outer scalar
//' factor of a composite gradient contains sum(v^2).
//'
//' @param v Numeric vector.
//' @return Scalar double equal to sum(v^2).
//' @export
// ---- Threaded reduction support (add-threaded-reduction-kernels, #2) -------
// Read an R option as a double, returning `dflt` when unset/NA. Never raises.
static double dat_opt_double(const char* name, double dflt) {
  SEXP val = Rf_GetOption1(Rf_install(name));
  if (val == R_NilValue) return dflt;
  double x = Rf_asReal(val);
  return ISNAN(x) ? dflt : x;
}

// Worker count for threaded reductions: DefDiff.jit_threads option, else
// min(8, hardware_concurrency()). Always >= 1.
static int dat_reduce_workers() {
  double t = dat_opt_double("DefDiff.jit_threads", -1.0);
  int nt;
  if (t >= 1.0) {
    nt = static_cast<int>(t);
  } else {
    unsigned hw = std::thread::hardware_concurrency();
    nt = (hw == 0u) ? 4 : static_cast<int>(std::min(hw, 8u));
  }
  return nt < 1 ? 1 : nt;
}

// Thread the reduction only at/above DefDiff.reduce_threshold (default 1e7)
// with > 1 worker; below that the single-threaded vDSP path runs unchanged.
static bool dat_reduce_threaded(R_xlen_t n) {
  double thr = dat_opt_double("DefDiff.reduce_threshold", 1e7);
  return static_cast<double>(n) >= thr && dat_reduce_workers() > 1;
}

// [[Rcpp::export]]
double fast_sum_sq(Rcpp::NumericVector v) {
#ifdef __APPLE__
  R_xlen_t n = v.size();
  if (n == 0) return 0.0;
  const double* vp = REAL(v);
  if (!dat_reduce_threaded(n)) {
    double result = 0.0;
    vDSP_svesqD(vp, 1, &result, static_cast<vDSP_Length>(n));
    return result;
  }
  // Parallel partial-reduction: each worker runs vDSP_svesqD on a disjoint
  // slice into its own partial; the main thread sums the partials.
  int nt = dat_reduce_workers();
  if (static_cast<R_xlen_t>(nt) > n) nt = static_cast<int>(n);
  std::vector<double> partials(static_cast<size_t>(nt), 0.0);
  std::vector<std::thread> workers;
  workers.reserve(static_cast<size_t>(nt));
  R_xlen_t chunk = (n + nt - 1) / nt;
  for (int t = 0; t < nt; ++t) {
    R_xlen_t lo = static_cast<R_xlen_t>(t) * chunk;
    R_xlen_t hi = std::min(n, lo + chunk);
    if (lo >= hi) break;
    workers.emplace_back([=, &partials]() {
      double p = 0.0;
      vDSP_svesqD(vp + lo, 1, &p, static_cast<vDSP_Length>(hi - lo));
      partials[static_cast<size_t>(t)] = p;
    });
  }
  for (auto& w : workers) w.join();
  double result = 0.0;
  for (double p : partials) result += p;
  return result;
#else
  Rcpp::stop("fast_sum_sq: Apple Accelerate backend not available on this platform");
#endif
}

//' Fast sum-of-powers via Apple Accelerate vForce + vDSP
//'
//' Computes \code{sum(v^k)} for an integer power \code{k >= 2} in two passes:
//' \code{vvpow} forms the elementwise power into a scratch buffer, then
//' \code{vDSP_sveD} sums it. Generalizes \code{fast_sum_sq} (the k = 2 special
//' case) to arbitrary integer powers; the gradient side (k * v^(k-1)) is
//' handled separately by the scalar-power dispatcher. SIMD accumulation order
//' differs from R's stock \code{sum}, so the result matches to within ~1e-10
//' relative. On non-macOS platforms an error is raised.
//'
//' @param v Numeric vector.
//' @param k Integer power (>= 2).
//' @return Scalar double equal to sum(v^k).
//' @export
// [[Rcpp::export]]
double fast_sum_pow(Rcpp::NumericVector v, int k) {
#ifdef __APPLE__
  R_xlen_t n = v.size();
  if (n == 0) return 0.0;
  const double* vp = REAL(v);
  const double kd = static_cast<double>(k);
  if (!dat_reduce_threaded(n)) {
    int n_int = static_cast<int>(n);
    std::vector<double> powed(static_cast<size_t>(n));
    std::vector<double> expo(static_cast<size_t>(n), kd);
    vvpow(powed.data(), expo.data(), vp, &n_int);   // powed[i] = v[i]^k
    double result = 0.0;
    vDSP_sveD(powed.data(), 1, &result, static_cast<vDSP_Length>(n));
    return result;
  }
  // Parallel partial-reduction: each worker allocates its own per-slice scratch
  // for the vvpow output, then sums its slice; the main thread sums the partials.
  int nt = dat_reduce_workers();
  if (static_cast<R_xlen_t>(nt) > n) nt = static_cast<int>(n);
  std::vector<double> partials(static_cast<size_t>(nt), 0.0);
  std::vector<std::thread> workers;
  workers.reserve(static_cast<size_t>(nt));
  R_xlen_t chunk = (n + nt - 1) / nt;
  for (int t = 0; t < nt; ++t) {
    R_xlen_t lo = static_cast<R_xlen_t>(t) * chunk;
    R_xlen_t hi = std::min(n, lo + chunk);
    if (lo >= hi) break;
    workers.emplace_back([=, &partials]() {
      int m = static_cast<int>(hi - lo);
      std::vector<double> powed(static_cast<size_t>(m));
      std::vector<double> expo(static_cast<size_t>(m), kd);
      vvpow(powed.data(), expo.data(), vp + lo, &m);
      double p = 0.0;
      vDSP_sveD(powed.data(), 1, &p, static_cast<vDSP_Length>(m));
      partials[static_cast<size_t>(t)] = p;
    });
  }
  for (auto& w : workers) w.join();
  double result = 0.0;
  for (double p : partials) result += p;
  return result;
#else
  Rcpp::stop("fast_sum_pow: Apple Accelerate backend not available on this platform");
#endif
}

//' vForce cosine (Tier 2c)
//' @param v Numeric vector
//' @return Numeric vector with cos(v) elementwise via Apple Accelerate vvcos
//' @export
// [[Rcpp::export]]
Rcpp::NumericVector fast_vv_cos(Rcpp::NumericVector v) {
#ifdef __APPLE__
  return dat_vv_dispatch(v, vvcos);
#else
  Rcpp::stop("fast_vv_cos: Apple Accelerate vForce not available");
#endif
}

//' vForce sine (Tier 2c)
//' @param v Numeric vector
//' @return Numeric vector with sin(v) elementwise via Apple Accelerate vvsin
//' @export
// [[Rcpp::export]]
Rcpp::NumericVector fast_vv_sin(Rcpp::NumericVector v) {
#ifdef __APPLE__
  return dat_vv_dispatch(v, vvsin);
#else
  Rcpp::stop("fast_vv_sin: Apple Accelerate vForce not available");
#endif
}

//' vForce exponential (Tier 2c)
//' @param v Numeric vector
//' @return Numeric vector with exp(v) elementwise via Apple Accelerate vvexp
//' @export
// [[Rcpp::export]]
Rcpp::NumericVector fast_vv_exp(Rcpp::NumericVector v) {
#ifdef __APPLE__
  return dat_vv_dispatch(v, vvexp);
#else
  Rcpp::stop("fast_vv_exp: Apple Accelerate vForce not available");
#endif
}

//' vForce logarithm (Tier 2c)
//' @param v Numeric vector
//' @return Numeric vector with log(v) elementwise via Apple Accelerate vvlog
//' @export
// [[Rcpp::export]]
Rcpp::NumericVector fast_vv_log(Rcpp::NumericVector v) {
#ifdef __APPLE__
  return dat_vv_dispatch(v, vvlog);
#else
  Rcpp::stop("fast_vv_log: Apple Accelerate vForce not available");
#endif
}

//' vForce hyperbolic tangent (Tier 2c)
//' @param v Numeric vector
//' @return Numeric vector with tanh(v) elementwise via Apple Accelerate vvtanh
//' @export
// [[Rcpp::export]]
Rcpp::NumericVector fast_vv_tanh(Rcpp::NumericVector v) {
#ifdef __APPLE__
  return dat_vv_dispatch(v, vvtanh);
#else
  Rcpp::stop("fast_vv_tanh: Apple Accelerate vForce not available");
#endif
}

//' vForce square root (Tier 2c)
//' @param v Numeric vector
//' @return Numeric vector with sqrt(v) elementwise via Apple Accelerate vvsqrt
//' @export
// [[Rcpp::export]]
Rcpp::NumericVector fast_vv_sqrt(Rcpp::NumericVector v) {
#ifdef __APPLE__
  return dat_vv_dispatch(v, vvsqrt);
#else
  Rcpp::stop("fast_vv_sqrt: Apple Accelerate vForce not available");
#endif
}

#ifdef __APPLE__
extern "C" {
  // vDSP scalar-vector divide (Tier 2e). Computes out[i] = A / B[i] where
  // A is the scalar and B is the input vector.
  void vDSP_svdivD(const double *__A,
                   const double *__B, vDSP_Stride __IB,
                   double *__C, vDSP_Stride __IC,
                   vDSP_Length __N);
  // vDSP vector-vector arithmetic (Tier 2b).
  void vDSP_vaddD(const double *__A, vDSP_Stride __IA,
                  const double *__B, vDSP_Stride __IB,
                  double *__C, vDSP_Stride __IC,
                  vDSP_Length __N);
  void vDSP_vsubD(const double *__B, vDSP_Stride __IB,
                  const double *__A, vDSP_Stride __IA,
                  double *__C, vDSP_Stride __IC,
                  vDSP_Length __N);  // computes A - B (note: vsub reverses)
  void vDSP_vsmaD(const double *__A, vDSP_Stride __IA,
                  const double *__B,
                  const double *__C, vDSP_Stride __IC,
                  double *__D, vDSP_Stride __ID,
                  vDSP_Length __N);  // D = B * A + C (scalar B times vector A plus vector C)
  // vDSP vector-vector elementwise multiply (Tier 3 fix 3b).
  void vDSP_vmulD(const double *__A, vDSP_Stride __IA,
                  const double *__B, vDSP_Stride __IB,
                  double *__C, vDSP_Stride __IC,
                  vDSP_Length __N);
  // vDSP vector-vector elementwise divide (Tier 4 family-2).
  // *** REVERSED-ARGUMENT CONVENTION ***: vDSP_vdivD's signature is
  //   (B, IB, A, IA, C, IC, N)  computing  C = A / B
  // The DIVIDEND is the SECOND vDSP arg, DIVISOR is the FIRST. The fast_vec_div
  // wrapper inverts this so callers can write fast_vec_div(num, denom)
  // intuitively. See fast_vec_div for the swap.
  void vDSP_vdivD(const double *__B, vDSP_Stride __IB,
                  const double *__A, vDSP_Stride __IA,
                  double *__C, vDSP_Stride __IC,
                  vDSP_Length __N);
}
#endif

//' Fast vector-vector add via Apple Accelerate vDSP (Tier 2b)
//' @param v Numeric vector
//' @param w Numeric vector of same length
//' @return Numeric vector v + w (elementwise)
//' @export
// [[Rcpp::export]]
Rcpp::NumericVector fast_vec_add(Rcpp::NumericVector v, Rcpp::NumericVector w) {
#ifdef __APPLE__
  R_xlen_t n = v.size();
  if (w.size() != n) Rcpp::stop("fast_vec_add: length mismatch");
  Rcpp::NumericVector out(Rcpp::no_init(n));
  if (n > 0) vDSP_vaddD(REAL(v), 1, REAL(w), 1, REAL(out), 1, static_cast<vDSP_Length>(n));
  return out;
#else
  Rcpp::stop("fast_vec_add: Apple Accelerate not available");
#endif
}

//' Fast vector-vector subtract via Apple Accelerate vDSP (Tier 2b)
//' @param v Numeric vector (minuend)
//' @param w Numeric vector of same length (subtrahend)
//' @return Numeric vector v - w (elementwise)
//' @export
// [[Rcpp::export]]
Rcpp::NumericVector fast_vec_sub(Rcpp::NumericVector v, Rcpp::NumericVector w) {
#ifdef __APPLE__
  R_xlen_t n = v.size();
  if (w.size() != n) Rcpp::stop("fast_vec_sub: length mismatch");
  Rcpp::NumericVector out(Rcpp::no_init(n));
  // vDSP_vsubD computes A - B where signature is (B, A, out): C[i] = A[i] - B[i]
  // To compute v - w, pass w as first arg (B) and v as second (A).
  if (n > 0) vDSP_vsubD(REAL(w), 1, REAL(v), 1, REAL(out), 1, static_cast<vDSP_Length>(n));
  return out;
#else
  Rcpp::stop("fast_vec_sub: Apple Accelerate not available");
#endif
}

//' Fast scalar-multiply-add via Apple Accelerate vDSP (Tier 2b)
//' @param s Scalar
//' @param v Numeric vector
//' @param w Numeric vector of same length as v
//' @return Numeric vector s * v + w (elementwise, fused single-pass via vDSP_vsmaD)
//' @export
// [[Rcpp::export]]
Rcpp::NumericVector fast_vec_smadd(double s, Rcpp::NumericVector v, Rcpp::NumericVector w) {
#ifdef __APPLE__
  R_xlen_t n = v.size();
  if (w.size() != n) Rcpp::stop("fast_vec_smadd: length mismatch");
  Rcpp::NumericVector out(Rcpp::no_init(n));
  if (n > 0) vDSP_vsmaD(REAL(v), 1, &s, REAL(w), 1, REAL(out), 1, static_cast<vDSP_Length>(n));
  return out;
#else
  Rcpp::stop("fast_vec_smadd: Apple Accelerate not available");
#endif
}

//' Fast scalar-divide-vector via Apple Accelerate vDSP (Tier 2e)
//' @param s Scalar numerator
//' @param v Numeric vector denominator (each element used as divisor)
//' @return Numeric vector where result[i] = s / v[i]
//' @export
// [[Rcpp::export]]
Rcpp::NumericVector fast_scalar_div(double s, Rcpp::NumericVector v) {
#ifdef __APPLE__
  R_xlen_t n = v.size();
  Rcpp::NumericVector out(Rcpp::no_init(n));
  if (n > 0) {
    vDSP_svdivD(&s, REAL(v), 1, REAL(out), 1, static_cast<vDSP_Length>(n));
  }
  return out;
#else
  Rcpp::stop("fast_scalar_div: Apple Accelerate not available");
#endif
}

//' Fast vector-vector elementwise multiply via Apple Accelerate vDSP (Tier 3)
//' @param v Numeric vector
//' @param w Numeric vector of same length
//' @return Numeric vector with v[i] * w[i] elementwise
//' @export
// [[Rcpp::export]]
Rcpp::NumericVector fast_vec_mul(Rcpp::NumericVector v, Rcpp::NumericVector w) {
#ifdef __APPLE__
  R_xlen_t n = v.size();
  if (w.size() != n) Rcpp::stop("fast_vec_mul: length mismatch");
  Rcpp::NumericVector out(Rcpp::no_init(n));
  if (n > 0) vDSP_vmulD(REAL(v), 1, REAL(w), 1, REAL(out), 1, static_cast<vDSP_Length>(n));
  return out;
#else
  Rcpp::stop("fast_vec_mul: Apple Accelerate not available");
#endif
}

//' Fast vector-vector elementwise divide via Apple Accelerate vDSP (Tier 4 family-2)
//'
//' Computes the elementwise quotient `numerator / denominator`. Wrapper around
//' `vDSP_vdivD`.
//'
//' *** Argument-order note ***: vDSP_vdivD itself takes (denominator, numerator,
//' out, n) — the dividend is the SECOND vDSP arg, divisor is the FIRST. This
//' wrapper inverts the convention so callers pass `(numerator, denominator)`
//' intuitively. A regression would manifest as wrong-direction division
//' (e.g., `fast_vec_div(c(6, 8, 10), c(2, 4, 5))` returning `c(0.333, 0.5, 0.5)`
//' instead of `c(3, 2, 2)`).
//'
//' @param numerator Numeric vector (the dividend)
//' @param denominator Numeric vector of same length as numerator (the divisor)
//' @return Numeric vector with `numerator[i] / denominator[i]` elementwise
//' @export
// [[Rcpp::export]]
Rcpp::NumericVector fast_vec_div(Rcpp::NumericVector numerator,
                                 Rcpp::NumericVector denominator) {
#ifdef __APPLE__
  R_xlen_t n = numerator.size();
  if (denominator.size() != n) Rcpp::stop("fast_vec_div: length mismatch");
  Rcpp::NumericVector out(Rcpp::no_init(n));
  // SWAP: vDSP expects (divisor, dividend, out). We pass denominator AS divisor
  // (first vDSP arg) and numerator AS dividend (second vDSP arg).
  if (n > 0) vDSP_vdivD(REAL(denominator), 1, REAL(numerator), 1,
                        REAL(out), 1, static_cast<vDSP_Length>(n));
  return out;
#else
  Rcpp::stop("fast_vec_div: Apple Accelerate not available");
#endif
}

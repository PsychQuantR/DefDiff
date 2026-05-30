## verify.R
## Three-layer verification of a candidate gradient.

#' Verify a candidate gradient through three independent checks
#'
#' Runs three layered checks against a candidate gradient `gf` of a scalar
#' function `f`:
#' 1. **Syntactic** — `level(gf)` is at most `level(f)` (closure check).
#' 2. **Numeric** — sample random vectors and compare `gf(v)` against a
#'    centered finite-difference approximation of `f` at `v`.
#' 3. **Cross-strategy** — recompute the gradient via the `Deriv` package
#'    (if available) and compare on the same sample points.
#'
#' @param f A function: the scalar-valued original (vector argument).
#' @param gf A function: the candidate gradient (same argument signature).
#' @param n_samples Integer. Number of random vectors to evaluate at.
#' @param sample_dim Integer. Dimension of the sample vectors.
#' @param tol Numeric. Tolerance for numeric agreement.
#' @return An S3 object of class `DefDiff_verify_result` with named slots
#'   `$syntactic`, `$numeric`, `$cross_strategy`.
#' @export
#' @examples
#' f  <- function(v) sum(v^2)
#' gf <- function(v) 2 * v
#' result <- verify_grad(f, gf)
#' print(result)
verify_grad <- function(f, gf, n_samples = 100L, sample_dim = 3L, tol = 1e-6) {
  if (!is.function(f) || !is.function(gf)) {
    .dat_stop("DefDiff_not_definable",
              "verify_grad() requires both `f` and `gf` to be functions.")
  }
  set.seed(1234L)
  samples <- matrix(stats::runif(n_samples * sample_dim, -1, 1),
                    nrow = n_samples, ncol = sample_dim)
  result <- list(
    syntactic      = .verify_syntactic(f, gf),
    numeric        = .verify_numeric(f, gf, samples, tol),
    cross_strategy = .verify_cross_strategy(f, gf, samples, tol)
  )
  structure(result, class = "DefDiff_verify_result")
}

# Layer 1: level(gf) is at most level(f).
.verify_syntactic <- function(f, gf) {
  lf  <- level(f)
  lgf <- level(gf)
  rank_f  <- .level_rank(lf)
  rank_gf <- .level_rank(lgf)
  pass <- !is.na(rank_f) && !is.na(rank_gf) && rank_gf <= rank_f
  list(pass = pass, level_of_f = lf, level_of_gf = lgf)
}

# Layer 2: numeric finite-difference comparison.
.verify_numeric <- function(f, gf, samples, tol) {
  n     <- nrow(samples)
  d     <- ncol(samples)
  eps   <- 1e-5
  max_err <- 0
  for (i in seq_len(n)) {
    v <- samples[i, ]
    fd <- .central_difference(f, v, eps)
    ag <- tryCatch(as.numeric(gf(v)),
                   error = function(e) rep(NA_real_, d))
    if (any(is.na(ag)) || length(ag) != d) {
      return(list(pass = FALSE, max_abs_error = NA_real_,
                  reason = "gf() returned wrong shape or errored"))
    }
    err <- max(abs(ag - fd))
    if (err > max_err) max_err <- err
  }
  list(pass = max_err < tol, max_abs_error = max_err)
}

.central_difference <- function(f, v, eps) {
  d <- length(v)
  out <- numeric(d)
  for (j in seq_len(d)) {
    v_plus  <- v; v_plus[j]  <- v_plus[j]  + eps
    v_minus <- v; v_minus[j] <- v_minus[j] - eps
    out[j] <- (f(v_plus) - f(v_minus)) / (2 * eps)
  }
  out
}

# Layer 3: cross-strategy oracle via Deriv package, if available.
.verify_cross_strategy <- function(f, gf, samples, tol) {
  if (!requireNamespace("Deriv", quietly = TRUE)) {
    return(list(status = "skipped", reason = "Deriv not installed"))
  }
  oracle <- tryCatch(Deriv::Deriv(f), error = function(e) NULL)
  if (is.null(oracle)) {
    return(list(status = "skipped", reason = "Deriv::Deriv() failed on f"))
  }
  n <- nrow(samples); d <- ncol(samples)
  max_err <- 0
  for (i in seq_len(n)) {
    v <- samples[i, ]
    ag  <- tryCatch(as.numeric(gf(v)),     error = function(e) rep(NA_real_, d))
    org <- tryCatch(as.numeric(oracle(v)), error = function(e) rep(NA_real_, d))
    if (any(is.na(ag)) || any(is.na(org)) || length(ag) != length(org)) {
      return(list(status = "error", reason = "shape mismatch with Deriv oracle"))
    }
    err <- max(abs(ag - org))
    if (err > max_err) max_err <- err
  }
  list(pass = max_err < tol, oracle = "Deriv", max_abs_error = max_err)
}

#' Print method for `DefDiff_verify_result`
#'
#' @param x A `DefDiff_verify_result` object returned by `verify_grad()`.
#' @param ... Ignored.
#' @return Invisibly returns `x`.
#' @export
print.DefDiff_verify_result <- function(x, ...) {
  cat("DAT gradient verification result\n")
  cat("================================\n")
  s <- x$syntactic
  cat(sprintf("Syntactic: %s  (level(f) = %s, level(gf) = %s)\n",
              if (s$pass) "PASS" else "FAIL", s$level_of_f, s$level_of_gf))
  n <- x$numeric
  if (n$pass) {
    cat(sprintf("Numeric: PASS  (max abs error = %.3g)\n", n$max_abs_error))
  } else {
    cat(sprintf("Numeric: FAIL  (max abs error = %s%s)\n",
                if (is.na(n$max_abs_error)) "NA" else sprintf("%.3g", n$max_abs_error),
                if (!is.null(n$reason)) paste0(" - ", n$reason) else ""))
  }
  c <- x$cross_strategy
  if (!is.null(c$status) && c$status == "skipped") {
    cat(sprintf("Cross-strategy: SKIPPED  (%s)\n", c$reason))
  } else if (!is.null(c$status) && c$status == "error") {
    cat(sprintf("Cross-strategy: ERROR  (%s)\n", c$reason))
  } else if (isTRUE(c$pass)) {
    cat(sprintf("Cross-strategy: PASS  (oracle = %s, max abs error = %.3g)\n",
                c$oracle, c$max_abs_error))
  } else {
    cat(sprintf("Cross-strategy: FAIL  (oracle = %s, max abs error = %.3g)\n",
                c$oracle, c$max_abs_error))
  }
  invisible(x)
}

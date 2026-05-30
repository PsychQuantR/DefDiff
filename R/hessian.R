## hessian.R
## Higher-order derivative operator implementing closure thesis
## `hessian : L_i -> L_{i-2}` on top of grad. A fast-path dispatcher handles
## separable forms (`sum(<elementwise>(v))`), quadratic forms
## (`crossprod(v, A %*% v)`), and scaled quadratics with exact closed forms.
## Bodies that match no fast-path pattern fall through to the recursive
## Jacobian-of-gradient walker (`.hessian_recursive` in hessian_inner.R),
## which operationally closes the Hessian over the catalog already closed by
## the grad engine. Genuinely unsupported shapes raise hessian_not_supported.

#' Symbolic Hessian (S3 generic)
#'
#' Compute the symbolic Hessian of a scalar-valued function of a vector
#' variable. Returns a function whose evaluation produces an n×n numeric
#' matrix where n = length of the input vector. A fast-path table handles
#' separable, quadratic, and scaled-quadratic forms with exact closed forms;
#' other bodies fall through to a recursive Jacobian-of-gradient walker.
#' Shapes the walker cannot construct raise `hessian_not_supported`.
#'
#' @param x An R function with a single vector argument returning a scalar.
#' @param vars Character vector of variable names; if NULL, derived from
#'   `names(formals(x))`. Must have length 1 in v0.0.3.
#' @param ... Reserved for future use.
#' @return An R function with the same formals as `x` that, when called on
#'   a numeric vector, returns the Hessian matrix.
#' @export
#' @examples
#' hf <- hessian(function(v) sum(v^2))
#' hf(c(1, 2, 3))             # 3x3 diagonal with 2s
#'
#' hf2 <- hessian(function(v) sum(sin(v)))
#' hf2(c(0.5, 1.0, 1.5))      # 3x3 diagonal -sin(v)
hessian <- function(x, vars = NULL, ...) {
  UseMethod("hessian")
}

#' @export
#' @rdname hessian
hessian.default <- function(x, vars = NULL, ...) {
  .dat_stop(
    "DefDiff_not_definable",
    paste0("hessian() does not have a method for class ",
           paste(class(x), collapse = "/"))
  )
}

#' @export
#' @rdname hessian
hessian.function <- function(x, vars = NULL, ...) {
  if (is.null(vars)) vars <- names(formals(x))

  body_expr <- body(x)
  blocker <- .control_flow_block(body_expr)
  if (!is.na(blocker)) {
    .dat_stop("DefDiff_not_definable",
              paste0("Function body contains unsupported construct `", blocker,
                     "`; only straight-line scalar expressions are supported."))
  }
  body_expr <- .strip_paren(body_expr)

  if (length(vars) == 1L) {
    pattern <- .recognize_hessian_pattern(body_expr, vars)
    if (is.null(pattern)) {
      # Fast-path dispatcher found no match: fall through to the recursive
      # Jacobian-of-gradient walker (add-hessian-recursive-walker). It raises
      # hessian_not_supported when the gradient is computable but its shape is
      # outside the recognized rules, and propagates the grad engine's condition
      # (DefDiff_not_definable / DefDiff_unknown_generator) when the gradient itself is
      # unsupported.
      hexpr <- .hessian_recursive(body_expr, vars)
    } else {
      hexpr <- .construct_hessian_body(pattern, vars, environment(x))
    }
  } else {
    # Multi-variable block Hessian (add-multi-variable-hessian): assemble a
    # named list of named lists, where block [[a]][[b]] is the Jacobian of the
    # per-variable gradient grad_a(f) w.r.t. b. Each block is constructed by the
    # generalized walker, which treats the other vector variables as constants.
    # The fast-path single-variable dispatcher is skipped.
    build_row <- function(a) {
      blocks <- lapply(vars, function(b) .hessian_block(body_expr, a, b, vars))
      names(blocks) <- vars
      as.call(c(list(quote(list)), blocks))
    }
    rows <- lapply(vars, build_row)
    names(rows) <- vars
    hexpr <- as.call(c(list(quote(list)), rows))
  }

  new_fn <- function() NULL
  formals(new_fn) <- formals(x)
  body(new_fn) <- hexpr
  environment(new_fn) <- environment(x)
  new_fn
}

# .recognize_hessian_pattern(body_expr, var)
#
# Pattern dispatcher. Returns one of:
#   list(kind = "elementwise", inner_fn = <name>, k = <int_or_NA>)
#   list(kind = "quadratic_form", matrix_expr = <expr>)
#   list(kind = "scaled_quadratic", scalar = <num>)
# or NULL if no pattern matches.
.recognize_hessian_pattern <- function(body_expr, var) {
  body_expr <- .strip_paren(body_expr)

  # Detect scaled quadratic: `c * sum(v^2)` or `sum(v^2) * c` (literal c)
  if (is.call(body_expr) && length(body_expr) == 3L &&
      identical(body_expr[[1L]], as.name("*"))) {
    lhs <- .strip_paren(body_expr[[2L]])
    rhs <- .strip_paren(body_expr[[3L]])
    if (is.numeric(lhs) && length(lhs) == 1L && is.finite(lhs) &&
        .is_sum_sq(rhs, var)) {
      return(list(kind = "scaled_quadratic", scalar = as.double(lhs)))
    }
    if (is.numeric(rhs) && length(rhs) == 1L && is.finite(rhs) &&
        .is_sum_sq(lhs, var)) {
      return(list(kind = "scaled_quadratic", scalar = as.double(rhs)))
    }
  }

  # Detect quadratic form: `crossprod(v, A %*% v)`
  if (is.call(body_expr) && length(body_expr) == 3L &&
      identical(body_expr[[1L]], as.name("crossprod"))) {
    a1 <- .strip_paren(body_expr[[2L]])
    a2 <- .strip_paren(body_expr[[3L]])
    if (is.symbol(a1) && identical(as.character(a1), var) &&
        is.call(a2) && length(a2) == 3L &&
        identical(a2[[1L]], as.name("%*%"))) {
      inner_lhs <- .strip_paren(a2[[2L]])
      inner_rhs <- .strip_paren(a2[[3L]])
      if (is.symbol(inner_rhs) && identical(as.character(inner_rhs), var)) {
        return(list(kind = "quadratic_form", matrix_expr = inner_lhs))
      }
    }
  }

  # Detect elementwise: `sum(<atom>(v))` or `sum(v^k)`
  if (is.call(body_expr) && length(body_expr) == 2L &&
      identical(body_expr[[1L]], as.name("sum"))) {
    inner <- .strip_paren(body_expr[[2L]])

    # sum(v^k)
    if (is.call(inner) && length(inner) == 3L &&
        identical(inner[[1L]], as.name("^"))) {
      base <- .strip_paren(inner[[2L]])
      exponent <- .strip_paren(inner[[3L]])
      if (is.symbol(base) && identical(as.character(base), var) &&
          is.numeric(exponent) && length(exponent) == 1L &&
          is.finite(exponent) && exponent >= 2L &&
          exponent == as.integer(exponent)) {
        return(list(kind = "elementwise", inner_fn = "^",
                    k = as.integer(exponent)))
      }
    }

    # sum(<fn>(v)) for fn in {sin, cos, exp, log, tanh}
    if (is.call(inner) && length(inner) == 2L && is.symbol(inner[[1L]])) {
      fn_name <- as.character(inner[[1L]])
      if (fn_name %in% c("sin", "cos", "exp", "log", "tanh")) {
        arg <- .strip_paren(inner[[2L]])
        if (is.symbol(arg) && identical(as.character(arg), var)) {
          return(list(kind = "elementwise", inner_fn = fn_name, k = NA_integer_))
        }
      }
    }
  }

  NULL
}

# .is_sum_sq(expr, var): TRUE iff expr is `sum(v^2)` with given var.
.is_sum_sq <- function(expr, var) {
  expr <- .strip_paren(expr)
  if (!is.call(expr) || length(expr) != 2L) return(FALSE)
  if (!identical(expr[[1L]], as.name("sum"))) return(FALSE)
  inner <- .strip_paren(expr[[2L]])
  if (!is.call(inner) || length(inner) != 3L) return(FALSE)
  if (!identical(inner[[1L]], as.name("^"))) return(FALSE)
  base <- .strip_paren(inner[[2L]])
  exponent <- .strip_paren(inner[[3L]])
  is.symbol(base) && identical(as.character(base), var) &&
    is.numeric(exponent) && length(exponent) == 1L && exponent == 2
}

# .diag_vec(d_expr, vsym): build a diag() call whose dimension is pinned to
# length(vsym). Without the explicit nrow, a length-1 input makes diag() treat
# its single value as a *dimension* (e.g. diag(c(2)) -> 2x2, diag(c(0.6)) -> 0x0)
# rather than a 1x1 diagonal matrix. `diag(vec, nrow = n)` is identical to
# `diag(vec)` for n >= 2, so this only corrects the n=1 case.
.diag_vec <- function(d_expr, vsym) {
  bquote(diag(.(d_expr), nrow = length(.(vsym))))
}

# .construct_hessian_body(pattern, var, env): builds the new function's body
# expression based on the recognized pattern kind.
.construct_hessian_body <- function(pattern, var, env) {
  vsym <- as.name(var)
  switch(
    pattern$kind,
    "elementwise" = {
      switch(
        pattern$inner_fn,
        "^" = {
          k <- pattern$k
          if (k == 2L) {
            .diag_vec(bquote(rep(2, length(.(vsym)))), vsym)
          } else {
            .diag_vec(bquote(.(k * (k - 1L)) * .(vsym)^.(k - 2L)), vsym)
          }
        },
        "sin"  = .diag_vec(bquote(-sin(.(vsym))), vsym),
        "cos"  = .diag_vec(bquote(-cos(.(vsym))), vsym),
        "exp"  = .diag_vec(bquote(exp(.(vsym))), vsym),
        "log"  = .diag_vec(bquote(-1 / .(vsym)^2), vsym),
        "tanh" = .diag_vec(bquote(-2 * tanh(.(vsym)) * (1 - tanh(.(vsym))^2)), vsym)
      )
    },
    "quadratic_form" = {
      mexpr <- pattern$matrix_expr
      bquote(.(mexpr) + t(.(mexpr)))
    },
    "scaled_quadratic" = {
      .diag_vec(bquote(rep(.(2 * pattern$scalar), length(.(vsym)))), vsym)
    }
  )
}

# .hessian_not_supported(body_expr): raise the hessian_not_supported condition.
.hessian_not_supported <- function(body_expr) {
  msg <- paste0(
    "hessian(): expression body `", deparse(body_expr)[[1L]],
    "` does not match any recognized hessian pattern. ",
    "Recognized patterns include sum(v^k), sum(sin(v)), sum(cos(v)), ",
    "sum(exp(v)), sum(log(v)), sum(tanh(v)), crossprod(v, A %*% v), ",
    "scaled quadratic forms, and scalar-denominator quotients such as ",
    "sum(v * exp(v)) / sum(exp(v)). Quotients with a vector-valued ",
    "denominator remain unsupported. See ?hessian."
  )
  cond <- structure(
    class = c("hessian_not_supported", "simpleError", "error", "condition"),
    list(message = msg, call = sys.call(-1L))
  )
  stop(cond)
}

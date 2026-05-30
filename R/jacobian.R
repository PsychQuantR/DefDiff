## jacobian.R
## Vector-output Jacobian (add-vector-output-jacobian, Option A: symbolic
## per-output-component). For f : R^n -> R^m whose body is an explicit assembly
## of m scalar-valued catalog components (primarily `c(g_1, ..., g_m)`), the
## Jacobian J[i, j] = d f_i / d v_j is the m grad rows `grad g_i` stacked by
## `rbind`. Differentiation is 100% delegated to the existing grad engine
## (`.grad_expr`); the only new machinery is the vector-assembly shape detector
## and the row stacker. No runtime tape — the returned body is a closed-form
## matrix AST over the declared catalog, one dimension wider than `grad`,
## mirroring the closed-form matrix AST `hessian` already produces.

#' Symbolic vector-output Jacobian (S3 generic)
#'
#' Compute the symbolic Jacobian of a vector-valued function of a vector
#' variable. For `f : R^n -> R^m` whose body is an explicit assembly of scalar
#' catalog components (e.g. `c(g_1, ..., g_m)`), returns a function evaluating
#' to the `m x n` matrix `J[i, j] = d f_i / d v_j`. A scalar-output body
#' degrades to a `1 x n` row. Stays closed-form (no runtime tape), consistent
#' with the closure thesis.
#'
#' @param x An R object: function or call (from `quote()`).
#' @param vars Character vector of variable names. For function input,
#'   defaults to `names(formals(x))`. Single variable in this version.
#' @param ... Reserved for future use.
#' @return The symbolic Jacobian. Type matches input (function -> function,
#'   call -> call).
#' @export
#' @examples
#' jf <- jacobian(function(v) c(sum(v), sum(v^2)))
#' jf(c(1, 2, 3))                       # rbind(c(1,1,1), c(2,4,6))
#'
#' jacobian(quote(c(sum(v), sum(v^2))), "v")
jacobian <- function(x, vars = NULL, ...) {
  UseMethod("jacobian")
}

#' @export
#' @rdname jacobian
jacobian.default <- function(x, vars = NULL, ...) {
  .dat_stop(
    "DefDiff_not_definable",
    paste0("jacobian() does not have a method for class ", paste(class(x), collapse = "/"))
  )
}

# .jacobian_not_supported(body_expr): raise the jacobian_not_supported condition
# for a vector-valued body that is not an explicit assembly of catalog-closable
# scalar components.
.jacobian_not_supported <- function(body_expr) {
  msg <- paste0(
    "jacobian(): body `", deparse(body_expr)[[1L]],
    "` is not an explicit assembly of catalog-closable scalar components. ",
    "Only `c(g_1, ..., g_m)` of scalar-output expressions (and a scalar-output ",
    "body, which degrades to a 1xn row) are supported; an implicit vector ",
    "output is out of scope. See ?jacobian."
  )
  cond <- structure(
    class = c("jacobian_not_supported", "simpleError", "error", "condition"),
    list(message = msg, call = sys.call(-1L))
  )
  stop(cond)
}

# .jacobian_rows(body_expr, var) — return a list of per-component gradient row
# ASTs (each a length-n vector AST from the grad engine). Dispatches on the
# vector-assembly shape:
#   - `c(g_1, ..., g_m)`  -> one row per component
#   - scalar-output body  -> single row (1xn degrade)
#   - otherwise (a vector-valued body that is not an explicit c(...) assembly)
#     -> jacobian_not_supported
# Out-of-catalog or control-flow components propagate the grad engine's
# condition (`DefDiff_not_definable` / `DefDiff_unknown_generator`) unchanged.
.jacobian_rows <- function(body_expr, var) {
  if (is.call(body_expr) && is.symbol(body_expr[[1L]]) &&
      identical(as.character(body_expr[[1L]]), "c")) {
    components <- as.list(body_expr)[-1L]
    return(lapply(components, function(g) .grad_expr(.strip_paren(g), var)))
  }
  # Not a c(...) assembly: only a scalar-valued body is a valid (degrading)
  # input. A vector-valued body that is not an explicit assembly is out of
  # scope. `.hess_shape` classifies the grain; "scalar" routes to the grad
  # engine (which itself raises dat_* for an out-of-catalog scalar body).
  if (.hess_shape(body_expr, var) == "scalar") {
    return(list(.grad_expr(body_expr, var)))
  }
  .jacobian_not_supported(body_expr)
}

# .jacobian_matrix_ast(rows, var) — stack per-component row ASTs into an
# m x n matrix AST. Each row is coerced to numeric so a constant-component
# gradient (a bare 0) recycles to the variable's length under rbind.
.jacobian_matrix_ast <- function(rows, var) {
  coerced <- lapply(rows, function(r) bquote(as.numeric(.(r))))
  as.call(c(list(quote(rbind)), coerced))
}

#' @export
#' @rdname jacobian
jacobian.function <- function(x, vars = NULL, ...) {
  if (is.null(vars)) vars <- names(formals(x))
  if (length(vars) != 1L) {
    .jacobian_not_supported(body(x))
  }
  var <- vars[[1L]]
  body_expr <- body(x)
  blocker <- .control_flow_block(body_expr)
  if (!is.na(blocker)) {
    .dat_stop("DefDiff_not_definable",
              paste0("Function body contains unsupported construct `", blocker,
                     "`; only straight-line vector-output expressions are supported."))
  }
  body_expr <- .strip_paren(body_expr)
  rows <- .jacobian_rows(body_expr, var)
  new_fn <- function() NULL
  formals(new_fn) <- formals(x)
  body(new_fn) <- .jacobian_matrix_ast(rows, var)
  environment(new_fn) <- environment(x)
  new_fn
}

#' @export
#' @rdname jacobian
jacobian.call <- function(x, vars, ...) {
  if (missing(vars) || is.null(vars)) {
    .dat_stop("DefDiff_not_definable",
              "jacobian() on a call requires `vars` (character vector of variable names).")
  }
  if (length(vars) != 1L) {
    .jacobian_not_supported(x)
  }
  blocker <- .control_flow_block(x)
  if (!is.na(blocker)) {
    .dat_stop("DefDiff_not_definable",
              paste0("Expression contains unsupported construct `", blocker, "`."))
  }
  rhs <- .strip_paren(x)
  rows <- .jacobian_rows(rhs, vars[[1L]])
  .jacobian_matrix_ast(rows, vars[[1L]])
}

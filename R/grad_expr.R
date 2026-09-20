## grad_expr.R
## Accessor: recover the symbolic gradient AST from a gradient function,
## independent of which evaluation backend its body uses.

#' Recover the symbolic gradient of a gradient function
#'
#' \code{\link{grad}} applied to a \code{function} returns a gradient
#' \emph{function}. On macOS its \code{body()} is typically a fast-path kernel
#' call (e.g. \code{fast_scalar_mul(2, v)}) rather than the readable symbolic
#' AST, so \code{body(gf)} is not a reliable way to inspect the gradient. The
#' symbolic gradient is preserved separately as an attribute regardless of the
#' evaluation backend (plain AST, vDSP/vForce kernel, Metal, or a future fused
#' evaluator); \code{grad_expr()} returns it.
#'
#' @param gf A gradient function returned by \code{grad(<function>)}.
#' @return The symbolic gradient AST: a single call (e.g. \code{2 * v}) for a
#'   one-variable gradient, or a named list of calls keyed by variable name for
#'   a multi-variable gradient.
#' @seealso \code{\link{grad}}
#' @export
#' @examples
#' gf <- grad(function(v) sum(v^3))
#' grad_expr(gf)                      # 3 * v^2  (even if body(gf) is a kernel call)
grad_expr <- function(gf) {
  e <- attr(gf, "grad_expr")
  if (is.null(e)) {
    .dat_stop(
      "DefDiff_not_gradient",
      "`gf` is not a gradient function from grad(<function>) (no grad_expr attribute)."
    )
  }
  e
}

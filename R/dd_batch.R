## dd_batch.R
## Reading-A convenience for the multi-dimensional batch protocol (#4):
## store a formula's gradient ONCE, apply it across inputs of differing
## dimension. This is sugar over grad() — the dimension-free reuse it relies on
## is already provided by the symbolic gradient (n-free) plus the fused-JIT
## warm-up cache whose key excludes `n` (one compiled kernel serves all lengths).

#' Store a formula's gradient once, evaluate it across many dimensions
#'
#' \code{dd_batch(f)} computes \code{\link{grad}(f)} a single time — storing the
#' dimension-free symbolic gradient and (on macOS) reusing a single compiled
#' kernel via the AST-hash cache, whose key excludes the vector length \code{n}.
#' It returns a function that maps that one stored gradient over a collection of
#' input vectors of possibly differing length, embodying "define the formula
#' once, apply it to any dimension".
#'
#' The returned function accepts either a list of numeric vectors or several
#' vectors passed directly (variadic). The stored symbolic gradient is attached
#' as the \code{grad_expr} attribute so it remains inspectable
#' (e.g. \code{attr(b, "grad_expr")} is \code{3 * v^2} for \code{sum(v^3)}).
#'
#' Scope: single-vector-argument scalar functions (the \code{f(x)} use case).
#'
#' @param f A scalar-valued function of one vector argument, as accepted by
#'   \code{\link{grad}}.
#' @param ... Passed through to \code{\link{grad}}.
#' @return A function \code{batch(...)} that returns a list of gradient vectors,
#'   one per input, reusing the single stored gradient across dimensions.
#' @seealso \code{\link{grad}}, \code{\link{grad_expr}}
#' @export
#' @examples
#' b <- dd_batch(function(v) sum(v^3))
#' b(c(1, 2, 3), c(0.5, -0.5))        # list( 3*v^2 for each input )
#' attr(b, "grad_expr")               # 3 * v^2 (stored once, dimension-free)
dd_batch <- function(f, ...) {
  gf <- grad(f, ...)                    # store the formula's gradient ONCE
  # Single-vector-argument contract: a multi-variable f makes grad() return a
  # function of >1 argument, so the lapply(inputs, gf) below would call it with a
  # single argument and silently return a wrong-shaped list (#4 verify HIGH).
  # Fail fast instead.
  if (length(formals(gf)) != 1L) {
    .dat_stop("DefDiff_not_definable",
              paste0("dd_batch supports single-vector-argument scalar functions; ",
                     "grad(f) takes ", length(formals(gf)), " arguments. ",
                     "For multi-variable gradients call grad() directly and map yourself."))
  }
  batch <- function(...) {
    inputs <- list(...)
    # Accept either dd_batch_fn(list(v1, v2)) or dd_batch_fn(v1, v2). Exclude
    # data.frame (is.list is TRUE for it) so a single df isn't column-unwrapped
    # into per-column "inputs" (#4 verify MEDIUM).
    if (length(inputs) == 1L && is.list(inputs[[1L]]) && !is.data.frame(inputs[[1L]]))
      inputs <- inputs[[1L]]
    if (length(inputs) == 0L) {
      .dat_stop("DefDiff_not_definable",
                "dd_batch's returned function expects at least one numeric vector.")
    }
    lapply(inputs, gf)                  # reuse the one stored gradient across dims
  }
  attr(batch, "grad_expr") <- attr(gf, "grad_expr")
  batch
}

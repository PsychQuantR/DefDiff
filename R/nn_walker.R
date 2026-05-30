## nn_walker.R
## Tier 5 Option B-lite (`add-nn-walker-extension`): focused walker shape
## extension for arbitrary-depth elementwise-matmul chains.
##
## `.elementwise_matmul_chain_grad(expr, var)` recursively walks expressions
## of the form  f_k(W_k %*% f_{k-1}(W_{k-1} %*% ... f_1(W_1 %*% v)))  and
## returns list(value, jacobian) where:
##   value    -- AST evaluating to the chain's m-vector value
##   jacobian -- AST evaluating to the m × n Jacobian d(value)/d(v)
##
## Each recursive case follows backprop chain rule:
##   bare v        -> jacobian sentinel (NULL); next %*% emits W directly
##   W %*% inner   -> (W %*% value_in, W %*% jac_in)
##   f(inner)      -> (f(value_in), f'(value_in) * jac_in)   [row-broadcast]
##   anything else -> raise DefDiff_not_definable (out of chain scope)
##
## Hooked from .sum_rule in generators.R; final sum collapses jacobian via
## colSums to produce the n-vector gradient.

.elementwise_matmul_chain_grad <- function(expr, var) {
  expr <- .strip_paren(expr)

  # Case 1: bare differentiation variable (sentinel)
  if (.is_var(expr, var)) {
    return(list(value = expr, jacobian = NULL))
  }

  if (!is.call(expr)) {
    .dat_stop("DefDiff_not_definable",
              paste0("elementwise_matmul_chain: leaf is not bare variable: ",
                     deparse(expr)))
  }

  head_sym <- expr[[1L]]
  if (!is.symbol(head_sym)) {
    .dat_stop("DefDiff_not_definable",
              "elementwise_matmul_chain: non-symbol function head")
  }

  # Case 2: matrix multiply W %*% inner
  if (identical(head_sym, quote(`%*%`)) && length(expr) == 3L) {
    W <- expr[[2L]]
    inner <- expr[[3L]]
    if (.contains_var(W, var)) {
      .dat_stop("DefDiff_not_definable",
                paste0("elementwise_matmul_chain: %*% first operand contains var: ",
                       deparse(W)))
    }
    sub <- .elementwise_matmul_chain_grad(inner, var)
    if (is.null(sub$jacobian)) {
      # Inner was bare v: new value = W %*% v, jacobian = W (no I_n materialization)
      return(list(value = bquote(.(W) %*% .(sub$value)), jacobian = W))
    }
    return(list(
      value = bquote(.(W) %*% .(sub$value)),
      jacobian = bquote(.(W) %*% .(sub$jacobian))
    ))
  }

  # Case 3: elementwise vForce wrap f(inner) for f in {sin, cos, exp, tanh}
  if (length(expr) == 2L) {
    fn_name <- as.character(head_sym)
    if (fn_name %in% c("sin", "cos", "exp", "tanh")) {
      sub <- .elementwise_matmul_chain_grad(expr[[2L]], var)
      if (is.null(sub$jacobian)) {
        # vForce(bare v) — not a chain; let single-layer fast path handle it
        .dat_stop("DefDiff_not_definable",
                  "elementwise_matmul_chain: vForce of bare var is not a chain")
      }
      # f'(value) AST with sign per function. Wrap with as.numeric() so the
      # result is a length-m vector (not an m×1 matrix from W %*% v). R's
      # `vec * matrix` recycles vec column-wise, giving `diag(vec) %*% matrix`
      # semantics for an m-vec and m×n matrix — but `matrix * matrix` requires
      # conformable dimensions, hence the as.numeric() cast.
      f_deriv_val <- switch(fn_name,
        sin  = bquote(as.numeric(cos(.(sub$value)))),
        cos  = bquote(as.numeric(-sin(.(sub$value)))),
        exp  = bquote(as.numeric(exp(.(sub$value)))),
        tanh = bquote(as.numeric(1 - tanh(.(sub$value))^2)))
      return(list(
        value = bquote(.(head_sym)(.(sub$value))),
        jacobian = bquote(.(f_deriv_val) * .(sub$jacobian))
      ))
    }
  }

  # Case 4: anything else — out of chain scope
  .dat_stop("DefDiff_not_definable",
            paste0("elementwise_matmul_chain: unsupported expression shape: ",
                   deparse(expr)))
}

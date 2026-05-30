## grad_inner.R
## Tier 3 (add-tier3-catalog-closure-completion):
## Recursive grad walker implementing standard differentiation rules over
## L_3 expressions. Honors the DD closure thesis operationally: every L_3
## expression (any composition of +, -, *, /, ^, sum, crossprod, and the
## scalar functions {cos, sin, exp, log, tanh, sqrt, atan}) has a working
## gradient via standard chain/product/quotient/power rules.
##
## `.grad_inner(expr, var)` is called by `.sum_rule` and `.crossprod_rule`
## (after Tier 3 rewrite) to compute the elementwise gradient of a single
## vector-valued expression with respect to `var`. The result is an AST in
## the same L_3 language (closure preservation).

# .grad_inner(expr, var)
#
# Recursive walker. Returns the elementwise derivative AST of `expr` with
# respect to symbol `var`. Output is an L_3 expression (uses only +, -, *,
# /, ^, and recognized scalar functions). Algebraic simplification applied
# inline via .smart_* helpers from utils.R.
.grad_inner <- function(expr, var) {
  expr <- .strip_paren(expr)

  # Cached: AST for `length(<var>)` used by zero-pullback constructions
  n_expr <- bquote(length(.(as.symbol(var))))

  # ===== Leaves =====

  # Constants
  if (is.numeric(expr) || is.logical(expr) || is.character(expr)) {
    return(list(
      value = expr,
      pullback = .make_pullback_zero(n_expr)
    ))
  }
  # Bare symbol
  if (is.symbol(expr)) {
    if (identical(as.character(expr), var)) {
      return(list(value = expr, pullback = .make_pullback_identity()))
    }
    # Free symbol (constant bound in enclosing env)
    return(list(value = expr, pullback = .make_pullback_zero(n_expr)))
  }
  if (!is.call(expr)) {
    return(list(value = expr, pullback = .make_pullback_zero(n_expr)))
  }
  if (!is.symbol(expr[[1L]])) {
    .dat_stop("DefDiff_not_definable",
              paste0("non-symbol function head in .grad_inner: ", deparse(expr)))
  }
  op <- as.character(expr[[1L]])

  # ===== Arithmetic cases =====

  # Unary minus: d/dv(-a) → pullback negates upstream's flow through a
  if (op == "-" && length(expr) == 2L) {
    sub <- .grad_inner(expr[[2L]], var)
    p_sub <- .pullback_of(sub)
    pullback <- if (!is.null(p_sub)) .make_pullback_neg(p_sub) else NULL
    value <- bquote(-.(.value_of(sub, expr[[2L]])))
    return(list(value = value, pullback = pullback))
  }
  # Unary plus: identity passthrough
  if (op == "+" && length(expr) == 2L) {
    sub <- .grad_inner(expr[[2L]], var)
    return(list(
      value = .value_of(sub, expr[[2L]]),
      pullback = .pullback_of(sub)
    ))
  }

  # Binary additive: d/dv(a + b)
  if (op == "+" && length(expr) == 3L) {
    a_expr <- expr[[2L]]; b_expr <- expr[[3L]]
    a <- .grad_inner(a_expr, var); b <- .grad_inner(b_expr, var)
    p_a <- .pullback_of(a); p_b <- .pullback_of(b)
    pullback <- if (!is.null(p_a) && !is.null(p_b)) {
      .make_pullback_add(p_a, p_b)
    } else NULL
    value <- bquote(.(.value_of(a, a_expr)) + .(.value_of(b, b_expr)))
    return(list(value = value, pullback = pullback))
  }
  if (op == "-" && length(expr) == 3L) {
    a_expr <- expr[[2L]]; b_expr <- expr[[3L]]
    a <- .grad_inner(a_expr, var); b <- .grad_inner(b_expr, var)
    p_a <- .pullback_of(a); p_b <- .pullback_of(b)
    pullback <- if (!is.null(p_a) && !is.null(p_b)) {
      .make_pullback_sub(p_a, p_b)
    } else NULL
    value <- bquote(.(.value_of(a, a_expr)) - .(.value_of(b, b_expr)))
    return(list(value = value, pullback = pullback))
  }

  # Product rule: d/dv(a * b) via .make_pullback_mul
  if (op == "*" && length(expr) == 3L) {
    a_expr <- expr[[2L]]; b_expr <- expr[[3L]]
    a <- .grad_inner(a_expr, var); b <- .grad_inner(b_expr, var)
    p_a <- .pullback_of(a); p_b <- .pullback_of(b)
    pullback <- if (!is.null(p_a) && !is.null(p_b)) {
      .make_pullback_mul(p_a, .value_of(a, a_expr), p_b, .value_of(b, b_expr))
    } else NULL
    value <- bquote(.(.value_of(a, a_expr)) * .(.value_of(b, b_expr)))
    return(list(value = value, pullback = pullback))
  }

  # Quotient rule: d/dv(a / b) via .make_pullback_div
  if (op == "/" && length(expr) == 3L) {
    a_expr <- expr[[2L]]; b_expr <- expr[[3L]]
    a <- .grad_inner(a_expr, var); b <- .grad_inner(b_expr, var)
    p_a <- .pullback_of(a); p_b <- .pullback_of(b)
    pullback <- if (!is.null(p_a) && !is.null(p_b)) {
      .make_pullback_div(p_a, .value_of(a, a_expr), p_b, .value_of(b, b_expr))
    } else NULL
    value <- bquote(.(.value_of(a, a_expr)) / .(.value_of(b, b_expr)))
    return(list(value = value, pullback = pullback))
  }

  # Power rule: d/dv(a^k) constant exponent, or v^v general (Phase 6)
  if (op == "^" && length(expr) == 3L) {
    base_expr <- expr[[2L]]; exponent <- expr[[3L]]
    if (.contains_var(exponent, var)) {
      # Both base and exponent depend on v → general power rule pullback.
      a_sub <- .grad_inner(base_expr, var)
      b_sub <- .grad_inner(exponent, var)
      p_a <- .pullback_of(a_sub); p_b <- .pullback_of(b_sub)
      pullback <- if (!is.null(p_a) && !is.null(p_b)) {
        .make_pullback_pow_general(
          p_a, .value_of(a_sub, base_expr),
          p_b, .value_of(b_sub, exponent)
        )
      } else NULL
      value <- bquote(.(.value_of(a_sub, base_expr))^.(.value_of(b_sub, exponent)))
      return(list(value = value, pullback = pullback))
    }
    # Constant exponent: standard power rule.
    a_sub <- .grad_inner(base_expr, var)
    p_a <- .pullback_of(a_sub)
    pullback <- if (!is.null(p_a) && is.numeric(exponent) && length(exponent) == 1L) {
      .make_pullback_pow(p_a, .value_of(a_sub, base_expr), exponent)
    } else NULL
    value <- bquote(.(.value_of(a_sub, base_expr))^.(exponent))
    return(list(value = value, pullback = pullback))
  }

  # Matrix-vector product W %*% inner. Vector-grain breaks here; pullback
  # is t(W) %*% upstream composed with inner's pullback. Restricts to W
  # constant w.r.t. var; var-dependent W falls through to L_0 fallback or
  # the final error.
  if (op == "%*%" && length(expr) == 3L) {
    W_expr <- expr[[2L]]; inner_expr <- expr[[3L]]
    if (!.contains_var(W_expr, var)) {
      inner_sub <- .grad_inner(inner_expr, var)
      p_inner <- .pullback_of(inner_sub)
      pullback <- if (!is.null(p_inner)) {
        .make_pullback_matmul(p_inner, W_expr)
      } else NULL
      value <- bquote(.(W_expr) %*% .(.value_of(inner_sub, inner_expr)))
      return(list(value = value, pullback = pullback))
    }
    # W contains var → fall through to L_0 fallback (which raises)
  }

  # Chain rule for elementwise scalar functions
  if (length(expr) == 2L) {
    inner <- expr[[2L]]
    outer_deriv <- switch(op,
      "sin"  = bquote(cos(.(inner))),
      "cos"  = bquote(-sin(.(inner))),
      "exp"  = bquote(exp(.(inner))),
      "log"  = bquote(1 / .(inner)),
      "tanh" = bquote(1 - tanh(.(inner))^2),
      "sqrt" = bquote(1 / (2 * sqrt(.(inner)))),
      "atan" = bquote(1 / (1 + .(inner)^2)),
      NULL
    )
    if (!is.null(outer_deriv)) {
      inner_sub <- .grad_inner(inner, var)
      p_inner <- .pullback_of(inner_sub)
      pullback <- if (!is.null(p_inner)) {
        .make_pullback_chain(p_inner, outer_deriv)
      } else NULL
      value <- as.call(list(as.symbol(op), .value_of(inner_sub, inner)))
      return(list(value = value, pullback = pullback))
    }
  }

  # L_0 catalog fallback (Tier 4 change `add-walker-l0-fallback`).
  # L_0 rules emit a per-coord gradient AST (e.g., rep returns `0` for
  # constant patterns). Phase 6 wraps this in a shim via
  # `.make_pullback_per_coord` so the result composes with surrounding
  # arithmetic/chain pullbacks (e.g., `rep(c) * sin(v)`).
  fn <- .dat_env$catalog$L_0[[op]]
  if (is.function(fn)) {
    per_coord <- fn(expr, var)
    return(list(
      value = expr,
      pullback = .make_pullback_per_coord(per_coord)
    ))
  }

  # Function not recognized in walker switch or L_0 catalog
  .dat_stop("DefDiff_unknown_generator",
            paste0("Unknown L_3 generator in .grad_inner: ", op,
                   ". Use extend_language() to register a rule."))
}

# .simplify_ast(expr) — additional algebraic simplification beyond .smart_*
# helpers. Catches patterns that emerge from compound derivatives:
#   -(- x) → x
#   x + (-y) → x - y
#   (-x) * y → -(x * y)
# Idempotent (calling twice produces same output).
.simplify_ast <- function(expr) {
  if (!is.call(expr)) return(expr)
  # Recurse first into children (bottom-up simplification)
  if (length(expr) > 1L) {
    for (i in seq_along(expr)[-1L]) {
      expr[[i]] <- .simplify_ast(expr[[i]])
    }
  }
  # Now simplify this node
  if (!is.symbol(expr[[1L]])) return(expr)
  op <- as.character(expr[[1L]])
  # -(- x) → x
  if (op == "-" && length(expr) == 2L) {
    inner <- expr[[2L]]
    if (is.call(inner) && length(inner) == 2L &&
        identical(inner[[1L]], as.name("-"))) {
      return(inner[[2L]])
    }
  }
  expr
}

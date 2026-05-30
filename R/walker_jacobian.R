## walker_jacobian.R
## Tier 5 Option B Phase 1: pullback helper module.
##
## Reverse-mode AD pullback constructors used by the future shape-extended
## walker. A "pullback" here is an R function:
##
##   pullback :: upstream_ast -> input_grad_ast
##
## where:
##   upstream_ast  -- AST evaluating to ∂L/∂(this_subexpr_value)
##   input_grad_ast -- AST evaluating to ∂L/∂v (the differentiation variable)
##
## Composition is natural at the AST level: outer_pullback(inner_pullback(...)).
## No intermediate closure-of-closure AST emission — pullbacks are R functions
## that build their result AST directly.
##
## To finalize a gradient, the caller invokes pullback(rep(1, length(value)))
## for a sum reduction (upstream of a sum is 1 broadcast over all coords).

# Identity: for bare v, ∂L/∂v = upstream
.make_pullback_identity <- function() {
  function(upstream_ast) upstream_ast
}

# Zero: for constants, ∂L/∂v = 0 (vector of length n_expr)
.make_pullback_zero <- function(n_expr) {
  force(n_expr)
  function(upstream_ast) bquote(rep(0, .(n_expr)))
}

# Add: ∂L/∂v from a+b is p_a(upstream) + p_b(upstream)
.make_pullback_add <- function(p_a, p_b) {
  force(p_a); force(p_b)
  function(upstream_ast) {
    bquote(.(p_a(upstream_ast)) + .(p_b(upstream_ast)))
  }
}

# Subtract: ∂L/∂v from a-b is p_a(upstream) - p_b(upstream)
.make_pullback_sub <- function(p_a, p_b) {
  force(p_a); force(p_b)
  function(upstream_ast) {
    bquote(.(p_a(upstream_ast)) - .(p_b(upstream_ast)))
  }
}

# Multiply (product rule): ∂L/∂v from a*b is p_a(upstream*b) + p_b(upstream*a)
.make_pullback_mul <- function(p_a, val_a, p_b, val_b) {
  force(p_a); force(val_a); force(p_b); force(val_b)
  function(upstream_ast) {
    g_for_a <- bquote(.(upstream_ast) * .(val_b))
    g_for_b <- bquote(.(upstream_ast) * .(val_a))
    bquote(.(p_a(g_for_a)) + .(p_b(g_for_b)))
  }
}

# Divide (quotient rule): ∂L/∂v from a/b is p_a(g/b) - p_b(g*a/b^2)
.make_pullback_div <- function(p_a, val_a, p_b, val_b) {
  force(p_a); force(val_a); force(p_b); force(val_b)
  function(upstream_ast) {
    g_for_a <- bquote(.(upstream_ast) / .(val_b))
    g_for_b <- bquote(.(upstream_ast) * .(val_a) / .(val_b)^2)
    bquote(.(p_a(g_for_a)) - .(p_b(g_for_b)))
  }
}

# Power (constant k): ∂L/∂v from a^k is p_a(g * k * a^(k-1))
.make_pullback_pow <- function(p_a, val_a, k) {
  force(p_a); force(val_a); force(k)
  function(upstream_ast) {
    k_minus_1 <- k - 1
    factor <- bquote(.(upstream_ast) * .(k) * .(val_a)^.(k_minus_1))
    p_a(factor)
  }
}

# Chain rule (elementwise f): ∂L/∂v from f(a) is p_a(g * f'(val_a))
# `f_deriv_ast` is the AST of f'(val_a) already constructed by the caller
# (e.g., for sin: f_deriv_ast = bquote(cos(.(val_a))))
.make_pullback_chain <- function(p_inner, f_deriv_ast) {
  force(p_inner); force(f_deriv_ast)
  function(upstream_ast) {
    g_for_inner <- bquote(.(upstream_ast) * .(f_deriv_ast))
    p_inner(g_for_inner)
  }
}

# Matmul backprop: ∂L/∂v from W%*%x is p_inner(t(W) %*% upstream)
# Note: upstream comes in as a vector; t(W) %*% upstream needs upstream as a
# matrix or vector. If upstream is m-vec, t(W) %*% upstream is n-vec.
.make_pullback_matmul <- function(p_inner, W) {
  force(p_inner); force(W)
  function(upstream_ast) {
    g_for_inner <- bquote(t(.(W)) %*% .(upstream_ast))
    p_inner(g_for_inner)
  }
}

# Unary negation: ∂L/∂v from -a is -p_a(upstream)
.make_pullback_neg <- function(p_a) {
  force(p_a)
  function(upstream_ast) {
    bquote(-.(p_a(upstream_ast)))
  }
}

# General power rule a^b where both a and b depend on v:
#   d/dv(a^b) = b·a^(b-1)·a' + a^b·log(a)·b'
# Reverse-mode: ∂L/∂v = p_a(g·b·a^(b-1)) + p_b(g·a^b·log(a))
.make_pullback_pow_general <- function(p_a, val_a, p_b, val_b) {
  force(p_a); force(val_a); force(p_b); force(val_b)
  function(upstream_ast) {
    g_for_a <- bquote(.(upstream_ast) * .(val_b) * .(val_a)^(.(val_b) - 1))
    g_for_b <- bquote(.(upstream_ast) * .(val_a)^.(val_b) * log(.(val_a)))
    bquote(.(p_a(g_for_a)) + .(p_b(g_for_b)))
  }
}

# Bridge for L_0 catalog rules that emit a per-coord gradient AST. Wrap the
# emitted AST as a pullback closure `function(g) g * per_coord_ast`. When
# invoked with the sum-reduction upstream `rep(1, n)`, this yields exactly
# the per-coord AST — preserving legacy semantics — while still composing
# with arithmetic and chain pullbacks (e.g., `rep(c) * sin(v)` recovers
# the correct gradient via standard product-rule pullback composition).
.make_pullback_per_coord <- function(per_coord_ast) {
  force(per_coord_ast)
  function(upstream_ast) {
    bquote(.(upstream_ast) * .(per_coord_ast))
  }
}

# ============================================================================
# Shim accessors (Phase 6: walker returns shim for every recognized case).
# ----------------------------------------------------------------------------
# `.grad_inner` returns a shim `list(value = AST, pullback = R-function)`
# for every case it recognizes (leaves, arithmetic, chain library, matmul,
# L_0 fallback wrapped via `.make_pullback_per_coord`). The accessors below
# remain defensive in case future generators emit a bare AST, but the
# walker itself no longer produces one.
# ============================================================================

# Extract the pullback R function, or NULL if the sub-walker returned bare AST.
.pullback_of <- function(x) {
  if (is.list(x) && !is.null(x$pullback)) x$pullback else NULL
}

# Extract the value AST. For bare returns, fall back to the original
# sub-expression (semantically equivalent: bare returns are gradient ASTs,
# not value ASTs, so the sub-expression itself represents the value).
.value_of <- function(x, fallback_expr) {
  if (is.list(x) && !is.null(x$value)) x$value else fallback_expr
}

# ============================================================================
# Consumer-side finalizer (Phase 6: pullback is the single source of truth).
# ----------------------------------------------------------------------------
# Used by `.sum_rule` and `.crossprod_rule` to convert a walker result into
# the final scalar-output gradient AST.
#
# Dispatch:
#   - Shim (list with `value`) → invoke `pullback(rep(1, length(value)))`.
#     This applies the chain rule from the scalar `sum`/`crossprod` output
#     back to the input variable.
#   - Shim with NULL pullback → unrecognized generator combination; raise
#     DefDiff_not_definable so .sum_rule's tryCatch surfaces it.
#   - Bare AST (defensive) → return as-is. Walker no longer emits these for
#     recognized generators, but this preserves legacy behavior in case a
#     custom L_0 rule registered via `extend_language()` returns bare.
# ============================================================================
.finalize_reduction_grad <- function(walker_result, inner_expr) {
  if (is.list(walker_result) && !is.null(walker_result$value)) {
    pullback <- walker_result$pullback
    if (is.null(pullback)) {
      .dat_stop("DefDiff_not_definable",
                paste0("sum(", deparse(inner_expr),
                       "): walker returned a shim with no pullback"))
    }
    upstream <- bquote(rep(1, length(.(walker_result$value))))
    return(pullback(upstream))
  }
  walker_result
}

## grad.R
## S3 generic dispatch + recursive AST walker for symbolic differentiation.

#' Symbolic gradient (S3 generic)
#'
#' Compute the symbolic gradient of a scalar-valued function of a vector
#' variable. Dispatches over function, call, and expression input types and
#' returns a derivative of the same R type as the input.
#'
#' @param x An R object: function, call (from `quote()`), or expression.
#' @param vars Character vector of variable names. For function input,
#'   defaults to `names(formals(x))`.
#' @param ... Reserved for future use.
#' @return The symbolic gradient. Type matches input. For \code{function}
#'   input the result is a gradient \emph{function}; on macOS its \code{body()}
#'   is typically a fast-path kernel call rather than the symbolic AST, so
#'   recover the symbolic form with \code{\link{grad_expr}}.
#' @export
#' @examples
#' # Function input
#' gf <- grad(function(v) sum(v^2))
#' gf(c(1, 2, 3))                     # c(2, 4, 6)
#'
#' # Call input
#' grad(quote(sum(v^2)), "v")          # 2 * v
#'
#' # Composition
#' grad(quote(sin(sum(v^2))), "v")     # cos(sum(v^2)) * (2 * v)
grad <- function(x, vars = NULL, ...) {
  UseMethod("grad")
}

#' @export
#' @rdname grad
grad.default <- function(x, vars = NULL, ...) {
  .dat_stop(
    "DefDiff_not_definable",
    paste0("grad() does not have a method for class ", paste(class(x), collapse = "/"))
  )
}

# Top-level entry wrappers (add-l4-integral-implicit-nodes, Decision 3):
# alpha-rename every L_4 binder once, differentiate, restore the names.
# Recursive callers keep using .grad_expr / .grad_body_for_var unchanged.
.grad_expr_top <- function(expr, var) {
  ren <- .alpha_rename_binders(expr)
  .restore_binder_names(.grad_expr(ren$expr, var), ren$map)
}
.grad_body_for_var_top <- function(body_expr, var) {
  ren <- .alpha_rename_binders(body_expr)
  .restore_binder_names(.grad_body_for_var(ren$expr, var), ren$map)
}

# .grad_body_for_var(body_expr, var) — build the gradient BODY AST for a
# single variable `var`, applying the full Tier 1-5 fast-path dispatch chain.
# `body_expr` must already be control-flow-checked and paren-stripped.
# Returns the body AST that a gradient function would evaluate. Shared by both
# the single-variable and multi-variable paths of grad.function.
.grad_body_for_var <- function(body_expr, var) {
  gexpr <- .grad_expr(body_expr, var)
  base <- .grad_body_base(gexpr, var)
  # Backend B (add-fused-jit-grad-backend, #2): route genuinely multi-pass
  # fusable arithmetic gradients to the fused single-pass JIT kernel above a
  # size threshold, falling back to `base` below threshold or on any
  # compile/run failure. grad_expr() inspectability is unaffected — the
  # symbolic gradient is stored separately in grad.function.
  if (.jit_path_available() && .fused_should_fire(gexpr, var)) {
    # Metal-eligible = the canonical `<scalar> * <var>` Metal handles; the fused
    # window is then capped below metal_threshold (see .build_fused_body).
    metal_eligible <- !is.null(.try_normalize_scalar_var_product(gexpr, var))
    fused <- tryCatch(.build_fused_body(gexpr, var, base, metal_eligible),
                      error = function(e) NULL)
    if (!is.null(fused)) return(fused)
  }
  base
}

# .grad_body_base(gexpr, var) — pre-Backend-B fast-path dispatch chain (Tier
# 1-5); returns the body AST, or the raw symbolic gradient on no match.
.grad_body_base <- function(gexpr, var) {
  # Dispatch priority chain (Tier 1 strict → Tier 2a normalizer → Tier 2d
  # composite → Tier 2c scaled-elementwise → Tier 2c bare elementwise →
  # Tier 2e reciprocal → Tier 3b scalar-pow → generic R). All fast-path tiers
  # require macOS Accelerate backend.
  if (.fast_path_available()) {
    canonical <- .try_normalize_scalar_var_product(gexpr, var)
    if (!is.null(canonical)) {
      # add-metal-backend: runtime size-threshold dispatch. Route the canonical
      # Tier 1 `<scalar> * <var>` body to the Metal GPU kernel only when Metal
      # is available AND the vector is large enough to amortize the
      # double<->float32 conversion passes (default 1e9; Metal loses to vDSP
      # below that). Otherwise the unchanged vDSP fast_scalar_mul path runs, so
      # below-threshold / non-macOS / CPU-only calls are numerically identical
      # to before. `.metal_path_available` is exported so this body resolves it
      # without `:::`.
      return(bquote(
        if (.metal_path_available() &&
            length(.(canonical$var)) >= .dat_opt_pos_num("DefDiff.metal_threshold", 1e9L))
          metal_scalar_mul(.(canonical$scalar), .(canonical$var))
        else
          fast_scalar_mul(.(canonical$scalar), .(canonical$var))
      ))
    }
    composite <- .try_normalize_scalar_var_product_with_outer(gexpr, var)
    if (!is.null(composite)) {
      if (identical(composite$inner_kind, "elementwise")) {
        return(bquote({
          s_outer <- .(composite$outer_expr)
          fast_scalar_mul(s_outer, .(as.name(composite$kernel_name))(.(composite$var)))
        }))
      }
      # add-simplify-extensions: scalar-power inner (outer * (c * v^k), k>=2).
      # Build the v^k product via chained fast_vec_mul (k times the base),
      # mirroring the bare scalar-power path, and fold the substituted outer
      # scalar into the fast_scalar_mul coefficient.
      if (identical(composite$inner_kind, "scalar_pow")) {
        k <- composite$exponent
        vsym <- composite$var
        chained <- bquote(fast_vec_mul(.(vsym), .(vsym)))  # v^2
        for (i in seq_len(k - 2L)) {
          chained <- bquote(fast_vec_mul(.(chained), .(vsym)))
        }
        return(bquote({
          s_outer <- .(composite$outer_expr)
          fast_scalar_mul(s_outer * .(composite$scalar), .(chained))
        }))
      }
      return(bquote({
        s_outer <- .(composite$outer_expr)
        fast_scalar_mul(s_outer * .(composite$scalar), .(composite$var))
      }))
    }
    ew_scaled <- .try_normalize_scalar_var_elementwise(gexpr, var)
    if (!is.null(ew_scaled)) {
      return(bquote(fast_scalar_mul(
        .(ew_scaled$scalar),
        .(as.name(ew_scaled$kernel_name))(.(ew_scaled$var)))))
    }
    ew_bare <- .try_dispatch_elementwise(gexpr, var)
    if (!is.null(ew_bare)) {
      return(bquote(.(as.name(ew_bare$kernel_name))(.(ew_bare$var))))
    }
    # Tier 2e fix 1: reciprocal-vForce dispatch
    recip <- .try_normalize_reciprocal_vforce(gexpr, var)
    if (!is.null(recip)) {
      return(bquote(fast_scalar_div(
        .(recip$numerator),
        .(as.name(recip$kernel_name))(.(recip$var)))))
    }
    # Tier 3 fix 3b: scalar-power dispatch <scalar> * <var>^<int_lit>
    pow <- .try_normalize_scalar_pow(gexpr, var)
    if (!is.null(pow)) {
      k <- pow$exponent
      vsym <- pow$var
      if (k == 2L) {
        return(bquote({
          pow_v <- fast_vec_mul(.(vsym), .(vsym))
          fast_scalar_mul(.(pow$scalar), pow_v)
        }))
      }
      # k >= 3: chain fast_vec_mul (k-1) times
      chained <- bquote(fast_vec_mul(.(vsym), .(vsym)))  # v^2
      for (i in seq_len(k - 2L)) {
        chained <- bquote(fast_vec_mul(.(chained), .(vsym)))
      }
      return(bquote(fast_scalar_mul(.(pow$scalar), .(chained))))
    }
  }
  gexpr
}

#' @export
#' @rdname grad
grad.function <- function(x, vars = NULL, ...) {
  # Default to all formal arguments → the full gradient. A subset is selected
  # by passing `vars` explicitly.
  if (is.null(vars)) vars <- names(formals(x))
  body_expr <- body(x)
  blocker <- .control_flow_block(body_expr)
  if (!is.na(blocker)) {
    .dat_stop("DefDiff_not_definable",
              paste0("Function body contains unsupported construct `", blocker,
                     "`; only straight-line scalar expressions are supported."))
  }
  body_expr <- .strip_paren(body_expr)
  new_fn <- function() NULL
  formals(new_fn) <- formals(x)
  if (length(vars) == 1L) {
    # Single variable: bare-vector contract (unchanged).
    body(new_fn) <- .grad_body_for_var_top(body_expr, vars)
    # Preserve the symbolic gradient separately so grad_expr() can recover it
    # regardless of the body's evaluation backend (the body may be a fast-path
    # kernel call, not the readable AST). See grad_expr() and issue #3.
    attr(new_fn, "grad_expr") <- .grad_expr_top(body_expr, vars)
  } else {
    # Multiple variables: one gradient body per variable, assembled into a
    # named list keyed by variable name (order preserved).
    elements <- lapply(vars, function(vn) .grad_body_for_var_top(body_expr, vn))
    names(elements) <- vars
    body(new_fn) <- as.call(c(list(quote(list)), elements))
    sym <- lapply(vars, function(vn) .grad_expr_top(body_expr, vn))
    names(sym) <- vars
    attr(new_fn, "grad_expr") <- sym
  }
  if (.has_binder_call(body(new_fn))) {
    # Generated bodies call `DefDiff::integral` / `DefDiff::implicit`, so no
    # user binding of those names (function or variable) can capture them
    # (verify #23 rounds 2–3c); base `::` itself is trusted like every other
    # base operator a body uses. grad_expr() keeps the public names; the
    # environment stays the user's. `body<-` rebuilds the closure and drops
    # attributes, so re-attach.
    sym_attr <- attr(new_fn, "grad_expr")
    body(new_fn) <- .rewrite_binder_heads(body(new_fn))
    attr(new_fn, "grad_expr") <- sym_attr
  }
  environment(new_fn) <- environment(x)
  new_fn
}

#' @export
#' @rdname grad
grad.call <- function(x, vars, ...) {
  if (missing(vars) || is.null(vars)) {
    .dat_stop("DefDiff_not_definable",
              "grad() on a call requires `vars` (character vector of variable names).")
  }
  blocker <- .control_flow_block(x)
  if (!is.na(blocker)) {
    .dat_stop("DefDiff_not_definable",
              paste0("Expression contains unsupported construct `", blocker, "`."))
  }
  rhs <- .strip_paren(x)
  if (length(vars) == 1L) {
    # Single variable: bare AST (unchanged).
    return(.grad_expr_top(rhs, vars))
  }
  # Multiple variables: named list of per-variable gradient ASTs.
  result <- lapply(vars, function(vn) .grad_expr_top(rhs, vn))
  names(result) <- vars
  result
}

#' @export
#' @rdname grad
grad.expression <- function(x, vars, ...) {
  if (length(x) != 1L) {
    .dat_stop("DefDiff_not_definable",
              "grad() supports single-element expression objects in v0.1.")
  }
  result <- grad.call(x[[1L]], vars = vars, ...)
  if (is.list(result)) {
    # Multiple variables: grad.call returned a named list of ASTs; wrap each
    # in an expression object, preserving names.
    out <- lapply(result, as.expression)
    names(out) <- names(result)
    return(out)
  }
  # Single variable: bare expression (unchanged).
  as.expression(result)
}

#' @export
#' @rdname grad
#' @examples
#' # One-sided formula
#' grad(~ sum(v^2), "v")              # ~ 2 * v
#'
#' # Two-sided formula preserves LHS
#' grad(y ~ sum(v^2), "v")            # y ~ 2 * v
#'
#' # vars inferred when only one free variable
#' grad(~ sum(v^2))                   # ~ 2 * v
grad.formula <- function(x, vars = NULL, ...) {
  # R formulas are calls with `~` as head: length 2 = one-sided, 3 = two-sided.
  if (length(x) == 2L) {
    rhs <- x[[2L]]
    lhs <- NULL
  } else if (length(x) == 3L) {
    lhs <- x[[2L]]
    rhs <- x[[3L]]
  } else {
    .dat_stop("DefDiff_not_definable",
              "formula input must be one-sided (`~ rhs`) or two-sided (`lhs ~ rhs`).")
  }

  # vars inference: if NULL, derive ALL free variables from the RHS. A RHS
  # with multiple free variables defaults to the full gradient.
  if (is.null(vars)) {
    candidates <- all.vars(rhs)
    if (length(candidates) == 0L) {
      .dat_stop("DefDiff_not_definable",
                "RHS contains no free variables; gradient is trivially 0. Pass `vars` explicitly to confirm intent.")
    }
    vars <- candidates
  }

  blocker <- .control_flow_block(rhs)
  if (!is.na(blocker)) {
    .dat_stop("DefDiff_not_definable",
              paste0("Formula RHS contains unsupported construct `", blocker, "`."))
  }

  rhs_stripped <- .strip_paren(rhs)
  env <- environment(x)

  # Helper: rebuild a formula of the same arity, preserving LHS + env.
  build_formula <- function(grad_ast) {
    fcall <- if (is.null(lhs)) call("~", grad_ast) else call("~", lhs, grad_ast)
    as.formula(fcall, env = env)
  }

  if (length(vars) == 1L) {
    # Single variable: bare formula (unchanged), LHS preserved.
    return(build_formula(.grad_expr_top(rhs_stripped, vars)))
  }
  # Multiple variables: named list of per-variable formulas.
  result <- lapply(vars, function(vn) build_formula(.grad_expr_top(rhs_stripped, vn)))
  names(result) <- vars
  result
}

# stats::as.formula imported indirectly via base namespace; declare for NAMESPACE.
#' @importFrom stats as.formula
NULL

# ----- Internal recursive walker -----

# .grad_expr(expr, var) — return the gradient expression of `expr` with
# respect to `var`. The variable is identified by name (character). Pattern
# dispatch happens via the generator catalog: each function name resolves
# to a rule which encodes its own chain rule logic.
.grad_expr <- function(expr, var) {
  # Pre-grad algebraic fold (Tier 4 `add-algebraic-simplifier`): apply
  # always-safe identities so catalog dispatch sees the simplified form.
  # This restores Tier 1 fast-path opportunity for cases like
  # exp(log(sum(v^2))) that would otherwise route through the slower
  # Tier 3 walker fall-through path.
  expr <- .algebraic_simplify(.strip_paren(expr))

  # Atoms: constants → 0; the variable symbol itself reaches here only via
  # rules that recurse on un-decomposed arguments. We return a structural
  # zero/one without attempting to encode the identity Jacobian (v0.1 scope
  # restricts grad() inputs to scalar-valued expressions; pattern matching
  # intercepts vector-valued occurrences earlier).
  if (is.symbol(expr)) {
    if (identical(as.character(expr), var)) {
      .dat_stop(
        "DefDiff_not_definable",
        paste0("Variable `", var, "` appears at a position where a scalar ",
               "expression is required by v0.1 catalog. Wrap inside a scalar ",
               "reduction such as sum() or crossprod().")
      )
    }
    return(0)
  }
  if (is.numeric(expr) || is.logical(expr) || is.character(expr)) {
    return(0)
  }
  if (!is.call(expr)) {
    .dat_stop("DefDiff_not_definable",
              paste0("Unsupported AST node of class ", paste(class(expr), collapse = "/")))
  }

  head_sym <- expr[[1L]]
  if (!is.symbol(head_sym)) {
    .dat_stop("DefDiff_not_definable",
              "Non-symbol function head not supported in v0.1.")
  }
  fname <- as.character(head_sym)

  # Built-in linear combinators (+ - * / ^) live in L_0 / L_1; everything
  # else (sum, crossprod, sin, etc.) is looked up via the catalog.
  rule <- .lookup_derivative(fname)
  if (is.null(rule)) {
    .dat_stop(
      "DefDiff_unknown_generator",
      paste0("Unknown generator `", fname, "`. Use extend_language() to ",
             "register a rule, or check that the function is part of the ",
             "intended L_i catalog.")
    )
  }
  rule(expr, var)
}

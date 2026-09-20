## l4_nodes.R
## Language level L_4 (add-l4-integral-implicit-nodes, #23): the two binder
## nodes `integral(f, t, a, b)` and `implicit(F, y, lower, upper)`.
##
## A binder node is a plain R call whose second element is an unevaluated
## body and whose third element is the bound symbol. Everything else in the
## walker assumes every symbol occurrence is free; the helpers here are the
## only place that knows about binding (Decision 2), plus a one-time
## alpha-renaming at grad() entry so that substitutions downstream never
## capture (Decision 3).

.l4_binder_heads <- c("integral", "implicit")

# Canonical binder name for a call head, or NA. Accepts the public symbol
# (`integral`) and the namespace-qualified form `DefDiff::integral` that
# generated gradient bodies use: a `::` call performs no lookup of `integral`
# / `implicit` in the user's environment, so no user binding of those names —
# function or variable — can capture the generated code (verify #23 rounds
# 2–3c). The body still relies on base `::`, exactly as every DefDiff body
# relies on base `+`, `exp`, `sum`; redefining those in the user's environment
# is outside the package's trust model. grad_expr() and grad(<call>) keep the
# public symbol.
.l4_head_name <- function(h) {
  if (is.symbol(h)) {
    n <- as.character(h)
    return(if (n %in% .l4_binder_heads) n else NA_character_)
  }
  if (is.call(h) && length(h) == 3L && is.symbol(h[[1L]]) &&
      identical(as.character(h[[1L]]), "::") &&
      is.symbol(h[[2L]]) && identical(as.character(h[[2L]]), "DefDiff") &&
      is.symbol(h[[3L]])) {
    n <- as.character(h[[3L]])
    return(if (n %in% .l4_binder_heads) n else NA_character_)
  }
  NA_character_
}

# A bound / interval value must be a single non-NA number (±Inf allowed for
# integral bounds). Strings coerce to non-finite inside stats::integrate and
# silently switch it to an infinite-range transform (verify #23 round 2, R2-1).
.check_bound_value <- function(x, slot, fn) {
  if (!is.numeric(x) || length(x) != 1L || is.na(x)) {
    .dat_stop("DefDiff_not_definable",
              paste0(fn, "(): `", slot, "` must be a single non-NA number, got `",
                     deparse1(x), "` (", class(x)[1L], ")."))
  }
  invisible(TRUE)
}

# TRUE iff `fname` names an L_4 binder node.
.is_binder_head <- function(fname) {
  is.character(fname) && length(fname) == 1L && fname %in% .l4_binder_heads
}

# TRUE iff `expr` is a call whose head is an L_4 binder node.
.is_binder_call <- function(expr) {
  is.call(expr) && !is.na(.l4_head_name(expr[[1L]]))
}

# Validate the shape of a binder call: exactly (head, body, symbol, a, b) with
# a bare symbol in the binder slot. Raises DefDiff_not_definable otherwise so
# that arity / string-literal mistakes never reach the walkers (verify #23,
# findings 2 and 5: a string literal in the binder slot used to skip renaming
# and return a silently wrong derivative).
.check_binder_node <- function(expr) {
  op <- .l4_head_name(expr[[1L]])
  if (length(expr) != 5L) {
    .dat_stop("DefDiff_not_definable",
              paste0(op, "() node requires exactly 4 arguments (",
                     if (op == "integral") "f, t, a, b" else "F, y, lower, upper",
                     "); got ", length(expr) - 1L, " in `", deparse1(expr), "`."))
  }
  if (!is.symbol(expr[[3L]])) {
    .dat_stop("DefDiff_not_definable",
              paste0(op, "(): the bound-variable slot must be a bare symbol, got `",
                     deparse1(expr[[3L]]), "` in `", deparse1(expr), "`."))
  }
  invisible(TRUE)
}

# TRUE iff `expr` contains a binder CALL (head position), scanning call heads
# recursively. Unlike all.names(), a plain variable that happens to be named
# `integral` does not count (verify #23 finding 4).
.has_binder_call <- function(expr) {
  !is.na(.first_binder_head(expr))
}

# Register the L_4 tier. The derivative rules live in `.grad_inner` (they need
# the elementwise walker, Decision 9); the catalog entries here make `level()`
# and `extend_language()` recognize the heads and route top-level calls.
.register_L4 <- function() {
  .dat_env$catalog$L_4[["integral"]] <- function(expr, var) .integral_rule(expr, var)
  .dat_env$catalog$L_4[["implicit"]] <- function(expr, var) .implicit_rule(expr, var)
}

# Top-level catalog rules: finalize the walker shim with a unit upstream.
.integral_rule <- function(expr, var) .l4_finalize_node(expr, var)
.implicit_rule <- function(expr, var) .l4_finalize_node(expr, var)

.l4_finalize_node <- function(expr, var) {
  res <- .l4_grad_inner(expr, var)
  .l4_clean(.simplify_ast(.pullback_of(res)(1)))
}

# Elementwise derivative AST of `sub` w.r.t. `var` via the elementwise walker
# (Decision 9): the top-level .grad_expr rejects bare scalar variables, the
# walker does not. Returns the literal 0 when `sub` does not depend on `var`.
.l4_deriv <- function(sub, var) {
  if (!.contains_var(sub, var)) return(0)
  res <- .grad_inner(sub, var)
  pb <- .pullback_of(res)
  if (is.null(pb)) {
    .dat_stop("DefDiff_not_definable",
              paste0("Cannot differentiate `", deparse1(sub), "` inside an L_4 node."))
  }
  .l4_clean(.simplify_ast(pb(1)))
}

# Fold the structural artifacts the pullback machinery leaves behind when the
# upstream is the unit scalar: `rep(0, length(x))` additive terms, `1 * x`,
# `x * 1`, `x^1`, `-(0)`. Bottom-up, idempotent. Keeps the output inside the
# catalog (`length` is not a generator) and readable for grad_expr().
.l4_clean <- function(expr) {
  if (!is.call(expr)) return(expr)
  for (i in seq_along(expr)[-1L]) {
    if (is.null(expr[[i]])) next
    expr[[i]] <- .l4_clean(expr[[i]])
  }
  if (!is.symbol(expr[[1L]])) return(expr)
  op <- as.character(expr[[1L]])
  is_zero <- function(x) {
    identical(x, 0) || identical(x, 0L) ||
      (is.call(x) && identical(x[[1L]], quote(rep)) && length(x) == 3L &&
         (identical(x[[2L]], 0) || identical(x[[2L]], 0L)))
  }
  is_one <- function(x) identical(x, 1) || identical(x, 1L)
  if (op == "+" && length(expr) == 3L) {
    if (is_zero(expr[[2L]])) return(expr[[3L]])
    if (is_zero(expr[[3L]])) return(expr[[2L]])
  }
  if (op == "-" && length(expr) == 3L) {
    if (is_zero(expr[[3L]])) return(expr[[2L]])
    if (is_zero(expr[[2L]])) return(.l4_clean(bquote(-.(expr[[3L]]))))
  }
  if (op == "-" && length(expr) == 2L) {
    if (is_zero(expr[[2L]])) return(0)
    inner <- expr[[2L]]
    if (is.call(inner) && identical(inner[[1L]], quote(`-`)) && length(inner) == 2L) {
      return(inner[[2L]])
    }
  }
  if (op == "*" && length(expr) == 3L) {
    if (is_one(expr[[2L]])) return(expr[[3L]])
    if (is_one(expr[[3L]])) return(expr[[2L]])
    if (is_zero(expr[[2L]]) || is_zero(expr[[3L]])) return(0)
  }
  if (op == "^" && length(expr) == 3L && is_one(expr[[3L]])) return(expr[[2L]])
  expr
}

.is_infinite_bound <- function(x) {
  if (is.numeric(x) && length(x) == 1L && is.infinite(x)) return(TRUE)
  # `-Inf` parses as the call `-`(Inf)
  is.call(x) && identical(x[[1L]], quote(`-`)) && length(x) == 2L &&
    is.numeric(x[[2L]]) && is.infinite(x[[2L]])
}

# Pullback for a scalar-valued node with derivative AST `d`: upstream * d,
# with the unit upstream folded away for readable top-level output.
.make_pullback_l4 <- function(d, var) {
  force(d); force(var)
  if (identical(d, 0)) return(.make_pullback_zero(bquote(length(.(as.symbol(var))))))
  function(upstream_ast) {
    if (identical(upstream_ast, 1) || identical(upstream_ast, 1L)) return(d)
    bquote(.(upstream_ast) * .(d))
  }
}

# Walker case for L_4 binder nodes. Precondition (Decision 3): bound symbols
# have been alpha-renamed at grad() entry, so `var` is never a bound symbol
# here and substitution into the body cannot capture.
.l4_grad_inner <- function(expr, var) {
  .check_binder_node(expr)
  op <- .l4_head_name(expr[[1L]])
  bsym <- as.character(expr[[3L]])
  if (identical(bsym, var)) {
    # `var` is bound by this node: no free occurrence inside, derivative 0
    # (bounds may still depend on var — handled below for integral).
    if (op == "implicit") {
      return(list(value = expr, pullback = .make_pullback_l4(0, var)))
    }
  }
  if (op == "integral") {
    body <- expr[[2L]]; a <- expr[[4L]]; b <- expr[[5L]]
    d_body <- if (identical(bsym, var)) 0 else .l4_deriv(body, var)
    term_int <- if (identical(d_body, 0)) 0 else
      as.call(list(as.symbol("integral"), d_body, as.symbol(bsym), a, b))
    term_b <- if (.is_infinite_bound(b) || !.contains_var(b, var)) 0 else
      .smart_mul(.subst_symbol(body, bsym, b), .l4_deriv(b, var))
    term_a <- if (.is_infinite_bound(a) || !.contains_var(a, var)) 0 else
      .smart_mul(.subst_symbol(body, bsym, a), .l4_deriv(a, var))
    d <- .smart_sub(.smart_add(term_int, term_b), term_a)
    return(list(value = expr, pullback = .make_pullback_l4(d, var)))
  }
  if (op == "implicit") {
    F_expr <- expr[[2L]]
    if (!.contains_var(F_expr, var)) {
      return(list(value = expr, pullback = .make_pullback_l4(0, var)))
    }
    dF_var <- .l4_deriv(F_expr, var)
    dF_y   <- .l4_deriv(F_expr, bsym)
    num <- .subst_symbol(dF_var, bsym, expr)
    den <- .subst_symbol(dF_y,   bsym, expr)
    d <- bquote(-(.(num)) / (.(den)))
    return(list(value = expr, pullback = .make_pullback_l4(d, var)))
  }
  .dat_stop("DefDiff_not_definable", paste0("Unknown L_4 node: ", op))
}

# Second-order operators do not support L_4 nodes yet (#24).
.first_binder_head <- function(expr) {
  if (!is.call(expr)) return(NA_character_)
  if (.is_binder_call(expr)) return(.l4_head_name(expr[[1L]]))
  # A compound head (`(f)(x)`, `g(x)(y)`) is itself a call — scan it too.
  if (is.call(expr[[1L]])) {
    h <- .first_binder_head(expr[[1L]])
    if (!is.na(h)) return(h)
  }
  for (i in seq_along(expr)[-1L]) {
    if (is.null(expr[[i]])) next
    h <- .first_binder_head(expr[[i]])
    if (!is.na(h)) return(h)
  }
  NA_character_
}

# Rewrite public binder heads to `DefDiff::<head>` (generated bodies only).
.rewrite_binder_heads <- function(expr) {
  if (!is.call(expr)) return(expr)
  if (is.symbol(expr[[1L]])) {
    fname <- as.character(expr[[1L]])
    if (fname %in% .l4_binder_heads) {
      expr[[1L]] <- call("::", as.symbol("DefDiff"), as.symbol(fname))
    }
  } else if (is.call(expr[[1L]])) {
    expr[[1L]] <- .rewrite_binder_heads(expr[[1L]])
  }
  for (i in seq_along(expr)[-1L]) {
    if (is.null(expr[[i]])) next
    expr[[i]] <- .rewrite_binder_heads(expr[[i]])
  }
  expr
}

.refuse_l4_nodes <- function(expr, operator) {
  head <- .first_binder_head(expr)
  if (!is.na(head)) {
    .dat_stop("DefDiff_not_definable",
              paste0(operator, "() does not support the L_4 binder node `", head,
                     "` yet (second-order rules are tracked in issue #24)."))
  }
  invisible(TRUE)
}

# ---------------------------------------------------------------------------
# Binder helpers (Decision 3): alpha-rename once at grad() entry, then every
# downstream substitution can be naive.
# ---------------------------------------------------------------------------

# Structural substitution of `sym` (character) by `replacement` (AST) in
# `expr`. Naive on purpose: callers guarantee (via .alpha_rename_binders) that
# `sym` is not bound anywhere inside `expr`, so no capture can occur.
.subst_symbol <- function(expr, sym, replacement) {
  if (is.symbol(expr)) {
    if (identical(as.character(expr), sym)) return(replacement)
    return(expr)
  }
  if (is.call(expr)) {
    # Skip the call head (position 1): a bound variable named like a function
    # (`t`, `exp`) must not rewrite `t(A)` / `exp(x)` (verify #23 finding 3).
    # Skip NULL elements: `call[[i]] <- NULL` DELETES the element in R, which
    # shrinks the call mid-loop (verify #23 finding 1; `function()` nodes carry
    # NULL formals / srcref slots).
    for (i in seq_along(expr)[-1L]) {
      if (is.null(expr[[i]])) next
      expr[[i]] <- .subst_symbol(expr[[i]], sym, replacement)
    }
    # A compound head (`g(x)(t)`) is itself a call and may contain the symbol;
    # only a *symbol* head is exempt.
    if (is.call(expr[[1L]])) expr[[1L]] <- .subst_symbol(expr[[1L]], sym, replacement)
    return(expr)
  }
  expr
}

# Rename the bound symbol of every binder node in `expr` to a fresh symbol
# `.b<n>` that does not occur anywhere in the input. Returns
# list(expr = renamed AST, map = named list fresh_name -> original_name).
.alpha_rename_binders <- function(expr) {
  taken <- all.names(expr)
  counter <- 0L
  map <- list()
  fresh <- function() {
    repeat {
      counter <<- counter + 1L
      cand <- paste0(".b", counter)
      if (!(cand %in% taken)) { taken <<- c(taken, cand); return(cand) }
    }
  }
  walk <- function(e) {
    if (!is.call(e)) return(e)
    if (.is_binder_call(e)) {
      .check_binder_node(e)
      old <- as.character(e[[3L]])
      new <- fresh()
      map[[new]] <<- old
      e[[2L]] <- .subst_symbol(e[[2L]], old, as.symbol(new))
      e[[3L]] <- as.symbol(new)
    }
    for (i in seq_along(e)[-1L]) {
      if (is.null(e[[i]])) next
      e[[i]] <- walk(e[[i]])
    }
    e
  }
  list(expr = walk(expr), map = map)
}

# Restore original bound names recorded by .alpha_rename_binders, but only
# when the original name does not occur anywhere else in `expr` (a free
# occurrence would be captured). Fresh names that cannot be restored stay.
.restore_binder_names <- function(expr, map) {
  if (length(map) == 0L) return(expr)
  for (fresh_name in names(map)) {
    original <- map[[fresh_name]]
    others <- setdiff(all.names(expr), fresh_name)
    if (original %in% others) next
    expr <- .subst_symbol(expr, fresh_name, as.symbol(original))
  }
  expr
}

# ---------------------------------------------------------------------------
# Evaluators (Decision 1): the language is its own evaluation language, so a
# gradient-function body containing L_4 nodes must run as plain R.
# ---------------------------------------------------------------------------

#' Definite integral node
#'
#' `integral(f, t, a, b)` is the L_4 binder node for \eqn{\int_a^b f(t)\,dt}.
#' `f` is captured unevaluated and integrated over the bound symbol `t` with
#' [stats::integrate()]; every other symbol in `f` is looked up in the calling
#' environment. Bounds may be finite numbers, expressions, or `-Inf` / `Inf`.
#'
#' Precondition for differentiation with an infinite bound: the integrand must
#' vanish at that limit. The Leibniz rule used by [grad()] omits the boundary
#' term at `-Inf` / `Inf`; if the integrand does not decay there the symbolic
#' derivative is wrong, which [verify_grad()] detects as a finite-difference
#' mismatch.
#'
#' @param f Integrand expression in the bound symbol `t` (unevaluated).
#' @param t Bound symbol (bare name).
#' @param a,b Lower and upper bounds.
#' @return The numeric value of the integral.
#' @seealso [implicit()], [grad()]
#' @export
#' @examples
#' integral(exp(-t), t, 0, Inf)   # 1
integral <- function(f, t, a, b) {
  body_expr <- substitute(f)
  tsym <- substitute(t)
  if (!is.symbol(tsym)) {
    .dat_stop("DefDiff_not_definable",
              paste0("integral(): the bound-variable slot must be a bare symbol, got `",
                     deparse1(tsym), "`."))
  }
  sym <- as.character(tsym)
  .check_bound_value(a, "a", "integral"); .check_bound_value(b, "b", "integral")
  env <- parent.frame()
  user_error <- NULL
  integrand <- function(x) {
    vapply(x, function(xi) {
      e <- new.env(parent = env)
      assign(sym, xi, envir = e)
      # Any condition raised while evaluating the body (user typos, stop(),
      # the scalar-shape check below) is remembered so the quadrature wrapper
      # re-raises it unchanged instead of re-labelling it as a quadrature
      # failure. Handler classes do not reliably match through integrate()'s
      # C frames, hence the closure variable.
      tryCatch({
        val <- eval(body_expr, e)
        if (!is.numeric(val) || length(val) != 1L) {
          .dat_stop("DefDiff_not_definable",
                    paste0("integral(): the integrand must be scalar-valued at each ",
                           sym, "; `", deparse1(body_expr), "` returned length ",
                           length(val), ". Vector free parameters are outside the L_4 ",
                           "evaluator's contract."))
        }
        as.numeric(val)
      }, error = function(err) { user_error <<- err; stop(err) })
    }, numeric(1))
  }
  # Garbage option values (NA / non-numeric / <= 0) fall back to the default
  # instead of surfacing as a quadrature failure (verify #23 finding 11).
  rel_tol <- .dat_opt_pos_num("DefDiff.integrate_rel_tol", 1e-8)
  # Only quadrature failures are re-labelled; errors raised by the user's own
  # integrand (typos, non-functions, stop()) and DefDiff conditions propagate.
  res <- tryCatch(
    stats::integrate(integrand, lower = a, upper = b, rel.tol = rel_tol,
                     subdivisions = 200L),
    error = function(e) {
      if (!is.null(user_error)) stop(user_error)
      .dat_stop("DefDiff_not_definable",
                paste0("integral(", deparse1(body_expr), ", ", sym, ", ",
                       deparse1(a), ", ", deparse1(b), ") failed in quadrature: ",
                       conditionMessage(e)))
    }
  )
  res$value
}

#' Implicit function node
#'
#' `implicit(F, y, lower, upper)` is the L_4 binder node for the root
#' \eqn{y^*} of \eqn{F(y) = 0} located in `[lower, upper]`. `F` is captured
#' unevaluated; every symbol other than the bound symbol `y` is looked up in
#' the calling environment. The root is found with [stats::uniroot()].
#'
#' The interval must bracket a sign change of `F`; otherwise a
#' `DefDiff_not_definable` condition is raised. If several roots lie in the
#' interval, which one is returned is up to `uniroot()` — choose the interval
#' so that it contains exactly one root. [grad()] differentiates this node with
#' the implicit function theorem, whose precondition is \eqn{\partial F /
#' \partial y \neq 0} at the root; this is not checked symbolically, and a
#' degenerate root yields a non-finite derivative (caught by [verify_grad()]).
#'
#' @param F Equation left-hand side in the bound symbol `y` (unevaluated).
#' @param y Bound symbol (bare name).
#' @param lower,upper Numeric root interval.
#' @return The numeric root.
#' @seealso [integral()], [grad()]
#' @export
#' @examples
#' theta <- 4
#' implicit(y^2 - theta, y, 0, 10)   # 2
implicit <- function(F, y, lower, upper) {
  body_expr <- substitute(F)
  ysym <- substitute(y)
  if (!is.symbol(ysym)) {
    .dat_stop("DefDiff_not_definable",
              paste0("implicit(): the bound-variable slot must be a bare symbol, got `",
                     deparse1(ysym), "`."))
  }
  sym <- as.character(ysym)
  .check_bound_value(lower, "lower", "implicit"); .check_bound_value(upper, "upper", "implicit")
  if (!is.finite(lower) || !is.finite(upper) || lower >= upper) {
    .dat_stop("DefDiff_not_definable",
              paste0("implicit(): the interval must be finite with lower < upper; got [",
                     format(lower), ", ", format(upper), "]."))
  }
  env <- parent.frame()
  fn <- function(yy) {
    e <- new.env(parent = env)
    assign(sym, yy, envir = e)
    val <- eval(body_expr, e)
    if (!is.numeric(val) || length(val) != 1L) {
      .dat_stop("DefDiff_not_definable",
                paste0("implicit(): F must be scalar-valued at each ", sym, "; `",
                       deparse1(body_expr), "` returned length ", length(val), "."))
    }
    as.numeric(val)
  }
  f_lo <- fn(lower); f_hi <- fn(upper)
  if (!is.finite(f_lo) || !is.finite(f_hi) || sign(f_lo) * sign(f_hi) > 0) {
    .dat_stop("DefDiff_not_definable",
              paste0("implicit(", deparse1(body_expr), ", ", sym,
                     "): no sign change of F on [", lower, ", ", upper,
                     "] (F(lower) = ", format(f_lo), ", F(upper) = ",
                     format(f_hi), "); choose an interval bracketing one root."))
  }
  stats::uniroot(fn, lower = lower, upper = upper, tol = 1e-10)$root
}

## simplify.R
## Pre-grad algebraic simplifier — Tier 4 change `add-algebraic-simplifier`.
##
## `.algebraic_simplify(expr)` walks an R AST bottom-up and folds 9 always-safe
## algebraic identities. Hooked into `.grad_expr` entry so catalog dispatch
## sees the simplified form; this restores Tier 1 fast-path opportunity for
## expressions like `exp(log(sum(v^2)))` that would otherwise route through
## the Tier 3 walker fall-through path.
##
## Idempotent: each fold produces a form that doesn't re-match the same rule.

.algebraic_simplify <- function(expr) {
  # Strip top-level parens up front so subsequent matches see the bare AST.
  expr <- .strip_paren(expr)
  if (!is.call(expr)) return(expr)
  # L_4 binder nodes are opaque to the simplifier (Decision 6): its rules
  # assume a symbol absent from the free-variable set is a constant, which is
  # false for the bound symbol inside the body.
  if (.is_binder_call(expr)) return(expr)

  # Recurse first into arguments (bottom-up: simplify children before parent).
  # Strip parens after each child-recurse so e.g. (x) anywhere collapses to x.
  for (i in seq_along(expr)[-1L]) {
    expr[[i]] <- .strip_paren(.algebraic_simplify(expr[[i]]))
  }

  if (!is.symbol(expr[[1L]])) return(expr)
  op <- as.character(expr[[1L]])

  # Rule 10 (add-simplify-extensions): literal-only constant folding. A
  # binary arithmetic node whose operands (after child recursion) are all
  # finite numeric/integer literals collapses to its evaluated scalar. Folds
  # only when the result is finite, so 1/0 stays 1/0 (no Inf literal injected)
  # and IEEE semantics are preserved at runtime. Re-association of literal
  # siblings under an associative operator (e.g. 2*x*4 → 8*x) is deliberately
  # deferred (design open question 2) to avoid shifting variable-term
  # accumulation order; only fully-literal sub-trees fold.
  if (op %in% c("+", "-", "*", "/", "^") && length(expr) == 3L &&
      .all_literals(expr)) {
    # eval() here is safe by the .all_literals guard: the node contains only
    # numeric/integer literals combined with {+,-,*,/,^} — no symbols, no
    # function calls. It evaluates pure arithmetic (e.g. 2 + 3), never input.
    folded <- tryCatch(eval(expr),
                       error = function(e) NULL, warning = function(w) NULL)
    if (!is.null(folded) && length(folded) == 1L &&
        is.numeric(folded) && is.finite(folded)) {
      return(folded)
    }
  }

  # Rule 11 (add-simplify-extensions): sqrt(x^2) → abs(x). EXACT over the
  # reals (|x|, never the wrong `→ x` which fails for x < 0). Dormant unless
  # `abs` is in the L_0 catalog — emitting abs() when the engine can't
  # differentiate it would turn a working gradient into an unknown-generator
  # raise, so the fold is gated on abs presence.
  if (op == "sqrt" && length(expr) == 2L) {
    inner <- expr[[2L]]
    if (is.call(inner) && length(inner) == 3L &&
        is.symbol(inner[[1L]]) && as.character(inner[[1L]]) == "^" &&
        (identical(inner[[3L]], 2) || identical(inner[[3L]], 2L)) &&
        .abs_registered()) {
      return(bquote(abs(.(inner[[2L]]))))
    }
  }

  # Rule 12 (add-simplify-extensions): conservative trig identities. Both are
  # domain-TOTAL (hold for all real x), so they preserve value and gradient
  # everywhere. (a) sin(x)^2 + cos(x)^2 → 1 (either order, same syntactic x);
  # (b) 2 * sin(x) * cos(x) → sin(2 * x) (literal 2 in any factor position).
  # Domain-narrowing identities (tan, half-angle) are excluded.
  if (op == "+" && length(expr) == 3L) {
    x <- .pythag_arg(expr[[2L]], expr[[3L]])
    if (!is.null(x)) return(1)
  }
  if (op == "*" && length(expr) == 3L) {
    x <- .double_angle_arg(expr)
    if (!is.null(x)) return(bquote(sin(2 * .(x))))
  }

  # Rule 1: exp(log(x)) → x   (inverse pair; AST-level fold safe because
  # log of non-positive R values returns NaN at evaluation time anyway)
  if (op == "exp" && length(expr) == 2L) {
    inner <- expr[[2L]]
    if (is.call(inner) && length(inner) == 2L &&
        is.symbol(inner[[1L]]) && as.character(inner[[1L]]) == "log") {
      return(inner[[2L]])
    }
  }

  # Rule 2: log(exp(x)) → x   (inverse pair; always safe)
  if (op == "log" && length(expr) == 2L) {
    inner <- expr[[2L]]
    if (is.call(inner) && length(inner) == 2L &&
        is.symbol(inner[[1L]]) && as.character(inner[[1L]]) == "exp") {
      return(inner[[2L]])
    }
  }

  # Rules 3 & 4: power identities.  Only fold when exponent is the literal
  # 1 or 0 (numeric or integer); symbolic exponents left as-is.
  if (op == "^" && length(expr) == 3L) {
    exponent <- expr[[3L]]
    if (identical(exponent, 1) || identical(exponent, 1L)) return(expr[[2L]])
    if (identical(exponent, 0) || identical(exponent, 0L)) return(1)
  }

  # Rules 5 & 6: additive identity (0+x, x+0)
  if (op == "+" && length(expr) == 3L) {
    a <- expr[[2L]]; b <- expr[[3L]]
    if (identical(a, 0) || identical(a, 0L)) return(b)
    if (identical(b, 0) || identical(b, 0L)) return(a)
  }

  # Rule 7: additive identity for subtraction (x-0)
  # Note: 0-x → -x is NOT in this rule set — would emit a unary minus call,
  # changing the AST shape in ways downstream dispatchers don't expect.
  if (op == "-" && length(expr) == 3L) {
    b <- expr[[3L]]
    if (identical(b, 0) || identical(b, 0L)) return(expr[[2L]])
  }

  # Rules 8 & 9: multiplicative identity & annihilator (1*x, x*1, 0*x, x*0)
  # Annihilator (0*x → 0) checked before identity to avoid recursion on 0*1.
  if (op == "*" && length(expr) == 3L) {
    a <- expr[[2L]]; b <- expr[[3L]]
    if (identical(a, 0) || identical(a, 0L)) return(0)
    if (identical(b, 0) || identical(b, 0L)) return(0)
    if (identical(a, 1) || identical(a, 1L)) return(b)
    if (identical(b, 1) || identical(b, 1L)) return(a)
  }

  expr
}

# .all_literals(node): TRUE iff node is a finite numeric/logical literal or an
# arithmetic call over {+,-,*,/,^} whose operands are all literals. Used to gate
# constant folding so only fully-literal sub-trees evaluate.
.all_literals <- function(node) {
  if (is.numeric(node) || is.logical(node)) return(TRUE)
  if (!is.call(node) || !is.symbol(node[[1L]])) return(FALSE)
  if (!(as.character(node[[1L]]) %in% c("+", "-", "*", "/", "^"))) return(FALSE)
  all(vapply(as.list(node)[-1L], .all_literals, logical(1)))
}

# .abs_registered(): TRUE iff an `abs` rule is present in the active L_0 catalog
# (gates the dormant sqrt(x^2) → abs(x) fold).
.abs_registered <- function() {
  !is.null(.dat_env$catalog) && !is.null(.dat_env$catalog$L_0[["abs"]])
}

# .is_fn_call(node, fn): TRUE iff node is `fn(arg)` (single-argument call).
.is_fn_call <- function(node, fn) {
  is.call(node) && length(node) == 2L && is.symbol(node[[1L]]) &&
    as.character(node[[1L]]) == fn
}

# .sq_of(node, fn): if node is `fn(x)^2`, return x; else NULL.
.sq_of <- function(node, fn) {
  if (is.call(node) && length(node) == 3L && is.symbol(node[[1L]]) &&
      as.character(node[[1L]]) == "^" &&
      (identical(node[[3L]], 2) || identical(node[[3L]], 2L)) &&
      .is_fn_call(node[[2L]], fn)) {
    return(node[[2L]][[2L]])
  }
  NULL
}

# .pythag_arg(a, b): if {a, b} == {sin(x)^2, cos(x)^2} with the same syntactic
# x (either order), return x; else NULL.
.pythag_arg <- function(a, b) {
  sa <- .sq_of(a, "sin"); cb <- .sq_of(b, "cos")
  if (!is.null(sa) && !is.null(cb) && identical(sa, cb)) return(sa)
  ca <- .sq_of(a, "cos"); sb <- .sq_of(b, "sin")
  if (!is.null(ca) && !is.null(sb) && identical(ca, sb)) return(ca)
  NULL
}

# .double_angle_arg(expr): if expr is a product whose flattened factors are
# exactly {literal 2, sin(x), cos(x)} with the same syntactic x, return x;
# else NULL.
.double_angle_arg <- function(expr) {
  factors <- list()
  collect <- function(e) {
    if (is.call(e) && length(e) == 3L && is.symbol(e[[1L]]) &&
        as.character(e[[1L]]) == "*") {
      collect(e[[2L]]); collect(e[[3L]])
    } else {
      factors[[length(factors) + 1L]] <<- e
    }
  }
  collect(expr)
  if (length(factors) != 3L) return(NULL)
  is_two <- vapply(factors, function(f) identical(f, 2) || identical(f, 2L), logical(1))
  if (sum(is_two) != 1L) return(NULL)
  rest <- factors[!is_two]
  sin_x <- NULL; cos_x <- NULL
  for (f in rest) {
    if (.is_fn_call(f, "sin")) sin_x <- f[[2L]]
    else if (.is_fn_call(f, "cos")) cos_x <- f[[2L]]
  }
  if (!is.null(sin_x) && !is.null(cos_x) && identical(sin_x, cos_x)) return(sin_x)
  NULL
}

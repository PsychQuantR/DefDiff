## utils.R
## Shared helpers for AST inspection and expression building.

# TRUE if x is a symbol equal to the variable named by `var`.
.is_var <- function(x, var) {
  is.symbol(x) && identical(as.character(x), var)
}

# TRUE if expr contains the variable `var` anywhere in its AST.
.contains_var <- function(expr, var) {
  if (is.symbol(expr)) return(identical(as.character(expr), var))
  if (is.numeric(expr) || is.logical(expr) || is.character(expr)) return(FALSE)
  if (is.call(expr)) {
    # L_4 binder nodes (integral / implicit): the third element is a bound
    # symbol. An occurrence of `var` inside the body (second element) is bound
    # when `var` equals that symbol and therefore not free; the bounds /
    # interval (elements 4+) are always scanned. See l4_nodes.R, Decision 2.
    binder <- .is_binder_call(expr)
    if (binder) {
      .check_binder_node(expr)
      if (identical(as.character(expr[[3L]]), var)) {
        for (i in seq_along(expr)[-c(1L, 2L, 3L)]) {
          if (.contains_var(expr[[i]], var)) return(TRUE)
        }
        return(FALSE)
      }
    }
    for (i in seq_along(expr)[-1L]) {
      if (binder && i == 3L) next
      if (is.null(expr[[i]])) next
      if (.contains_var(expr[[i]], var)) return(TRUE)
    }
    return(FALSE)
  }
  FALSE
}

# Convert a character variable name to a symbol.
.as_var <- function(var) {
  as.symbol(var)
}

# Strip outer parentheses recursively.
.strip_paren <- function(expr) {
  while (is.call(expr) && identical(expr[[1L]], quote(`(`))) {
    expr <- expr[[2L]]
  }
  expr
}

# Normalise an R call object: strip parentheses, drop wrapping noise.
#
# Returns the expression unchanged for non-call inputs; for calls, removes
# `(`-wrappers and recurses one layer for the head.
parse_expr <- function(expr) {
  .strip_paren(expr)
}

# Detect control-flow constructs that cannot be differentiated symbolically.
# Returns the offending construct name, or NA_character_ if none found.
.control_flow_block <- function(expr) {
  blockers <- c("if", "for", "while", "repeat", "function", "<-", "=", "{")
  if (is.call(expr)) {
    if (is.symbol(expr[[1L]])) {
      head <- as.character(expr[[1L]])
      if (head %in% blockers) return(head)
    } else {
      # Compound / qualified head (e.g. `DefDiff::integral(...)` in a generated
      # gradient body): scan the head as a subexpression instead of coercing
      # it to a character vector (which would give a length-3 condition).
      sub <- .control_flow_block(expr[[1L]])
      if (!is.na(sub)) return(sub)
    }
    for (i in seq_along(expr)[-1L]) {
      if (is.null(expr[[i]])) next
      sub <- .control_flow_block(expr[[i]])
      if (!is.na(sub)) return(sub)
    }
  }
  NA_character_
}

# Build a numeric literal zero of length compatible with `var` at runtime.
.zero_like_var <- function(var) {
  bquote(rep(0, length(.(.as_var(var)))))
}

# Smart product: simplify obvious cases (multiplying by 1 or 0).
.smart_mul <- function(a, b) {
  if (identical(a, 1) || identical(a, 1L)) return(b)
  if (identical(b, 1) || identical(b, 1L)) return(a)
  if (identical(a, 0) || identical(a, 0L)) return(0)
  if (identical(b, 0) || identical(b, 0L)) return(0)
  bquote(.(a) * .(b))
}

# Smart addition: simplify multiplying by 0.
.smart_add <- function(a, b) {
  if (identical(a, 0) || identical(a, 0L)) return(b)
  if (identical(b, 0) || identical(b, 0L)) return(a)
  bquote(.(a) + .(b))
}

# Smart subtraction.
.smart_sub <- function(a, b) {
  if (identical(b, 0) || identical(b, 0L)) return(a)
  if (identical(a, 0) || identical(a, 0L)) return(bquote(-.(b)))
  bquote(.(a) - .(b))
}

# Smart unary negation.
.smart_neg <- function(a) {
  if (identical(a, 0) || identical(a, 0L)) return(0)
  bquote(-.(a))
}

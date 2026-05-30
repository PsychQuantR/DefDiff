## fast_dispatch.R
## Tier 1 (add-vdsp-fast-path): strict pattern predicate + platform check.
## Tier 2a (add-vdsp-fast-path-tier2a-normalization): AST normalizer
## extending pattern coverage to isomorphic variants (commutative swap,
## unary-minus folding, outer negation, integer-literal coercion). All
## variants reduce to canonical `<scalar> * <var>` and dispatch through
## existing fast_scalar_mul kernel — no new C++ code.

# Cached Metal-availability state (add-metal-backend). Session-local; reset by
# devtools::load_all (a fresh namespace gets a fresh env).
.dat_metal_state <- new.env(parent = emptyenv())

#' Is the Metal compute backend available? (internal, exported for dispatch)
#'
#' Returns TRUE only on macOS, with a Metal-enabled build, when the metallib
#' loads and the compute pipeline initializes (result cached). Returns FALSE
#' otherwise and never raises, so the gradient fast path can fall back to the
#' vDSP / base-R kernels. Exported (despite the dot prefix) so the gradient
#' bodies the engine emits can reference it without `:::`.
#'
#' @return TRUE if a Metal scalar-multiply pipeline is ready, FALSE otherwise.
#' @export
.metal_path_available <- function() {
  cached <- .dat_metal_state$available
  if (!is.null(cached)) return(cached)
  ok <- tryCatch({
    is_mac <- identical(Sys.info()[["sysname"]], "Darwin")
    has_fn <- exists("metal_scalar_mul_init", where = asNamespace("DefDiff"),
                     inherits = FALSE)
    lib <- if (is_mac && has_fn) {
      system.file("metal", "scalar_mul.metallib", package = "DefDiff")
    } else ""
    is_mac && has_fn && nzchar(lib) && file.exists(lib) &&
      isTRUE(metal_scalar_mul_init(lib))
  }, error = function(e) FALSE)
  .dat_metal_state$available <- ok
  ok
}

# .is_scalar_var_product(expr, var)
#
# Returns TRUE iff `expr` is a call of the form `<num> * <var_sym>` where
# the literal is a length-1 finite numeric and the symbol's name equals
# `var`. Returns FALSE for any other input without raising any condition.
.is_scalar_var_product <- function(expr, var) {
  if (!is.call(expr)) return(FALSE)
  if (length(expr) != 3L) return(FALSE)
  if (!identical(expr[[1L]], as.name("*"))) return(FALSE)

  lit <- expr[[2L]]
  sym <- expr[[3L]]

  if (!is.numeric(lit)) return(FALSE)
  if (length(lit) != 1L) return(FALSE)
  if (!is.finite(lit)) return(FALSE)

  if (!is.symbol(sym)) return(FALSE)
  if (!identical(as.character(sym), var)) return(FALSE)

  TRUE
}

# .fast_path_available()
#
# Runtime check for vDSP backend availability. Returns TRUE on macOS where
# fast_scalar_mul is linked against Apple Accelerate; FALSE elsewhere.
.fast_path_available <- function() {
  identical(Sys.info()[["sysname"]], "Darwin")
}

# .extract_literal_scalar(x)
#
# Helper for normalizer: extracts a double-valued scalar from an AST node
# that is either a bare finite numeric/integer literal or a unary-minus
# call wrapping such a literal. Returns NULL otherwise. Integer literals
# are coerced to double per Design Decision 2.
.extract_literal_scalar <- function(x) {
  if (is.numeric(x) && length(x) == 1L && is.finite(x)) {
    return(as.double(x))
  }
  if (is.call(x) && length(x) == 2L && identical(x[[1L]], as.name("-"))) {
    inner <- x[[2L]]
    if (is.numeric(inner) && length(inner) == 1L && is.finite(inner)) {
      return(-as.double(inner))
    }
  }
  NULL
}

# .try_normalize_scalar_var_product(expr, var, depth = 0L)
#
# Recognizes isomorphic variants of canonical `<scalar> * <var>` and
# reduces to canonical form. Returns list(scalar = <double>, var = <symbol>)
# on success, NULL on failure. Never raises a condition.
#
# Recognized variants (Design Behavior 1):
#   - Strict canonical: `<num> * <var>`
#   - Commutative swap: `<var> * <num>`
#   - Unary-minus on literal: `(-)(<num>) * <var>` or `<var> * (-)(<num>)`
#   - Outer negation: `(-)( <inner> )` where inner matches any of the above
#   - Integer literals: coerced to double via .extract_literal_scalar
#
# Depth cap (Design Decision 4): one outer negation peel permitted; deeper
# patterns return NULL.
.try_normalize_scalar_var_product <- function(expr, var, depth = 0L) {
  # R parser wraps the argument of unary minus in parens when the argument
  # is itself a call: quote(-(2 * v)) has expr[[2]] == quote((2 * v)).
  # Strip them via the engine's existing helper so recursion sees the
  # un-wrapped inner expression.
  expr <- .strip_paren(expr)

  # Rule 5: outer negation `(-)( <inner> )`
  if (is.call(expr) && length(expr) == 2L && identical(expr[[1L]], as.name("-"))) {
    if (depth >= 1L) return(NULL)
    inner <- .try_normalize_scalar_var_product(expr[[2L]], var, depth + 1L)
    if (is.null(inner)) return(NULL)
    return(list(scalar = -inner$scalar, var = inner$var))
  }

  # Rules 1-4: must be `<X> * <Y>` form
  if (!is.call(expr)) return(NULL)
  if (length(expr) != 3L) return(NULL)
  if (!identical(expr[[1L]], as.name("*"))) return(NULL)

  lhs <- expr[[2L]]
  rhs <- expr[[3L]]

  # Rules 1 + 3: lhs is scalar (possibly unary-minus wrapped), rhs is var
  if (is.symbol(rhs) && identical(as.character(rhs), var)) {
    s <- .extract_literal_scalar(lhs)
    if (!is.null(s)) return(list(scalar = s, var = rhs))
  }

  # Rules 2 + 4: lhs is var, rhs is scalar (possibly unary-minus wrapped)
  if (is.symbol(lhs) && identical(as.character(lhs), var)) {
    s <- .extract_literal_scalar(rhs)
    if (!is.null(s)) return(list(scalar = s, var = lhs))
  }

  NULL
}

# .is_scalar_evaluable(expr, var)
#
# Tier 2d helper. TRUE iff `expr` is a recognized scalar-evaluable AST per
# dat-performance spec: sum(<any>) | crossprod(<any>, <any>) | recursively
# cos|sin|exp|log|tanh|sqrt(<inner>) where <inner> is scalar-evaluable.
# Returns FALSE for any other input. Never raises a condition.
#
# Note: the recognition is structural — at runtime the outer expression
# evaluates to a scalar only if the recognized form is well-typed. We trust
# the dispatch path (gradient AST always well-typed scalar-of-vector).
.is_scalar_evaluable <- function(expr, var) {
  expr <- .strip_paren(expr)
  # Base: finite numeric literal is scalar-evaluable
  if (is.numeric(expr) && length(expr) == 1L && is.finite(expr)) return(TRUE)
  # A bare symbol that is NOT the variable: assume scalar-evaluable (constant
  # bound in enclosing env). This is structural; runtime will fail if it
  # turns out vector-typed.
  if (is.symbol(expr)) {
    return(!identical(as.character(expr), var))
  }
  if (!is.call(expr)) return(FALSE)
  if (!is.symbol(expr[[1L]])) return(FALSE)
  fname <- as.character(expr[[1L]])

  # Base reductions: sum, crossprod
  if (fname == "sum") return(TRUE)
  if (fname == "crossprod" && length(expr) == 3L) return(TRUE)

  # Recursive scalar functions of scalar
  if (fname %in% c("cos", "sin", "exp", "log", "tanh", "sqrt") &&
      length(expr) == 2L) {
    return(.is_scalar_evaluable(expr[[2L]], var))
  }

  # Tier 2e fix 3a: scalar arithmetic compositions. For ops in {+, -, *, /, ^},
  # all operands must be scalar-evaluable. Unary (length=2) and binary
  # (length=3) forms both recognized.
  if (fname %in% c("+", "-", "*", "/", "^")) {
    if (length(expr) == 2L) {
      return(.is_scalar_evaluable(expr[[2L]], var))
    }
    if (length(expr) == 3L) {
      return(.is_scalar_evaluable(expr[[2L]], var) &&
             .is_scalar_evaluable(expr[[3L]], var))
    }
  }

  FALSE
}

# .try_normalize_scalar_var_product_with_outer(expr, var)
#
# Tier 2d composite normalizer. Recognizes `<scalar_evaluable> * <Tier_2a>`
# (and commutative swap). Returns list(outer_expr, scalar, var) on success,
# NULL otherwise. Never raises a condition.
.try_normalize_scalar_var_product_with_outer <- function(expr, var) {
  expr <- .strip_paren(expr)
  if (!is.call(expr)) return(NULL)
  if (length(expr) != 3L) return(NULL)
  if (!identical(expr[[1L]], as.name("*"))) return(NULL)

  lhs <- .strip_paren(expr[[2L]])
  rhs <- .strip_paren(expr[[3L]])

  # Try outer = lhs, inner = rhs
  if (.is_scalar_evaluable(lhs, var)) {
    # First try Tier 2a inner (canonical scalar-var product)
    inner <- .try_normalize_scalar_var_product(rhs, var)
    if (!is.null(inner)) {
      return(list(inner_kind = "scalar",
                  outer_expr = .substitute_sum_sq(lhs, var),
                  scalar = inner$scalar, var = inner$var))
    }
    # Tier 2e fix 3b: also try Tier 2c elementwise inner
    inner_ew <- .try_dispatch_elementwise(rhs, var)
    if (!is.null(inner_ew)) {
      return(list(inner_kind = "elementwise",
                  outer_expr = .substitute_sum_sq(lhs, var),
                  kernel_name = inner_ew$kernel_name, var = inner_ew$var))
    }
    # add-simplify-extensions: scalar-power inner `<lit> * <var>^k` (k>=2).
    # This is the shape of grad(<outer>(sum(v^k))) for k>=3, where the
    # var-part is v^(k-1); substituting sum-powers in the outer scalar makes
    # fast_sum_pow reachable from a real gradient body.
    inner_pow <- .try_normalize_scalar_pow(rhs, var)
    if (!is.null(inner_pow)) {
      return(list(inner_kind = "scalar_pow",
                  outer_expr = .substitute_sum_sq(lhs, var),
                  scalar = inner_pow$scalar, exponent = inner_pow$exponent,
                  var = inner_pow$var))
    }
  }

  # Try outer = rhs, inner = lhs (commutative)
  if (.is_scalar_evaluable(rhs, var)) {
    inner <- .try_normalize_scalar_var_product(lhs, var)
    if (!is.null(inner)) {
      return(list(inner_kind = "scalar",
                  outer_expr = .substitute_sum_sq(rhs, var),
                  scalar = inner$scalar, var = inner$var))
    }
    inner_ew <- .try_dispatch_elementwise(lhs, var)
    if (!is.null(inner_ew)) {
      return(list(inner_kind = "elementwise",
                  outer_expr = .substitute_sum_sq(rhs, var),
                  kernel_name = inner_ew$kernel_name, var = inner_ew$var))
    }
    inner_pow <- .try_normalize_scalar_pow(lhs, var)
    if (!is.null(inner_pow)) {
      return(list(inner_kind = "scalar_pow",
                  outer_expr = .substitute_sum_sq(rhs, var),
                  scalar = inner_pow$scalar, exponent = inner_pow$exponent,
                  var = inner_pow$var))
    }
  }

  NULL
}

# .substitute_sum_sq(expr, var)
#
# Tier 2d extension: substitute `sum(<var>^2)` patterns inside the outer
# scalar expression with `fast_sum_sq(<var>)` call. Eliminates intermediate
# v^2 allocation (800MB at n=1e8) and per-element R `^` overhead by routing
# to vDSP_svesqD single-pass sum-of-squares.
#
# Only substitutes the exact pattern; more general substitutions (sum(v*v),
# crossprod(v, v) → fast_sum_sq) left to future Tier.
.substitute_sum_sq <- function(expr, var) {
  if (!is.call(expr)) return(expr)
  if (length(expr) == 2L && identical(expr[[1L]], as.name("sum"))) {
    inner <- .strip_paren(expr[[2L]])
    if (is.call(inner) && length(inner) == 3L &&
        identical(inner[[1L]], as.name("^"))) {
      base <- .strip_paren(inner[[2L]])
      exponent <- .strip_paren(inner[[3L]])
      # add-simplify-extensions: generalize from k == 2 to any integer k >= 2.
      # k == 2 keeps the single-pass fast_sum_sq (vDSP_svesqD); k >= 3 routes to
      # the two-pass fast_sum_pow (vvpow + vDSP_sveD). The original (double)
      # exponent literal is preserved in the emitted call. Non-integer / k < 2 /
      # compound-base fall through unchanged.
      if (is.symbol(base) && identical(as.character(base), var) &&
          is.numeric(exponent) && length(exponent) == 1L &&
          is.finite(exponent) && exponent == round(exponent) && exponent >= 2) {
        if (exponent == 2) {
          return(bquote(fast_sum_sq(.(base))))
        }
        return(bquote(fast_sum_pow(.(base), .(exponent))))
      }
    }
  }
  # Tier 2e fix 2: also substitute crossprod(<var>, <var>) -> fast_sum_sq(<var>)
  # (mathematically equivalent for double-precision input; vDSP_svesqD avoids
  # allocating the 1x1 matrix that crossprod returns).
  if (length(expr) == 3L && identical(expr[[1L]], as.name("crossprod"))) {
    a1 <- .strip_paren(expr[[2L]])
    a2 <- .strip_paren(expr[[3L]])
    if (is.symbol(a1) && is.symbol(a2) &&
        identical(as.character(a1), var) &&
        identical(as.character(a2), var)) {
      return(bquote(fast_sum_sq(.(a1))))
    }
  }
  # Tier 2e fix 4: substitute bare vForce calls <fn>(<var>) -> fast_vv_<fn>(<var>)
  # when fn is in the Tier 2c kernel map. Speeds up outer scalar expressions
  # like `1 - tanh(sum(sin(v)))^2` by routing `sin(v)` to vForce SIMD.
  if (length(expr) == 2L && is.symbol(expr[[1L]])) {
    fname <- as.character(expr[[1L]])
    arg <- .strip_paren(expr[[2L]])
    if (is.symbol(arg) && identical(as.character(arg), var) &&
        !is.null(.elementwise_kernel_map[[fname]])) {
      return(call(.elementwise_kernel_map[[fname]], arg))
    }
  }
  for (i in seq_along(expr)[-1L]) {
    expr[[i]] <- .substitute_sum_sq(expr[[i]], var)
  }
  expr
}

# Tier 2c: vForce elementwise dispatch ----------------------------------
# Maps recognized elementwise function names to their fast_vv_* kernel.
.elementwise_kernel_map <- list(
  cos  = "fast_vv_cos",
  sin  = "fast_vv_sin",
  exp  = "fast_vv_exp",
  log  = "fast_vv_log",
  tanh = "fast_vv_tanh",
  sqrt = "fast_vv_sqrt"
)

# .try_dispatch_elementwise(expr, var)
#
# Tier 2c bare-elementwise recognizer. TRUE iff expr is a recognized
# elementwise call `<fn>(<var>)` where fn is in the Tier 2c kernel map.
# Returns list(kernel_name, var) on success, NULL otherwise.
.try_dispatch_elementwise <- function(expr, var) {
  expr <- .strip_paren(expr)
  if (!is.call(expr)) return(NULL)
  if (length(expr) != 2L) return(NULL)
  if (!is.symbol(expr[[1L]])) return(NULL)
  fn_name <- as.character(expr[[1L]])
  if (is.null(.elementwise_kernel_map[[fn_name]])) return(NULL)
  arg <- .strip_paren(expr[[2L]])
  if (!is.symbol(arg) || !identical(as.character(arg), var)) return(NULL)
  list(kernel_name = .elementwise_kernel_map[[fn_name]], var = arg)
}

# .try_normalize_scalar_var_elementwise(expr, var)
#
# Tier 2c scaled-elementwise recognizer. Recognizes `<num> * <fn>(<var>)`
# and commutative swap. Returns list(scalar, kernel_name, var) on success,
# NULL otherwise.
.try_normalize_scalar_var_elementwise <- function(expr, var) {
  expr <- .strip_paren(expr)

  # Outer negation on bare elementwise: -<fn>(<var>) → scalar = -1
  if (is.call(expr) && length(expr) == 2L && identical(expr[[1L]], as.name("-"))) {
    inner <- .try_dispatch_elementwise(.strip_paren(expr[[2L]]), var)
    if (!is.null(inner)) {
      return(list(scalar = -1, kernel_name = inner$kernel_name, var = inner$var))
    }
  }

  if (!is.call(expr)) return(NULL)
  if (length(expr) != 3L) return(NULL)
  if (!identical(expr[[1L]], as.name("*"))) return(NULL)

  lhs <- .strip_paren(expr[[2L]])
  rhs <- .strip_paren(expr[[3L]])

  # Try lhs = scalar, rhs = elementwise call
  s <- .extract_literal_scalar(lhs)
  if (!is.null(s)) {
    inner <- .try_dispatch_elementwise(rhs, var)
    if (!is.null(inner)) {
      return(list(scalar = s, kernel_name = inner$kernel_name, var = inner$var))
    }
  }

  # Try rhs = scalar, lhs = elementwise call (commutative)
  s <- .extract_literal_scalar(rhs)
  if (!is.null(s)) {
    inner <- .try_dispatch_elementwise(lhs, var)
    if (!is.null(inner)) {
      return(list(scalar = s, kernel_name = inner$kernel_name, var = inner$var))
    }
  }

  NULL
}

# Tier 2e fix 1: reciprocal vForce dispatch.
# Recognizes <num> / (<num> * <vForce>(v)) and <num> / <vForce>(v) patterns.
.try_normalize_reciprocal_vforce <- function(expr, var) {
  expr <- .strip_paren(expr)
  if (!is.call(expr)) return(NULL)
  if (length(expr) != 3L) return(NULL)
  if (!identical(expr[[1L]], as.name("/"))) return(NULL)

  numerator_node <- .strip_paren(expr[[2L]])
  denom <- .strip_paren(expr[[3L]])

  num <- .extract_literal_scalar(numerator_node)
  if (is.null(num)) return(NULL)

  # Case A: denom is bare elementwise <vForce>(v)
  inner_ew <- .try_dispatch_elementwise(denom, var)
  if (!is.null(inner_ew)) {
    return(list(numerator = num,
                kernel_name = inner_ew$kernel_name,
                var = inner_ew$var))
  }

  # Case B: denom is <num> * <vForce>(v) — collapse to (num/inner_num)
  if (is.call(denom) && length(denom) == 3L &&
      identical(denom[[1L]], as.name("*"))) {
    lhs <- .strip_paren(denom[[2L]])
    rhs <- .strip_paren(denom[[3L]])
    s <- .extract_literal_scalar(lhs)
    if (!is.null(s)) {
      inner <- .try_dispatch_elementwise(rhs, var)
      if (!is.null(inner) && s != 0) {
        return(list(numerator = num / s,
                    kernel_name = inner$kernel_name,
                    var = inner$var))
      }
    }
    s <- .extract_literal_scalar(rhs)
    if (!is.null(s)) {
      inner <- .try_dispatch_elementwise(lhs, var)
      if (!is.null(inner) && s != 0) {
        return(list(numerator = num / s,
                    kernel_name = inner$kernel_name,
                    var = inner$var))
      }
    }
  }

  NULL
}

# Tier 3 fix 3b: scalar-power dispatch.
# Recognizes <num_lit> * <var_sym>^<int_lit> (k >= 2) — exactly what the
# Tier 3 recursive walker produces for sum(v^k) gradients (after smart_mul
# folding: k * v^(k-1)).
.try_normalize_scalar_pow <- function(expr, var) {
  expr <- .strip_paren(expr)
  if (!is.call(expr)) return(NULL)
  if (length(expr) != 3L) return(NULL)
  if (!identical(expr[[1L]], as.name("*"))) return(NULL)

  lhs <- .strip_paren(expr[[2L]])
  rhs <- .strip_paren(expr[[3L]])

  # Extract scalar literal (left)
  s <- .extract_literal_scalar(lhs)
  if (is.null(s)) {
    # Try commutative: scalar on right
    s <- .extract_literal_scalar(rhs)
    if (is.null(s)) return(NULL)
    pow_part <- lhs
  } else {
    pow_part <- rhs
  }
  pow_part <- .strip_paren(pow_part)

  # pow_part must be <var>^<int_lit>
  if (!is.call(pow_part) || length(pow_part) != 3L) return(NULL)
  if (!identical(pow_part[[1L]], as.name("^"))) return(NULL)

  base <- .strip_paren(pow_part[[2L]])
  exponent <- .strip_paren(pow_part[[3L]])

  if (!is.symbol(base) || !identical(as.character(base), var)) return(NULL)
  if (!is.numeric(exponent) || length(exponent) != 1L || !is.finite(exponent)) return(NULL)
  k <- as.integer(exponent)
  if (k != exponent || k < 2L) return(NULL)  # only integer k >= 2

  list(scalar = s, var = base, exponent = k)
}

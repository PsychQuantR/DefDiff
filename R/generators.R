## generators.R
## Default generator catalog for L_0, L_1, L_2, L_3.
##
## Each generator rule has signature `function(expr, var) -> expression`
## and returns the gradient expression. Pattern matching for compound
## expressions (e.g. `sum(v^2)` as norm-squared, `crossprod(v, A %*% v)`
## as quadratic form) happens inside the relevant rule.

# ========== L_0: pure vector space (linear combinations) ==========

.register_L0 <- function() {
  # ∇(a + b) = ∇a + ∇b
  .dat_env$catalog$L_0[["+"]] <- function(expr, var) {
    a <- expr[[2L]]; b <- expr[[3L]]
    .smart_add(.grad_expr(a, var), .grad_expr(b, var))
  }

  # Unary or binary `-`
  .dat_env$catalog$L_0[["-"]] <- function(expr, var) {
    if (length(expr) == 2L) {
      .smart_neg(.grad_expr(expr[[2L]], var))
    } else {
      .smart_sub(.grad_expr(expr[[2L]], var), .grad_expr(expr[[3L]], var))
    }
  }

  # Scalar multiplication / Hadamard product.
  # Constant-factor case: c * expr → c * ∇expr.
  # Both-contain-var case: product rule (valid for scalar args).
  .dat_env$catalog$L_0[["*"]] <- function(expr, var) {
    a <- expr[[2L]]; b <- expr[[3L]]
    a_has <- .contains_var(a, var)
    b_has <- .contains_var(b, var)
    if (!a_has && !b_has) return(0)
    if (!a_has) return(.smart_mul(a, .grad_expr(b, var)))
    if (!b_has) return(.smart_mul(b, .grad_expr(a, var)))
    .smart_add(.smart_mul(b, .grad_expr(a, var)),
               .smart_mul(a, .grad_expr(b, var)))
  }

  # Scalar division a / b. Three cases:
  #   1. Neither contains var → 0
  #   2. b constant w.r.t. var → grad(a) / b (existing fast path preserved)
  #   3. b contains var → inline quotient rule (a'b - ab') / b^2 with
  #      .grad_expr for sub-derivatives (Tier 4 change
  #      `close-top-level-division-gap`).
  #      Using .grad_expr (not .grad_inner) so sub-derivatives go through
  #      the full catalog dispatch — .grad_inner doesn't know L_1 generators
  #      like sum/crossprod, but sub-expressions can legitimately contain them.
  .dat_env$catalog$L_0[["/"]] <- function(expr, var) {
    a <- expr[[2L]]; b <- expr[[3L]]
    a_has <- .contains_var(a, var)
    b_has <- .contains_var(b, var)
    if (!a_has && !b_has) return(0)
    if (!b_has) return(bquote(.(.grad_expr(a, var)) / .(b)))
    tryCatch({
      da <- .grad_expr(a, var)
      db <- .grad_expr(b, var)
      bquote((.(da) * as.numeric(.(b)) - as.numeric(.(a)) * .(db)) / as.numeric(.(b))^2)
    }, DefDiff_unknown_generator = function(e) {
      .dat_stop("DefDiff_unknown_generator",
                paste0("Top-level division ", deparse(expr),
                       " contains unrecognized generator: ",
                       conditionMessage(e)))
    })
  }

  # rep() data-shape primitive (Tier 4 change `add-rep-generator`).
  # When the value being repeated does not contain the differentiation
  # variable, the output is a constant vector and gradient is 0.
  # Variable in first argument would change output shape across coordinates
  # (sparse-Jacobian semantics) — outside single-var scalar contract.
  # The second argument (times / length.out) may contain length(v) etc.
  # because length is a fixed property at gradient evaluation time, not a
  # value-level dependence; we only check the first argument.
  .dat_env$catalog$L_0[["rep"]] <- function(expr, var) {
    values <- expr[[2L]]
    if (!.contains_var(values, var)) return(0)
    .dat_stop("DefDiff_not_definable",
              paste0("rep() with variable-dependent first argument is ",
                     "outside DD's single-variable scalar-output contract; ",
                     "would change output-shape semantics across coordinates."))
  }

  # Note: %*% is intentionally NOT registered at L_0. The pre-existing L_2
  # entry in .register_L2() handles the bare-matmul raise + correct level()
  # classification (matmul expressions are L_2 per the DAT hierarchy). The
  # three supported patterns (sum(W %*% v), crossprod(v, W %*% v),
  # crossprod(W %*% v, c)) fire BEFORE catalog lookup via fast-path branches
  # in .sum_rule and .crossprod_rule. Walker-context calls (e.g., the inner
  # of sum(tanh(W %*% v))) hit walker's L_0 fallback miss and raise generic
  # DefDiff_unknown_generator wrapped to DefDiff_not_definable by .sum_rule.
}

# ========== L_1: inner product ==========

.register_L1 <- function() {
  .dat_env$catalog$L_1[["sum"]]       <- .sum_rule
  .dat_env$catalog$L_1[["crossprod"]] <- .crossprod_rule
  .dat_env$catalog$L_1[["^"]]         <- .pow_rule
}

# Shared helper for Tier 4 family-2 Pattern B (product of two vForces).
# Given a variable symbol and two vForce function names from {sin, cos, exp},
# returns the gradient AST for sum(f(v) * g(v)) [also valid for the
# mathematically-equivalent crossprod(f(v), g(v))]. Returns NULL if either
# function name is outside the supported set.
#
# The function-pair is normalized via sort() so (cos, sin) and (sin, cos)
# both dispatch to the same key. Sign handling: (cos, cos) requires a
# leading negative since both derivatives carry a sign change.
.product_of_vforces_grad <- function(vsym, fn_a, fn_b) {
  if (!(fn_a %in% c("sin", "cos", "exp")) ||
      !(fn_b %in% c("sin", "cos", "exp"))) {
    return(NULL)
  }
  pair_key <- paste(sort(c(fn_a, fn_b)), collapse = "_")
  switch(pair_key,
    "sin_sin" = bquote(fast_vec_add(
      fast_vec_mul(fast_vv_cos(.(vsym)), fast_vv_sin(.(vsym))),
      fast_vec_mul(fast_vv_sin(.(vsym)), fast_vv_cos(.(vsym))))),
    "cos_sin" = bquote(fast_vec_sub(
      fast_vec_mul(fast_vv_cos(.(vsym)), fast_vv_cos(.(vsym))),
      fast_vec_mul(fast_vv_sin(.(vsym)), fast_vv_sin(.(vsym))))),
    "exp_sin" = bquote(fast_vec_add(
      fast_vec_mul(fast_vv_cos(.(vsym)), fast_vv_exp(.(vsym))),
      fast_vec_mul(fast_vv_sin(.(vsym)), fast_vv_exp(.(vsym))))),
    "cos_cos" = bquote(fast_scalar_mul(-1, fast_vec_add(
      fast_vec_mul(fast_vv_sin(.(vsym)), fast_vv_cos(.(vsym))),
      fast_vec_mul(fast_vv_cos(.(vsym)), fast_vv_sin(.(vsym)))))),
    "cos_exp" = bquote(fast_vec_sub(
      fast_vec_mul(fast_vv_cos(.(vsym)), fast_vv_exp(.(vsym))),
      fast_vec_mul(fast_vv_sin(.(vsym)), fast_vv_exp(.(vsym))))),
    "exp_exp" = bquote(fast_vec_add(
      fast_vec_mul(fast_vv_exp(.(vsym)), fast_vv_exp(.(vsym))),
      fast_vec_mul(fast_vv_exp(.(vsym)), fast_vv_exp(.(vsym))))),
    NULL)
}

.sum_rule <- function(expr, var) {
  if (length(expr) != 2L) {
    .dat_stop("DefDiff_not_definable", "sum() with multiple arguments is not supported in v0.1.")
  }
  inner <- .strip_paren(expr[[2L]])
  vsym  <- .as_var(var)

  # sum(v^k) → k * v^(k-1)
  if (is.call(inner) && identical(inner[[1L]], quote(`^`)) && .is_var(inner[[2L]], var)) {
    k <- inner[[3L]]
    if (identical(k, 2) || identical(k, 2L)) return(bquote(2 * .(vsym)))
    return(bquote(.(k) * .(vsym)^.(k - 1)))
  }

  # sum(v * w) with v the variable and w constant → w
  if (is.call(inner) && identical(inner[[1L]], quote(`*`))) {
    a <- inner[[2L]]; b <- inner[[3L]]
    if (.is_var(a, var) && .is_var(b, var)) return(bquote(2 * .(vsym)))
    if (.is_var(a, var) && !.contains_var(b, var)) return(b)
    if (.is_var(b, var) && !.contains_var(a, var)) return(a)
  }

  # sum(v * <vforce_fn>(v)) — product-of-var-vForce fast path (Tier 4
  # `add-product-vforce-fastpath`). Family 1 of the walker-fast-paths
  # series. Gradient via product rule: d/dv sum(v * f(v)) = f(v) + v*f'(v).
  # Emit as composition of existing fast kernels — no new C++.
  if (is.call(inner) && length(inner) == 3L && identical(inner[[1L]], quote(`*`))) {
    a <- inner[[2L]]; b <- inner[[3L]]
    vforce_call <- NULL
    # Bare var on left, vForce(var) on right
    if (.is_var(a, var) && is.call(b) && length(b) == 2L &&
        is.symbol(b[[1L]]) && .is_var(b[[2L]], var)) {
      vforce_call <- b
    } else if (.is_var(b, var) && is.call(a) && length(a) == 2L &&
               is.symbol(a[[1L]]) && .is_var(a[[2L]], var)) {
      vforce_call <- a
    }
    if (!is.null(vforce_call)) {
      fn_name <- as.character(vforce_call[[1L]])
      fast_ast <- switch(fn_name,
        sin = bquote(fast_vec_add(fast_vv_sin(.(vsym)),
                                  fast_vec_mul(.(vsym), fast_vv_cos(.(vsym))))),
        cos = bquote(fast_vec_sub(fast_vv_cos(.(vsym)),
                                  fast_vec_mul(.(vsym), fast_vv_sin(.(vsym))))),
        exp = bquote(fast_vec_add(fast_vv_exp(.(vsym)),
                                  fast_vec_mul(.(vsym), fast_vv_exp(.(vsym))))),
        NULL)
      if (!is.null(fast_ast)) return(fast_ast)
    }
  }

  # sum(<vforce_fn1>(v) * <vforce_fn2>(v)) — product-of-two-vForces fast
  # path (Tier 4 `add-walker-family2-fastpath`, family 2 Pattern B).
  # Gradient via product rule: d/dv (f(v) * g(v)) = f'(v)*g(v) + f(v)*g'(v).
  # Commutative; pair normalized via .product_of_vforces_grad helper.
  if (is.call(inner) && length(inner) == 3L && identical(inner[[1L]], quote(`*`))) {
    a <- inner[[2L]]; b <- inner[[3L]]
    if (is.call(a) && length(a) == 2L && is.symbol(a[[1L]]) &&
        .is_var(a[[2L]], var) &&
        is.call(b) && length(b) == 2L && is.symbol(b[[1L]]) &&
        .is_var(b[[2L]], var)) {
      fast_ast <- .product_of_vforces_grad(vsym,
                                           as.character(a[[1L]]),
                                           as.character(b[[1L]]))
      if (!is.null(fast_ast)) return(fast_ast)
    }
  }

  # sum(v / <vforce_fn>(v)) — quotient-with-vForce-denom fast path (Tier 4
  # `add-walker-family2-fastpath`, family 2 Pattern A). Gradient via
  # quotient rule: d/dv sum(v/f(v)) = (f(v) - v*f'(v)) / f(v)^2.
  # NOT commutative for `/` — only numerator-is-var case supported.
  if (is.call(inner) && length(inner) == 3L && identical(inner[[1L]], quote(`/`)) &&
      .is_var(inner[[2L]], var) &&
      is.call(inner[[3L]]) && length(inner[[3L]]) == 2L &&
      is.symbol(inner[[3L]][[1L]]) && .is_var(inner[[3L]][[2L]], var)) {
    fn_name <- as.character(inner[[3L]][[1L]])
    fast_ast <- switch(fn_name,
      sin = bquote(fast_vec_div(
        fast_vec_sub(fast_vv_sin(.(vsym)),
                     fast_vec_mul(.(vsym), fast_vv_cos(.(vsym)))),
        fast_vec_mul(fast_vv_sin(.(vsym)), fast_vv_sin(.(vsym))))),
      cos = bquote(fast_vec_div(
        fast_vec_add(fast_vv_cos(.(vsym)),
                     fast_vec_mul(.(vsym), fast_vv_sin(.(vsym)))),
        fast_vec_mul(fast_vv_cos(.(vsym)), fast_vv_cos(.(vsym))))),
      exp = bquote(fast_vec_div(
        fast_vec_sub(fast_vv_exp(.(vsym)),
                     fast_vec_mul(.(vsym), fast_vv_exp(.(vsym)))),
        fast_vec_mul(fast_vv_exp(.(vsym)), fast_vv_exp(.(vsym))))),
      NULL)
    if (!is.null(fast_ast)) return(fast_ast)
  }

  # sum(v^2 +/- <vforce_fn>(v)) — additive composition fast path (Tier 4
  # `add-walker-family3-fastpath`, family 3 Pattern A). Gradient via
  # sum/difference rule: d/dv (v^2 +/- f(v)) = 2*v +/- f'(v).
  # Supports both operand orders + both +/- operators. Only v^2 (k=2);
  # other powers fall through.
  if (is.call(inner) && length(inner) == 3L &&
      (identical(inner[[1L]], quote(`+`)) || identical(inner[[1L]], quote(`-`)))) {
    op <- inner[[1L]]; a <- inner[[2L]]; b <- inner[[3L]]
    # Detect which operand is v^2 (the power-2 term) and which is vForce.
    is_v2 <- function(e) {
      is.call(e) && length(e) == 3L && identical(e[[1L]], quote(`^`)) &&
        .is_var(e[[2L]], var) &&
        (identical(e[[3L]], 2) || identical(e[[3L]], 2L))
    }
    is_vforce <- function(e) {
      is.call(e) && length(e) == 2L && is.symbol(e[[1L]]) &&
        as.character(e[[1L]]) %in% c("sin", "cos", "exp") &&
        .is_var(e[[2L]], var)
    }
    v2_first <- is_v2(a) && is_vforce(b)
    v2_second <- is_vforce(a) && is_v2(b)
    if (v2_first || v2_second) {
      vforce_call <- if (v2_first) b else a
      fn_name <- as.character(vforce_call[[1L]])
      # f'(v) AST and whether its sign is naturally negative (only cos→-sin).
      f_deriv <- switch(fn_name,
        sin = bquote(fast_vv_cos(.(vsym))),  # +cos
        cos = bquote(fast_vv_sin(.(vsym))),  # -sin (sign carried below)
        exp = bquote(fast_vv_exp(.(vsym))))  # +exp
      f_deriv_negative <- (fn_name == "cos")
      two_v <- bquote(fast_scalar_mul(2, .(vsym)))
      # Determine emit operator from input op, operand order, and f' sign.
      #   sum(v^2 + f(v)): grad = 2v + f' → use add (or sub if f'<0)
      #   sum(v^2 - f(v)): grad = 2v - f' → use sub (or add if f'<0)
      #   sum(f(v) + v^2): grad = f' + 2v → same as +order with operands swapped, equals fast_vec_add(2v, f') for commutativity
      #   sum(f(v) - v^2): grad = f' - 2v → fast_vec_sub(f', 2v)  (order-sensitive!)
      input_is_plus <- identical(op, quote(`+`))
      if (v2_first) {
        # v^2 first: 2v <op> f'  (mind f_deriv_negative)
        # For `+`: 2v + f'   → if f'<0 emit sub(2v, |f'|), else add(2v, f')
        # For `-`: 2v - f'   → if f'<0 emit add(2v, |f'|), else sub(2v, f')
        if (input_is_plus) {
          fast_ast <- if (f_deriv_negative) bquote(fast_vec_sub(.(two_v), .(f_deriv)))
                      else bquote(fast_vec_add(.(two_v), .(f_deriv)))
        } else {
          fast_ast <- if (f_deriv_negative) bquote(fast_vec_add(.(two_v), .(f_deriv)))
                      else bquote(fast_vec_sub(.(two_v), .(f_deriv)))
        }
      } else {
        # vForce first: f' <op> 2v   (commutative for +, ORDER MATTERS for -)
        if (input_is_plus) {
          # f' + 2v = 2v + f' (commutative); emit same shape as v2_first +
          fast_ast <- if (f_deriv_negative) bquote(fast_vec_sub(.(two_v), .(f_deriv)))
                      else bquote(fast_vec_add(.(two_v), .(f_deriv)))
        } else {
          # f' - 2v : NOT commutative. Emit with f' as left operand.
          # If f'<0 (cos case): result is -sin - 2v = -(sin + 2v), use scalar_mul(-1, add)
          fast_ast <- if (f_deriv_negative)
            bquote(fast_scalar_mul(-1, fast_vec_add(.(f_deriv), .(two_v))))
          else
            bquote(fast_vec_sub(.(f_deriv), .(two_v)))
        }
      }
      return(fast_ast)
    }
  }

  # sum(<vforce_fn>(k * v)) — chain-through-scalar-multiply fast path (Tier 4
  # `add-walker-family3-fastpath`, family 3 Pattern B). Gradient via chain
  # rule: d/dv f(k*v) = k * f'(k*v). Emit as body block to bind k*v once.
  # k MUST be a numeric literal (not symbol or expression).
  if (is.call(inner) && length(inner) == 2L && is.symbol(inner[[1L]])) {
    fn_name <- as.character(inner[[1L]])
    if (fn_name %in% c("sin", "cos", "exp")) {
      arg <- .strip_paren(inner[[2L]])
      if (is.call(arg) && length(arg) == 3L && identical(arg[[1L]], quote(`*`))) {
        # Detect (k * v) or (v * k) with k a numeric literal.
        k <- NULL
        if (is.numeric(arg[[2L]]) && length(arg[[2L]]) == 1L && .is_var(arg[[3L]], var)) {
          k <- as.numeric(arg[[2L]])
        } else if (is.numeric(arg[[3L]]) && length(arg[[3L]]) == 1L && .is_var(arg[[2L]], var)) {
          k <- as.numeric(arg[[3L]])
        }
        if (!is.null(k)) {
          signed_k <- if (fn_name == "cos") -k else k
          deriv_fn <- switch(fn_name,
            sin = as.name("fast_vv_cos"),
            cos = as.name("fast_vv_sin"),
            exp = as.name("fast_vv_exp"))
          return(bquote({
            kv <- fast_scalar_mul(.(k), .(vsym))
            fast_scalar_mul(.(signed_k), .(deriv_fn)(kv))
          }))
        }
      }
    }
  }

  # sum(W %*% v) — linear matrix-vector form (Tier 4 `add-matmul-generator`).
  # Detection: inner is `W %*% v` where v is the var, W doesn't contain v.
  # Gradient: d/dv_k sum_i (Wv)_i = d/dv_k sum_i sum_j W[i,j] v[j]
  #         = sum_i W[i,k] = colSums(W)[k]
  # Emit in canonical matrix form: t(W) %*% rep(1, nrow(W))
  if (is.call(inner) && length(inner) == 3L &&
      identical(inner[[1L]], quote(`%*%`)) &&
      .is_var(inner[[3L]], var) &&
      !.contains_var(inner[[2L]], var)) {
    W <- inner[[2L]]
    return(bquote(t(.(W)) %*% rep(1, nrow(.(W)))))
  }

  # sum(v) → rep(1, length(v))
  if (.is_var(inner, var)) return(bquote(rep(1, length(.(vsym)))))

  # sum(<elementwise>(v)) for fn in {sin, cos, exp, log, tanh, sqrt}
  # Tier 2c catalog extension: enables grad(function(v) sum(sin(v))) etc.
  # Gradient AST emitted in standard symbolic form; Tier 2c dispatch hook
  # in fast_dispatch.R routes the elementwise call to fast_vv_* vForce kernel.
  if (is.call(inner) && length(inner) == 2L && is.symbol(inner[[1L]])) {
    fn_name <- as.character(inner[[1L]])
    inner_arg <- .strip_paren(inner[[2L]])
    if (.is_var(inner_arg, var)) {
      deriv <- switch(fn_name,
                      sin  = bquote(cos(.(vsym))),
                      cos  = bquote(-sin(.(vsym))),
                      exp  = bquote(exp(.(vsym))),
                      log  = bquote(1 / .(vsym)),
                      tanh = bquote(1 - tanh(.(vsym))^2),
                      sqrt = bquote(1 / (2 * sqrt(.(vsym)))),
                      NULL)
      if (!is.null(deriv)) return(deriv)
    }
  }

  # sum(<elementwise>(W %*% v)) — single-layer NN forward (Tier 5 Option A,
  # `add-elementwise-matmul-fastpath`). Gradient: t(W) %*% f'(W %*% v).
  # Emits a body block binding Wv (and z for tanh) to avoid recomputation.
  # Scope: f ∈ {sin, cos, exp, tanh}; W constant; v bare. Multi-layer
  # composition (sum(f(W2 %*% f(W1 %*% v)))) NOT closed — Tier 5 Option B
  # walker extension territory.
  if (is.call(inner) && length(inner) == 2L && is.symbol(inner[[1L]])) {
    fn_name <- as.character(inner[[1L]])
    if (fn_name %in% c("sin", "cos", "exp", "tanh")) {
      arg <- .strip_paren(inner[[2L]])
      if (is.call(arg) && length(arg) == 3L &&
          identical(arg[[1L]], quote(`%*%`)) &&
          !.contains_var(arg[[2L]], var) &&
          .is_var(arg[[3L]], var)) {
        W <- arg[[2L]]
        fast_ast <- switch(fn_name,
          sin  = bquote({
            Wv <- .(W) %*% .(vsym)
            t(.(W)) %*% fast_vv_cos(Wv)
          }),
          cos  = bquote({
            Wv <- .(W) %*% .(vsym)
            -t(.(W)) %*% fast_vv_sin(Wv)
          }),
          exp  = bquote({
            Wv <- .(W) %*% .(vsym)
            t(.(W)) %*% fast_vv_exp(Wv)
          }),
          tanh = bquote({
            Wv <- .(W) %*% .(vsym)
            z <- fast_vv_tanh(Wv)
            t(.(W)) %*% (1 - z^2)
          }))
        if (!is.null(fast_ast)) return(fast_ast)
      }
    }
  }

  # sum(<elementwise-matmul chain>) — multi-layer NN forward (Tier 5
  # Option B-lite, `add-nn-walker-extension`). Subsumes the single-layer
  # case but only fires for chains that the simpler Option A above didn't
  # already handle (2+ matmuls or anything Option A's narrow shape rejected).
  # Helper raises DefDiff_not_definable for non-chain shapes; we catch and fall
  # through to the existing walker path.
  if (is.call(inner)) {
    head_sym <- inner[[1L]]
    is_chain_shape <- (identical(head_sym, quote(`%*%`)) && length(inner) == 3L) ||
      (length(inner) == 2L && is.symbol(head_sym) &&
       as.character(head_sym) %in% c("sin", "cos", "exp", "tanh"))
    if (is_chain_shape) {
      chain_result <- tryCatch(
        .elementwise_matmul_chain_grad(inner, var),
        DefDiff_not_definable = function(e) NULL
      )
      if (!is.null(chain_result) && !is.null(chain_result$jacobian)) {
        return(bquote(as.numeric(colSums(.(chain_result$jacobian)))))
      }
    }
  }

  # sum(constant) → 0
  if (!.contains_var(inner, var)) return(0)

  # Tier 3 closure completion: any remaining L_3 expression that didn't match
  # the fast-path rules above delegates to the recursive .grad_inner walker.
  # This honors the closure thesis operationally — every declared L_3
  # expression has a working gradient via standard differentiation rules.
  # Phase 4 (add-walker-shape-extension): .finalize_reduction_grad selects
  # the legacy AST when available, else invokes pullback with rep(1, length(value)).
  tryCatch(
    .simplify_ast(.finalize_reduction_grad(.grad_inner(inner, var), inner)),
    DefDiff_unknown_generator = function(e) {
      .dat_stop("DefDiff_not_definable",
                paste0("sum(", deparse(inner),
                       ") contains unrecognized generator: ", conditionMessage(e)))
    }
  )
}

.crossprod_rule <- function(expr, var) {
  # Single-argument form: crossprod(a) ≡ crossprod(a, a)
  if (length(expr) == 2L) {
    a <- .strip_paren(expr[[2L]])
    if (.is_var(a, var)) return(bquote(2 * .(.as_var(var))))
    if (is.call(a) && identical(a[[1L]], quote(`%*%`)) && .is_var(a[[3L]], var)) {
      A <- a[[2L]]
      return(bquote(2 * t(.(A)) %*% (.(A) %*% .(.as_var(var)))))
    }
    if (!.contains_var(a, var)) return(0)
    .dat_stop("DefDiff_not_definable",
              paste0("crossprod(", deparse(a), ") not in v0.1 catalog."))
  }

  a <- .strip_paren(expr[[2L]])
  b <- .strip_paren(expr[[3L]])
  a_has <- .contains_var(a, var)
  b_has <- .contains_var(b, var)
  vsym  <- .as_var(var)

  if (!a_has && !b_has) return(0)

  # crossprod(v, v) → 2v
  if (.is_var(a, var) && .is_var(b, var)) return(bquote(2 * .(vsym)))

  # crossprod(v, w) with w constant → w  (and symmetric)
  if (.is_var(a, var) && !b_has) return(b)
  if (.is_var(b, var) && !a_has) return(a)

  # crossprod(v, A %*% v) → (A + t(A)) %*% v  (quadratic form)
  if (.is_var(a, var) && is.call(b) && identical(b[[1L]], quote(`%*%`)) && .is_var(b[[3L]], var)) {
    A <- b[[2L]]
    return(bquote((.(A) + t(.(A))) %*% .(vsym)))
  }
  if (.is_var(b, var) && is.call(a) && identical(a[[1L]], quote(`%*%`)) && .is_var(a[[3L]], var)) {
    A <- a[[2L]]
    return(bquote((.(A) + t(.(A))) %*% .(vsym)))
  }

  # crossprod(A %*% v, A %*% v) → 2 t(A) %*% (A %*% v)
  if (is.call(a) && identical(a[[1L]], quote(`%*%`)) && .is_var(a[[3L]], var) &&
      is.call(b) && identical(b[[1L]], quote(`%*%`)) && .is_var(b[[3L]], var) &&
      identical(a[[2L]], b[[2L]])) {
    A <- a[[2L]]
    return(bquote(2 * t(.(A)) %*% (.(A) %*% .(vsym))))
  }

  # crossprod(W %*% v, c) for constant c (Tier 4 `add-matmul-generator`).
  # Gradient: d/dv c · (Wv) = t(W) %*% c (direction-projection pattern).
  # Symmetric variant crossprod(c, W %*% v) gives same gradient.
  if (is.call(a) && identical(a[[1L]], quote(`%*%`)) && .is_var(a[[3L]], var) &&
      !.contains_var(a[[2L]], var) && !b_has) {
    W <- a[[2L]]
    return(bquote(t(.(W)) %*% .(b)))
  }
  if (is.call(b) && identical(b[[1L]], quote(`%*%`)) && .is_var(b[[3L]], var) &&
      !.contains_var(b[[2L]], var) && !a_has) {
    W <- b[[2L]]
    return(bquote(t(.(W)) %*% .(a)))
  }

  # crossprod(v, <vforce_fn>(v)) — product-of-var-vForce fast path (Tier 4
  # `add-product-vforce-fastpath`). Same gradient as sum(v * f(v)) since
  # crossprod(v, f(v)) = sum(v * f(v)). Emit identical fast-kernel AST.
  vforce_call <- NULL
  if (.is_var(a, var) && is.call(b) && length(b) == 2L &&
      is.symbol(b[[1L]]) && .is_var(b[[2L]], var)) {
    vforce_call <- b
  } else if (.is_var(b, var) && is.call(a) && length(a) == 2L &&
             is.symbol(a[[1L]]) && .is_var(a[[2L]], var)) {
    vforce_call <- a
  }
  if (!is.null(vforce_call)) {
    fn_name <- as.character(vforce_call[[1L]])
    fast_ast <- switch(fn_name,
      sin = bquote(fast_vec_add(fast_vv_sin(.(vsym)),
                                fast_vec_mul(.(vsym), fast_vv_cos(.(vsym))))),
      cos = bquote(fast_vec_sub(fast_vv_cos(.(vsym)),
                                fast_vec_mul(.(vsym), fast_vv_sin(.(vsym))))),
      exp = bquote(fast_vec_add(fast_vv_exp(.(vsym)),
                                fast_vec_mul(.(vsym), fast_vv_exp(.(vsym))))),
      NULL)
    if (!is.null(fast_ast)) return(fast_ast)
  }

  # crossprod(<vforce_fn1>(v), <vforce_fn2>(v)) — product-of-two-vForces
  # fast path (Tier 4 `add-walker-family2-fastpath`, family 2 Pattern B).
  # Mathematically equivalent to sum(f(v) * g(v)); shares the same gradient
  # AST helper. Commutative; pair normalized via .product_of_vforces_grad.
  if (is.call(a) && length(a) == 2L && is.symbol(a[[1L]]) &&
      .is_var(a[[2L]], var) &&
      is.call(b) && length(b) == 2L && is.symbol(b[[1L]]) &&
      .is_var(b[[2L]], var)) {
    fast_ast <- .product_of_vforces_grad(vsym,
                                         as.character(a[[1L]]),
                                         as.character(b[[1L]]))
    if (!is.null(fast_ast)) return(fast_ast)
  }

  # Tier 3 closure: crossprod(a(v), b(v)) for elementwise vector-valued a, b
  # is mathematically sum(a(v) * b(v)). Use product rule:
  # d/dv (a(v) * b(v)) = a'(v) * b(v) + a(v) * b'(v).
  # Delegate to .grad_inner on the product form. Phase 4: finalize via the
  # consumer-side .finalize_reduction_grad to dispatch legacy vs pullback.
  product_expr <- bquote(.(a) * .(b))
  tryCatch(
    .simplify_ast(.finalize_reduction_grad(.grad_inner(product_expr, var), product_expr)),
    DefDiff_unknown_generator = function(e) {
      .dat_stop("DefDiff_not_definable",
                paste0("crossprod(", deparse(a), ", ", deparse(b),
                       ") contains unrecognized generator: ", conditionMessage(e)))
    }
  )
}

# Power rule for scalar exponents on the variable.
# Handles scalar `x^k` (k constant) — used both for L_0 polynomial and L_1
# vector-element cases (the sum_rule has its own fast-path).
.pow_rule <- function(expr, var) {
  base <- expr[[2L]]; exponent <- expr[[3L]]
  if (.contains_var(exponent, var)) {
    .dat_stop("DefDiff_not_definable",
              "Variable in exponent position is not in v0.1 catalog.")
  }
  if (!.contains_var(base, var)) return(0)
  # ∇(b^k) = k * b^(k-1) * ∇b
  d_base <- .grad_expr(base, var)
  factor <- bquote(.(exponent) * .(base)^.(exponent - 1))
  .smart_mul(factor, d_base)
}

# ========== L_2: named operators (level-tagging entries) ==========

.register_L2 <- function() {
  # Matrix-vector product. Bare A %*% v is vector-valued, not scalar; the
  # crossprod_rule above handles the scalar wrappings (quadratic forms).
  # Registering it here lets level() recognise L_2 expressions.
  .dat_env$catalog$L_2[["%*%"]] <- function(expr, var) {
    .dat_stop(
      "DefDiff_not_definable",
      "Bare matrix-vector product A %*% v is vector-valued; wrap inside crossprod() or sum() for a scalar gradient."
    )
  }

  .dat_env$catalog$L_2[["t"]] <- function(expr, var) {
    .dat_stop("DefDiff_not_definable",
              "Bare t() is matrix-valued; wrap inside a scalar reduction.")
  }
}

# ========== L_3: analytic scalar functions ==========

.register_L3 <- function() {
  outer_derivatives <- list(
    sin  = function(arg) bquote(cos(.(arg))),
    cos  = function(arg) bquote(-sin(.(arg))),
    exp  = function(arg) bquote(exp(.(arg))),
    log  = function(arg) bquote(1 / .(arg)),
    sqrt = function(arg) bquote(1 / (2 * sqrt(.(arg)))),
    tanh = function(arg) bquote(1 - tanh(.(arg))^2),
    atan = function(arg) bquote(1 / (1 + (.(arg))^2))
  )
  # Factory captures `d_outer` properly via force() and returns a closure;
  # avoids the for-loop late-binding pitfall and the locked-binding error
  # from using <<- inside local() in package namespaces.
  make_rule <- function(d_outer) {
    force(d_outer)
    function(expr, var) {
      arg <- .strip_paren(expr[[2L]])
      if (!.contains_var(arg, var)) return(0)
      d_arg <- .grad_expr(arg, var)
      .smart_mul(d_outer(arg), d_arg)
    }
  }
  for (fname in names(outer_derivatives)) {
    .dat_env$catalog$L_3[[fname]] <- make_rule(outer_derivatives[[fname]])
  }
}

## fused_jit.R
## Backend B (add-fused-jit-grad-backend): lower a fusable arithmetic gradient
## AST into a single-pass C++ kernel, compiled at runtime via the system clang,
## dlopen'd, cached, and applied across worker threads. This file holds the
## R-side lowering and fusability analysis; the threaded native driver lives in
## src/fused_jit_runtime.cpp and the runtime-compile/cache glue below.
##
## Two senses of dimension-independence are preserved (issue #2):
##   1. symbolic — grad_expr() still returns the n-free `3 * v^2` (set in grad.R)
##   2. kernel   — the compiled kernel takes n as a RUNTIME argument; the cache
##                 key is the canonical per-element residual + dtype, NEVER n.

# Default size threshold below which the fused JIT path is not worth the
# compile + dispatch + thread-spawn overhead (the existing vDSP / plain-R path
# runs instead). Calibrated against the 12-cell jax benchmark (#2): the fixed
# overhead (~0.3-0.5 ms) dominates at n=1e6 (fused sum(v^3) = 1.03 ms LOSES to
# the base path there), while fused wins decisively from n=1e7 (0.68 ms vs vDSP
# 2.77 ms). The crossover is ~3-5e6; 1e7 is the conservative "only fuse where
# the win is large and certain" default.
.DAT_JIT_THRESHOLD_DEFAULT <- 1e7L

# Auto-tune coarse floor: below this length, base always wins (the fused path's
# thread-spawn + dispatch fixed overhead dominates), so never probe — just run
# base. Avoids the tiny-n thread-spawn trap that inflated #4's small-n numbers.
.DAT_AUTOTUNE_FLOOR <- 1e5

# .is_fusable_elementwise(expr, var)
#
# TRUE iff `expr` is a fusable element-wise arithmetic gradient body. A
# subexpression that does not reference `var` except inside a reduction
# (sum / crossprod) is "scalar" (hoistable) — recognized via the existing
# .is_scalar_evaluable. After hoisting, every remaining per-element node must
# be the variable, a numeric constant, or an arithmetic combinator
# {+, -, *, /, ^, unary -}. A transcendental applied directly to the vector
# (e.g. cos(v)), a matrix product, or a second vector variable makes it
# non-fusable. Never raises.
.is_fusable_elementwise <- function(expr, var) {
  expr <- .strip_paren(expr)

  # Scalar subexpression (constant, non-var symbol, sum(...), cos(scalar), ...)
  # is hoistable to the R preamble — fusable as a leaf.
  if (.is_scalar_evaluable(expr, var)) return(TRUE)

  # The variable element itself.
  if (is.symbol(expr)) return(identical(as.character(expr), var))

  if (is.numeric(expr) || is.logical(expr)) return(TRUE)

  if (!is.call(expr)) return(FALSE)
  if (!is.symbol(expr[[1L]])) return(FALSE)
  fname <- as.character(expr[[1L]])

  # Power: base must be fusable and the exponent must be scalar (so `v^2` is
  # fusable but `2^v` — a per-element transcendental — is not).
  if (fname == "^" && length(expr) == 3L) {
    return(.is_fusable_elementwise(expr[[2L]], var) &&
             .is_scalar_evaluable(.strip_paren(expr[[3L]]), var))
  }

  # Arithmetic combinators: every operand must be fusable. Covers unary minus
  # (length 2) and binary + - * / (length 3).
  if (fname %in% c("+", "-", "*", "/")) {
    if (length(expr) == 2L) return(.is_fusable_elementwise(expr[[2L]], var))
    if (length(expr) == 3L) {
      return(.is_fusable_elementwise(expr[[2L]], var) &&
               .is_fusable_elementwise(expr[[3L]], var))
    }
  }

  FALSE
}

# .fused_var_count(expr, var) — number of times the bare variable symbol
# appears outside a scalar (reduction) context. Used to skip trivial 1-pass
# shapes (bare `v`, `-v`) that are not worth JIT-compiling.
.fused_var_count <- function(expr, var) {
  expr <- .strip_paren(expr)
  if (.is_scalar_evaluable(expr, var)) return(0L)
  if (is.symbol(expr)) return(if (identical(as.character(expr), var)) 1L else 0L)
  if (!is.call(expr)) return(0L)
  total <- 0L
  for (i in seq_along(expr)[-1L]) total <- total + .fused_var_count(expr[[i]], var)
  total
}

# .fused_has_var_pow(expr, var) — TRUE if a `var^k` (var base) node appears,
# i.e. the residual is nonlinear in the variable and would be multi-pass.
.fused_has_var_pow <- function(expr, var) {
  expr <- .strip_paren(expr)
  # A var^k inside a reduction (sum(v^2)) is a hoisted scalar, not a per-element
  # power — skip scalar subtrees so 1-pass shapes like cos(sum(v^2))*2*v are
  # not mis-classified as multi-pass.
  if (.is_scalar_evaluable(expr, var)) return(FALSE)
  if (!is.call(expr)) return(FALSE)
  if (is.symbol(expr[[1L]]) && identical(as.character(expr[[1L]]), "^") &&
      length(expr) == 3L) {
    base <- .strip_paren(expr[[2L]])
    if (is.symbol(base) && identical(as.character(base), var)) return(TRUE)
  }
  for (i in seq_along(expr)[-1L]) {
    if (.fused_has_var_pow(expr[[i]], var)) return(TRUE)
  }
  FALSE
}

# .fused_should_fire(expr, var) — gate the fused branch to genuinely multi-pass
# fusable shapes: fusable AND (var appears >= 2 times OR a var-power is present).
# Trivial 1-pass shapes (bare v, -v, scalar*v already caught upstream) are skipped.
.fused_should_fire <- function(expr, var) {
  if (!.is_fusable_elementwise(expr, var)) return(FALSE)
  if (.fused_var_count(expr, var) < 1L) return(FALSE)  # pure scalar — no benefit
  e <- .strip_paren(expr)
  # Peel a single outer unary minus so -(3*v^2) / -(2*v) still qualify (and a
  # bare -v collapses to v -> excluded as a trivial copy).
  if (is.call(e) && is.symbol(e[[1L]]) &&
      identical(as.character(e[[1L]]), "-") && length(e) == 2L) {
    e <- .strip_paren(e[[2L]])
  }
  # Fire on any binary arithmetic op on the variable — single-pass (2*v) and
  # multi-pass (3*v^2, s*v + t*v, cos(sum(v^2))*2*v) alike. Bare `v` (a symbol,
  # a no-op copy) and pure scalars are excluded.
  is.call(e) && is.symbol(e[[1L]]) &&
    as.character(e[[1L]]) %in% c("+", "-", "*", "/", "^") && length(e) == 3L
}

# .c_double_literal(x) — format an R numeric as a round-trippable C double
# literal (always containing a '.' or 'e' so it is not treated as an int).
.c_double_literal <- function(x) {
  s <- sprintf("%.17g", as.double(x))
  if (!grepl("[.eEnN]", s)) s <- paste0(s, ".0")  # 3 -> 3.0; leaves inf/nan alone
  s
}

# .lower_fused_cpp(expr, var) — lower a fusable arithmetic gradient AST into a
# single-pass C++ kernel. Returns a list:
#   residual      : C expression string in terms of v[i] and s[0..m-1]
#   scalar_exprs  : list of R expressions (one per hoisted scalar), in order;
#                   reductions already substituted to fast_sum_sq / fast_sum_pow
#   cpp_src       : full extern "C" kernel source
#   cache_key     : canonical, dimension-free key (residual string + dtype)
#   kernel_name   : the C symbol name
# Assumes .is_fusable_elementwise(expr, var) is TRUE.
.lower_fused_cpp <- function(expr, var) {
  scalars <- new.env(parent = emptyenv())
  scalars$keys <- character(0)   # deparsed scalar exprs, for dedup + order
  scalars$exprs <- list()

  hoist <- function(node) {
    key <- paste(deparse(node), collapse = " ")
    idx <- match(key, scalars$keys)
    if (is.na(idx)) {
      # Substitute sum(v^k)/crossprod to the fast single-pass reductions so the
      # preamble reuses the A intrinsics rather than allocating v^k.
      scalars$exprs[[length(scalars$exprs) + 1L]] <- .substitute_sum_sq(node, var)
      scalars$keys <- c(scalars$keys, key)
      idx <- length(scalars$keys)
    }
    sprintf("s[%d]", idx - 1L)
  }

  emit <- function(node) {
    node <- .strip_paren(node)
    # Numeric constants inline (3 -> 3.0); checked before .is_scalar_evaluable
    # so literals are not needlessly hoisted into the runtime scalar vector.
    if (is.numeric(node)) return(.c_double_literal(node))
    # The variable element.
    if (is.symbol(node) && identical(as.character(node), var)) return("v[i]")
    # Any other scalar subexpression (non-var symbol, sum(...), cos(scalar), ...)
    # is hoisted to the R preamble.
    if (.is_scalar_evaluable(node, var)) return(hoist(node))
    if (!is.call(node)) {
      .dat_stop("DefDiff_not_definable", "fused lowering hit a non-callable node.")
    }
    fname <- as.character(node[[1L]])
    if (fname == "^" && length(node) == 3L) {
      exp_node <- .strip_paren(node[[3L]])
      base_c <- emit(node[[2L]])
      # Integer exponent k in [2, 8] -> repeated multiplication (clang
      # auto-vectorizes; avoids the slower std::pow call).
      if (is.numeric(exp_node) && length(exp_node) == 1L && is.finite(exp_node) &&
          exp_node == round(exp_node) && exp_node >= 1 && exp_node <= 8) {
        k <- as.integer(exp_node)
        if (k == 1L) return(base_c)
        return(paste0("(", paste(rep(base_c, k), collapse = " * "), ")"))
      }
      return(sprintf("std::pow(%s, %s)", base_c, emit(exp_node)))
    }
    if (fname == "-" && length(node) == 2L) {
      return(sprintf("(-%s)", emit(node[[2L]])))
    }
    if (fname %in% c("+", "-", "*", "/") && length(node) == 3L) {
      return(sprintf("(%s %s %s)", emit(node[[2L]]), fname, emit(node[[3L]])))
    }
    .dat_stop("DefDiff_not_definable",
              paste0("fused lowering hit a non-fusable node `", fname, "`."))
  }

  residual <- emit(expr)
  kernel_name <- "dat_fused_kernel"
  cpp_src <- paste0(
    "#include <cmath>\n",
    'extern "C" void ', kernel_name,
    "(const double* v, double* out, long n, const double* s) {\n",
    "  for (long i = 0; i < n; ++i) out[i] = ", residual, ";\n",
    "}\n"
  )
  list(
    residual = residual,
    scalar_exprs = scalars$exprs,
    cpp_src = cpp_src,
    cache_key = paste0("f64:", residual),
    kernel_name = kernel_name
  )
}

# ===== Runtime compile + warm-up cache + threaded dispatch =====

# Session-scoped JIT state (reset on namespace reload). `cache` maps cache_key
# -> list(addr = <NativeSymbol externalptr>) on success or the string "FAILED".
# `compile_count` counts actual clang invocations (cache misses) for tests.
.dat_jit_state <- new.env(parent = emptyenv())
.dat_jit_state$cache <- new.env(parent = emptyenv())
.dat_jit_state$compile_count <- 0L
.dat_jit_state$available <- NULL
# Auto-tune learned path-choice cache: key "(cache_key, log10(n) bucket)" ->
# "base" | "fused". probe_count counts the timed base-vs-fused probes (tests).
.dat_jit_state$pathchoice <- new.env(parent = emptyenv())
.dat_jit_state$probe_count <- 0L

#' Is the fused-JIT backend available? (internal, exported for dispatch)
#'
#' TRUE only on macOS with a system \code{clang++} present, unless disabled via
#' \code{options(DefDiff.jit_disable = TRUE)}. Result cached. Never raises.
#' Exported (despite the dot prefix) so emitted gradient bodies resolve it
#' without \code{:::}, mirroring \code{.metal_path_available}.
#' @return TRUE if the JIT path can be used, FALSE otherwise.
#' @export
.jit_path_available <- function() {
  if (isTRUE(getOption("DefDiff.jit_disable", FALSE))) return(FALSE)
  cached <- .dat_jit_state$available
  if (!is.null(cached)) return(cached)
  ok <- tryCatch(
    identical(Sys.info()[["sysname"]], "Darwin") && nzchar(Sys.which("clang++")),
    error = function(e) FALSE)
  .dat_jit_state$available <- ok
  ok
}

# Default worker-thread count: the spike showed fusion needs threads to beat
# XLA (single core is bandwidth-ceiling-limited). Cap at 8 physical cores.
.dat_jit_default_threads <- function() {
  nc <- tryCatch(parallel::detectCores(logical = FALSE), error = function(e) NA_integer_)
  if (is.na(nc) || nc < 1L) nc <- 4L
  as.integer(min(8L, nc))
}

# .dat_fused_compile(cache_key, cpp_src, kernel_name)
# Compile cpp_src to a standalone .so via system clang, dyn.load it, and return
# the kernel's NativeSymbol externalptr. Cached by cache_key; failures cached as
# "FAILED" so we never recompile a known-bad shape. Returns NULL on failure.
.dat_fused_compile <- function(cache_key, cpp_src, kernel_name) {
  hit <- .dat_jit_state$cache[[cache_key]]
  if (!is.null(hit)) {
    if (identical(hit, "FAILED")) return(NULL)
    return(hit$addr)
  }
  addr <- tryCatch({
    src <- tempfile(fileext = ".cpp")
    so  <- tempfile(fileext = .Platform$dynlib.ext)
    writeLines(cpp_src, src)
    .dat_jit_state$compile_count <- .dat_jit_state$compile_count + 1L
    st <- system2("clang++",
                  c("-O3", "-mcpu=native", "-std=c++17", "-shared", "-fPIC",
                    shQuote(src), "-o", shQuote(so)),
                  stdout = FALSE, stderr = FALSE)
    if (!identical(st, 0L) || !file.exists(so)) stop("clang compile failed")
    info <- dyn.load(so)
    getNativeSymbolInfo(kernel_name, PACKAGE = info)$address
  }, error = function(e) NULL)
  if (is.null(addr)) {
    .dat_jit_state$cache[[cache_key]] <- "FAILED"
    return(NULL)
  }
  .dat_jit_state$cache[[cache_key]] <- list(addr = addr)
  addr
}

#' Run a fused gradient kernel, or NULL to signal fallback (internal, exported)
#'
#' Compiles (cached) and applies the fused kernel across worker threads.
#' Returns the gradient vector, or \code{NULL} on any failure so the caller
#' falls back to the existing vDSP / plain-R path. Exported so emitted gradient
#' bodies can call it without \code{:::}. Never raises.
#' @param cache_key,cpp_src,kernel_name Lowering outputs from \code{.lower_fused_cpp}.
#' @param v Numeric input vector.
#' @param scalars Numeric vector of hoisted scalars (possibly length 0).
#' @return Numeric gradient vector, or NULL to request fallback.
#' @export
.dat_fused_try <- function(cache_key, cpp_src, kernel_name, v, scalars) {
  tryCatch({
    if (!.jit_path_available()) return(NULL)
    addr <- .dat_fused_compile(cache_key, cpp_src, kernel_name)
    if (is.null(addr)) return(NULL)
    nthreads <- as.integer(getOption("DefDiff.jit_threads", .dat_jit_default_threads()))
    dat_fused_apply(addr, as.double(v),
                    if (length(scalars)) as.double(scalars) else double(0),
                    nthreads)
  }, error = function(e) NULL)
}

# .build_fused_body(gexpr, var, base_body)
# Emit the dispatched gradient body for a fusable multi-pass gradient: compute
# the hoisted scalars, then above the size threshold call the fused kernel,
# else (or on fused failure / below threshold) evaluate `base_body` (the
# existing vDSP / plain-R emission). `n` is read at runtime via length(v), so
# the same emitted body and compiled kernel serve every dimension.
#' Choose base vs fused gradient path by learned measurement (internal, exported)
#'
#' Auto-tune: on the first call for a given (gradient shape, log10(n) bucket),
#' time `base()` and `fused()` once each end-to-end and cache the faster; later
#' calls in that bucket reuse the learned winner (free + optimal). The crossover
#' depends on n (only known at call time) and on the machine, and the two paths
#' have opposite cost structures, so no single path is always fastest — auto-tune
#' measures which wins rather than predicting with a static constant.
#'
#' Reverts to the static threshold gate when `DefDiff.jit_threshold` is set or
#' `DefDiff.autotune=FALSE` (reproducible benchmarks + path-forcing tests). Below
#' `.DAT_AUTOTUNE_FLOOR` always base (the tiny-n thread-spawn trap). Metal lane
#' preserved at `n >= metal_threshold` for Metal-eligible shapes. Returns the
#' gradient vector. Exported so emitted bodies call it unqualified.
#' @keywords internal
#' @export
.dat_auto_dispatch <- function(cache_key, v, fused, base, metal_eligible = FALSE) {
  n <- length(v)

  # Override: explicit threshold OR autotune off -> the prior static gate.
  thr_opt <- getOption("DefDiff.jit_threshold", NULL)
  if (!is.null(thr_opt) || !isTRUE(getOption("DefDiff.autotune", TRUE))) {
    thr <- if (!is.null(thr_opt)) as.numeric(thr_opt) else .DAT_JIT_THRESHOLD_DEFAULT
    fire <- n >= thr
    if (isTRUE(metal_eligible)) fire <- fire && n < .dat_opt_pos_num("DefDiff.metal_threshold", 1e9L)
    if (fire) { r <- fused(); if (!is.null(r)) return(r) }
    return(base())
  }

  if (n < .dat_opt_pos_num("DefDiff.autotune_floor", .DAT_AUTOTUNE_FLOOR))
    return(base())                                                  # tiny n: always base
  if (isTRUE(metal_eligible) && n >= .dat_opt_pos_num("DefDiff.metal_threshold", 1e9L))
    return(base())                                                  # preserve Metal lane

  st <- .dat_jit_state
  key <- paste0(cache_key, "@", floor(2 * log10(n)))               # half-decade bucket
  reprobe <- max(1L, as.integer(min(.dat_opt_pos_num("DefDiff.autotune_reprobe", 64), .Machine$integer.max)))
  reprobe_secs <- .dat_opt_pos_num("DefDiff.autotune_reprobe_secs", 60)
  learned <- st$pathchoice[[key]]
  if (!is.null(learned)) {
    learned$uses <- learned$uses + 1L
    # Re-probe on use-count (#5 HIGH — fast for single-shape hot loops) OR wall-clock
    # age (#7 — shape-independent: a multi-shape workload never accumulates `reprobe`
    # uses on one bucket, so use-count alone never recovers it). Either trigger re-measures.
    aged <- !is.null(learned$probed_at) &&
      (as.numeric(Sys.time()) - learned$probed_at) > reprobe_secs
    if (learned$uses %% reprobe != 0L && !aged) {
      st$pathchoice[[key]] <- learned
      if (identical(learned$choice, "fused")) { r <- fused(); if (!is.null(r)) return(r) }
      return(base())
    }
  }
  uses0 <- if (is.null(learned)) 1L else learned$uses
  now <- as.numeric(Sys.time())

  # Probe: repeat-K-until-measurable timing of both paths; reps>1 takes the min to
  # filter transient spikes. Returns the winner's already-computed value, so only
  # the loser path is extra work. tie within 10% -> base (lower-variance default).
  st$probe_count <- st$probe_count + 1L
  reps <- max(1L, as.integer(min(.dat_opt_pos_num("DefDiff.autotune_reps", 3), .Machine$integer.max)))
  rb <- .dat_probe_timed(base)
  rf <- .dat_probe_timed(fused)
  if (is.null(rf$val)) {                                            # fused unavailable
    st$pathchoice[[key]] <- list(choice = "base", uses = uses0, probed_at = now)
    return(rb$val)
  }
  if (!.dat_paths_agree(rb$val, rf$val)) {                          # correctness cross-check
    warning("DefDiff auto-tune: base and fused gradients disagree for '", key,
            "' — using base (possible lowering bug).", call. = FALSE)
    st$pathchoice[[key]] <- list(choice = "base", uses = uses0, probed_at = now)
    return(rb$val)
  }
  tb <- rb$t; tf <- rf$t
  if (reps > 1L) for (i in seq_len(reps - 1L)) {
    tb <- min(tb, .dat_probe_timed(base)$t)
    tf <- min(tf, .dat_probe_timed(fused)$t)
  }
  winner <- if (tf < tb * 0.90) "fused" else "base"
  st$pathchoice[[key]] <- list(choice = winner, uses = uses0, probed_at = now)
  if (identical(winner, "fused")) rf$val else rb$val
}

# ---- auto-tune probe helpers (internal; placed AFTER .dat_auto_dispatch so the
# `#' @export` roxygen block binds to .dat_auto_dispatch, not these — #6 verify).

#' Coerced finite-positive option read (internal, exported for generated bodies)
#'
#' A garbage option (NA / non-numeric / <= 0) falls back to the default instead
#' of erroring deep in the probe (#6). Exported because gradient-function bodies
#' generated by `grad()` reference it without `:::`.
#' @param name Option name.
#' @param default Fallback value.
#' @return A finite positive number.
#' @keywords internal
#' @export
.dat_opt_pos_num <- function(name, default) {
  val <- suppressWarnings(as.numeric(getOption(name, default))[1L])
  if (length(val) == 0L || is.na(val) || !is.finite(val) || val <= 0) default else val
}

# Repeat-K-until-measurable timing. proc.time elapsed is coarse (~1-16ms), so a
# single small-n run quantizes to 0 and the probe "defaults" instead of measuring
# (#5 verify MEDIUM). Doubling loop until cumulative elapsed clears a resolution
# floor; per-iter = total / iters. Returns the FIRST iter's value (the thunk is a
# pure fn of v). Capped at max_iter. fused() NULL on the first call propagates.
.dat_probe_timed <- function(thunk) {
  floor_s  <- .dat_opt_pos_num("DefDiff.autotune_min_probe_s", 0.005)
  max_iter <- 1024L
  first_val <- NULL
  iters <- 0L
  k <- 1L
  t0 <- proc.time()[["elapsed"]]
  repeat {
    for (i in seq_len(k)) {
      out <- thunk()
      if (iters == 0L && i == 1L) {
        if (is.null(out)) return(list(t = NA_real_, val = NULL))   # unavailable
        first_val <- out
      }
    }
    iters <- iters + k
    elapsed <- proc.time()[["elapsed"]] - t0
    # 2s hard wall-cap bounds a pathological autotune_min_probe_s (#7); the
    # iteration cap alone doesn't bound wall-time at huge n.
    if (elapsed >= floor_s || elapsed >= 2 || iters >= max_iter)
      return(list(t = elapsed / iters, val = first_val))
    k <- k * 2L
  }
}

# Bounded-sample agreement check. A lowering divergence is systematic, so a capped
# sample catches it without a full 1e8 pass (#5 verify LOW). check.attributes=FALSE:
# the base path keeps the input's names/attrs but the fused C kernel returns a bare
# vector, so attribute comparison would spuriously "disagree" on a named input and
# wrongly disable fused (#6 verify MEDIUM).
.dat_paths_agree <- function(a, b) {
  if (length(a) != length(b)) return(FALSE)
  n <- length(a)
  idx <- if (n <= 2e4L) seq_len(n)
         else unique(c(seq_len(1e4L), as.integer(seq.int(1L, n, length.out = 1e4L))))
  isTRUE(all.equal(a[idx], b[idx], tolerance = 1e-8, check.attributes = FALSE))
}

.build_fused_body <- function(gexpr, var, base_body, metal_eligible = FALSE) {
  L <- .lower_fused_cpp(gexpr, var)
  vsym <- as.name(var)
  scalar_call <- if (length(L$scalar_exprs) == 0L) {
    quote(double(0))
  } else {
    as.call(c(list(quote(c)), L$scalar_exprs))
  }
  # Emit base + fused as thunks; .dat_auto_dispatch learns (per shape per machine)
  # which wins at this n. metal_eligible (canonical <scalar>*<var>) is threaded
  # through so the dispatcher preserves the Metal GPU lane at n >= metal_threshold.
  bquote({
    .dat_s <- .(scalar_call)
    .dat_auto_dispatch(
      .(L$cache_key), .(vsym),
      function() .dat_fused_try(.(L$cache_key), .(L$cpp_src), .(L$kernel_name), .(vsym), .dat_s),
      function() .(base_body),
      .(isTRUE(metal_eligible)))
  })
}

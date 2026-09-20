## dd_freeze.R
## #8: bake the learned base-vs-fused dispatch decision into the emitted body.
## The dual of auto-tune (#5-#7): auto-tune LEARNS the winner at runtime and pays
## ~1.6us of R-level dispatch per call; dd_freeze takes the learned answer and
## regenerates the closure body as the bare winning call -- zero getOption, zero
## cache lookup, zero dispatcher. Opt-in; the default dispatch path is untouched.

#' Freeze a gradient function's dispatch decision for a known input size
#'
#' \code{dd_freeze(gf, n_hint)} post-processes a \code{\link{grad}} gradient
#' function: it resolves which evaluation path (base vDSP vs fused-JIT kernel)
#' wins at \code{n_hint} -- reading the auto-tune learned choice, or probing once
#' through the existing dispatcher if unlearned -- and returns a new function
#' whose body is that bare winning call. This removes the per-call dispatch
#' overhead (\code{getOption} reads, cache lookups, the dispatcher call itself,
#' and for canonical shapes the Metal-window check), leaving only the structural
#' floor of an R call plus one \code{.Call}.
#'
#' Trade-offs (by design): the frozen function gives up runtime adaptivity --
#' auto-tune no longer re-selects paths for it, and later \code{options()}
#' changes (e.g. \code{DefDiff.metal_threshold}) do not apply. It is
#' session-scoped: do not serialize frozen functions across package versions
#' (kernel signatures may drift). A wrong \code{n_hint} misplaces performance
#' but never correctness -- every path computes the same gradient. If
#' \code{n_hint} is large and the path is unlearned, the one-time probe
#' allocates a vector of that length.
#'
#' Gradients whose body has no dispatcher (non-fusable shapes such as
#' \code{sum(sin(v))}, already a bare kernel call) are returned unchanged with
#' a message. Unrecognized body shapes are returned unchanged with a warning
#' (fail-open to correct-but-unfrozen, never to wrong).
#'
#' @param gf A gradient function produced by \code{\link{grad}}.
#' @param n_hint The vector length the frozen function will be called with
#'   (required -- the winning path depends on the size bucket).
#' @return A function computing the same gradient with a specialized body.
#'   Attributes: \code{grad_expr} (preserved) and \code{dd_frozen = n_hint}.
#' @seealso \code{\link{grad}}, \code{\link{dd_batch}}
#' @export
#' @examples
#' gf <- grad(function(v) sum(v^3))
#' gfz <- dd_freeze(gf, n_hint = 1e3)
#' gfz(c(1, 2, 3))                     # 3*v^2, minimal-body dispatch
#' attr(gfz, "dd_frozen")              # 1000
dd_freeze <- function(gf, n_hint) {
  if (!is.function(gf) || is.null(attr(gf, "grad_expr"))) {
    .dat_stop("DefDiff_not_definable",
              "dd_freeze expects a DefDiff gradient function produced by grad().")
  }
  if (length(formals(gf)) != 1L) {
    .dat_stop("DefDiff_not_definable",
              "dd_freeze supports single-vector-argument gradients (like dd_batch).")
  }
  if (missing(n_hint)) {
    .dat_stop("DefDiff_not_definable",
              paste0("dd_freeze requires n_hint (the winning path is size-dependent); ",
                     "a silent default would freeze a quietly suboptimal path."))
  }
  n_hint <- suppressWarnings(as.numeric(n_hint)[1L])
  if (length(n_hint) == 0L || is.na(n_hint) || !is.finite(n_hint) || n_hint < 1) {
    .dat_stop("DefDiff_not_definable", "dd_freeze: n_hint must be a finite positive length.")
  }
  # Floor AFTER validation: a fractional n_hint would compute the bucket on the
  # unfloored value while the probe vector truncates, mismatching near half-decade
  # boundaries (#8 verify) -- freeze and probe must see the same integer length.
  n_hint <- floor(n_hint)

  b <- body(gf)
  disp <- NULL
  pre <- NULL                                     # the `.dat_s <- <scalars>` preamble
  if (is.call(b) && identical(b[[1L]], as.name("{"))) {
    for (s in as.list(b)[-1L]) {
      if (is.call(s) && identical(s[[1L]], as.name("<-")) &&
          identical(s[[2L]], as.name(".dat_s"))) pre <- s
      if (is.call(s) && is.symbol(s[[1L]]) &&
          identical(as.character(s[[1L]]), ".dat_auto_dispatch")) { disp <- s; break }
    }
  }
  if (is.null(disp)) {
    if (!is.null(attr(gf, "dd_frozen"))) {
      message("dd_freeze: already frozen (at n_hint = ", attr(gf, "dd_frozen"),
              "); re-freezing is not supported -- re-derive with grad() and freeze again.")
    } else {
      message("dd_freeze: this gradient has no dispatcher to remove (already a minimal body); ",
              "returning it unchanged.")
    }
    return(gf)
  }

  # Validate arity BEFORE indexing: a drifted emission with fewer args would
  # otherwise raise an unclassed subscript error instead of the fail-open
  # warning path (#8 verify).
  if (length(disp) != 6L) {
    warning("dd_freeze: unrecognized emitted-body shape; returning the gradient unchanged ",
            "(correct but unfrozen).", call. = FALSE)
    return(gf)
  }
  cache_key <- disp[[2L]]
  fused_th  <- disp[[4L]]
  base_th   <- disp[[5L]]
  metal_flag <- isTRUE(disp[[6L]])
  shapes_ok <- is.character(cache_key) && length(cache_key) == 1L &&
    is.call(fused_th) && identical(fused_th[[1L]], as.name("function")) &&
    is.call(base_th)  && identical(base_th[[1L]],  as.name("function")) &&
    is.call(fused_th[[3L]]) &&
    identical(fused_th[[3L]][[1L]], as.name(".dat_fused_try")) &&
    !is.null(pre)
  if (!shapes_ok) {
    warning("dd_freeze: unrecognized emitted-body shape; returning the gradient unchanged ",
            "(correct but unfrozen).", call. = FALSE)
    return(gf)
  }
  fused_call <- fused_th[[3L]]                    # .dat_fused_try(key, src, kernel, v, .dat_s)
  base_body  <- base_th[[3L]]

  # Resolve the winner for n_hint, mirroring the dispatcher's own decision order.
  winner <- "base"
  metal_thr <- .dat_opt_pos_num("DefDiff.metal_threshold", 1e9L)
  thr_opt <- getOption("DefDiff.jit_threshold", NULL)
  if (!is.null(thr_opt) || !isTRUE(getOption("DefDiff.autotune", TRUE))) {
    # Static-gate override active: honor the user's explicit config instead of
    # probing (the dispatcher's override branch never writes pathchoice, so a
    # probe here would learn nothing and silently bake base -- #8 verify MEDIUM).
    thr <- if (!is.null(thr_opt)) suppressWarnings(as.numeric(thr_opt)[1L])
           else .DAT_JIT_THRESHOLD_DEFAULT
    if (!is.na(thr) && is.finite(thr) && n_hint >= thr &&
        !(metal_flag && n_hint >= metal_thr)) winner <- "fused"
  } else if (metal_flag && n_hint >= metal_thr) {
    # Metal lane: the dispatcher returns base() here WITHOUT learning, so a probe
    # is a guaranteed-futile allocation of length n_hint (#8 verify) -- skip it.
    # winner stays "base"; resolve_metal below keeps the Metal if for this regime.
  } else if (n_hint >= .dat_opt_pos_num("DefDiff.autotune_floor", .DAT_AUTOTUNE_FLOOR)) {
    # Read the learned choice or probe ONCE through the existing dispatcher
    # (reusing the #6/#7 machinery: repeat-K timer + agreement check).
    key <- paste0(cache_key, "@", floor(2 * log10(n_hint)))
    learned <- .dat_jit_state$pathchoice[[key]]
    if (is.null(learned)) {
      if (n_hint > 1e7) {
        message("dd_freeze: probing an unlearned bucket at n_hint = ",
                format(n_hint, big.mark = ",", scientific = FALSE),
                " allocates a vector of that length once; note the dispatch ",
                "overhead freeze removes is negligible at large n.")
      }
      # Deterministic probe input: values are irrelevant to timing/agreement,
      # and runif() would advance the user's global RNG stream as a side effect.
      probe_v <- seq_len(n_hint) / n_hint
      invisible(gf(probe_v))
      rm(probe_v)
      learned <- .dat_jit_state$pathchoice[[key]]
    }
    if (!is.null(learned) && identical(learned$choice, "fused")) winner <- "fused"
  }

  # Canonical base bodies carry a Metal-window if; resolve it at freeze time
  # (n_hint below the threshold -> take the cpu arm directly). Necessary to reach
  # the bare-.Call floor; the metal arm case (n_hint >= 1e9) keeps the if as-is.
  resolve_metal <- function(bb) {
    if (is.call(bb) && identical(bb[[1L]], as.name("if"))) {
      cond <- bb[[2L]]
      if (is.call(cond) && identical(cond[[1L]], as.name("&&")) &&
          is.call(cond[[2L]]) &&
          identical(cond[[2L]][[1L]], as.name(".metal_path_available"))) {
        if (n_hint < .dat_opt_pos_num("DefDiff.metal_threshold", 1e9L)) return(bb[[4L]])
        return(bb)
      }
    }
    bb
  }
  bare_base <- resolve_metal(base_body)

  frozen_body <- if (identical(winner, "fused")) {
    # Keep the NULL fallback: graceful degradation if the kernel cache is cold
    # and compilation fails; the in-.dat_fused_try cache lookup is ~0.1us.
    bquote({
      .(pre)
      .dat_out <- .(fused_call)
      if (is.null(.dat_out)) .dat_out <- .(bare_base)
      .dat_out
    })
  } else if (".dat_s" %in% all.names(bare_base)) {
    # Defensive (unreachable in today's emitter, asserted by tests): if a future
    # base emission references the wrapper preamble, keep it rather than emit a
    # body with an unbound .dat_s (#8 verify, codex HIGH).
    bquote({ .(pre); .(bare_base) })
  } else {
    bare_base                                     # no preamble: base computes its own scalars
  }

  out <- gf
  attrs <- attributes(gf)                         # `body<-` drops function attributes
  body(out) <- frozen_body
  attributes(out) <- attrs                        # restore everything (incl. grad_expr)
  attr(out, "dd_frozen") <- n_hint
  out
}

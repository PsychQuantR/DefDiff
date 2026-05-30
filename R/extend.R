## extend.R
## Package-internal catalog state + user-facing extension API.
##
## .dat_env stores the active generator catalog. Mutation is session-local;
## reload (devtools::load_all()) resets to defaults via .onLoad().

#' @keywords internal
.dat_env <- new.env(parent = emptyenv())

# Valid language tiers
.dat_levels <- c("L_0", "L_1", "L_2", "L_3")

# Numeric ordering for tier comparison (L_0 < L_1 < L_2 < L_3 < unknown)
.level_rank <- function(level) {
  if (is.null(level) || length(level) != 1L) {
    return(NA_integer_)
  }
  ranks <- c(L_0 = 0L, L_1 = 1L, L_2 = 2L, L_3 = 3L, unknown = 4L)
  rank <- ranks[level]
  if (is.na(rank)) NA_integer_ else unname(rank)
}

#' Extend the active generator catalog
#'
#' Registers a new generator at the specified language tier. After
#' registration, [grad()] and [level()] recognise the generator. The
#' mutation is session-local: reloading the package via
#' `devtools::load_all()` resets the catalog to defaults.
#'
#' @param level Character. One of `"L_0"`, `"L_1"`, `"L_2"`, `"L_3"`.
#' @param name Character vector of length 1. The R function name to register.
#' @param derivative A function `function(arg_expr, arg_grad)` that returns
#'   an R expression for the chain-rule derivative. `arg_expr` is the call
#'   passed as the function's first argument; `arg_grad` is the recursive
#'   gradient of that argument.
#' @return Invisibly returns `TRUE` on success.
#' @export
#' @examples
#' \dontrun{
#' extend_language(
#'   "L_3", "erf",
#'   function(x, dx) bquote(2 / sqrt(pi) * exp(-(.(x))^2) * .(dx))
#' )
#' grad(quote(erf(v)), "v")
#' }
extend_language <- function(level, name, derivative) {
  if (!is.character(level) || length(level) != 1L || !(level %in% .dat_levels)) {
    .dat_stop(
      "DefDiff_invalid_extension",
      paste0(
        "Invalid `level` argument: must be one of L_0, L_1, L_2, L_3; got ",
        deparse(level)
      )
    )
  }
  if (!is.character(name) || length(name) != 1L || nzchar(name) == FALSE) {
    .dat_stop(
      "DefDiff_invalid_extension",
      paste0("Invalid `name` argument: must be a length-1 character; got ", deparse(name))
    )
  }
  if (!is.function(derivative)) {
    .dat_stop(
      "DefDiff_invalid_extension",
      paste0("Invalid `derivative` argument: must be a function; got ", typeof(derivative))
    )
  }
  .dat_env$catalog[[level]][[name]] <- derivative
  invisible(TRUE)
}

#' Inspect the active generator catalog
#'
#' @param level Optional character. If supplied (one of `"L_0"`, `"L_1"`,
#'   `"L_2"`, `"L_3"`), returns a character vector of generator names at
#'   that tier. If `NULL` (default), returns a named list keyed by tier.
#' @return A named list (when `level` is `NULL`) or a character vector.
#' @export
#' @examples
#' language_catalog()
#' language_catalog("L_3")
language_catalog <- function(level = NULL) {
  if (is.null(level)) {
    lapply(.dat_env$catalog, names)
  } else {
    if (!(level %in% .dat_levels)) {
      .dat_stop(
        "DefDiff_invalid_extension",
        paste0("Unknown level: ", deparse(level))
      )
    }
    names(.dat_env$catalog[[level]])
  }
}

# Internal: typed condition raiser
.dat_stop <- function(class, message) {
  cond <- structure(
    class = c(class, "error", "condition"),
    list(message = message, call = sys.call(-1))
  )
  stop(cond)
}

# Internal: initialise catalog with default generators
.onLoad <- function(libname, pkgname) {
  register_default_catalog()
}

#' Reset the catalog to default generators
#'
#' Internal entry point for tests and reload. Wipes any user-added
#' generators and restores the shipped default catalog (L_0 through L_3).
#'
#' @return Invisibly `TRUE`.
#' @keywords internal
register_default_catalog <- function() {
  .dat_env$catalog <- list(
    L_0 = list(),
    L_1 = list(),
    L_2 = list(),
    L_3 = list()
  )
  .register_L0()
  .register_L1()
  .register_L2()
  .register_L3()
  invisible(TRUE)
}

# Look up a generator's tier in the active catalog.
# Returns one of "L_0", "L_1", "L_2", "L_3", or NA_character_ if absent.
.lookup_level <- function(name) {
  for (lvl in .dat_levels) {
    if (!is.null(.dat_env$catalog[[lvl]][[name]])) {
      return(lvl)
    }
  }
  NA_character_
}

# Look up a generator's derivative function in the active catalog.
# Returns the function or NULL if absent.
.lookup_derivative <- function(name) {
  for (lvl in .dat_levels) {
    fn <- .dat_env$catalog[[lvl]][[name]]
    if (!is.null(fn)) return(fn)
  }
  NULL
}

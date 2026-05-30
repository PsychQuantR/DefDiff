## helper-numderiv.R
## dat exports grad/hessian/jacobian as S3 generics with `.function` methods.
## numDeriv exports the SAME generic names (dispatching function-class args to
## `<generic>.default`). Because R's S3 method table is global by
## `generic.class`, dat's `<generic>.function` HIJACKS numDeriv's generic for
## function input whenever dat is loaded — so `numDeriv::grad(f, x)` wrongly
## dispatches to `DefDiff:::grad.function`. (This surfaces only for the INSTALLED
## package, not under devtools::load_all, which is why R CMD check catches it.)
## These accessors call numDeriv's own methods directly. Lazy (resolved at call
## time) so a missing numDeriv only matters inside skip_if_not_installed tests.
nd_grad     <- function(...) getFromNamespace("grad.default",     "numDeriv")(...)
nd_hessian  <- function(...) getFromNamespace("hessian.default",  "numDeriv")(...)
nd_jacobian <- function(...) getFromNamespace("jacobian.default", "numDeriv")(...)

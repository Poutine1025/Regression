#' @useDynLib regressionCpp
#' @export
LinearReg <- function(dat) {
  linear_regression(dat)
}
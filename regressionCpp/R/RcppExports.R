# Generated by using Rcpp::compileAttributes() -> do not edit by hand
# Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

col_ridge_2 <- function(Y, X, lambda) {
    .Call('_regressionCpp_col_ridge_2', PACKAGE = 'regressionCpp', Y, X, lambda)
}

linear_regression <- function(data) {
    .Call('_regressionCpp_linear_regression', PACKAGE = 'regressionCpp', data)
}

logistic_regression <- function(data) {
    .Call('_regressionCpp_logistic_regression', PACKAGE = 'regressionCpp', data)
}


#include "linear_regression.hpp"
#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]

// [[Rcpp::export]]
Rcpp::List linear_regression(Eigen::MatrixXd data){
  Rcpp::List ret;
  LinearRegResult res;
  res = LinearReg(data);
  ret["Estimate"] = res.Estimate;
  ret["Variance"] = res.Variance;
  ret["FittedValues"] = res.Fittedvalues;
  return ret;
}

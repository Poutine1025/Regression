#include <RcppEigen.h>
using namespace Rcpp;

//[[Rcpp::export]]
Eigen::MatrixXd col_ridge_2(Eigen::MatrixXd Y, Eigen::MatrixXd X, Eigen::VectorXd lambda) {
  long nY = Y.cols();
  long p = X.cols();
  Eigen::MatrixXd beta(p, nY);
  for(int i = 0; i < nY; ++i){
    beta.col(i) = (X.transpose() * X + lambda(i) * Eigen::MatrixXd::Identity(p, p)).inverse() * X.transpose() * Y.col(i);
  }
  return(beta);
}
//
//  Linear_Regression.hpp
//  cpptraining
//
//  Created by 徐杨见琛 on 11/22/19.
//  Copyright © 2019 Yangjianchen Xu. All rights reserved.
//

#ifndef linear_regression_hpp
#define linear_regression_hpp

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <time.h>
#include <RcppEigen.h>
using namespace Eigen;
using namespace std;

struct LinearRegResult{
    Eigen::VectorXd Estimate;
    Eigen::MatrixXd Variance;
    Eigen::VectorXd Fittedvalues;
};

struct LinearRegResult LinearReg(Eigen::MatrixXd data){
  struct LinearRegResult res;
  long n, p;
  n = data.rows();
  p = data.cols();
  Eigen::VectorXd Y(n);
  Eigen::VectorXd Yhat(n);
  Eigen::MatrixXd X(n,p);
  Eigen::VectorXd beta(p);
  Eigen::MatrixXd Covbeta(p,p);
  double sigma2;
  
  Y << data.leftCols(1);
  Eigen::VectorXd Intercept = Eigen::VectorXd::Constant(n, 1, 1);
  X << Intercept, data.rightCols(p-1);
  beta = (X.transpose()*X).inverse()*X.transpose()*Y;
  Yhat = X*beta;
  sigma2 = ((Y-Yhat).squaredNorm())/(n-p);
  Covbeta = sigma2*(X.transpose()*X).inverse();
  
  res.Estimate = beta;
  res.Variance = Covbeta;
  res.Fittedvalues = Yhat;
  return res;
}

#endif /* linear_regression_hpp */

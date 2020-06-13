#include <RcppEigen.h>
using namespace Rcpp;
// [[Rcpp::depends(RcppEigen)]]
struct LogisticInformation{
  double LogLikelihood;
  Eigen::VectorXd ScoreFun;
  Eigen::MatrixXd InformationMat;
};

struct LogisticRegResult{
  Eigen::VectorXd Estimate;
  Eigen::VectorXd StdErr;
  Eigen::VectorXd Zstat;
  Eigen::VectorXd pval;
  Eigen::VectorXd LowerCI;
  Eigen::VectorXd UpperCI;
};

double normalCDF(double value){
  return 0.5 * erfc(-value * M_SQRT1_2);
}

struct LogisticInformation LogisticMLE(Eigen::MatrixXd data, Eigen::VectorXd beta){
  struct LogisticInformation res;
  long n, p;
  n = data.rows();
  p = data.cols();
  Eigen::VectorXd Y(n);
  Eigen::MatrixXd X(n,p);
  Y << data.leftCols(1);
  Eigen::VectorXd Intercept = Eigen::VectorXd::Constant(n, 1, 1);
  X << Intercept, data.rightCols(p-1);
  //Calculate information
  double LogLikelihood;
  LogLikelihood = beta.transpose()*X.transpose()*Y;
  Eigen::VectorXd Score(p);
  Eigen::MatrixXd V(p,p);
  V = Eigen::MatrixXd::Zero(p, p);
  Score = Eigen::MatrixXd::Zero(p, 1);
  for (int i=1; i<(n+1); ++i) {
    double temp;
    temp = 0;
    temp = (X.row(i-1)*beta)(0);
    //Log-likelihood
    LogLikelihood = LogLikelihood - log(1+exp(temp));
    //Score function
    Score = Score + (Y(i-1)-exp(temp)/(1+exp(temp)))*X.row(i-1).transpose();
    //Information matrix
    V = V + exp(temp)/pow(1+exp(temp),2)*X.row(i-1).transpose()*X.row(i-1);
  }
  res.LogLikelihood = LogLikelihood;
  res.ScoreFun = Score;
  res.InformationMat = V;
  
  return res;
}

Eigen::VectorXd NewtonRaphson_Logistic(Eigen::MatrixXd data, Eigen::VectorXd beta, long MaxIter){
  double error;
  Eigen::VectorXd beta_new;
  Eigen::VectorXd beta_old;
  LogisticInformation temp_old;
  LogisticInformation temp_new;
  long Iter;
  beta_old = beta;
  error = 1;
  Iter = 0;
  temp_old = LogisticMLE(data, beta_old);
  while (error>=1e-6 && Iter<=MaxIter) {
    beta_new = beta_old + temp_old.InformationMat.inverse()*temp_old.ScoreFun;
    temp_new = LogisticMLE(data, beta_new);
    error = fabs((temp_new.LogLikelihood-temp_old.LogLikelihood)/temp_old.LogLikelihood);
    beta_old = beta_new;
    temp_old = temp_new;
    ++Iter;
  }
  return beta_old;
}

struct LogisticRegResult LogisticReg(Eigen::MatrixXd data){
  LogisticRegResult res;
  LogisticInformation model;
  long n, p;
  n = data.rows();
  p = data.cols();
  Eigen::VectorXd beta(p);
  Eigen::MatrixXd Cov(p,p);
  Eigen::VectorXd SE(p);
  Eigen::VectorXd Zstat(p);
  Eigen::VectorXd pval(p);
  Eigen::VectorXd LowerCI(p);
  Eigen::VectorXd UpperCI(p);
  
  beta = Eigen::VectorXd::Zero(p, 1);
  beta = NewtonRaphson_Logistic(data, beta, 100);
  model = LogisticMLE(data, beta);
  Cov = model.InformationMat.inverse();
  for(int i=0; i<p; ++i){
    SE(i) = sqrt(Cov(i,i));
    Zstat(i) = beta(i)/SE(i);
    pval(i) = 1 - normalCDF(fabs(Zstat(i)))+normalCDF(-fabs(Zstat(i)));
    LowerCI(i) = beta(i) - 1.96*SE(i);
    UpperCI(i) = beta(i) + 1.96*SE(i);
  }
  
  res.Estimate = beta;
  res.StdErr = SE;
  res.Zstat = Zstat;
  res.pval = pval;
  res.LowerCI = LowerCI;
  res.UpperCI = UpperCI;
  return res;
}

// [[Rcpp::export]]
Rcpp::List logistic_regression(Eigen::MatrixXd data){
  Rcpp::List ret;
  struct LogisticRegResult res;
  res = LogisticReg(data);
  Eigen::MatrixXd CI(data.cols(),2);
  CI.col(0) = res.LowerCI;
  CI.col(1) = res.UpperCI;
  ret["Estimate"] = res.Estimate;
  ret["StdErr"] = res.StdErr;
  ret["Zstat"] = res.Zstat;
  ret["pval"] = res.pval;
  ret["CI"] = CI;
  return ret;
}

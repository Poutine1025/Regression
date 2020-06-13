setwd("/Users/jianchen/Dropbox/My Mac (徐杨见琛的MacBook Pro)/Documents/GitHub/Regression")
library(devtools)
library(microbenchmark)
library(Rcpp)
library(RcppEigen)
load_all("regressionCpp")

training.data.raw <- read.csv('https://raw.githubusercontent.com/agconti/kaggle-titanic/master/data/train.csv')
data <- subset(training.data.raw,select=c(2,3,5,6,7,8,10))
data$Age[is.na(data$Age)] <- mean(data$Age,na.rm=T)
data$Sex=ifelse(data$Sex=="female",1,0)
data.Cpp=as.matrix(data)

#Linear regression
microbenchmark(model_linear_Rcpp=LinearReg(data.Cpp),
               model_linear=lm(Survived~., data = data), times = 100)

#Logistic regression
microbenchmark(model_logistic_Rcpp=LogisticReg(data.Cpp),
               model_logistic=glm(Survived~., data = data, family = binomial(link = "logit")))


n <- 100
Y <- matrix(rnorm(n*20),nrow=20)
X <- scale(matrix(rnorm(20*2),ncol=2))
lambda <- runif(n,.1,2)
library(MASS)
colRidge1 <- function(Y, X, lambda) {
  df <- as.data.frame(X)
  n <- ncol(Y)
  beta <- matrix(nrow=2,ncol=n)
  stopifnot(length(lambda) == n)
  for (j in seq_len(n)) {
    beta[,j] <- coef(lm.ridge(Y[,j] ~ 0 + V1 + V2, data=df, lambda=lambda[j]))
  }
  beta
}
microbenchmark(colRidge1(Y, X, lambda), colRidge2(Y, X, lambda), times=10)

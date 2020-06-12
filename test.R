setwd("/Users/jianchen/Dropbox/My Mac (徐杨见琛的MacBook Pro)/Documents/GitHub/Regression")
library(devtools)
library(microbenchmark)
load_all("regressionCpp")

training.data.raw <- read.csv('https://raw.githubusercontent.com/agconti/kaggle-titanic/master/data/train.csv')
data <- subset(training.data.raw,select=c(2,3,5,6,7,8,10))
data$Age[is.na(data$Age)] <- mean(data$Age,na.rm=T)
data$Sex=ifelse(data$Sex=="female",1,0)
data.Cpp=as.matrix(data)

#Linear regression
microbenchmark(model_linear_Rcpp=LinearReg(data.Cpp),
               model_linear=lm(Survived~., data = data))

#Logistic regression
microbenchmark(model_logistic_Rcpp=LogisticReg(data.Cpp),
               model_logistic=glm(Survived~., data = data, family = binomial(link = "logit")))

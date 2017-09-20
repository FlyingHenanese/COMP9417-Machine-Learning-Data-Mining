# #environment build
install.packages("arules")
install.packages("e1071")
install.packages("C50")
install.packages("class")
install.packages("nnet")
install.packages("randomForest")
install.packages("caret")

#0 Preparation
data_num=read.csv("bank-full-num1.csv", header=TRUE, sep = ";")
data_num=data_num[,-11]
data_num_1=data_num[,-16]
cor(data_num_1)#corralation check
cor(data_num_1$poutcome,data_num$y) #keep out poutcome var; remove pdays previous
cor(data_num_1$pdays,data_num$y)
cor(data_num_1$previous,data_num$y)



#1 Assoication rule learning (Apriori Algorithm)
library(arules)
myfacdata=read.csv("bank-full-fac2.csv", header=TRUE, sep = ";")
rules <- apriori(myfacdata,parameter = list(minlen=5, supp=0.3, conf=0.95),appearance = list(rhs=c("y=yes", "y=no"),default="lhs"),control = list(verbose=F))
rules.sorted=sort(rules,by="lift")
subset.matrix <- is.subset(rules.sorted, rules.sorted)
subset.matrix[lower.tri(subset.matrix, diag=T)] <- NA
redundant <- colSums(subset.matrix, na.rm=T) >= 1
rules.pruned <- rules.sorted[!redundant]
inspect(rules.pruned)

#2 K-nearest-neighbours Algorithm(KNN)
library(class)
library(caret)
mydata=read.csv("bank-full-num1.csv", header=TRUE, sep = ";")
mytestdata=read.csv("bank-num1.csv", header=TRUE, sep = ";")
mydata=as.matrix(mydata)
mytestdata =as.matrix(mytestdata)
mydata=mydata[,-11]
mytestdata=mytestdata[,-11]
cl=mydata[,16]
mydata.knn<-knn(mydata, mytestdata, cl, k = 9, prob=TRUE, use.all=TRUE)
confusionMatrix(table(mydata.knn, mytestdata[,16]))



#3 Naive Bayes
library(e1071)
my_data_fac=read.csv("bank-full-fac2.csv", header=TRUE, sep = ";")
my_testdata_fac=read.csv("bank-fac2.csv", header=TRUE, sep = ";")
model <- naiveBayes(y ~ ., data =my_data_fac)
pred <- predict(model, my_testdata_fac)
confusionMatrix(table(pred, my_testdata_fac $y))


#4 Support Vector Machine(SVM)
library(e1071)
mydata=read.csv("bank-full-num1.csv", header=TRUE, sep = ";")
mytestdata=read.csv("bank-num1.csv", header=TRUE, sep = ";")
mydata=mydata[,-15]
mydata=mydata[,-15]
mydata=mydata[,-11]
mytestdata=mytestdata[,-15]
mytestdata=mytestdata[,-15]
mytestdata=mytestdata[,-11]
model <- svm(y ~ ., data = mydata)
pred <- predict(model, mytestdata[,-15])
confusionMatrix(table(pred, mytestdata$y))


#5 Decision Tree
library(C50)
mydata=read.csv("bank-full.csv", header=TRUE, sep = ";")
mytestdata=read.csv("bank.csv", header=TRUE, sep = ";")
TreeModel <- C5.0(y ~ ., data = mydata)
pred<-predict(TreeModel,mytestdata[,-17])
confusionMatrix(table(pred, mytestdata[,17]))


# 6 Bagging (random forest)
library(randomForest)
bank_full=read.csv("bank-full.csv", header=TRUE, sep = ";")
bank_test=read.csv("bank.csv", header=TRUE, sep = ";")
rf <- randomForest(y ~ ., data=bank_full, ntree=100, proximity=TRUE)
bank_test<-read.csv("bank.csv",header=TRUE,sep=";")
confusionMatrix(table(predict(rf,bank_test), bank_test$y))

#7 Boosting
library(C50)
mydata=read.csv("bank-full.csv", header=TRUE, sep = ";")
mytestdata=read.csv("bank.csv", header=TRUE, sep = ";")
TreeModel <- C5.0(y~., data = mydata, trials = 100) ##number of boosting iteration
pred<-predict(TreeModel,mytestdata)
confusionMatrix(table(pred, mytestdata$y))


#9 Neural networks
library(nnet)
mydata=read.csv("bank-full-num1.csv", header=TRUE, sep = ";")
mytestdata=read.csv("bank-num1.csv", header=TRUE, sep = ";")
nn <- nnet(y ~ ., data = mydata, size = 2, rang = 0.1,decay = 5e-4, maxit = 200)
confusionMatrix(table(mytestdata$y, predict(nn, mytestdata, type = "class")))









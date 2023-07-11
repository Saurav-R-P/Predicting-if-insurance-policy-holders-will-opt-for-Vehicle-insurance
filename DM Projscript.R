library("tidyverse")
library("psych")
library("caret")
library("FNN")
library("ISLR")
library("tree")
library("randomForest")
library("neuralnet")
library("ROCR")
library("e1071")
library("gains")
library("ggplot2")
library("reshape2")
library("rpart")
library("rpart.plot")
library("corrplot")
library("mlr")
library("ROSE")

train <- read.csv("train.csv", stringsAsFactors = TRUE)
test <- read.csv("test.csv", stringsAsFactors = TRUE)

head(train)
head(test)

dim(train)
dim(test)

sum(is.null(train))

numerical_columns <- c("Age", "Region_Code","Annual_Premium","Vintage")
categorical_columns <- c("Gender","Driving_License","Previously_Insured","Vehicle_Age","Vehicle_Damage","Response")

summary(train[numerical_columns])
describe(train[numerical_columns])

##Response Histogram
ggplot(train, aes(x=as.factor(Response))) + geom_bar()
table(train$Response)

##Age Histogram
ggplot(train, aes(Age)) +geom_histogram(binwidth = 2, color = "black", fill = "steelblue")

##Age Boxplot
ggplot(train, aes(y=Age)) + geom_boxplot()

##Age vs Premium
ggplot(train, aes(x=Age, y=Annual_Premium)) + geom_point() + geom_smooth(method=lm, se=FALSE)

##Gender Bar
ggplot(train, aes(x = Gender, y = Response)) + geom_bar(stat = "identity", col="darkblue")

##Gender and Response Bar
barplot(table(train$Gender,train$Response), main="Car Distribution by Gears and VS", xlab="Number of Gears", beside=TRUE, legend = rownames(table(train$Gender,train$Response)), col=c("darkblue","red"))

##Driving License and Gender
table(train$Gender,train$Driving_License)
barplot(table(train$Gender,train$Driving_License), main="Car Distribution by Gears and VS", xlab="Number of Gears", beside=TRUE, legend = rownames(table(train$Gender,train$Driving_License)), col=c("darkblue","red"))

##Already Insured Bar
ggplot(train, aes(x=as.factor(Previously_Insured))) + geom_bar()

##Vehicle Age Bar
ggplot(train, aes(x=Vehicle_Age)) + geom_bar()

##Response And Vehicle Age
table(train$Vehicle_Age,train$Response)
barplot(table(train$Vehicle_Age,train$Response), main="Car Distribution by Gears and VS", xlab="Number of Gears", beside=TRUE, legend = rownames(table(train$Vehicle_Age,train$Response)), col=c("darkblue","red","Grey"))

##Damaged Vehicle Bar
ggplot(train, aes(x=Vehicle_Damage)) + geom_bar()

##Damaged Vehicle And Response
table(train$Vehicle_Damage,train$Response)
barplot(table(train$Vehicle_Damage,train$Response), main="Car Distribution by Gears and VS", xlab="Number of Gears", beside=TRUE, legend = rownames(table(train$Vehicle_Damage,train$Response)), col=c("darkblue","red"))

##Premium Dist
ggplot(train, aes(Annual_Premium)) +geom_histogram(binwidth = 5000, color = "black", fill = "steelblue")

##Premium Boxplot
ggplot(train, aes(y=Annual_Premium)) + geom_boxplot()

##Vintage Dist
ggplot(train, aes(Vintage)) +geom_histogram(color = "black", fill = "steelblue")


train["Gender"] <- as.numeric(ifelse(train$Gender == "Male", 1, 0))
train["Vehicle_Damage"] <- as.numeric(ifelse(train$Vehicle_Damage == "Yes", 1, 0))
train_dummies <- createDummyFeatures(train, cols = c("Vehicle_Age"))

colnames(train_dummies)[12] <- c("Vehicle_Age_lt_1_Year")
colnames(train_dummies)[13] <- c("Vehicle_Age_gt_2_Year")
colnames(train_dummies)[14] <- c("Vehicle_Age_1_2_Year")

num_feat = c('Age','Vintage')
cat_feat = c('Gender', 'Driving_License', 'Previously_Insured', 'Vehicle_Age_lt_1_Year','Vehicle_Age_1_2_Year','Vehicle_Age_gt_2_Years','Vehicle_Damage_Yes','Region_Code','Policy_Sales_Channel')

a <- cor(train_dummies)

normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x))) }

train_d_n <- as.data.frame(lapply(train_dummies[2:14], normalize))

#Sampling
train_over <- ovun.sample(Response ~ ., data = train_d_n, method = "under")$data
table(train_over$Response)
train.index <- sample(rownames(train_over), nrow(train_over)*0.6)
train.df <- train_over[train.index,]
valid.df <- train_over[!rownames(train_over) %in% train.index,]


#test set modification
test["Gender"] <- as.numeric(ifelse(test$Gender == "Male", 1, 0))
test["Vehicle_Damage"] <- as.numeric(ifelse(test$Vehicle_Damage == "Yes", 1, 0))
test_dummies <- createDummyFeatures(test, cols = c("Vehicle_Age"))

colnames(test_dummies)[11] <- c("Vehicle_Age_lt_1_Year")
colnames(test_dummies)[12] <- c("Vehicle_Age_gt_2_Year")
colnames(test_dummies)[13] <- c("Vehicle_Age_1_2_Year")


summary(train_d_n)


#KNN

T_train.df	<-	train.df$Response
T_valid.df	<-	valid.df$Response

Training <-model.matrix(~Age+Gender+Age+Driving_License+Region_Code+Previously_Insured+Vehicle_Damage+Annual_Premium+Policy_Sales_Channel+Vintage+Response+Vehicle_Age_lt_1_Year+Vehicle_Age_gt_2_Year+Vehicle_Age_1_2_Year,data=train.df)	

Validation <-model.matrix(~Age+Gender+Age+Driving_License+Region_Code+Previously_Insured+Vehicle_Damage+Annual_Premium+Policy_Sales_Channel+Vintage+Response+Vehicle_Age_lt_1_Year+Vehicle_Age_gt_2_Year+Vehicle_Age_1_2_Year,data=valid.df)	



library(class)

knn <- knn(train = Training, test = Validation, cl=T_train.df,k=305)
knn

table(knn	,T_valid.df)



#log reg

model <- glm(Response~., data = train.df, family = binomial)
summary(model)

predicted <- predict(model, valid.df, type="response")

library(InformationValue)

optCutOff <- optimalCutoff(valid.df$Response, predicted)

plotROC(valid.df$Response, predicted)

sensitivity(valid.df$Response, predicted, threshold = optCutOff)

specificity(valid.df$Response, predicted, threshold = optCutOff)

Concordance(valid.df$Response, predicted)

confusionMatrix(valid.df$Response, predicted, threshold = optCutOff)

#Naive Bayes
NaiveB <- naiveBayes(Response~., data = train.df)
class(NaiveB)

pred <- predict(NaiveB, valid.df)
predict(NaiveB, newdata = valid.df,type = "raw")


nb.pred <-predict(NaiveB, newdata = valid.df, type = "raw")
pred.val <-prediction(nb.pred[,2], valid.df$Response)

confusionMatrix(pred, valid.df$Response, threshold = optCutOff)

table(pred)


#Decision Tree


Dtree <-rpart(Response ~ ., data = train.df, method = "class")
rpart.plot(Dtree, main = "Classification Tree")

tree.pred <-predict(Dtree, valid.df)
pred.val <-prediction(tree.pred[, 2], valid.df$Response)

prediction <-predict(Dtree, valid.df, type = 'class')

table_pred <- table(valid.df$Response, prediction)
table_pred

accuracy_Test <- sum(diag(table_pred)) / sum(table_pred)
print(paste('Accuracy for test', accuracy_Test))

#Neural Network

require(neuralnet)
library(neuralnet)

nn <- neuralnet(Response~ ., data=train.df, hidden=3,act.fct = "logistic",linear.output = FALSE)

plot(nn)


#Random Forest
rf <- randomForest(Response ~ ., data = train.df)
rf.pred <- predict(rf, valid.df)
pred.val <- prediction(rf.pred[,2], valid.df$Response)

confusionMatrix(rf.pred, valid.df$Response)

table(rf.pred, valid.df$Response)

varImpPlot(rf, type = 1)

rf
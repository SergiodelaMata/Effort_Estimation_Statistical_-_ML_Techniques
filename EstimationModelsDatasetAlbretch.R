install.packages("scales")
install.packages("Metrics")
install.packages("Hmisc")
install.packages("ftsa")
install.packages("tidyverse")
install.packages("ISLR")
install.packages("caTools")
install.packages("h2o")
install.packages('bit64', repos = "https://cran.rstudio.com")
install.packages("tidyverse")
library(scales)
library(Metrics)
library(Hmisc)
library(ftsa)
library(tidyverse)
library(ISLR)
library(caTools)
library(h2o)
library(bit64)
library(tidyverse)
file <- "D:\\User\\Escritorio\\albretch.csv"
data <- read.csv(file) #Building data frame from a dataset
num_elements <- nrow(data) #Number of elements of the dataset
data <- data[sample(num_elements),] #Randomly reordered
#For Logistic Regression
data <- within(data, relation <- data$size/data$effort)
data <- within(data, binomialRelation <- ifelse(data$size/data$effort < 200, 0, 1))

#Generate different sets of training and testing data according to timing
time <- as.numeric(Sys.time())
set.seed(time)
split <- sample.split(data$effort, SplitRatio =  0.70)

datatrain <- subset(data, split == TRUE) #Training data for the estimation model
datatest <- subset(data, split == FALSE) #Test data for the estimation model

#-------------------------------------------------------------------------------
#Correlation Coefficient
cor(datatest$effort, datatest$size)
#Covariance 
cov(datatest$effort, datatest$size)

#-------------------------------------------------------------------------------
#Linear regression
lineal_regression <- lm(effort~size, datatrain)
predictions_lm <- predict(lineal_regression, datatest)
predictions_lm
#Representation linear regression from data
plot(lineal_regression)
text(lineal_regression)

#summary(lineal_regression)

#-------------------------------------------------------------------------------
#Accuracy error measures:
#Mean Square Error
mse(datatest$effort, predictions_lm)

#Mean Absolute Error
mae(datatest$effort, predictions_lm)

#Relative Absolute Error
rae(datatest$effort, predictions_lm)

#Median Absolute Error
mdae(datatest$effort, predictions_lm)

#Mean Absolute Scaled Error
mase(datatest$effort, predictions_lm)

#-------------------------------------------------------------------------------
#Accuracy relative error measures:
#Mean Relative Absolute Error
error(predictions_lm, datatrain$effort, true = datatest$effort, method = "mrae", giveall = FALSE)

#Median Relative Absolute Error
error(predictions_lm, datatrain$effort, true = datatest$effort, method = "mdrae", giveall = FALSE)

#Geometric Mean Relative Absolute Error
error(predictions_lm, datatrain$effort, true = datatest$effort, method = "gmrae", giveall = FALSE)

#Relative Mean Absolute Error
error(predictions_lm, datatrain$effort, true = datatest$effort, method = "relmae", giveall = FALSE)

#Relative Mean Square Error
error(predictions_lm, datatrain$effort, true = datatest$effort, method = "relmse", giveall = FALSE)

#Relative Squared Error
rse(datatest$effort, predictions_lm)

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#Logistic Regression
logistic_regression <- glm(binomialRelation~effort, data = data, family = "binomial")
predictions_glm <- predict(logistic_regression, datatest)
predictions_glm
ggplot(data = data, aes(x = effort, y = binomialRelation)) +
  geom_point(aes(color = as.factor(binomialRelation)), shape = 1) +
  stat_function(fun = function(x){predict(logistic_regression, newdata = data.frame(size = x),
                                          type = "response")}) +
  theme_bw() +
  labs(title = "Logistic Regression", y = "Probability effort") +
  theme(legend.position = "none")

#-------------------------------------------------------------------------------
#Accuracy error measures:
#Mean Square Error
mse(datatest$effort, predictions_glm)

#Mean Absolute Error
mae(datatest$effort, predictions_glm)

#Relative Absolute Error
rae(datatest$effort, predictions_glm)

#Median Absolute Error
mdae(datatest$effort, predictions_glm)

#Mean Absolute Scaled Error
mase(datatest$effort, predictions_glm)

#-------------------------------------------------------------------------------
#Accuracy relative error measures:
#Mean Relative Absolute Error
error(predictions_glm, datatrain$effort, true = datatest$effort, method = "mrae", giveall = FALSE)

#Median Relative Absolute Error
error(predictions_glm, datatrain$effort, true = datatest$effort, method = "mdrae", giveall = FALSE)

#Geometric Mean Relative Absolute Error
error(predictions_glm, datatrain$effort, true = datatest$effort, method = "gmrae", giveall = FALSE)

#Relative Mean Absolute Error
error(predictions_glm, datatrain$effort, true = datatest$effort, method = "relmae", giveall = FALSE)

#Relative Mean Square Error
error(predictions_glm, datatrain$effort, true = datatest$effort, method = "relmse", giveall = FALSE)

#Relative Squared Error
rse(datatest$effort, predictions_glm)

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#Neural Networks
#table(datatrain$effort)
h2o.init(nthreads = -1)

classifier <- h2o.deeplearning(y = 'effort',
                              training_frame = as.h2o(datatrain),
                              activation = 'Rectifier',
                              hidden = c(5, 5),
                              epochs = 100,
                              train_samples_per_iteration = -2)
prediction_nnn <- t(h2o.predict(classifier, newdata = as.h2o(datatest)))
prediction_nnn

#-------------------------------------------------------------------------------
#Accuracy error measures:
#Mean Square Error
mse(datatest$effort, t(prediction_nnn))

#Mean Absolute Error
mae(datatest$effort, t(prediction_nnn))

#Relative Absolute Error
rae(datatest$effort, prediction_nnn)
#rae(datatest$effort, t(prediction_nnn))

#Median Absolute Error
mdae(datatest$effort, t(as_tibble(prediction_nnn)))
#t(prediction_nnn) #Transpose of the h2oFrame
#t(as_tibble(prediction_nnn)) #Transpose of the transformed prediction tables from h2oFrame to dataFrame

#Mean Absolute Scaled Error
mase(datatest$effort, prediction_nnn)

#-------------------------------------------------------------------------------
#Accuracy relative error measures:
#Mean Relative Absolute Error
error(as.numeric(as_tibble(prediction_nnn)), datatrain$effort, true = datatest$effort, method = "mrae", giveall = FALSE)

#Median Relative Absolute Error
error(as.numeric(as_tibble(prediction_nnn)), datatrain$effort, true = datatest$effort, method = "mdrae", giveall = FALSE)

#Geometric Mean Relative Absolute Error
error(as.numeric(as_tibble(prediction_nnn)), datatrain$effort, true = datatest$effort, method = "gmrae", giveall = FALSE)

#Relative Mean Absolute Error
error(as.numeric(as_tibble(prediction_nnn)), datatrain$effort, true = datatest$effort, method = "relmae", giveall = FALSE)

#Relative Mean Square Error
error(as.numeric(as_tibble(prediction_nnn)), datatrain$effort, true = datatest$effort, method = "relmse", giveall = FALSE)

#Relative Squared Error
rse(datatest$effort, prediction_nnn)

#-------------------------------------------------------------------------------
#Cross Validation:
#Function to show all the results shown above, but using cross validation
myStudy <- function(data, datatrain, datatest)
{
  print("Initial dataset:")
  print(data)
  print("Training data:")
  print(datatrain)
  print("Test data:")
  print(datatest)
  #-------------------------------------------------------------------------------
  #Correlation Coefficient from the training data (As there will be only one element at the test data, it can't be made from the test data)
  print("Correlation Coefficient:")
  print(cor(datatrain$effort, datatrain$size))
  #Covariance  from the training data (As there will be only one element at the test data, it can't be made from the test data)
  print("Covariance:")
  print(cov(datatrain$effort, datatrain$size))
  
  #-------------------------------------------------------------------------------
  #Linear regression
  print("LINEAR REGRESSION:")
  lineal_regression <- lm(effort~size, datatrain)
  predictions_lm <- predict(lineal_regression, datatest)
  predictions_lm
  
  #-------------------------------------------------------------------------------
  #Accuracy error measures:
  #Mean Square Error
  print("Mean Square Error:")
  print(mse(datatest$effort, predictions_lm))
  
  #Mean Absolute Error
  print("Mean Absolute Error:")
  print(mae(datatest$effort, predictions_lm))
  
  #Relative Absolute Error
  print("Relative Absolute Error:")
  print(rae(datatest$effort, predictions_lm))
  
  #Median Absolute Error
  print("Median Absolute Error:")
  print(mdae(datatest$effort, predictions_lm))
  
  #Mean Absolute Scaled Error
  print("Mean Absolute Scaled Error")
  print(mase(datatest$effort, predictions_lm))
  
  #-------------------------------------------------------------------------------
  #Accuracy relative error measures:
  #Mean Relative Absolute Error
  print("Mean Relative Absolute Error:")
  print(error(predictions_lm, datatrain$effort, true = datatest$effort, method = "mrae", giveall = FALSE))
  
  #Median Relative Absolute Error
  print("Median Relative Absolute Error:")
  print(error(predictions_lm, datatrain$effort, true = datatest$effort, method = "mdrae", giveall = FALSE))
  
  #Geometric Mean Relative Absolute Error
  print("Geometric Mean Relative Absolute Error:")
  print(error(predictions_lm, datatrain$effort, true = datatest$effort, method = "gmrae", giveall = FALSE))
  
  #Relative Mean Absolute Error
  print("Relative Mean Absolute Error:")
  print(error(predictions_lm, datatrain$effort, true = datatest$effort, method = "relmae", giveall = FALSE))
  
  #Relative Mean Square Error
  print("Relative Mean Square Error:")
  print(error(predictions_lm, datatrain$effort, true = datatest$effort, method = "relmse", giveall = FALSE))
  
  #Relative Squared Error
  print("Relative Squared Error according to the test data:")
  print(rse(datatest$effort, predictions_lm))
  print("Relative Squared Error according to the train data:")
  print(rse(datatrain$effort, predictions_lm))
  #-------------------------------------------------------------------------------
  #-------------------------------------------------------------------------------
  #Logistic Regression
  print("LOGISTIC REGRESSION:")
  logistic_regression <- glm(binomialRelation~effort, data = data, family = "binomial")
  predictions_glm <- predict(logistic_regression, datatest)
  predictions_glm
  
  #-------------------------------------------------------------------------------
  #Accuracy error measures:
  #Mean Square Error
  print("Mean Square Error:")
  print(mse(datatest$effort, predictions_glm))
  
  #Mean Absolute Error
  print("Mean Absolute Error:")
  print(mae(datatest$effort, predictions_glm))
  
  #Relative Absolute Error
  print("Relative Absolute Error:")
  print(rae(datatest$effort, predictions_glm))
  
  #Median Absolute Error
  print("Median Absolute Error:")
  print(mdae(datatest$effort, predictions_glm))
  
  #Mean Absolute Scaled Error
  print("Mean Absolute Scaled Error")
  print(mase(datatest$effort, predictions_glm))
  
  #-------------------------------------------------------------------------------
  #Accuracy relative error measures:
  #Mean Relative Absolute Error
  print("Mean Relative Absolute Error:")
  print(error(predictions_glm, datatrain$effort, true = datatest$effort, method = "mrae", giveall = FALSE))
  
  #Median Relative Absolute Error
  print("Median Relative Absolute Error:")
  print(error(predictions_glm, datatrain$effort, true = datatest$effort, method = "mdrae", giveall = FALSE))
  
  #Geometric Mean Relative Absolute Error
  print("Geometric Mean Relative Absolute Error:")
  print(error(predictions_glm, datatrain$effort, true = datatest$effort, method = "gmrae", giveall = FALSE))
  
  #Relative Mean Absolute Error
  print("Relative Mean Absolute Error:")
  print(error(predictions_glm, datatrain$effort, true = datatest$effort, method = "relmae", giveall = FALSE))
  
  #Relative Mean Square Error
  print("Relative Mean Square Error:")
  print(error(predictions_glm, datatrain$effort, true = datatest$effort, method = "relmse", giveall = FALSE))
  
  #Relative Squared Error
  print("Relative Squared Error according to the test data:")
  print(rse(datatest$effort, predictions_glm))
  print("Relative Squared Error according to the train data:")
  print(rse(datatrain$effort, predictions_glm))
  
  #-------------------------------------------------------------------------------
  #-------------------------------------------------------------------------------
  #Neural Networks
  print("NEURAL NETWORKS:")
  h2o.init(nthreads = -1)
  
  classifier <- h2o.deeplearning(y = 'effort',
                                 training_frame = as.h2o(datatrain),
                                 activation = 'Rectifier',
                                 hidden = c(5, 5),
                                 epochs = 100,
                                 train_samples_per_iteration = -2)
  prediction_nnn <- t(h2o.predict(classifier, newdata = as.h2o(datatest)))
  prediction_nnn
  
  #-------------------------------------------------------------------------------
  #Accuracy error measures:
  #Mean Square Error
  print("Mean Square Error:")
  print(mse(datatest$effort, t(prediction_nnn)))
  
  #Mean Absolute Error
  print("Mean Absolute Error:")
  print(mae(datatest$effort, t(prediction_nnn)))
  
  #Relative Absolute Error
  print("Relative Absolute Error:")
  print(rae(datatest$effort, prediction_nnn))
  #rae(datatest$effort, t(prediction_nnn))
  
  #Median Absolute Error
  print("Median Absolute Error:")
  print(mdae(datatest$effort, t(as_tibble(prediction_nnn))))

  #Mean Absolute Scaled Error
  print("Mean Absolute Scaled Error")
  print(mase(datatest$effort, prediction_nnn))
  
  #-------------------------------------------------------------------------------
  #Accuracy relative error measures:
  #Mean Relative Absolute Error
  print("Mean Relative Absolute Error:")
  print(error(as.numeric(as_tibble(prediction_nnn)), datatrain$effort, true = datatest$effort, method = "mrae", giveall = FALSE))
  
  #Median Relative Absolute Error
  print("Median Relative Absolute Error:")
  print(error(as.numeric(as_tibble(prediction_nnn)), datatrain$effort, true = datatest$effort, method = "mdrae", giveall = FALSE))
  
  #Geometric Mean Relative Absolute Error
  print("Geometric Mean Relative Absolute Error:")
  print(error(as.numeric(as_tibble(prediction_nnn)), datatrain$effort, true = datatest$effort, method = "gmrae", giveall = FALSE))
  
  #Relative Mean Absolute Error
  print("Relative Mean Absolute Error:")
  print(error(as.numeric(as_tibble(prediction_nnn)), datatrain$effort, true = datatest$effort, method = "relmae", giveall = FALSE))
  
  #Relative Mean Square Error
  print("Relative Mean Square Error:")
  print(error(as.numeric(as_tibble(prediction_nnn)), datatrain$effort, true = datatest$effort, method = "relmse", giveall = FALSE))
  
  #Relative Squared Error
  print("Relative Squared Error according to the test data:")
  print(rse(datatest$effort, prediction_nnn))
  print("Relative Squared Error according to the train data:")
  print(rse(datatrain$effort, prediction_nnn))
}
num_partitions <- num_elements #Number of partitions can be changed (In this situation, as there are only 10 elements, all the possibilities can be studied)
folds <- cut(seq(2, num_elements), breaks = num_elements, labels=FALSE)
for(i in 1: num_partitions){
  testIndexes <- which(folds==i, arr.ind = TRUE)
  
  datatrain <- data[-testIndexes, ] #Training data for the estimation model
  datatest <- data[testIndexes, ] #Test data for the estimation model
  myStudy(data, datatrain, datatest)
}



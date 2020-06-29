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
file <- "D:\\User\\Escritorio\\Effort_Estimation_Statistical_&_ML_Techniques\\Effort_Estimation_Statistical_-_ML_Techniques\\ISBSGv10.AttributesSelected_795Instances_14Att_Ln_ProgLanWithoutGLs2.csv"
data <- read.csv(file) #Building data frame from a dataset
num_elements <- nrow(data) #Number of elements of the dataset

#Generate different sets of training and testing data according to timing
time <- as.numeric(Sys.time())
set.seed(time)
split <- sample.split(data$NormalisedWorkEffortLevel1, SplitRatio =  0.70)
data <- within(data, relation <- data$FunctionalSize/data$NormalisedWorkEffortLevel1)
data <- within(data, binomialRelation <- ifelse(data$FunctionalSize/data$NormalisedWorkEffortLevel1 < 0.5, 0, 1))

datatrain <- subset(data, split == TRUE) #Training data for the estimation model
datatest <- subset(data, split == FALSE) #Test data for the estimation model

#-------------------------------------------------------------------------------
#Correlation Coefficient
cor(datatest$NormalisedWorkEffortLevel1, datatest$FunctionalSize)
#Covariance 
cov(datatest$NormalisedWorkEffortLevel1, datatest$FunctionalSize)

#-------------------------------------------------------------------------------
#Linear regression
lineal_regression <- lm(NormalisedWorkEffortLevel1~FunctionalSize, datatrain)
predictions_lm <- predict(lineal_regression, datatest)
predictions_lm
#Representation linear regression from data
plot(lineal_regression)
text(lineal_regression)

#summary(lineal_regression)

#-------------------------------------------------------------------------------
#Accuracy error measures:
#Mean Square Error
mse(datatest$NormalisedWorkEffortLevel1, predictions_lm)

#Mean Absolute Error
mae(datatest$NormalisedWorkEffortLevel1, predictions_lm)

#Relative Absolute Error
rae(datatest$NormalisedWorkEffortLevel1, predictions_lm)

#Median Absolute Error
mdae(datatest$NormalisedWorkEffortLevel1, predictions_lm)

#Mean Absolute Scaled Error
mase(datatest$NormalisedWorkEffortLevel1, predictions_lm)

#-------------------------------------------------------------------------------
#Accuracy relative error measures:
#Mean Relative Absolute Error
error(predictions_lm, datatrain$NormalisedWorkEffortLevel1, true = datatest$NormalisedWorkEffortLevel1, method = "mrae", giveall = FALSE)

#Median Relative Absolute Error
error(predictions_lm, datatrain$NormalisedWorkEffortLevel1, true = datatest$NormalisedWorkEffortLevel1, method = "mdrae", giveall = FALSE)

#Geometric Mean Relative Absolute Error
error(predictions_lm, datatrain$NormalisedWorkEffortLevel1, true = datatest$NormalisedWorkEffortLevel1, method = "gmrae", giveall = FALSE)

#Relative Mean Absolute Error
error(predictions_lm, datatrain$NormalisedWorkEffortLevel1, true = datatest$NormalisedWorkEffortLevel1, method = "relmae", giveall = FALSE)

#Relative Mean Square Error
error(predictions_lm, datatrain$NormalisedWorkEffortLevel1, true = datatest$NormalisedWorkEffortLevel1, method = "relmse", giveall = FALSE)

#Relative Squared Error
rse(datatest$NormalisedWorkEffortLevel1, predictions_lm)

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#Logistic Regression
logistic_regression <- glm(binomialRelation~NormalisedWorkEffortLevel1, data = data, family = "binomial")
predictions_glm <- predict(logistic_regression, datatest)
predictions_glm
ggplot(data = data, aes(x = NormalisedWorkEffortLevel1, y = binomialRelation)) +
  geom_point(aes(color = as.factor(binomialRelation)), shape = 1) +
  stat_function(fun = function(x){predict(logistic_regression, newdata = data.frame(NormalisedWorkEffortLevel1 = x),
                                          type = "response")}) +
  theme_bw() +
  labs(title = "Logistic Regression", y = "Probability NormalisedWorkEffortLevel1") +
  theme(legend.position = "none")

#-------------------------------------------------------------------------------
#Accuracy error measures:
#Mean Square Error
mse(datatest$NormalisedWorkEffortLevel1, predictions_glm)

#Mean Absolute Error
mae(datatest$NormalisedWorkEffortLevel1, predictions_glm)

#Relative Absolute Error
rae(datatest$NormalisedWorkEffortLevel1, predictions_glm)

#Median Absolute Error
mdae(datatest$NormalisedWorkEffortLevel1, predictions_glm)

#Mean Absolute Scaled Error
mase(datatest$NormalisedWorkEffortLevel1, predictions_glm)

#-------------------------------------------------------------------------------
#Accuracy relative error measures:
#Mean Relative Absolute Error
error(predictions_glm, datatrain$NormalisedWorkEffortLevel1, true = datatest$NormalisedWorkEffortLevel1, method = "mrae", giveall = FALSE)

#Median Relative Absolute Error
error(predictions_glm, datatrain$NormalisedWorkEffortLevel1, true = datatest$NormalisedWorkEffortLevel1, method = "mdrae", giveall = FALSE)

#Geometric Mean Relative Absolute Error
error(predictions_glm, datatrain$NormalisedWorkEffortLevel1, true = datatest$NormalisedWorkEffortLevel1, method = "gmrae", giveall = FALSE)

#Relative Mean Absolute Error
error(predictions_glm, datatrain$NormalisedWorkEffortLevel1, true = datatest$NormalisedWorkEffortLevel1, method = "relmae", giveall = FALSE)

#Relative Mean Square Error
error(predictions_glm, datatrain$NormalisedWorkEffortLevel1, true = datatest$NormalisedWorkEffortLevel1, method = "relmse", giveall = FALSE)

#Relative Squared Error
rse(datatest$NormalisedWorkEffortLevel1, predictions_glm)

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#Neural Networks
#table(datatrain$NormalisedWorkEffortLevel1)
h2o.init(nthreads = -1)

classifier <- h2o.deeplearning(y = 'NormalisedWorkEffortLevel1',
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
mse(datatest$NormalisedWorkEffortLevel1, t(prediction_nnn))

#Mean Absolute Error
mae(datatest$NormalisedWorkEffortLevel1, t(prediction_nnn))

#Relative Absolute Error
rae(datatest$NormalisedWorkEffortLevel1, prediction_nnn)
#rae(datatest$NormalisedWorkEffortLevel1, t(prediction_nnn))

#Median Absolute Error
mdae(datatest$NormalisedWorkEffortLevel1, t(as_tibble(prediction_nnn)))
#t(prediction_nnn) #Transpose of the h2oFrame
#t(as_tibble(prediction_nnn)) #Transpose of the transformed prediction tables from h2oFrame to dataFrame

#Mean Absolute Scaled Error
mase(datatest$NormalisedWorkEffortLevel1, prediction_nnn)

#-------------------------------------------------------------------------------
#Accuracy relative error measures:
#Mean Relative Absolute Error
error(as.numeric(as_tibble(prediction_nnn)), datatrain$NormalisedWorkEffortLevel1, true = datatest$NormalisedWorkEffortLevel1, method = "mrae", giveall = FALSE)

#Median Relative Absolute Error
error(as.numeric(as_tibble(prediction_nnn)), datatrain$NormalisedWorkEffortLevel1, true = datatest$NormalisedWorkEffortLevel1, method = "mdrae", giveall = FALSE)

#Geometric Mean Relative Absolute Error
error(as.numeric(as_tibble(prediction_nnn)), datatrain$NormalisedWorkEffortLevel1, true = datatest$NormalisedWorkEffortLevel1, method = "gmrae", giveall = FALSE)

#Relative Mean Absolute Error
error(as.numeric(as_tibble(prediction_nnn)), datatrain$NormalisedWorkEffortLevel1, true = datatest$NormalisedWorkEffortLevel1, method = "relmae", giveall = FALSE)

#Relative Mean Square Error
error(as.numeric(as_tibble(prediction_nnn)), datatrain$NormalisedWorkEffortLevel1, true = datatest$NormalisedWorkEffortLevel1, method = "relmse", giveall = FALSE)

#Relative Squared Error
rse(datatest$NormalisedWorkEffortLevel1, prediction_nnn)

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
  print(cor(datatrain$NormalisedWorkEffortLevel1, datatrain$FunctionalSize))
  #Covariance  from the training data (As there will be only one element at the test data, it can't be made from the test data)
  print("Covariance:")
  print(cov(datatrain$NormalisedWorkEffortLevel1, datatrain$FunctionalSize))
  
  #-------------------------------------------------------------------------------
  #Linear regression
  print("LINEAR REGRESSION:")
  lineal_regression <- lm(NormalisedWorkEffortLevel1~FunctionalSize, datatrain)
  predictions_lm <- predict(lineal_regression, datatest)
  predictions_lm
  
  #-------------------------------------------------------------------------------
  #Accuracy error measures:
  #Mean Square Error
  print("Mean Square Error:")
  print(mse(datatest$NormalisedWorkEffortLevel1, predictions_lm))
  
  #Mean Absolute Error
  print("Mean Absolute Error:")
  print(mae(datatest$NormalisedWorkEffortLevel1, predictions_lm))
  
  #Relative Absolute Error
  print("Relative Absolute Error:")
  print(rae(datatest$NormalisedWorkEffortLevel1, predictions_lm))
  
  #Median Absolute Error
  print("Median Absolute Error:")
  print(mdae(datatest$NormalisedWorkEffortLevel1, predictions_lm))
  
  #Mean Absolute Scaled Error
  print("Mean Absolute Scaled Error")
  print(mase(datatest$NormalisedWorkEffortLevel1, predictions_lm))
  
  #-------------------------------------------------------------------------------
  #Accuracy relative error measures:
  #Mean Relative Absolute Error
  print("Mean Relative Absolute Error:")
  print(error(predictions_lm, datatrain$NormalisedWorkEffortLevel1, true = datatest$NormalisedWorkEffortLevel1, method = "mrae", giveall = FALSE))
  
  #Median Relative Absolute Error
  print("Median Relative Absolute Error:")
  print(error(predictions_lm, datatrain$NormalisedWorkEffortLevel1, true = datatest$NormalisedWorkEffortLevel1, method = "mdrae", giveall = FALSE))
  
  #Geometric Mean Relative Absolute Error
  print("Geometric Mean Relative Absolute Error:")
  print(error(predictions_lm, datatrain$NormalisedWorkEffortLevel1, true = datatest$NormalisedWorkEffortLevel1, method = "gmrae", giveall = FALSE))
  
  #Relative Mean Absolute Error
  print("Relative Mean Absolute Error:")
  print(error(predictions_lm, datatrain$NormalisedWorkEffortLevel1, true = datatest$NormalisedWorkEffortLevel1, method = "relmae", giveall = FALSE))
  
  #Relative Mean Square Error
  print("Relative Mean Square Error:")
  print(error(predictions_lm, datatrain$NormalisedWorkEffortLevel1, true = datatest$NormalisedWorkEffortLevel1, method = "relmse", giveall = FALSE))
  
  #Relative Squared Error
  print("Relative Squared Error according to the test data:")
  print(rse(datatest$NormalisedWorkEffortLevel1, predictions_lm))
  print("Relative Squared Error according to the train data:")
  print(rse(datatrain$NormalisedWorkEffortLevel1, predictions_lm))
  #-------------------------------------------------------------------------------
  #-------------------------------------------------------------------------------
  #Logistic Regression
  print("LOGISTIC REGRESSION:")
  logistic_regression <- glm(binomialRelation~NormalisedWorkEffortLevel1, data = data, family = "binomial")
  predictions_glm <- predict(logistic_regression, datatest)
  predictions_glm
  
  #-------------------------------------------------------------------------------
  #Accuracy error measures:
  #Mean Square Error
  print("Mean Square Error:")
  print(mse(datatest$NormalisedWorkEffortLevel1, predictions_glm))
  
  #Mean Absolute Error
  print("Mean Absolute Error:")
  print(mae(datatest$NormalisedWorkEffortLevel1, predictions_glm))
  
  #Relative Absolute Error
  print("Relative Absolute Error:")
  print(rae(datatest$NormalisedWorkEffortLevel1, predictions_glm))
  
  #Median Absolute Error
  print("Median Absolute Error:")
  print(mdae(datatest$NormalisedWorkEffortLevel1, predictions_glm))
  
  #Mean Absolute Scaled Error
  print("Mean Absolute Scaled Error")
  print(mase(datatest$NormalisedWorkEffortLevel1, predictions_glm))
  
  #-------------------------------------------------------------------------------
  #Accuracy relative error measures:
  #Mean Relative Absolute Error
  print("Mean Relative Absolute Error:")
  print(error(predictions_glm, datatrain$NormalisedWorkEffortLevel1, true = datatest$NormalisedWorkEffortLevel1, method = "mrae", giveall = FALSE))
  
  #Median Relative Absolute Error
  print("Median Relative Absolute Error:")
  print(error(predictions_glm, datatrain$NormalisedWorkEffortLevel1, true = datatest$NormalisedWorkEffortLevel1, method = "mdrae", giveall = FALSE))
  
  #Geometric Mean Relative Absolute Error
  print("Geometric Mean Relative Absolute Error:")
  print(error(predictions_glm, datatrain$NormalisedWorkEffortLevel1, true = datatest$NormalisedWorkEffortLevel1, method = "gmrae", giveall = FALSE))
  
  #Relative Mean Absolute Error
  print("Relative Mean Absolute Error:")
  print(error(predictions_glm, datatrain$NormalisedWorkEffortLevel1, true = datatest$NormalisedWorkEffortLevel1, method = "relmae", giveall = FALSE))
  
  #Relative Mean Square Error
  print("Relative Mean Square Error:")
  print(error(predictions_glm, datatrain$NormalisedWorkEffortLevel1, true = datatest$NormalisedWorkEffortLevel1, method = "relmse", giveall = FALSE))
  
  #Relative Squared Error
  print("Relative Squared Error according to the test data:")
  print(rse(datatest$NormalisedWorkEffortLevel1, predictions_glm))
  print("Relative Squared Error according to the train data:")
  print(rse(datatrain$NormalisedWorkEffortLevel1, predictions_glm))
  
  #-------------------------------------------------------------------------------
  #-------------------------------------------------------------------------------
  #Neural Networks
  print("NEURAL NETWORKS:")
  h2o.init(nthreads = -1)
  
  classifier <- h2o.deeplearning(y = 'NormalisedWorkEffortLevel1',
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
  print(mse(datatest$NormalisedWorkEffortLevel1, t(prediction_nnn)))
  
  #Mean Absolute Error
  print("Mean Absolute Error:")
  print(mae(datatest$NormalisedWorkEffortLevel1, t(prediction_nnn)))
  
  #Relative Absolute Error
  print("Relative Absolute Error:")
  print(rae(datatest$NormalisedWorkEffortLevel1, prediction_nnn))
  #rae(datatest$NormalisedWorkEffortLevel1, t(prediction_nnn))
  
  #Median Absolute Error
  print("Median Absolute Error:")
  print(mdae(datatest$NormalisedWorkEffortLevel1, t(as_tibble(prediction_nnn))))
  
  #Mean Absolute Scaled Error
  print("Mean Absolute Scaled Error")
  print(mase(datatest$NormalisedWorkEffortLevel1, prediction_nnn))
  
  #-------------------------------------------------------------------------------
  #Accuracy relative error measures:
  #Mean Relative Absolute Error
  print("Mean Relative Absolute Error:")
  print(error(as.numeric(as_tibble(prediction_nnn)), datatrain$NormalisedWorkEffortLevel1, true = datatest$NormalisedWorkEffortLevel1, method = "mrae", giveall = FALSE))
  
  #Median Relative Absolute Error
  print("Median Relative Absolute Error:")
  print(error(as.numeric(as_tibble(prediction_nnn)), datatrain$NormalisedWorkEffortLevel1, true = datatest$NormalisedWorkEffortLevel1, method = "mdrae", giveall = FALSE))
  
  #Geometric Mean Relative Absolute Error
  print("Geometric Mean Relative Absolute Error:")
  print(error(as.numeric(as_tibble(prediction_nnn)), datatrain$NormalisedWorkEffortLevel1, true = datatest$NormalisedWorkEffortLevel1, method = "gmrae", giveall = FALSE))
  
  #Relative Mean Absolute Error
  print("Relative Mean Absolute Error:")
  print(error(as.numeric(as_tibble(prediction_nnn)), datatrain$NormalisedWorkEffortLevel1, true = datatest$NormalisedWorkEffortLevel1, method = "relmae", giveall = FALSE))
  
  #Relative Mean Square Error
  print("Relative Mean Square Error:")
  print(error(as.numeric(as_tibble(prediction_nnn)), datatrain$NormalisedWorkEffortLevel1, true = datatest$NormalisedWorkEffortLevel1, method = "relmse", giveall = FALSE))
  
  #Relative Squared Error
  print("Relative Squared Error according to the test data:")
  print(rse(datatest$NormalisedWorkEffortLevel1, prediction_nnn))
  print("Relative Squared Error according to the train data:")
  print(rse(datatrain$NormalisedWorkEffortLevel1, prediction_nnn))
}
num_partitions <- floor(num_elements*0.01) #The number of the partitions to be studied can be changed regarding what the user wants
folds <- cut(seq(1, num_elements), breaks = num_elements, labels=FALSE)
for(i in 1: num_partitions){
  testIndexes <- which(folds==i, arr.ind = TRUE)
  
  datatrain <- data[-testIndexes, ] #Training data for the estimation model
  datatest <- data[testIndexes, ] #Test data for the estimation model
  myStudy(data, datatrain, datatest)
}

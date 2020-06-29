install.packages("tree")
library(tree)
data <- read.csv("D:\\User\\Escritorio\\Effort_Estimation_Statistical_&_ML_Techniques\\Effort_Estimation_Statistical_-_ML_Techniques\\atkinson.csv")

#Generate different sets of training and testing data according to timing
time <- as.numeric(Sys.time())
set.seed(time)
split <- sample.split(data$NormalisedWorkEffortLevel1, SplitRatio =  0.70)
data <- within(data, relation <- data$FunctionalSize/data$NormalisedWorkEffortLevel1)
data <- within(data, binomialRelation <- ifelse(data$FunctionalSize/data$NormalisedWorkEffortLevel1 < 0.5, 0, 1))

datatrain <- subset(data, split == TRUE) #Training data for the estimation model
datatest <- subset(data, split == FALSE) #Test data for the estimation model

#Regression tree setup
setup <- tree.control(nobs = nrow(datatrain), 
                      mincut = 5, 
                      minsize = 10,
                      mindev = 0.01)
#Regression Tree prediction and representation
regression_tree <- tree(ActEffort~IMT+IAT+IT+OMT+OAT+OT+ER+EA+ERA, datatrain, split = "deviance", control = setup)
plot(regression_tree)
text(regression_tree)
predict(regression_tree, datatest)

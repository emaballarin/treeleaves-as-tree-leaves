####################################################################################################
##                                                                                                ##
## |> The Leaf Identification Project <|                                                          ##
##                                                                                                ##
## A joint work of:                                                                               ##
##                 - Leonardo Arrighi <leonardo.arrighi@sissa.it>                                 ##
##                 - Emanuele Ballarin <emanuele@ballarin.cc>                                     ##
##                                                                                                ##
## Part of the "Introduction to Machine Learning" classes (Prof. Eric Medvet) coursework,         ##
## M.Sc. Programme in Data Science and Scientific Computing @ University of Trieste, SISSA, ICTP  ##
##                                                                                                ##
## Released under the MIT License (full text: https://ballarin.cc/legal/licenses/mit-collab.txt)  ##
##                                                                                                ##
####################################################################################################

# Comment out...
setwd("~/DSSC/mlda/MedvetExam/treeleaves-as-tree-leaves/src")


# LIBRARIES:
library(readr)
library(plyr)
library(dplyr)
library(rpart)
library(randomForest)
library(caret)
library(ipred)
library(e1071)
library(kernlab)
library(nnet)
library(obliqueRF)


#
# |> PREPARATORY PHASE <|
#

# DATA IMPORT:
leaves_dataset <- readr::read_delim(
    file = "../data/leaf.csv",
    delim = ",",
    col_names = FALSE,
    col_types = cols(
        X1 = "i",
        # We explicitly type columns
        X2 = "i",
        X3 = "d",
        X4 = "d",
        X5 = "d",
        X6 = "d",
        X7 = "d",
        X8 = "d",
        X9 = "d",
        X10 = "d",
        X11 = "d",
        X12 = "d",
        X13 = "d",
        X14 = "d",
        X15 = "d",
        X16 = "d"
    )
)

# DATA PREPARATION
colnames(leaves_dataset) <-
    c(
        "Species",
        # We give columns human-friendly names
        "SpecimenNr",
        "Eccentricity",
        "AspectRatio",
        "Elongation",
        "Solidity",
        "StochConvexity",
        "IsopFactor",
        "MaxIndDepth",
        "Lobedness",
        "AvgIntensity",
        "AvgContrast",
        "Smoothness",
        "TrdMoment",
        "Uniformity",
        "Entropy"
    )

# Snapshot the dataset
leaves_dataset_untouched <-
    leaves_dataset    # We perform an explicit copy as the dataset is light
# on memory

# DATA CLEANUP
leaves_dataset <-
    leaves_dataset %>% dplyr::select(-c("SpecimenNr"))    # We don't need that parameter
leaves_dataset$Species <-
    leaves_dataset$Species %>% dplyr::recode_factor(leaves_dataset$Species)    # We want to do
# classification, not regression!

# Global constants
kfcv_repetitions <- 1
kfcv_folds <- 5


#
# |> OUTER K-FOLD CROSS-VALIDATION - MODEL VALIDATION <|
#

# Config
outer_repetitions <- 1 #kfcv_repetitions
outer_folds <- 2 #kfcv_folds

# Loop over repetitions
for (outer_rep in seq(1, outer_repetitions))
{
    # RNG seeding
    set.seed(3456 + outer_rep - 1)

    # Shuffle dataset
    outer_rep_rows <- sample(nrow(leaves_dataset))
    leaves_dataset <- leaves_dataset[outer_rep_rows, ]

    # Draw
    outer_test_indexes <-
        createFolds(leaves_dataset$Species,    # Returns indexes, unshuffled
                    k = outer_folds)

    for (fold in seq(1, outer_folds))
    {
        test_set <- leaves_dataset[outer_test_indexes[fold][[1]], ]
        training_set <-
            leaves_dataset[-outer_test_indexes[fold][[1]], ]

        #
        # |> INNER K-FOLD CROSS-VALIDATION - HYPERPARAMETER TUNING <|
        #

        fitControl <- trainControl(method = "repeatedcv",
                                   number = kfcv_folds,
                                   repeats = kfcv_repetitions)

        set.seed(825 + (outer_rep - 1) * outer_folds + fold - 1)

        #
        # |> MODELS <|
        #

        ## Simple decision tree
        # sdtGrid <- expand.grid(maxdepth = (1:30))
        # sdtfit <- train(Species ~ Eccentricity + AspectRatio + Elongation + Solidity + StochConvexity + IsopFactor + MaxIndDepth + Lobedness + AvgIntensity + AvgContrast + Smoothness + TrdMoment + Uniformity + Entropy,
        #        data = training_set,
        #        method = "rpart2",
        #        trControl = fitControl,
        #        tuneGrid = sdtGrid,
        #        # Additional parameters
        #        metric = "Accuracy",
        #        minsplit = 2
        #        )

        ## Simple decision tree, with pruning => INEFFECTIVE! :/
        # sdtpGrid <- expand.grid(cp = (0:10)/200.0)
        # sdtpfit <- train(Species ~ Eccentricity + AspectRatio + Elongation + Solidity + StochConvexity + IsopFactor + MaxIndDepth + Lobedness + AvgIntensity + AvgContrast + Smoothness + TrdMoment + Uniformity + Entropy,
        #        data = training_set,
        #        method = "rpart",
        #        trControl = fitControl,
        #        tuneGrid = sdtpGrid,
        #        # Additional parameters
        #        metric = "Accuracy",
        #        minsplit = 2
        #        )

        # Simple decision tree, with bagging => NOT TUNABLE
        # sdtbafit <- train(Species ~ Eccentricity + AspectRatio + Elongation + Solidity + StochConvexity + IsopFactor + MaxIndDepth + Lobedness + AvgIntensity + AvgContrast + Smoothness + TrdMoment + Uniformity + Entropy,
        #        data = training_set,
        #        method = "treebag",
        #        trControl = fitControl,
        #        metric = "Accuracy",
        #        nbagg = 600,
        #        minsplit = 2
        #        )

        # # SVM with RBF
        # svmrbGrid <- expand.grid(C = c(3.7, 3.725, 3.75, 3.775, 3.8), sigma = c(0.18673, 0.3, 0.305, 0.31, 0.315, 0.32))
        # svmrbfit <- train(Species ~ Eccentricity + AspectRatio + Elongation + Solidity + StochConvexity + IsopFactor + MaxIndDepth + Lobedness + AvgIntensity + AvgContrast + Smoothness + TrdMoment + Uniformity + Entropy,
        #        data = training_set,
        #        method = "svmRadial",
        #        trControl = fitControl,
        #        tuneGrid = svmrbGrid,
        #        metric = "Accuracy",
        #        )

        # # Ridiculous neural network (i.e. with 1 HL only) -- Just for fun [Acc.: 0.7258308; Not worth it!]
        # nnetGrid <- expand.grid(size = c(30), decay = c(0.00005))
        # nnetfit <- train(Species ~ Eccentricity + AspectRatio + Elongation + Solidity + StochConvexity + IsopFactor + MaxIndDepth + Lobedness + AvgIntensity + AvgContrast + Smoothness + TrdMoment + Uniformity + Entropy,
        #        data = training_set,
        #        method = "nnet",
        #        trControl = fitControl,
        #        tuneGrid = nnetGrid,
        #        metric = "Accuracy",
        #        maxit = 2000,
        #        MaxNWts = 3000
        #        )

        # # Random Forest
        # rfGrid <- expand.grid(mtry = 2:13)    # Usually 6~10
        # rffit <- train(Species ~ Eccentricity + AspectRatio + Elongation + Solidity + StochConvexity + IsopFactor + MaxIndDepth + Lobedness + AvgIntensity + AvgContrast + Smoothness + TrdMoment + Uniformity + Entropy,
        #                data = training_set,
        #                method = "rf",
        #                trControl = fitControl,
        #                tuneGrid = rfGrid,
        #                verbose = FALSE,
        #                # Additional parameters
        #                ntree = 1500,    # >= 1000 actually; after [Breiman, 1999]
        #                minsplit = 2,
        #                metric = "Accuracy"
        #                )

        #
        # |> RF Variable importance selection (in short: the more the better!) <|
        #
        # resampledotf <- sample(nrow(leaves_dataset))
        # resampledotf_ds <- leaves_dataset[resampledotf,]
        # rfvsControl <- rfeControl(functions = rfFuncs, method = "repeatedcv", number = kfcv_folds, repeats = kfcv_repetitions)
        # rfvsRFE <- rfe(resampledotf_ds[2:15], resampledotf_ds[[1]], sizes = c(1:13), rfeControl = rfvsControl)
        # plot(rfvsRFE, type = c("g", "o"))







###### OK #######

        # An experiment...
        # customRF <- list(type = "Classification", library = "randomForest", loop = NULL)
        # customRF$parameters <- data.frame(parameter = c("mtry", "ntree", "nodesize"), class = rep("numeric", 3), label = c("mtry", "ntree", "nodesize")) # THIS!
        # customRF$grid <- function(x, y, len = NULL, search = "grid") {}
        # customRF$fit <- function(x, y, wts, param, lev, last, weights, classProbs, ...) {
        #     randomForest::randomForest(x, y, mtry = param$mtry, ntree = param$ntree, ...) # THIS!
        # }
        # customRF$predict <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
        #     predict(modelFit, newdata)
        # customRF$prob <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
        #     predict(modelFit, newdata, type = "prob")
        # customRF$sort <- function(x) x[order(x[,1]),]
        # customRF$levels <- function(x) x$classes

        #rfGrid <- expand.grid(ntree = (1:8)*50)
        # rffit <- train(Species ~ Eccentricity + AspectRatio + Elongation + Solidity + StochConvexity + IsopFactor + MaxIndDepth + Lobedness + AvgIntensity + AvgContrast + Smoothness + TrdMoment + Uniformity + Entropy,
        #                data = training_set,
        #                method = "rf",
        #                trControl = fitControl,
        #                #tuneGrid = rfGrid,
        #                verbose = FALSE,
        #                # Additional parameters
        #                ntree = 1000,
        #                metric = "Accuracy"
        #                )

        # rfGrid <- expand.grid(mtry = c(5:7), ntree = c(200, 500, 700), nodesize = c(10:1))
        # rffit <- train(Species ~ Eccentricity + AspectRatio + Elongation + Solidity + StochConvexity + IsopFactor + MaxIndDepth + Lobedness + AvgIntensity + AvgContrast + Smoothness + TrdMoment + Uniformity + Entropy,
        #                data = training_set,
        #                method = customRF,
        #                trControl = fitControl,
        #                tuneGrid = rfGrid,
        #                verbose = FALSE,
        #                # Additional parameters
        #                metric = "Accuracy"
        #                )
    }
}


#
# |> MODEL COMPARISON AND SELECTION <|
#


#
# |> BEST MODEL TRAINING ON FULL DATASET <|
# (on another file)
#

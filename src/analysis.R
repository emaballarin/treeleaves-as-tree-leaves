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


# LIBRARIES:
library(readr)
library(dplyr)
library(caret)


#
# |> PREPARATORY PHASE <|
#

# DATA IMPORT:
leaves_dataset <- readr::read_delim(file = "../data/leaf.csv",
                                    delim = ",",
                                    col_names = FALSE,
                                    col_types = cols(X1 = "i",    # We explicitly type columns
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
                                                     X16 = "d"))

# DATA PREPARATION
colnames(leaves_dataset) <- c("Species",            # We give columns human-friendly names
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
                              "Entropy")

# Snapshot the dataset
leaves_dataset_untouched <- leaves_dataset    # We perform an explicit copy as the dataset is light
                                              # on memory

# DATA CLEANUP
leaves_dataset <- leaves_dataset %>% dplyr::select(-c("SpecimenNr"))    # We don't need that


#
# |> OUTER K-FOLD CROSS-VALIDATION - MODEL VALIDATION <|
#

# RNG seeding
#set.seed(3456)

# Config
outer_folds <- 5
outer_repetitions <- 3

#
# SHUFFLE THE DATAFRAME HERE!
# In case multiple repetitions are needed, DON'T USE MULTIFOLDS.
# Just put everything in a loop and shuffle again.
#

# Draw
outer_test_indexes <- createFolds(leaves_dataset$Species,    # Returns indexes, unshuffled
                                  k = outer_folds)
for (i in seq(1,outer_folds)) {
    # Do something with the folds
    # i.e. put everything else into the loop
    # we know looping is bad practice in the context of R, but the alternative is code duplication.
}

# In general, the following works on ONE repetition...
#outer_leaves_dataset_train <- leaves_dataset[outer_train_indexes,]
#outer_leaves_dataset_test <- leaves_dataset[-outer_train_indexes,]
# ... and then shuffle.
# something has to be done for multiple repetitions:
# - looping
# (or)
# - `purrr::map`s




#
# |> INNER K-FOLD CROSS-VALIDATION - HYPERPARAMETER TUNING <|
#


#
# |> MODEL COMPARISON AND SELECTION <|
#


#
# |> BEST MODEL TRAINING ON FULL DATASET <|
#

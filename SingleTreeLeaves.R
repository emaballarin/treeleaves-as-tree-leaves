library(ggplot2)
library(tree)
library(ISLR)

leaves<- read.csv("leaf.csv", header=F)

colnames(leaves)<-c("Species", "SpecimenNumber", "Eccentricity", "AspectRatio", "Elongation", "Solidity", "StochasticConvexity", "IsoperimetricFactor", "MaximalIndentationDepth", "Lobedness", "AverageIntensity", "AverageContrast", "Smoothness", "ThirdMoment", "Uniformity", "Entropy")

attach(leaves)

tree.leaves=tree(Species~.-SpecimenNumber,leaves)

summary(tree.leaves)

plot(tree.leaves)
text(tree.leaves,pretty=0)
#note: Residual mean deviance is Devisance / (Nobs - Nnodes) = 11990/(340 - 21)

tree.leaves


#evaluate performaces
set.seed(2) #da valutare ovviamente
train=sample(1:nrow(leaves),200)
leaves.test=leaves[-train]
Species.test=Species[-train]

tree.leaves=tree(Species~.-SpecimenNumber,leaves,subset=train)
tree.pred=predict(tree.leaves,leaves.test)
#why it doesn't work :(
table(tree.pred, Species.test)
tree.pred

#FUN = in order to indicate that we want the error rate to guide cv and pruning, rather than default cv.tree() which is deviance
#cross validation
set.seed(3)
#cv.tree report the number of terminal nodes of each tree considered size as well as the the corrisponding error rate and the value of the cost complexity parameter used
cv.leaves=cv.tree(tree.leaves,FUN=prune.tree)
names(cv.leaves)                  
cv.leaves
#dev is the cv error rate
#5 is the lowest cv e.r.
#plot the error (size,k)
par(mfrow=c(1,2))
plot(cv.leaves$size, cv.leaves$dev, type="o")
plot(cv.leaves$k,cv.leaves$dev,type="o")

#pruning on 5
prune.leaves=prune.tree(tree.leaves,best=5)
plot(prune.leaves)
text(prune.leaves,pretty=0)

#test
tree.pred=predict(prune.leaves, leaves.test)
#why it doesn't work :(
table(tree.pred, Species.test)
tree.pred

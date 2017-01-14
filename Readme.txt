This repo was completed as part of the Advanced Topics in Artificial Intelligence subject taught at the Queensland University of Technology. 

The data: pickle file including 581012 rows of data. Each row including 54 variables of each forest cover type + type label. 

The solution: 4 different classification algorithms to be applied on the dataset. They are: Naives Bayes, Support Vector Machine, K Nearest Neighbors, and Decisions Trees

Overfitting was avoided using cross-validation. Training data was split into 10 smaller subsets. Each algorthim iterates through each set of 9 subsets. Results were averaged. 

Parameters tuning was also applied in the case of SVM, kNN, and Decision Trees. 

Obtained results indicated that SVM > Decision Trees => kNN => Naives Bayes in terms of precision. 

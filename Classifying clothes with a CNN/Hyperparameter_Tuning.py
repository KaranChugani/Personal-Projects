# -*- coding: utf-8 -*-

"""
This script selects the best optimizer function for the CNN by performing 3 fold
cross validation analysis on the training data and then selects optimal number of
filters for the convolutional layers
"""
import tensorflow as tf
import numpy as np
from sklearn.model_selection import KFold
from CNN_functions import create_model,create_model2
 


#%% IMPORTING DATA AND RESHAPING IT FOR INPUT IN CNN

mnist = tf.keras.datasets.fashion_mnist
(TrainData, TrainLabels), (_,_) = mnist.load_data()
# Reshaping data
TrainData=TrainData.reshape(60000, 28, 28, 1)
# Normalizing pixels
TrainData=TrainData / 255.0


#%% CREATING 3 FOLD CROSS VALIDATION INDICES
 
kf = KFold(n_splits=3)
kf.get_n_splits(TrainData) 


#%% TESTING THE MODEL WITH DIFFERENT OPTIMIZER FUNCTIONS USING CROSS-VALIDATION

AccuracyMatrix = np.zeros((3,7)) #Initializing accuracy matrix: 3 Folds * 7 OptimizerFunctions 
i = 0
j = 0

# Optimizer functions to be tested
OptimizerFunctions = {1:'SGD', 2:'RMSprop', 3:'Adagrad', 4:'Adadelta', 5:'Adam', 6:'Adamax',7:'Nadam'}


# This loop trains/tests the model with different folds and optimizer functions and 
# saves the results in the accuracy matrix

for train_index, test_index in kf.split(TrainData):
     TrainDataFold, TestDataFold = TrainData[train_index], TrainData[test_index]
     TrainLabelFold, TestLabelFold = TrainLabels[train_index], TrainLabels[test_index]
     
     for function in  OptimizerFunctions.values():
         model=create_model(function)
         model.fit(TrainDataFold, TrainLabelFold, epochs=5)
         Accuracy = model.evaluate(TestDataFold,TestLabelFold)
         AccuracyMatrix[i,j] = Accuracy[1]
         j = j + 1 
         
     j = 0 
     i = i + 1
    

MeanAccuracy = AccuracyMatrix.mean(0) # Finding the mean accuracy across the accuracy matrix
MaxAccuracy = np.max(MeanAccuracy) # Value of Max accuracy
BestOptimizer = np.argmax(MeanAccuracy) # Index of max accuracy

print("The best optimizer was " + OptimizerFunctions[BestOptimizer+1] + " with an accuracy of " + str(MaxAccuracy) )
     
     
     
#%% FINDING OPTIMAL NUMBER OF FILTERS USING THE BEST OPTIMIZER


Optimizer_Best = OptimizerFunctions[BestOptimizer+1] # Using best filter in next optimization task
     
FilterAccuracyMatrix = np.zeros((3,6)) #Initializing Filter accuracy matrix: 3 Folds * 6 Filter sizes
i = 0
j = 0

# Filter sizes functions to be tested
Filtern0List = [8,16,32,48,64,80]


# This loop trains/tests the model with different folds and optimizer functions and 
# saves the results in the accuracy matrix

for train_index, test_index in kf.split(TrainData):
     TrainDataFold, TestDataFold = TrainData[train_index], TrainData[test_index]
     TrainLabelFold, TestLabelFold = TrainLabels[train_index], TrainLabels[test_index]
     
     for FilterN0 in  Filtern0List:
         model=create_model2(Optimizer_Best,FilterN0)
         model.fit(TrainDataFold, TrainLabelFold, epochs=5)
         Accuracy = model.evaluate(TestDataFold,TestLabelFold)
         FilterAccuracyMatrix [i,j] = Accuracy[1]
         j = j + 1 
         
     j = 0 
     i = i + 1
     
     
MeanAccuracyFilter = AccuracyMatrix.mean(0) # Finding the mean accuracy across the accuracy matrix
MaxAccuracyFilter = np.max(MeanAccuracyFilter) # Value of Max accuracy
BestOptimizerFilter = np.argmax(MeanAccuracyFilter) # Index of max accuracy

print("The best filter size was " + str(Filtern0List[BestOptimizerFilter]) + " with an accuracy of " + str(MaxAccuracyFilter) )
     
     
     
     

         
model.summary()



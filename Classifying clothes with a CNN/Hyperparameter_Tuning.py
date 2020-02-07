# -*- coding: utf-8 -*-

"""
This script selects the best optimizer function for the CNN by performing 3 fold
cross validation analysis on the training data
"""
import tensorflow as tf
import numpy as np
from sklearn.model_selection import KFold
from CNNfunctions import create_model
 


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
         
#%% FINDING THE BEST MODEL


MeanAccuracy = AccuracyMatrix.mean(0) # Finding the mean accuracy across the accuracy matrix
MaxAccuracy = np.max(MeanAccuracy) # Value of Max accuracy
BestOptimizer = np.argmax(MeanAccuracy) # Index of max accuracy

print("The best optimizer was " + OptimizerFunctions[BestOptimizer+1] + " with an accuracy of " + str(MaxAccuracy) )
     





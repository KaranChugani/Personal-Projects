# -*- coding: utf-8 -*-
"""
# This script performs PCA analysis on the MNIST fashion plotting:
1) The variance explained using a different number of components
2) A scatter plot of the different items using the first two principal components
"""

import tensorflow as tf
import numpy as np
from sklearn.decomposition import PCA 
import matplotlib.pyplot as plt

#%% IMPORTING DATA AND RESHAPING IT FOR PCA ANALYSIS 

mnist = tf.keras.datasets.fashion_mnist
(_,_), (TestData, TestLabels) = mnist.load_data()
# Reshaping data
TestData=TestData.reshape(10000, 28, 28, 1)
# Normalizing pixels
TestData=TestData / 255.0
# Flattening image for PCA analysis
PCA_input = np.reshape(TestData,[10000,784], order='C')


#%% VARIANCE EXPLAINED PLOT

pca = PCA(n_components = len(PCA_input[0])-1)   # Setting n0 of PCA components to calculate
PCA_images = pca.fit_transform(PCA_input)       # Calculating PCA components
Variance = pca.explained_variance_ratio_        # Calculating variance explained by PCA components
PCAVariance = np.zeros(len(Variance))           
PCAVariance[0] = Variance[0]

# Calculating combined PCA variance explained 
for i in range(1,len(Variance)):
    PCAVariance[i] = PCAVariance[i-1] + Variance[i]
    
plt.plot(PCAVariance)

plt.xlabel('Number of Principal Components')
plt.ylabel('Variance explained %')

#%% 2D PCA SCATTER PLOT 


TestLabels = TestLabels + 1
# Colour dictionary
cdict = {1: 'r', 2: 'b', 3: 'g',4:'c',5:'m',6:'k',7:'gold',8:'brown',9:'navy',10:'silver'}
# Labels dictionary
labeldict = {1:'T-shirt', 2:'Trouser',3:'Pullover',4:'Dress',5:'Coat',6:'Sandal',7:'Shirt',8:'Sneaker',9:'Bag',10:'Ankle Boot'}

# Plotting first two components of each clothes item 
fig, ax = plt.subplots()
for g in np.unique(TestLabels):
    ix = np.where(TestLabels == g)
    ax.scatter(PCA_images[ix,0],PCA_images[ix,1], c = cdict[g], label = labeldict[g], s = 5)
ax.legend()
plt.show()

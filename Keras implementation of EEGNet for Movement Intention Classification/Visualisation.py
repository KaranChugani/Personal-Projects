# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 18:35:33 2020

@author: Karan
"""


import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np


    # Inputs:
    # Test_labels_full = Ground truth labels
    # Pred_labels_full = Predicted labels
    # Outputs: 
    # Confusion matrix plot figure


def EEGComfusionMatrix(Test_labels_full,Pred_labels_full):

    
    # Converting labels to np array for confusion matrix calculation
    Test_labels_full = np.array(Test_labels_full)
    Pred_labels_full = np.array(Pred_labels_full)
    
    Test_labels_full = np.reshape(Test_labels_full,9*288)
    Pred_labels_full = np.reshape(Pred_labels_full,9*288)
    
    
    # Calculating prediction labels and creating a confusion matrix
    cm = confusion_matrix(Test_labels_full,Pred_labels_full) 
    
    # Transform to df for easier plotting
    cm_df = pd.DataFrame(cm,
                         index = ['Right','Left','Foot','Tongue'], 
                         columns = ['Right','Left','Foot','Tongue'])
    
    # Plotting the figure
    plt.figure
    sns.heatmap(cm_df, annot=True)
    plt.title('CNN \nAccuracy:{0:.3f}'.format(accuracy_score(Test_labels_full,Pred_labels_full)))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    b, t = plt.ylim() 
    b += 0.5
    t -= 0.5
    plt.ylim(b, t) # update the ylim(bottom, top) values
    plt.show()
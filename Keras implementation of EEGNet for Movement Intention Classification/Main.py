# -*- coding: utf-8 -*-
"""
Created on Wed May 13 17:55:13 2020

@author: Karan
"""

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy import signal
from mat4py import loadmat
#import mne


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,callbacks



from tensorflow.keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model


from EEGModels import EEGModel
from EEGPreprocessing import EEGfilter,EWS,Data_extraction


# m = mne.io.read_raw_gdf('C:/Users/Karan/.spyder-py3/EEGv2/A01E.gdf',preload=True)


# Initialising lists to save results

TrainAcc = [] # Training accuracy iteration 1
ValAcc = [] # Validation accuracy iteration 1
TestAcc = [] # Test accuracy with iteration 1 model
TestAccFullModel = [] # Test accuracy with iteration 2 model

ValAccEpochs = [] # Validation accuracy on different epochs iteration 1

#%% CALLBACK INITIALISATION
    
# Callback 1: Stop training when val accuracy doesnt increase for a set number 
# of iterations 
Val_acc_stop = callbacks.EarlyStopping(monitor='val_accuracy', patience=20, mode='max',
baseline=None, restore_best_weights=True)


# Callback 2: Stop when train loss reaches train loss of iteration 1 
class myCallback(callbacks.Callback):
	def on_epoch_end(self, epoch, logs={}):
		if(logs.get('loss' ) < Train_loss):
			print("Training stopped")
			self.model.stop_training = True
            
# Instantiate a callback object
callbacks = myCallback()


#%% PARAMETER SELECTION

# Neural Network:

DrRate = 0.2 # Dropout rate
N_temp_filters = 80 # Number of temporal filters in the first convolutional layer
N_spatial_filters = 80 # Number of spatial filters in the first convolutional layer
temp_filter_len = 50 # Length of the temporal filter 
pool_size = 150 # Average pool size
pool_stride = 15 # Average pool stride
final_conv_filters = 12 # Filters in final layer

# Neural Network Training:

batch_size = 64 # Mini-batch size
lr = 0.01 * 0.01 # Learning rate
val_size = 0.20 # Percentage of samples used in validation set

# Filter characteristics:

lf = 0  # Low frequency
hf = 38  # High frequency
fs = 250 # Sampling frequency

# Exponentially weighted standardization:

decay_factor = 0.999 # Exponential decay factor
eps = 1e-4 # Epsilon value   

# Window start and end:

Win_Start_s = 1.5   # Window start seconds
Win_End_s =   6     # Window end seconds

# Window start and end in samples:

Win_Start = int(Win_Start_s*fs)
Win_End =   int(Win_End_s*fs)
Win_Len =   int(Win_End - Win_Start)


# Iterating through different subjects

for i in range (1,10):

    #%% DATA EXTRACTION
    
    # Training and evaluation BCI2A file paths
    Train_path = 'C:/Users/Karan/.spyder-py3/EEGv2/A0'+str(i)+'T.mat'
    Eval_path = 'C:/Users/Karan/.spyder-py3/EEGv2/A0'+str(i)+'E.mat'

    # Extracting windows and their respective labels for training and evaluation sets
    WindowData_train,Labelarray_train = Data_extraction(Train_path,lf,hf,fs,Win_Start,Win_End,decay_factor,eps)
    WindowData_test,Labelarray_test = Data_extraction(Eval_path,lf,hf,fs,Win_Start,Win_End,decay_factor,eps)
    
    
    #%% ORGNAIZING TRAIN/VAL/TEST SETS
        
    # Fitting data for NN input
    Data_train = np.array(WindowData_train)
    Labelarray_train = Labelarray_train - 1
    
    # Randomly splits training dataset into training a validation sets 
    X_train, X_val, y_train, y_val = train_test_split(Data_train,Labelarray_train,test_size=val_size,shuffle=True)
    
    # Obtaining dataset dimension 
    N_samples = len(Labelarray_train)
    N_samples_test = len(Labelarray_test)
    Train_size = len(y_train)
    Test_size = len(y_val)
    N_channels = len(X_train[0][0])
    
    # Reshaping data for NN input 
    X_train = np.reshape(X_train,(Train_size,Win_Len,N_channels,1))
    X_val = np.reshape(X_val,(Test_size,Win_Len,N_channels,1))
     
    # Merged Train + Validation set for training on iteration 2
    X_Train_It2 = np.reshape(Data_train,(N_samples,Win_Len,N_channels,1))
    
    # Test set 
    Test_data = np.array(WindowData_test)
    Test_data = np.reshape(Test_data,(N_samples_test,Win_Len,N_channels,1))
    Labelarray_test = Labelarray_test -1 
    

    
    #%% CREATING MODEL
    

    # N_channels = 22
    # X_train = np.zeros((100,1125, 22, 1))
    
    # Initializing model
    model1 = EEGModel(X_train.shape[1:],DrRate,N_channels,N_temp_filters,N_spatial_filters,temp_filter_len,pool_size,pool_stride,final_conv_filters)
    
    # Model summary
    model1.summary()
    
    # Model optimizer
    opt = keras.optimizers.Adam(learning_rate=lr)

    #Compiling model
    model1.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    
    #%% TRAINING MODEL   
    
    # Iteration 1
    history = model1.fit(X_train, y_train, epochs=2000, validation_data=(X_val,y_val),callbacks = [Val_acc_stop], shuffle=True, batch_size=batch_size)
    
    # Train loss and accuracy on best model iteration 1
    Train_loss,Train_acc = model1.evaluate(X_train,y_train)
    
    # Val loss and accuracy on best model iteration 1
    Val_loss,Val_acc = model1.evaluate(X_val,y_val)
    
    # Test loss and accuracy on best model iteration 1
    Test_loss_it1,Test_acc_it1 = model1.evaluate(Test_data,Labelarray_test)
    
    # Saving val_accuracy and train accuracy in a list
    Train_Val_epochs = np.transpose(np.vstack((history.history['accuracy'],history.history['val_accuracy'])))
    ValAccEpochs.append(Train_Val_epochs)
     
    # Iteration 2 of training 
    history2 = model1.fit(X_Train_It2,Labelarray_train, epochs=2000, callbacks=[callbacks])   
    
    # Test accuracy of model after second iteration
    Test_loss,Test_acc = model1.evaluate(Test_data,Labelarray_test)
    
    # Saving accuracy results in lists
    TrainAcc.append(Train_acc)
    ValAcc.append(Val_acc)
    TestAcc.append(Test_acc_it1)
    TestAccFullModel.append(Test_acc)
    
print(np.mean(TestAccFullModel))
print(np.mean(TestAcc))
print(np.mean(ValAcc))

    
    # Decrease stride in pooling layer 
    # Create confusion matrix 
    
    
    
    # # Plot training & validation accuracy values
    # plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    # plt.title('Model accuracy')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Test'], loc='upper left')
    # plt.show() 
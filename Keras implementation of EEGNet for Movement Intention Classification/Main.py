from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from tensorflow import keras
from tensorflow.keras import callbacks

from EEGModel import EEGModel,Callback_Iteration1
from EEGPreprocessing import Data_extraction
from Visualisation import EEGComfusionMatrix

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
lr = 0.0625 * 0.01 # Learning rate
val_size = 0.25 # Percentage of samples used in validation set
patience = 20 # Patience on training during iteration 1 

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

# Window length in samples
Win_Len =   int(Win_End_s*fs) - int(Win_Start_s*fs)

# Plot confusion 

plot_confusion = True # Plots confusion matrix of results of combined results 
                      # at the end of training 

#%% INITIALISING LISTS TO SAVE RESULTS

TrainAcc = [] # Training accuracy iteration 1
ValAcc = [] # Validation accuracy iteration 1
TestAccFullModel = [] # Test accuracy with iteration 2 model

Test_labels_full = [] # Ground truth labels
Pred_labels_full = [] # Predicted labels 

#%% CALLBACK INITIALISATION

# Callback 1: Stop training when val accuracy doesnt increase for a set number 
# of iterations 
Val_acc_stop = Callback_Iteration1(patience)
    
# Callback 2: Stop when train loss reaches train loss of iteration 1 
class myCallback(callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
    	if(logs.get('loss' ) < Train_loss):
    		print("Training stopped")
    		self.model.stop_training = True
                
# Instantiate a callback object
callbacks = myCallback()

   #%% DATA EXTRACTION

# Iterating through different subjects

for i in range (1,10):
    
    # Training and evaluation BCI2A file paths
    Train_path = 'C:/Users/Karan/.spyder-py3/EEGv2/A0'+str(i)+'T.mat'
    Eval_path = 'C:/Users/Karan/.spyder-py3/EEGv2/A0'+str(i)+'E.mat'

    # Extracting windows and their respective labels for training and evaluation sets
    WindowData_train,Labelarray_train = Data_extraction(Train_path,lf,hf,fs,Win_Start_s,Win_End_s,decay_factor,eps)
    WindowData_test,Labelarray_test = Data_extraction(Eval_path,lf,hf,fs,Win_Start_s,Win_End_s,decay_factor,eps)
    
    
    #%% ORGNAIZING TRAIN/VAL/TEST SETS
    
    # Randomly splits training dataset into training and validation sets 
    X_train, X_val, y_train, y_val = train_test_split(WindowData_train,Labelarray_train,test_size=val_size,shuffle=True)
    
    # Obtaining dataset dimension 
    N_samples = len(Labelarray_train)
    N_samples_test = len(Labelarray_test)
    Train_size = len(y_train)
    Test_size = len(y_val)
    N_channels = len(X_train[0][0])
    
    # Reshaping data for NN input 
    
    # Training set
    X_train = np.reshape(X_train,(Train_size,Win_Len,N_channels,1))
    
    # Validation set
    X_val = np.reshape(X_val,(Test_size,Win_Len,N_channels,1))
     
    # Merged Train + Validation set for training on iteration 2
    X_Train_It2 = np.reshape(WindowData_train,(N_samples,Win_Len,N_channels,1))
    
    # Test set 
    Test_data = np.reshape(WindowData_test,(N_samples_test,Win_Len,N_channels,1))
   
    #%% CREATING MODEL
 
    # Initializing model
    model1 = EEGModel(X_train.shape[1:],DrRate,N_channels,N_temp_filters,N_spatial_filters,temp_filter_len,pool_size,pool_stride,final_conv_filters)
    
    #  Model summary
    # model1.summary()
    
    # Model optimizer
    opt = keras.optimizers.Adam(learning_rate=lr)

    #Compiling model
    model1.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    
    #%% TRAINING MODEL   
    
    # Iteration 1 of training
    history = model1.fit(X_train, y_train, epochs=2000, validation_data=(X_val,y_val),callbacks = [Val_acc_stop], shuffle=True, batch_size=batch_size)
    
    # Saving best training and validation accuracy 
    Train_loss,Train_acc = model1.evaluate(X_train,y_train)
    Val_acc = np.max(history.history['val_accuracy'])
    
    # Iteration 2 of training 
    history2 = model1.fit(X_Train_It2,Labelarray_train, epochs=2000, callbacks=[callbacks])   
    
    # Test accuracy of model after second iteration
    Results = model1.predict(Test_data)
    PredictionLabels = Results.argmax(axis=-1)
    Test_acc = accuracy_score(PredictionLabels,Labelarray_test)
  
    #%% SAVING RESULTS AND LABELS
    
    TrainAcc.append(Train_acc)
    ValAcc.append(Val_acc)
    TestAccFullModel.append(Test_acc)
    
    Test_labels_full.append(Labelarray_test)
    Pred_labels_full.append(PredictionLabels)
    
    
  #%% CONFUSION MATRIX OF COMBINED RESULTS  
    
if plot_confusion == True:
    
    EEGComfusionMatrix(Test_labels_full,Pred_labels_full)


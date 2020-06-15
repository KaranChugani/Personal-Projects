# -*- coding: utf-8 -*-
"""
Created on Tue May 19 18:41:34 2020

@author: Karan
"""


from scipy import signal
import numpy as np
from mat4py import loadmat


# --------------------------------------------------------------------------------------------------------


# Applies and bandpass or lowpass filter to a raw EEG signal

    # Inputs:
    # Data = Raw EEG data
    # lf,hf = Low and high pass frequencies respectively
    # fs = Sampling frequency
    # Outputs: 
    # FiltData = Filtered EEG signal

def EEGfilter(Data,lf,hf,fs):
       
    # Calculating low and high pass frequencies as fractions
    nyq = 0.5 * fs
    low = lf/nyq
    high = hf/nyq

    # A bandpass or lowpass filter is applied depending on low cut frequency
    if lf > 0:
        b, a = signal.butter(3,[low, high], btype='bandpass')
    else:
        b, a = signal.butter(3,high, btype='lowpass')
                             
    FiltData = signal.lfilter(b,a,Data,axis=0)   

    return FiltData



# --------------------------------------------------------------------------------------------------------

 # Applies exponentially weighted standardization (EWS) to a given signal

    # Inputs:
    # Data = EEG data
    # decay_factor = EWS decay factor
    # eps = Epsilon
    # Outputs: 
    # DataArray = Standardized EEG signal

def EWS(Data,decay_factor,eps):
    
    I_df = 1 - decay_factor  # Decay factor - 1
    
    # Initializing post processing array
    SigLen = len(Data)
    DataArray = np.zeros(SigLen)

    # Each channel is standardized separately
    for i in range(len(Data[0])):

        Signal = Data[:,i]
        NewSig = np.zeros(SigLen)

        # Obtaining initial mean and variance using first 1000 samples
        ut= np.mean(Signal[0:1000])
        var = np.var(Signal[0:1000])

        # Calculating standardized value for each sample in the signal
        for i in range(SigLen):
            
            ut = I_df*Signal[i] + decay_factor*ut
            var = I_df*np.square(Signal[i]-ut) + decay_factor*var
            
            NewSig[i] = (Signal[i] - ut)/ np.maximum(np.sqrt(var),eps)
        
        # Adding standardized channel to main array
        DataArray = np.vstack((DataArray,NewSig))
    
    # Transposing final array 
    DataArray = np.transpose(DataArray)
    DataArray = np.delete(DataArray,0,axis=1)
    
    return DataArray


# --------------------------------------------------------------------------------------------------------

# This function preprocesses the EEG data for a given BCI Dataset 2a session 
# by filtering  using a digital bandpass/lowpass and applying EWS. It outputs
# a EEG data window and a label for trial in the session, each in a separate array.
    
    # Inputs:
    # path = Session saved in .mat format
    # lf,hf = Low and high pass frequencies respectively
    # fs = Sampling frequency
    # Win_Start = Start of the window of interest in samples
    # Win_End = End of the window of interest in samples
    # decay_factor = EWS decay factor
    # eps = Epsilon
    # Outputs: 
    # WindowData = Np Array with trialwise windowed EEG signal
    # Labelarray = Output movement for each trial 


def Data_extraction(path,lf,hf,fs,Win_Start_s,Win_End_s,decay_factor,eps):
    
    # Window start and end in samples
    Win_Start = int(Win_Start_s*fs)
    Win_End =   int(Win_End_s*fs)
    
   # Loading data 
    WindowData = []
    MatFile = loadmat(path)
    TrialList = MatFile['data']
    Labelarray = []
    
    Triallen = len(TrialList)
    
    for i in range(Triallen-6,Triallen):
        
        #Unpacking EEG Data
        Iteration=TrialList[i]
        RawData=Iteration['X']
        RawData = np.array(RawData)
        
        #Unpacking movement list and time
        label_time = Iteration['trial']
        label = Iteration['y']
        
        # Multipying by 10^6 (Same as paper)
        RawData = RawData * 1e6
        
        # Filtering data
        FiltData = EEGfilter(RawData,lf,hf,fs)
        
        # Exponentially weighted standardization
        DataArray = EWS(FiltData,decay_factor,eps)
        
        # Creating windows and label arrays for each individual trial 
        for j in range(len(label)):
            
            t = int(np.array(label_time[j]))
            #if int(np.array(artifacts[j])) == 0:
            #if np.max(np.abs(RawData[t+Win_Start:t+Win_End,0:25])) < 800:      
            WindowData.append(DataArray[t+Win_Start:t+Win_End,0:22])
            Labelarray.append(int(np.array(label[j])))
                
    # Converting label array to numpy and setting up labels for NN input
    Labelarray = np.array(Labelarray)
    Labelarray = Labelarray - 1
    
    # Converting data for NN input
    WindowData = np.array(WindowData)
    
            
    return WindowData,Labelarray
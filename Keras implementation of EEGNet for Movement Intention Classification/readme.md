
# **Project description**

This project implements the Shallow ConvNet described in [1] using the Keras API in the Tensorflow deep learning framework. The model is specially built to classify motor movement imagination from raw EEG with minimal preprocessing.

## Dataset:

The dataset used for this project was the BCI Competition IV dataset 2a, which can be downloaded from the following link: [Dataset](http://bnci-horizon-2020.eu/database/data-sets)

In brief, the dataset consists of 9 subjects performing 4 types of movement imagination which are listed below:

0.	Left hand
1.	Right hand
2.	Feet
3.	Tongue


For each subject two different sessions were recorded in different days. In each session the subject performed 6 runs separated by short breaks. Each run consisted of 48 trials (12 for each movement type) adding up to 288 trials for each session. The first session performed by the subject was used for training/validation whilst the second was exclusively used for testing. More information on the dataset can be found here [2].

## Preprocessing

Minimal preprocessing was applied to the dataset in order to test the capabilities of this system for end to end deep learning. The signal was digitally filtered using a 3rd order butterworth filter, with the best results being obtained when applying a lowpass filter at 38 Hz. After this electrode-wise exponentially weighted standardization(EWS), as described in section A7 of [1] was applied to standardize the continous data, with the decay factor having a value of 0.999 and the initial mean and the variance being obtained out of the first 1000 values of the signal. (Epsilon maybe)

## Shallow ConvNet















[1] Schirrmeister, Robin & Springenberg, Jost & Fiederer, Lukas & Glasstetter, Martin & Eggensperger, Katharina & Tangermann, Michael & Hutter, Frank & Burgard, Wolfram & Ball, Tonio. (2017). Deep learning with convolutional neural networks for EEG decoding and visualization: Convolutional Neural Networks in EEG Analysis. Human Brain Mapping. 38. 10.1002/hbm.23730. 




[2] Brunner C, Leeb R, Müller‐Putz G, Schlögl A, Pfurtscheller G (2008): BCI Competition 2008–Graz Data Set A. Institute for Knowledge Discovery (Laboratory of Brain‐Computer Interfaces), Graz University of Technology. pp 136–142




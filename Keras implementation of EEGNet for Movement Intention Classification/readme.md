
# **Project description**

This project implements the Shallow ConvNet described in [1] using the Keras API in the Tensorflow deep learning framework. The model is specially built to classify motor movement imagination from raw EEG with minimal preprocessing. Libraries needed;

## Dataset:

The dataset used for this project was the BCI Competition IV dataset 2a, which can be downloaded from the following link: [Dataset](http://bnci-horizon-2020.eu/database/data-sets)

In brief, the dataset consists of 9 subjects performing 4 types of movement imagination which are listed below:

0.	Left hand
1.	Right hand
2.	Feet
3.	Tongue

For each subject two different sessions were recorded in different days. In each session the subject performed 6 runs separated by short breaks. Each run consisted of 48 trials (12 for each movement type) adding up to 288 trials for each session. The first session performed by the subject was used for training/validation whilst the second was exclusively used for testing. More information on the dataset can be found here [2].

## Preprocessing

Minimal preprocessing was applied to the dataset in order to test the capabilities of this system for end to end deep learning. The signal was digitally filtered using a 3rd order butterworth filter, with the best results being obtained when applying a lowpass filter at 38 Hz. After this electrode-wise exponentially weighted standardization(EWS), as described in section A7 of [1] was applied to standardize the continous data, with the decay factor having a value of 0.999 and the initial mean and the variance being obtained out of the first 1000 values of the signal. For each trial the signal was windowed from 0.5s before trial onset to 4s after trial onset. As the sampling frequency was 250 Hz each window had a total of 4.5s * 250 samples/s = 1125 samples. (Epsilon maybe)

## Shallow ConvNet

The architechture of the network is based on the model described in figure 2 of [1]. The input to the network were the 22 EEG channels windowed in the region of interest, thus the input being of shape: Batch-size * 1125 * 22 * 1. The first block  consisted of two separated convolutional layers, the first performed convolutions through the temporal plane whilst the latter performed a convolution throught the spatial plane (electrodes). This was followed by a batch normalisation layer and a squaring non-linearity, which the authors of the paper suggest could better extract bandpower features in the signal. Following the squaring non-linearity were an average pooling layer, a log activation function and a final convolutional layer. The output of this layer was flattened and a dense layer with 4 neurons which used a sigmoid activation was used as the output layer. A summary of the model can be seen below:

## Model training 

The training strategy involved randomly splitting the training session into training/validation subsets, using an 80/20 split. The model was trained on the training subset and after each epoch, the accuracy of the model was tested on the validation subset. If the classification accuracy in the validation set stopped increasing for a set amount of epochs, training was stopped. The parameter which defined the amount of epochs the algorithm would wait before stopping training was called the patience and a value of 20 was found to work best. The weights of the model that performed best on this first iteration of training were saved. After this, the training of the model continued using both the training and validation subsets. This second iteration of training stopped when the training loss went below the training loss in the first iteration.Best results were found using a Adam as the optimizer and a learning rate of 0.0625 * 10^-2 and a mini batch size of 64.











[1] Schirrmeister, Robin & Springenberg, Jost & Fiederer, Lukas & Glasstetter, Martin & Eggensperger, Katharina & Tangermann, Michael & Hutter, Frank & Burgard, Wolfram & Ball, Tonio. (2017). Deep learning with convolutional neural networks for EEG decoding and visualization: Convolutional Neural Networks in EEG Analysis. Human Brain Mapping. 38. 10.1002/hbm.23730. 




[2] Brunner C, Leeb R, Müller‐Putz G, Schlögl A, Pfurtscheller G (2008): BCI Competition 2008–Graz Data Set A. Institute for Knowledge Discovery (Laboratory of Brain‐Computer Interfaces), Graz University of Technology. pp 136–142




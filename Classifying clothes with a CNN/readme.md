# **Project description**

This document will describe the steps taken build a Convolutional Neural Network for an clothes recognition task.

## Dataset:
The dataset used in this task is the Fashion-MNIST, consisting of 70000 (60000 training and 10000 testing) images of different clothes items from the Zalando catalogue. Each image has 28*28 dimensions and is in greyscale. There are 10 types of clothes items in this dataset: 

0.	T-shirt/top
1.	Trouser
2.	Pullover
3.	Dress
4.	Coat
5.	Sandal
6.	Shirt
7.	Sneaker
8.	Bag
9.	Ankle boot

More information on this dataset can be found in: [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist)

## Using PCA to explore the dataset:

In order to get a sense of the data PCA (Principal Component Analysis), which is a technique that can reduce the dimensionality of a dataset by applying orthogonal linear transformations, was used. In order to input the images into the PCA algorithm the 28*28 images where flattened out to form a vector of length 784, thus each observation having 784 features. First the proportion of variance explained by the principal components was calculated:

![](https://github.com/KaranChugani/Personal-Projects/blob/master/Classifying%20clothes%20with%20a%20CNN/Plots/Variance%20Explained.PNG)

As seen from the plot most of the variance is contained in the first components with the first 10 components containing around 73% of the variance and the first 2 principal components representing around 47% of the variance. Next, a scatter plot of the first two principal components was plot, with each clothes class being represented by a different colour.


![](https://github.com/KaranChugani/Personal-Projects/blob/master/Classifying%20clothes%20with%20a%20CNN/Plots/PCA.PNG)

The figure shows that the first two components give some separability in the dataset, however several classes appear quite close to each other in the feature space, which can give insights as to what classes could be misclassified with each other. For example, the sandal and the sneaker class are clustered in the same area and the Shirt and T-shirt class are scattered closely to each other 


## Convolutional Neural Network:

Convolutional neural networks (CNN) are a type of neural network that is commonly used in image classification problems. CNN´s first layers are normally convolutional and pooling layers, which extract the most relevant information from the images and reduce their dimensionality. This information is then fed into more classical dense layers. In this task the following architechture was used:

![](https://github.com/KaranChugani/Personal-Projects/blob/master/Classifying%20clothes%20with%20a%20CNN/Plots/CNNLAYERS.PNG)

Both the pooling and the convolutional layers used the smallest possible window sizes, as in this problem many clothes items had are very similar shape (eg: Sneaker and Sandal) which meant that small local features would be required to discriminate between objects with such similar shapes. 

## Cross-validation and optimizer selection

A selection test was performed in order to select the best optimizer function using the training dataset. To reduce the effects of overfitting 3 fold cross-validation was used. The functions tested were (more details on how each function works [here](https://keras.io/optimizers/)): 

1. SGD
2. RMSprop
3. Adagrad
4. Adadelta
5. Adam
6. Adamax
7. Nadam



## Results:

Using the parameters described as showing the best performance the CNN was trained using the Fashion MNIST training set. The resulting model was then evaluated using the testing data, with results being shown in the confusion matrix below:




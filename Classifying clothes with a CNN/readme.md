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

The figure shows that the first two components give some separability in the dataset, however several classes appear quite close to each other in the feature space, which can give insights as to what classes could be misclassified with each other. For example, the sandal and the sneaker class are clustered in the same area and the Shirt and T-shirt class are scattered closely to each other. The code used to create this plots can be found in Fashion_mnist_PCA_analysis.py.


## Convolutional Neural Network:

Convolutional neural networks (CNN) are a type of neural network that is commonly used in image classification problems. CNN´s first layers are normally convolutional and pooling layers, which extract the most relevant information from the images and reduce their dimensionality. In this CNN both the pooling and the convolutional layers used the smallest possible window sizes, as in this problem many clothes items had are very similar shape (eg: Sneaker and Sandal) which meant that small local features would be required to discriminate between objects with such similar shapes. Hyperparameter optimization was then used to find the best values for the number of filters in the convolutional layers and the best optimizer function

## Cross-validation and hyperparameter selection

A selection test was performed in order to select the best optimizer function using the training dataset. To reduce the effects of overfitting 3 fold cross-validation was used. The functions tested were (more details on how each function works [here](https://keras.io/optimizers/)): 

1. SGD
2. RMSprop
3. Adagrad
4. Adadelta
5. Adam
6. Adamax
7. Nadam

After the best optimizer was found a second test was performed to calculate the number of output filters in each convolution which gave the maximum classification accuracy. The filter numbers tested were: 8,16,32,48,64 and 80. The script, which is found in Hyperparameter_Tuning.py, showed that the "Adam" optimization method with a filter number of 80 gave the best performance, with only a slight increase in computation time. The architecture of the final model was: 

![](https://github.com/KaranChugani/Personal-Projects/blob/master/Classifying%20clothes%20with%20a%20CNN/Plots/CNNLAYERS.PNG)


## Results:

Using the optimal parameters the CNN was trained using the Fashion MNIST training set. The resulting model was then evaluated using the testing data, with results being shown in the confusion matrix below:

![](https://github.com/KaranChugani/Personal-Projects/blob/master/Classifying%20clothes%20with%20a%20CNN/Plots/Confusion%20Matrix.PNG)

As seen the model shows a high performance, with an accuracy of 91.6%. Most of the misclassification occurs between the T-shirt and Shirt classes, which was anticipated in the PCA scatter plot where the two classes were spread in the same area in the PC space. Other classes which were clustered in the same area such as the sandal and the sneaker, were differentiated with a excellent level of accuracy by the model with only 18 misclassifications (98.2% accuracy).

## References:

1. GitHub. (2020). zalandoresearch/fashion-mnist. [online] Available at: https://github.com/zalandoresearch/fashion-mnist [Accessed 8 Feb. 2020].

2. Keras.io. (2020). Optimizers - Keras Documentation. [online] Available at: https://keras.io/optimizers/ [Accessed 8 Feb. 2020].


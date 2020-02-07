# **Project description**

This document will describe the steps taken build a Convolutional Neural Network for an clothes recognition task.

## Dataset:
The dataset used in this task is the Fashion-MNIST, consisting of 70000 images of different clothes items from the Zalando catalogue. Each image has 28*28 dimensions and is in greyscale. There are 10 types of clothes items in this dataset: 

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

From the plot it can be seen that 

## Using PCA to explore the dataset:

In order to get a sense of the data PCA (Principal Component Analysis), which is a technique that can reduce the dimensionality of a dataset by applying orthogonal linear transformations, was used. In order to input the images into the PCA algorithm the 28*28 images where flattened out to form a vector of length 784, thus each observation having 784 features. First the proportion of variance explained by the principal components was calculated:

![](https://github.com/KaranChugani/Personal-Projects/blob/master/Classifying%20clothes%20with%20a%20CNN/Plots/Variance%20Explained.PNG)

As seen from the plot most of the variance is contained in the first components with the first 10 components containing around 73% of the variance and the first 2 principal components representing around 47% of the variance. Next, a scatter plot of the first two principal components was plot, with each clothes class being represented by a different colour.


![](https://github.com/KaranChugani/Personal-Projects/blob/master/Classifying%20clothes%20with%20a%20CNN/Plots/PCA.PNG)

The figure shows that the first two components give some separability in the dataset, however several classes appear quite close to each other in the feature space, which can give insights as to what classes could be misclassified with each other. For example, the sandal and the sneaker class are clustered in the same area and the Shirt and T-shirt class are scattered closely to each other 



## Convolutional Neural Network:

To make the decision as to which classifier to choose a literature review was made, specifically on papers that compared various classification techniques in the context of human activity recognition. A 2014 paper by researchers in the Ostrava Technical University compared the performance of various classifiers on three different Human Activity Recognition (HAR) datasets. Some of these classifiers are: Quadratic Discriminant Analysis (QDA), Random Forests (RF) and k-Nearest Neighbour (k-NN) amongst others. Results showed that k-NN and RF had the best performance [1]. Another paper published by researchers in Oulu University (Finland) and Waseda University (Japan) compared the performance of k-NN algorithm against a multilayer perceptron for the classification of 17 different activities. Results showed that the k-NN algorithm slightly outperformed the multilayer perceptron (92.89% against 89.76%) [2] . Due to its simplicity in design compared to other classifiers and its high classification accuracy, it was decided to implement the k-NN algorithm.

## Cross-validation and hyperparameter tuning

The k-NN algorithm works in a very simple way. First, a set of training data is mapped onto a feature space, without making any changes to the structure of the data. Then, to classify a new data point, a loop that finds the observations that are closest to the new testing observation (nearest neighbours) are found. This is where k comes in. If k = 1, the class assigned to the new observation is equal to the class of its nearest neighbour. If k > 1, a majority vote is taken amongst the k nearest to decide the class of the new observation. When there is a tie in the majority vote, the value of K is reduced until there was no longer a tie. The two main parameters that are used to tune the k-NN are the number k and the method used to calculate distance between different points. The classifier is implemented in KNN_ClassifierTest.m and uses MajorityVote.m and DistanceCalc.m as subroutines. 


## Results:




## References:

1. P. G. ,. T. P. Pavel Dohnálek, “Human activity recognition: classifier performance evaluation on multiple datasets,” JVE Journals,vol.16, no. 3, pp. 1523-1534, 2014.
2. S. P. F. Nakajima, “Feature Selection and Activity Recognition from Wearable Sensors,” International Symposium on UbiquitiousComputing Systems, vol. 4239, no. -, pp. 516-527, 2006.
3. M.-W. H. S.-W. K. a. C.-F. T. Li-Yu Hu, “The distance function effect on k-nearest neighbor classification for medical datasets,” Springerplus, Kaohsiung, 2016.© 

2020 GitHub, Inc.

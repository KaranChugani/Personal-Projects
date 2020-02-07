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


## PCA to explore the data:

The first step taken consisted in performing PCA (Principal Component Analysis), which is a technique that can reduce the dimensionality of a dataset by applying orthogonal linear transformations, which can transform large datasets into a new coordinate system where its features represent the differences amongst the directions with the greatest variances.
The advantage of PCA over using raw directly is that in most cases the number of principal components required to “explain” a high amount of variance in the data is much lower than the dimensionality of the original dataset, which can reduce the computational time it takes to train and test our classifier. Another advantage of PCA is that it allows the visualizing of how the different classes in our dataset are arranged, by plotting the first two or three principal components for each observation. A graphic observation of these components can give an indication of how well the different classes can be distinguished as well as showing outliers that don’t match the clustering patterns of the different classes. The code provided in Dataset_PCAVisualisation.m  performs PCA on the datasets and plots the first 3 and first 2 components in different figures. 

## Choosing a classifier:

To make the decision as to which classifier to choose a literature review was made, specifically on papers that compared various classification techniques in the context of human activity recognition. A 2014 paper by researchers in the Ostrava Technical University compared the performance of various classifiers on three different Human Activity Recognition (HAR) datasets. Some of these classifiers are: Quadratic Discriminant Analysis (QDA), Random Forests (RF) and k-Nearest Neighbour (k-NN) amongst others. Results showed that k-NN and RF had the best performance [1]. Another paper published by researchers in Oulu University (Finland) and Waseda University (Japan) compared the performance of k-NN algorithm against a multilayer perceptron for the classification of 17 different activities. Results showed that the k-NN algorithm slightly outperformed the multilayer perceptron (92.89% against 89.76%) [2] . Due to its simplicity in design compared to other classifiers and its high classification accuracy, it was decided to implement the k-NN algorithm.

## How does the K-Nearest Neighbours algorithms work:

The k-NN algorithm works in a very simple way. First, a set of training data is mapped onto a feature space, without making any changes to the structure of the data. Then, to classify a new data point, a loop that finds the observations that are closest to the new testing observation (nearest neighbours) are found. This is where k comes in. If k = 1, the class assigned to the new observation is equal to the class of its nearest neighbour. If k > 1, a majority vote is taken amongst the k nearest to decide the class of the new observation. When there is a tie in the majority vote, the value of K is reduced until there was no longer a tie. The two main parameters that are used to tune the k-NN are the number k and the method used to calculate distance between different points. The classifier is implemented in KNN_ClassifierTest.m and uses MajorityVote.m and DistanceCalc.m as subroutines. 

## Data preprocessing:

First, Principal component analysis was applied to the dataset, to reduce its dimensionality for faster training and testing of the data. As it was vital to not lose any important information from the data, the number of principal components that explain 99.99% of the variance were calculated. This number was found to be 11, so all tests were performed using the scores of the 11 first principal components as features for each of the observations. The code used to calculate the components required is in Components_reqCalculator.m.

## Cross-validation:

To ensure a fair test N-fold Cross validation was used. This is a technique that allows for the dataset to be split into N subsets of which N-1 will be used for training and 1 will be used for testing. Each round of the algorithm involves using one of these subsets for training and the other subset for testing. Results are averaged over several rounds, which can give a better estimate of the predictive power of the classifier due to the significant reduction in bias. All the following tests were performed using 5-fold cross validation. The code used to separate the datasets into 5 random folds is found in FiveCrossValidation.m and uses crossvalind.m (belongs to MATLAB) as a subroutine.  

## Hyperparameter tuning:

Testing was carried out in the KNN_HyperparameterTuning script. This test had the objective of comparing three different distance equations: Euclidean, Manhattan and Cosine similarity (Equations can be found in [3]); as well as finding the optimal value of K to find which combination would give the highest classification accuracy. The three distance metrics were compared against each other for values of k ranging from 1 to 10 using 5- fold cross validation. Results are shown in the figure below, where the Manhattan distance metric slightly outperforms the other two methods. Regarding K, the most optimal value was 1, as the accuracy slowly decreased as the value of K increased. Tests were performed in the KNN_HyperparameterTuning.m.

![alt text](https://github.com/KaranChugani/Personal-Projects/blob/master/Machine%20learning%20for%20Human%20Activity%20Recognition/Hyperparameter_tuning_results.PNG)

## Pipeline design: 

Considering the results obtained and the high amount of computational time required to find optimal hyperparameters using cross-validation, it was decided to preselect the values of the hyperparameters. Hence, the only pre-processing that would be performed would be converting the training data PCA components and selecting features that would explain 99.99% of the variance. The most optimal hyperparameters were K = 1 and Distance Metric = Manhattan. TrainKNN.m and KNNClassify.m were used to implememt the pipeline.

## References:

1. P. G. ,. T. P. Pavel Dohnálek, “Human activity recognition: classifier performance evaluation on multiple datasets,” JVE Journals,vol.16, no. 3, pp. 1523-1534, 2014.
2. S. P. F. Nakajima, “Feature Selection and Activity Recognition from Wearable Sensors,” International Symposium on UbiquitiousComputing Systems, vol. 4239, no. -, pp. 516-527, 2006.
3. M.-W. H. S.-W. K. a. C.-F. T. Li-Yu Hu, “The distance function effect on k-nearest neighbor classification for medical datasets,” Springerplus, Kaohsiung, 2016.© 

2020 GitHub, Inc.

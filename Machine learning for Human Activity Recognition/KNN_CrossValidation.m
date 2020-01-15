%% KNN algorithm

% This script performs cross validation tests for several values of K and
% for different distance metrics

load data.mat

% Distance metric: 1 = Manhattan, 2 = Euclidean and 3 = Cosine similarity
DM = 1;

% Set to 1 if you would like to plot the confusion matrix, 0 otherwise
CMPlot = 1;


%% Preprocessing data

%Perform PCA analysis of full training dataset
[~,PCA_score,~,~,PCA_explained,~] = pca(data(:,2:65));

%Finding number of components which explain 99.99% of the variance in the data
Variance_req = 99.99;
NumOfComp = Components_reqCalculator(Variance_req,PCA_explained);

%Merging data labels with the Number of components that explain 99.99% of
%variance
PCAfeatures = [data(:,1),PCA_score(:,1:NumOfComp)];

%Separating data in 5 training and testing folds
[TestChunk,TrainChunk] =  FiveCrossValidation(PCAfeatures);


%% Performing cross validation tests

for CV=1:5

testData = TrainChunk{CV}';
trainData = TestChunk{CV}';

%Range of K to test
for K = 1:1
[Accuracy(CV,K),PredictionLabels{CV,K}] = KNN_Classifier(DM,K,trainData,testData);
end

end






% Karan Chugani 











                  
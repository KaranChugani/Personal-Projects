function [Accuracy,PredictionLabels] = KNN_ClassifierTest(DM,K,trainData,testData)

% This function takes takes in distance metric and K hyperparameters + the
% training and testing dataset in order to compute the accuracy and
% prediction labels of the testing dataset.

% Inputs: DM = Distance Metric(1=Manhattan,2=Euclidean,3=Cosine)
%         K = Number of K nearest neighbours used for majority voting
%         trainData = Training dataset with each sample in a different 
%         column (First row = Label vector) 
%         testData = Testing dataset with each sample in a different column
%         (First row = Label vector) 
% Outputs: Accuracy = Accuracy of classifier on testing dataset
%          PredictionLabels = Predicted class labels for testing dataset. 

%Unpacking training and testing data
trainFea = trainData(2:end,:); 
trainLabel = trainData(1,:); 

testFea = testData(2:end,:);
TestLabel = testData(1,:); 

errorNo=0;

for i=1:size(testFea,2) 
    TestSamp = testFea(:,i);
    for j=1:size(trainFea,2)
        TrainSamp = trainFea(:,j);
        %Calculates the distance between each training and testing sample
        dist(j) = DistanceCalc(DM,TestSamp,TrainSamp);   
    end
    %Calculates K nearest values and assigns class after majority vote
    PredClass = MajorityVote(dist,K,trainLabel);
    PredictionLabels(i)=PredClass;     
     %Calculating classifiction accuracy 
     if PredictionLabels(i)~=TestLabel(i)                    
         errorNo=errorNo+1;
     end 
     Accuracy = 1 - (errorNo/length(PredictionLabels));
     dist = [];
end

% Karan Chugani 


function label = KNNClassify(input, parameters)

%This function uses the fed input data to classify a new set of given
%testing data using a K-NN classifier. Parameter K and the distance metric
%can be set beforehand. The algorithm first uses PCA to reduce the
%dimensionality of the data to reduce computational costs and then feeds
%the testing data into the classifier. 

% Inputs:  input = Incoming test data
%          parameters = Training dataset
% Outputs: label = Predicted label 

%% Parameter settings

% Value of K
K = 1; 

% Distance metric: 1 = Manhattan, 2 = Euclidean and 3 = Cosine similarity
DM = 1;

%% Data preprocessing

%Extracting training data and labels
Features_Train = parameters{1};
Trainlabels = parameters{2};

%Joining Training and Testing data for PCA analysis
PCAInput = [Features_Train;input];
Trainlength = length(Features_Train);

%Performing PCA analysis
[~,score,~,~,PCA_explained,~] = pca(PCAInput);

%Finding number of components which explain 99.99% of the variance in the data
Variance_req = 99.99;
NumOfComp = Components_reqCalculator(Variance_req,PCA_explained);

%Extracting PCA training and testing datasets
trainFea = score(1:Trainlength,1:NumOfComp)';
testFea = score(Trainlength+1:end,1:NumOfComp)';

%% Nearest neigbour computation

for i=1:size(testFea,2) 
    TestSamp = testFea(:,i);
    for j=1:size(trainFea,2)
        TrainSamp = trainFea(:,j);
        %Calculates the distance between each training and testing sample
        dist(j) = DistanceCalc(DM,TestSamp,TrainSamp);   
    end
    %Calculates K nearest values and assigns class after majority vote
    PredClass = MajorityVote(dist,K,Trainlabels);
    PredictionLabels(i)=PredClass;         
end

label =  PredictionLabels';

end

% Karan Chugani 
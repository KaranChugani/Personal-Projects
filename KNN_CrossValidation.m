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
[coeff,score,latent,tsquared,explained,mu] = pca(data(:,2:65));

%Finding number of components which explain 99.99% of the data
CompEx = 0;
ii = 1;

while CompEx < 99.99
CompEx = CompEx + explained(ii);
ii = ii + 1;
end

PCAfeatures = [data(:,1),score(:,1:ii)];

%Performing 5-fold cross validation
[TestChunk,TrainChunk] =  FiveCrossValidation(PCAfeatures);


%% Performing cross validation tests

%Number of cross-validation rounds
for CV=1:5

testData = TrainChunk{CV}';
trainData = TestChunk{CV}';

trainFea = trainData(2:ii+1,:); 
trainLabel = trainData(1,:); 

testFea = testData(2:ii+1,:);
TestLabel = testData(1,:); 

%Range of K to test
for K = 1:1

errorNo=0;
Class = zeros(5,1);


      for i=1:size(testFea,2) 
        TestNN = testFea(:,i);
        for j=1:size(trainFea,2)
            Train = trainFea(:,j);                                  
            switch DM                       %Distance metric is predifined by the user
                case 1                                
            Diff = TestNN-Train;
            dist(j)=norm(Diff,1);              
                case 2
            Diff = TestNN-Train;
            dist(j)=norm(Diff,2);
                case 3
            dist(j)=acos(TestNN'*Train/(norm(TestNN,2)*norm(Train,2)));
            end
        end
        [B,I] = mink(dist,K);
        for m = 1:K
            NNClass(m) = trainLabel(I(m)); %Finding class of k nearest values
        end
        
        for n = 1:K                        %Counting K nearest values
            if NNClass(n) == 1
                Class(1) = Class(1) + 1;
            end           
             if NNClass(n) == 2
                Class(2) = Class(2) + 1;
             end      
             if NNClass(n) == 3
                Class(3) = Class(3) + 1;
             end            
             if NNClass(n) == 4
                Class(4) = Class(4) + 1;
             end
              if NNClass(n) == 5
                Class(5) = Class(5) + 1;
             end              
        end          
        
                           
        idx = find( Class(:) == max(Class(:)) );    % Checks if there is a tie on the majority voting
        RemoveK = K;                                    
        while length(idx) > 1                           %Computes K-1 Nearest neighbour removing the furthest element
             RemoveClass = trainLabel(I(RemoveK));      %This process is repeated until the tie in the majority voting is broken
             Class(RemoveClass) = Class(RemoveClass)-1;
             RemoveK = RemoveK - 1;
             idx = find( Class(:) == max(Class(:)) );    % Checks if there is a tie on the majority voting
        end
        
        idx = [];
        
        [M,PredClass] = max(Class);                     %Assigns predicted class to max voted class
        classNo(i)=PredClass;                
        if classNo(i)~=TestLabel(i)                     %Classifiction accuracy testing
            errorNo=errorNo+1;
        end
        Class = zeros(5,1);     
      end
      
      KAccuracy(K,CV) = 1 - (errorNo/length(classNo));           %Stores classification accuracy of different values of K for different tests
      
      if CMPlot == 1
      
      CM = plotconfusion(GroundtruthNN(TestLabel),GroundtruthNN(classNo));
      
      end
      
      dist = [];

end

end


% Karan Chugani 01617930











                  
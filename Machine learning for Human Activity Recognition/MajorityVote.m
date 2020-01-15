function PredClass = MajorityVote(dist,K,trainLabel)

% This function selects a class for an individual test sample,

% Inputs: dist = Distance matrix between test sample point and all the
%         training samples,
%         K = Number of k nearest points to classify, 
%         trainLabel = Class labels for all of the training samples.
% Outputs:PredClass = Predicted class for the test sample

%Calculating closest sample
[~,I] = mink(dist,K);

%Initialzing class counter matrix
Class = zeros(5,1);
    
%Finding class of k nearest values
for m = 1:K
   NNClass(m) = trainLabel(I(m)); 
end
     
%Counting K nearest values
for n = 1:K                       
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

%Checks if there is a tie on the majority voting
idx = find( Class(:) == max(Class(:)) );
%Computes K-1 Nearest neighbour removing the furthest element if there is a
%tie.This process is repeated until the tie in the majority voting is broken
RemoveK = K;                                    
while length(idx) > 1                           
  RemoveClass = trainLabel(I(RemoveK));      
  Class(RemoveClass) = Class(RemoveClass)-1;
  RemoveK = RemoveK - 1;
  idx = find( Class(:) == max(Class(:)) );    
end

%Assigns predicted class to max voted class
[~,PredClass] = max(Class);   
        
end

% Karan Chugani 
        
                          
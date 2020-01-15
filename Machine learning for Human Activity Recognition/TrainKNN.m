function parameters = TrainKNN(input, label)

%This function arranges the training data and labels into a single cell for
%it to be fed into the ClassifyX function where all of the data
%preprocessing and classification is performed.

% Inputs:  input = Training feature vector
%          label = Training dataset labels
% Outputs: parameters = Merged labels and features

parameters = {input,label};

end



% Karan Chugani 
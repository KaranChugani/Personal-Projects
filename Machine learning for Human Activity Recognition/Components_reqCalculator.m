function NumOfComp = Components_reqCalculator(Variance_req,PCA_explained)

% This function calculates the PCA components required to explain X 
% percentage of variance.

% Inputs: Variance_req = Defines the amount of variance you want explained.
%         PCA_explained = Array which contains the percentage of variance
%         explained by each component in descending order
% Outputs: NumofComp = Minimun number of components that explain the
%          percentage of variance defined in Variance_req
    

CompEx = 0;
NumOfComp = 0;

while CompEx < Variance_req
NumOfComp = NumOfComp + 1;
CompEx = CompEx + PCA_explained(NumOfComp);
end

% Karan Chugani 
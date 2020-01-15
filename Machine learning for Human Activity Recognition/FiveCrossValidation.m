function [TestChunk,TrainChunk] =  FiveCrossValidation(data) 

%This function arranges the data in 5 different cell array folds, ready to perform
%crossvalidation using a 80/20 split.

%Input: data = Dataset to be analysed. Each training sample should be in a
%       different row, with all labels in the first column.
%Output:TestChunk = 5*1 cell array containing the 5 test folds
%       TrainChunk = 5*1 cell array containing the 5 training folds.

% Initializing counter for observations of each class
M1Count = 0;
M2Count = 0;
M3Count = 0;
M4Count = 0;
M5Count = 0;


%Adding classes into different arrays
for i = 1:length(data) 
    
    Movement = data(i,1);
    
    if Movement == 1
        M1Count = M1Count + 1;
        M1(M1Count,:) = data(i,:);
    end
    
    if Movement == 2   
        M2Count = M2Count + 1;
        M2(M2Count,:) = data(i,:);
    end
    
    if Movement == 3     
        M3Count = M3Count + 1;
        M3(M3Count,:) = data(i,:);
    end
    
    if Movement == 4 
        M4Count = M4Count + 1;
        M4(M4Count,:) = data(i,:);
    end
    
    if Movement == 5   
        M5Count = M5Count + 1;
        M5(M5Count,:) = data(i,:);
    end
    
end

%Performing random assigment of Cross validation folds
cvIndicesM1 = crossvalind('Kfold',M1Count,5);
cvIndicesM2 = crossvalind('Kfold',M2Count,5);
cvIndicesM3 = crossvalind('Kfold',M3Count,5);
cvIndicesM4 = crossvalind('Kfold',M4Count,5);
cvIndicesM5 = crossvalind('Kfold',M5Count,5);

cVIndicesTot = [cvIndicesM1;cvIndicesM2;cvIndicesM3;cvIndicesM4;cvIndicesM5];
Datasep = [M1;M2;M3;M4;M5];

% Arranging observations into 5 different folds

C1=1;
C2=1;
C3=1;
C4=1;
C5=1;

for i=1:length(cVIndicesTot)
    
    chunkindex = cVIndicesTot(i);  
    
    if chunkindex == 1
        cvChunk1(C1,:) = Datasep(i,:);
        C1 = C1 + 1;
    end
    if chunkindex == 2
        cvChunk2(C2,:) = Datasep(i,:);
        C2 = C2 + 1;
    end
    if chunkindex == 3
        cvChunk3(C3,:) = Datasep(i,:);
        C3 = C3 + 1;
    end
    if chunkindex == 4
        cvChunk4(C4,:) = Datasep(i,:);
        C4 = C4 + 1;
    end
    if chunkindex == 5
        cvChunk5(C5,:) = Datasep(i,:);
        C5 = C5 + 1;
    end
    
end


% Organizing folds to be used for training and testing in each iteration
% of the cross validationº

TestChunk{1} = cvChunk1;
TestChunk{2} = cvChunk2;
TestChunk{3} = cvChunk3;
TestChunk{4} = cvChunk4;
TestChunk{5} = cvChunk5;

TrainChunk{1} = [cvChunk1;cvChunk2;cvChunk3;cvChunk4];
TrainChunk{2} = [cvChunk1;cvChunk2;cvChunk3;cvChunk5];
TrainChunk{3} = [cvChunk1;cvChunk2;cvChunk4;cvChunk5];
TrainChunk{4} = [cvChunk1;cvChunk3;cvChunk4;cvChunk5];
TrainChunk{5} = [cvChunk2;cvChunk3;cvChunk4;cvChunk5];

end


    
    





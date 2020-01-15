%% PLOT PCA

% This script creates 3d and 2d plots of the first three and two principal
% components for the dataset provided

load data.mat

%% Preparing data for Principal Component Analisis

%Creating matrices for each of the classes

M1Count = 0;
M2Count = 0;
M3Count = 0;
M4Count = 0;
M5Count = 0;

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


%Performing PCA on data

DataSep = [M1;M2;M3;M4;M5];
[scores,coeffs] = pca(DataSep(:,2:65));


%Finding where each of the components starts and ends

M1End = M1Count;
M2Start = M1End+1;
M2End = M1Count+M2Count-1;
M3Start = M2End+1;
M3End = M1Count+M2Count+M3Count-2;
M4Start = M3End+1;
M4End = M1Count+M2Count+M3Count+M4Count-1;
M5Start = M4End+1;
M5End = M1Count+M2Count+M3Count+M4Count+M5Count;

%% Plotting principal components

%Plotting first 3 components for each class

figure(1)

scatter3(coeffs(1:M1End,1),coeffs(1:M1End,2),coeffs(1:M1End,3),'r');
hold on
scatter3(coeffs(M2Start:M2End,1),coeffs(M2Start:M2End,2),coeffs(M2Start:M2End,3),'b');
hold on
scatter3(coeffs(M3Start:M3End,1),coeffs(M3Start:M3End,2),coeffs(M3Start:M3End,3),'k');
hold on
scatter3(coeffs(M4Start:M4End,1),coeffs(M4Start:M4End,2),coeffs(M4Start:M4End,3),'c');
hold on
scatter3(coeffs(M5Start:M5End,1),coeffs(M5Start:M5End,2),coeffs(M5Start:M5End,3),'m');

legend('Sitting','Standing','Walking','Jogging','Martial Arts')

xlabel('PC1')
ylabel('PC2')
zlabel('PC3')

%Plotting first 2 components for each class

figure(2)

scatter(coeffs(1:M1End,1),coeffs(1:M1End,2),'r');
hold on
scatter(coeffs(M2Start:M2End,1),coeffs(M2Start:M2End,2),'b');
hold on
scatter(coeffs(M3Start:M3End,1),coeffs(M3Start:M3End,2),'k');
hold on
scatter(coeffs(M4Start:M4End,1),coeffs(M4Start:M4End,2),'c');
hold on
scatter(coeffs(M5Start:M5End,1),coeffs(M5Start:M5End,2),'m');

legend('Sitting','Standing','Walking','Jogging','Martial Arts')
xlabel('PC1')
ylabel('PC2')

% Karan Chugani 




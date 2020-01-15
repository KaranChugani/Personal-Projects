function dist = DistanceCalc(DM,TestNN,Train)

%This function calculates the distance between two points in an X
%dimensional space depending on the distance metric selected.

%Inputs:  DM = Distance metric(1=Manhattan,2=Euclidean,3=Cosine)
%         TestNN = Point1, Train = Point2
%Outputs: dist = Distance between two points

switch DM                       
   case 1                                
     Diff = TestNN-Train;
     dist=norm(Diff,1);              
    case 2
     Diff = TestNN-Train;
     dist=norm(Diff,2);
    case 3
     dist=acos(TestNN'*Train/(norm(TestNN,2)*norm(Train,2)));
end
 
end


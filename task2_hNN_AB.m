%
% Versin 0.9  (HS 06/03/2020)
%
function [Y] = task2_hNN_AB(X)
% Input:
%  X : N-by-D matrix of input vectors (in row-wise) (double)
% Output:
%  Y : N-by-1 vector of output (double)
% X = [2,5;1,1.5;2,3;5,1];

 %Because for polygon B a lot of lines intersect we need to cut it into two
 %different triangles in order to get the right polygon. W_B_UP are the
 %weights for the upper triangle, the other weights are the weights for the
 %lower triangle
 W_B_UP = [1;0.0416;-0.2819;1;-0.7469;0.8346;-1;0.3553;0.3096];
 W_B_DW = [1;-0.5294;0.1990;-1;0.8891;0.3469;1;-0.3553;-0.3096];
 
 %Importing weights for polygon A
 weights = importdata("task2_hNN_A_weights.txt",' ');
 W = weights.data;
 v = 1;
 u = 1;
 
 %for loop to read the file into 2 different weight matrix (hidden and last layer)
 for i = 1:length(W)
     if i <= 12
         Wl1(u) = W(i);
         u = u + 1;
     else
         Wl2(v) = W(i);
         v = v +1;
     end
 end

 %Matrix for the first layer of polygon B for both upper anf lower triangle
 layer1BU = reshape( W_B_UP, 3, []); 
 layer1BD = reshape( W_B_DW, 3, []);
 %Matrix for the first layer of polygon A
 layer1A = reshape(Wl1, 3, []);
 
 %Matrix for the second layer of polygon B (same for upper and lower)
 layer2B = reshape([-2,1,1,1],4, []);
 %Matrix for the second layer of polygon A
 layer2A = reshape(Wl2,5, []);
 
 %making sure that X is an N*2 matrix as asked
 X = reshape(X, [], 2); 
 
 hiddenU = [];
 hiddenD = [];
 hiddenA = [];
 %Finding the ouputs of the hidden layer for every point
 for i = (1:size(X,1))
   %For Polygon B
   hiddenU = [hiddenU; task2_hNeuron(layer1BU,X(i,:))];
   hiddenD = [hiddenD; task2_hNeuron(layer1BD,X(i,:))];
   %For polygon A
   hiddenA = [hiddenA; task2_hNeuron(layer1A,X(i,:))];
 end
 
 %reshaping the outputs: every row corresponds to the ouput of a point
 %For polygon B (3 boundaries per triangle)
 hiddenU = transpose(reshape(hiddenU,3,[]));
 hiddenD = transpose(reshape(hiddenD,3,[]));
 %For polygon A (4 boundaries)
 hiddenA = transpose(reshape(hiddenA,4,[]));
 
 outU= [];
 outD = [];
 %Finding the result when it goes through the second layer for Polygon B 
 for i = 1:size(hiddenU,1)
   outU = [outU; task2_hNeuron(layer2B,hiddenU(i,:))];
   outD = [outD; task2_hNeuron(layer2B,hiddenD(i,:))];
 end
 %Same as above but for Polygon A
 resultsA = [];
 for i = 1:size(hiddenA,1)
   resultsA = [resultsA; task2_hNeuron(layer2A,hiddenA(i,:))];
 end
 
 %concatenating to get the outputs of both triangle in a single matrix
 concatOut = cat(2,outU,outD);
 
 %For the point to be inside B we need it to be in either one of the
 %triangles so we design an OR gate
 resultsB = [];
  for j = 1:size(concatOut,1)
      if concatOut(j,1) == 1 | concatOut(j,2) == 1
          resultsB = [resultsB;1];
      else
          resultsB = [resultsB;0];
      end
  end
  
 %concatenating to get the outputs of both polygons in a single matrix
 results = cat(2,resultsA,resultsB);
 
 %For the points to be classified as 1 we need them to be inside B but
 %outside of B, so we look for a 0 in A and 1 in B
 Y = [];
 for x = 1:size(results,1)
     if results(x,1) == 0 & results(x,2) == 1
         Y = [Y;1];
     else
         Y = [Y;0];
     end
 end
end

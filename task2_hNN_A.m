%
% Versin 0.9  (HS 06/03/2020)
%
function [Y] = task2_hNN_A(X) 
% Input:
%  X : N-by-D matrix of input vectors (in row-wise) (double)
% Output:
%  Y : N-by-1 vector of output (double)

 %X = [1.5,3;2,2;1,1.8;2,2.8;1.9,2.4;2.5,2.5];
 
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
 % reshaping the obtained matrix to the right format 
 layer1 = reshape(Wl1, 3, []); % matrix for the first layer
 layer2 = reshape(Wl2,5, []); % matrix for the second layer
 
 %making sure that X is an N*2 matrix as asked
 X = reshape(X, [], 2); 
 
 hidden = [];
 %Finding the ouputs of the hidden layer for every point
 for i = (1:size(X,1))
   hidden = [hidden; task2_hNeuron(layer1,X(i,:))];
 end
 
 %reshaping the outputs: every row corresponds to the ouput of a point
 hidden = transpose(reshape(hidden,4,[]));
 
 Y = [];
 %Finding the final result when it goes through the second layer. 
 %This gives our Y
 for i = 1:size(hidden,1)
   Y =[Y; task2_hNeuron(layer2,hidden(i,:))];
 end
end

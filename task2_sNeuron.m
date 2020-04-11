%
% Versin 0.9  (HS 06/03/2020)
%
function [Y] = task2_sNeuron(W, X)
% Input:
%  X : N-by-D matrix of input vectors (in row-wise) (double)
%  W : (D+1)-by-1 vector of weights (double)
% Output:
%  Y : N-by-1 vector of output (double)
   Y = []; % Initializing output array 
  Xtr = transpose(X); % transposing x because x = (1,x')' 
  aug = ones(1, size(X,1)); % creating vector of ones to get augmented x
  Xaug = cat(1,aug,Xtr); % Adding the ones to get the 1,x' part
  y = transpose(W)* Xaug; % computing y(x) = h(w'x)
  y = transpose(y); % Transposing again because w = (w0,w')' and x = (1,x')'
   for i = (1:length(y)) % Looping through our new vector and assigning values with respect to the logistic sigmoid function
       denom = 1 + exp(-y(i));
       value = 1/denom;
       Y = [Y;value];
   end
end

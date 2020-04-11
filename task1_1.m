%
% Versin 0.9  (HS 06/03/2020)
%
function task1_1(X, Y)
% Input:
%  X : N-by-D data matrix (double)
%  Y : N-by-1 label vector (int32)
% Variables to save
%  S : D-by-D covariance matrix (double) to save as 't1_S.mat'
%  R : D-by-D correlation matrix (double) to save as 't1_R.mat'
  S = MyCov(X);
  R = MyCorrCov(S); 
  
  save('t1_S.mat', 'S');
  save('t1_R.mat', 'R');
end

function mu = MyMean(X) %This computes the mean mu given a matrix
  mu = sum(X) / size (X,1); %It sums X and divides it by the size of 1st dimension
end

function cov = MyCov(X)
  Substract_mean = bsxfun(@minus, X, MyMean(X)); %Substracts the mean from every element of X
  cov = (transpose(Substract_mean)*Substract_mean)/(size(X,1)); %computing our covariance matrix
end

function corr = MyCorrCov(X) %Function to compute correlation matrix from covariance matrix
  Standard_D =sqrt(diag(X)); %Gets the standard deviation of each Variable from the covariance matrix
  D = diag(Standard_D); %puts the result in a diagonal matrix
  D_inverse = inv(D); % inverting our matrix
  corr = D_inverse * X * D_inverse; %Computing the correlation matrix (normally we would need to transpose but since its diagonal it doesn't matter)
end

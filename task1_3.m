%
% Versin 0.9  (HS 06/03/2020)
%
function task1_3(Cov)
% Input:
%  Cov : D-by-D covariance matrix (double)
% Variales to save:
%  EVecs : D-by-D matrix of column vectors of eigen vectors (double)  
%  EVals : D-by-1 vector of eigen values (double)  
%  Cumvar : D-by-1 vector of cumulative variance (double)  
%  MinDims : 4-by-1 vector (int32)  
  [V,D] = eig(Cov);
  [EVals,index] = sort(diag(D),'descend'); %sorting the eigenvalues in descending order while keeping track of the indexes
  V2 = V(:,index); %reordering our eigenvectors using the indexes
  EVecs = V2;
  for i = (1:size(EVecs,1)) %loop from 1 to the first dimension of EVecs (doesn't matter which dimension, it's square)
      if EVecs(1,i) < 0 %checks if first element is negative
          EVecs (:,i) = -1 * EVecs(:,i); %if it's negative multiply by -1 to get opposit direction
      end
  end

  Cumvar = cumsum(EVals);% The cumulative variance is the cumulative sum of our eigen values
  
  percentage_difference = zeros(length(Cumvar),1);
  for i = (1:length(Cumvar)) % The percentage_difference array contains each entry i of CumVar as a percentage to the last element of CumVar
      percentage_difference(i) = 100-(100 *(Cumvar(length(Cumvar)) - Cumvar(i))./Cumvar(length(Cumvar)));
  end
  
  MinDims(1) = find(percentage_difference >= 70, 1, 'first'); % This allows us to find the first value where we cover 70% of the total variance using the values in percentage_difference
  MinDims(2) = find(percentage_difference >= 80, 1, 'first'); % Same as above but with 80
  MinDims(3) = find(percentage_difference >= 90, 1, 'first'); % Same as above but with 90
  MinDims(4) = find(percentage_difference >= 95, 1, 'first'); % Same as above but with 95
  save('t1_EVecs.mat', 'EVecs');
  save('t1_EVals.mat', 'EVals');
  save('t1_Cumvar.mat', 'Cumvar');
  save('t1_MinDims.mat', 'MinDims');
end

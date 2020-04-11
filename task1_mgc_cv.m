%
% Versin 0.9  (HS 06/03/2020)
%
function task1_mgc_cv(X, Y, CovKind, epsilon, Kfolds)
% Input:
%  X : N-by-D matrix of feature vectors (double)
%  Y : N-by-1 label vector (int32)
%  CovKind : scalar (int32)
%  epsilon : scalar (double)
%  Kfolds  : scalar (int32)
%
% Variables to save
%  PMap   : N-by-1 vector of partition numbers (int32)
%  Ms     : C-by-D matrix of mean vectors (double)
%  Covs   : C-by-D-by-D array of covariance matrices (double)
%  CM     : C-by-C confusion matrix (double)

%% THIS PART IS FOR THE PARTITION MAP
% The number of samples for a class i is N(i)
  N = zeros(1,size(unique(Y),1)); %acts like a counter(starting at 0)
  for i = (1:size(Y,1)) 
      k = Y(i); % We get the class number and then add 1 to its counter
      N(k) = N(k) + 1; 
  end

% The number of partitions we use is KFolds, so, Mc (where c is the class
% number) are given as follows. 
  M = zeros(1,size(unique(Y),1));
  for i = (1:length(N))
      M(i) = floor(N(i)/Kfolds);
  end
  
  MCounter = M; %Will allow us to keep track of the remaining places available
  
  % This is a brief description on class_to_partition: This vector
  % represents the partition decomposition. It has the size of unique(Y)
  % because every Y(j) is going to be in that range. So whenever we go
  % through Y(j), we check in the class_to_partition which partition it
  % should go to. When a parition is full for a certain class(i) we add 1
  % to Class_to_partition(i) to make sure it goes to the next partition. SO
  % we first start with a vector of ones because at first it will go to the
  % first partition and then our loop will update it.
  
  Class_to_partition = ones(size(unique(Y),1),1);
  PMap = zeros(size(Y,1),1);
  for i = (1:size(Y,1)) % Looping through Y species since Y gives the class Number of X the indexes are the same
      if (MCounter(Y(i)) > 0) % Checking if we can still
          PMap(i) = Class_to_partition(Y(i)); %Get the partition from the class_to_partition vector
          MCounter(Y(i)) = MCounter(Y(i)) -1;% Decrease by one the number of available places
      else %if its equal to 0, it means that the parition has reached full capacity for that class
          MCounter(Y(i)) = M(Y(i)); %We reset the counter for that class, as we are now in a new partition
          if (Class_to_partition(Y(i)) < Kfolds) % Because the last partition must contain everything that remains we only do this when we didn't reach the last one
              Class_to_partition(Y(i)) = Class_to_partition(Y(i))+1; %We increase by one the partition where numbers of class Y(i) should go
              PMap(i) = Class_to_partition(Y(i));
          else
              PMap(i) = Class_to_partition(Y(i));
          end 
      end
  end

  save('t1_mgc_5cv_PMap.mat', 'PMap');
  
  %% THIS PART DOES ALL THE REQUIRED COMPUTATIONS FOR EACH P AND COVKIND
  % Used for file names
  a = num2str(Kfolds); 
  b = num2str(CovKind); 
  
  %To simplify notation
  lt = length(unique(Y));
  
  % Intializing For the Confusion Matrix Part
  sumOfCM = zeros(lt,lt); % To prepare the final confusion matrix
  prior = (1./lt)*ones(lt,1); %prior probabilities (with respect to uniform prior distribution over class)
  for p = (1:Kfolds)
      % Below is the procedure to get Covs and Mean Vector
      Xcopy = X; %Making a copy of X so that we don't alter X
      Ycopy = Y; %same for Y
      Ms = [];
      sumOfCovs = zeros(size(MyCov(X))); %In case we need shared covariance
      i = 1; % Counter for the todelete list
      for j = (1:size(X,1))
          if (PMap(j) == p) %Checking if the X is part of the partition we don't want to consider
              todelete(i) = j; %Adding the vectors to delete at the end of the iteration, this avoids an index out of range for deletion
              i = i+1; %incresing our counter
          end
      end
      for x = (1:size(todelete)) %looping over the rows to delete
          Xcopy(todelete(x),:) = []; %Deleting the rows in our copy of the matrix
          Ycopy(todelete(x),:) = []; %same as above
      end 
 
      for c = (1:length(unique(Y))) %looping over all the classes
          Xclass=[];
          for i = (1:size(Ycopy,1)) %This loops groups in one matrix all Xs from the same partition and class (XClass)
              if Ycopy(i) == c
                  Xclass = [Xclass; Xcopy(i,:)];
              end
          end
          tempCov = MyCov(Xclass); %This is the covariance of our groupd Xs
          mean = MyMean(Xclass);%This is the mean
          regCov = tempCov + epsilon.*eye(size(tempCov)); %This performs the regularisation of covariance matrix
          Ms = [Ms; mean]; %Adds the mean to the mean vector
          
          if CovKind == 1 %If we want a full matrix
              Covs(c,:,:) =  regCov; %We leave the cov as it is and add it to Covs
          elseif CovKind == 2 %If we want a diagonal amtrix
              diagCov = diag(diag(regCov)); %We keep only the diagonal values and set the rest to zero
              Covs(c,:,:) = diagCov; 
          end
          sumOfCovs = sumOfCovs + regCov; %This is for the shared covariance matrix, we add all the cov matrices together
      end
      
      if CovKind == 3 %If we want a shared covariance matrix
          for c = (1:length(unique(Y)))
              Covs(c,:,:) = sumOfCovs * (1./(length(unique(Y)))); %Then we take our previously calculed sum, divide the result by the number of classes
          end % and set every element in Cov to be the final result
      end
      pstr = num2str(p);
      filenameCovs = ['t1_mgc_' a 'cv' pstr '_ck' b '_Covs.mat']; % Concatenation for file saving
      filenameMs = ['t1_mgc_' a 'cv' pstr '_Ms.mat']; %Same as above
      save(filenameCovs, 'Covs');
      save(filenameMs, 'Ms');
      
      % Classification experiment
       test_prob = zeros(length(Xcopy), lt);
       for c = 1:lt
           lik_k = gaussianMV(Ms(c,:), Covs(c,:,:), Xcopy);
           %Multiply with prior to get the joint probability 
           test_prob(:,c) = lik_k * prior(c);
       end
       % Assign each data point to the class with the highest probability, stores in variable class_pred 
       [~, test_pred] = max(test_prob, [], 2);
       % Compute confuction matrix using the ’confusionmat’ function
       CM = MyConfusionMat(Ycopy,int32(test_pred));

       sumOfCM = sumOfCM + CM;
       filenameConf = ['t1_mgc_' a 'cv' pstr '_ck' b '_CM.mat'];
       save(filenameConf, 'CM');
  end
  observations = sum(sum(sumOfCM)); % all the elements in the final matrix
  CM = (1./observations)*sumOfCM;
  accuracy = (trace(CM)); % Sum of the diagonal for accuracy
  disp(accuracy);
  L = num2str(Kfolds+1);
  filenameFinalCM = ['t1_mgc_' a 'cv' L '_ck' b '_CM.mat'];
  save(filenameFinalCM, 'CM');

end

%% Helper functions (To avoid built in functions)
function mu = MyMean(X) %This computes the mean mu given a matrix
  mu = sum(X) / size (X,1); %It sums X and divides it by the size of 1st dimension
end

function cov = MyCov(X) %Computes covariance matrix
  Substract_mean = bsxfun(@minus, X, MyMean(X)); %Substracts the mean from every element of X
  cov = (transpose(Substract_mean)*Substract_mean)/(size(X,1)); %computing our covariance matrix
end

function  y = gaussianMV(mu, covar, X) %Code taken from Lab 7 and modified
  [n,d] = size(X);
  %disp(d);
  [~,k,l] = size(covar);
  %disp(k); disp(l);
  % Check that the covariance matrix is the correct dimension 
  if ((l ~= d) | (k ~=d)) 
      error('Dimension of covariance matrix and data should match'); 
  end
  covar = squeeze(covar); % gets rid of the first dimension (=1);
  invcov = inv(covar); 
  mu = reshape(mu, 1, d); %Ensures mu is a row vector
  
  % Replicate mu and subtract from each data point 
  X = X - ones(n, 1)*mu; 
  fact = sum(((X*invcov).*X), 2);
  
  y = exp(-0.5*fact);
  y = y./sqrt((2*pi)^d*det(covar));
end

function CM = MyConfusionMat(knownGroup,predictedGroup)
[n,~] = size(knownGroup);
[k,~] = size(predictedGroup);
 if (n ~= k) 
      error('number of rows must match'); 
 end
 % We start by initializing our confusion matrix as only 0, we will then
 % increase the counts in the cells 
 numberOfClasses = length(unique(knownGroup));
 CM = zeros(numberOfClasses,numberOfClasses); 
 %First we need to find all occurences of the class, and get the indices
 %where we find the class. So we start by looping over all classes
 for c = 1:numberOfClasses
     knownIndex = find(knownGroup == c);
     % Once we have the indices for that specific group,We loop through it
     % and we look for what's there in our predicted group. So say the
     % index is A then we look at predictedGroup(A) and update our
     % confusion matrix by adding one each time we ecounter it.
     for a = 1:length(knownIndex)
         index = predictedGroup(knownIndex(a));
         CM(c,index) = CM(c,index) + 1;
     end
 end
end

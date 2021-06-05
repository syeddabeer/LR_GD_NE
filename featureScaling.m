function [X_norm, mu, sigma] = featureScaling(X)
%featureScaling Normalizes the features in X 
X_norm = X;
% size(X, 2) = # of columns in matrix X
% mu and sigma = row vector of size 1xn where n is the # of 
% rows in X (# of training data)
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));
% First, for each feature dimension, compute the mean
% of the feature and subtract it from the dataset,
% storing the mean value in mu. Next, compute the 
% standard deviation of each feature and divide
% each feature by it's standard deviation, storing
% the standard deviation in sigma. 
[rownumber columnnumber]=size(X_norm)  
numberOfColumnsInX_norm = columnnumber;  
for i = 1:numberOfColumnsInX_norm, 
	meanOfCurrentFeatureInX = mean(X(:, i));
	mu(:, i) = meanOfCurrentFeatureInX;
	X_norm(:, i) = X_norm(:, i)-mu(:, i);
	standardDeviationOfCurrentFeatureInX = std(X(:, i));
	sigma(:, i) = standardDeviationOfCurrentFeatureInX;
	X_norm(:, i) = X_norm(:, i) ./ sigma(:, i);
end
end

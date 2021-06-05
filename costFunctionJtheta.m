function J = costFunctionJtheta(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
% Initializing
m = length(y); % number of training examples
J = 0;
% Compute the cost(J) of a particular choice of theta
% X = mxn matrix
% theta = nx1 column vector
% errors = mx1 column vector
% y = mx1 column vector
errors = (X * theta) - y;
% square all elements individually within 
% column vector errors
% squareOfErrors = mx1 column vector
squareOfErrors = (errors).^2;
% sumOfSquareErrors = single number
sumOfSquareErrors = sum(squareOfErrors);
% J = single number
J = (1/(2 * m)) * sumOfSquareErrors;
end

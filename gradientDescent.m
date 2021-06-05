function [theta, J_history] = gradientDescent(X, y, theta, alpha, numberOfIterations)
%gradientDescent Performs gradient descent to learn theta


% Initializing
m = length(y); % number of training examples
J_history = zeros(numberOfIterations, 1); 


% Perform a single gradient step on the parameter vector theta. 

for iteration = 1:numberOfIterations
    errors = (X * theta) - y;
    newDecrement = (alpha * (1/m) * X' * errors); 
    theta = theta - newDecrement;
    % Save the cost J in every iteration    
    J_history(iteration) = costFunctionJtheta(X, y, theta);

end

end

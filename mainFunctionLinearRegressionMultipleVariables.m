%  Linear regression with multiple variables
%% Initialization

%% ================ Part 1: Feature Scaling ================

%% Clear and Close Figures
clear ; close all; clc

fprintf('Loading data ...\n');

%% Load Data
data = load('trainlotfrontagelotarea.txt');
X = data(:, 2);
y = data(:, 1);
m = length(y);

% Print out some data points
fprintf('First 10 examples from the dataset: \n');
fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

[X mu sigma] = featureScaling(X);

% Add intercept term to X
X = [ones(m, 1) X];


%% ================ Part 2: Gradient Descent ================
fprintf('Running gradient descent ...\n');

% Choose some alpha value
alpha1 = 0.001;
alpha2 = 0.01;
alpha3 = 1;
alpha4 = 1.4;
alpha5 = 0.005;
num_iters = 1000;

% Init Theta and Run Gradient Descent 
theta1 = zeros(2, 1);
[theta1, J_history1] = gradientDescent(X, y, theta1, alpha1, num_iters);
theta2 = zeros(2, 1);
[theta2, J_history2] = gradientDescent(X, y, theta2, alpha2, num_iters);
theta3 = zeros(2, 1);
[theta3, J_history3] = gradientDescent(X, y, theta3, alpha3, num_iters);




% Display gradient descent's result
fprintf('Theta computed from gradient descent for learning rate %f: \n', alpha1);
fprintf(' %f \n', theta1);
fprintf('\n');

fprintf('Theta computed from gradient descent for learning rate %f: \n', alpha2);
fprintf(' %f \n', theta2);
fprintf('\n');

fprintf('Theta computed from gradient descent for learning rate %f: \n', alpha3);
fprintf(' %f \n', theta3);
fprintf('\n');


% Init Theta and Run Gradient Descent 
theta4 = zeros(2, 1);
[theta4, J_history4] = gradientDescent(X, y, theta4, alpha4, num_iters);

theta5 = zeros(2, 1);
[theta5, J_history5] = gradientDescent(X, y, theta5, alpha5, num_iters);

% Display gradient descent's result
fprintf('Theta computed from gradient descent for learning rate %f: \n', alpha4);
fprintf(' %f \n', theta4);
fprintf('\n');

% Plot the convergence graph
figure;
plot(1:numel(J_history1), J_history1, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');
hold on
plot(1:numel(J_history2), J_history2, '-b', 'LineWidth', 2);
hold on
plot(1:numel(J_history3), J_history3, '-b', 'LineWidth', 2);


fprintf('Press enter to continue to graph for learning rate %f.\n', alpha4);
pause;
fprintf('\n');



% Plot the convergence graph
figure;
plot(1:numel(J_history4), J_history4, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');





%% ================ Part 3: Normal Equations ================

fprintf('Solving with normal equations...\n');
fprintf('\n');

% The following code computes the closed form solution for 
% linear regression using the normal equations. 

%% Load Data
data = csvread('trainlotfrontagelotarea.txt');
X = data(:, 2);
y = data(:, 1);
m = length(y);

% Add intercept term to X
X = [ones(m, 1) X];

% Calculate the parameters from the normal equation
theta = normalEquation(X, y); 

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);
fprintf('\n');




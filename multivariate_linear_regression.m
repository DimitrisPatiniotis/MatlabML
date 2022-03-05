% Multivariate Linear Regression

% Data includes 47 datapoints of housing prices in Portland, Oregon.
%
% x=(x1,x2)=(area, number of bedrooms)
% y= house prices


clear all; close all; clc

x = load('datasets/mv_regressionx.dat'); 
y = load('datasets/mv_regressiony.dat');

m = length(y);


% Add intercept term to x
x = [ones(m, 1), x];

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% 1. Using 'mvregress' function: "mvregress(x,y)" returns the estimated coefficients for a multivariate normal regression of the d-dimensional responses in Y on the design matrices in X.
beta1=mvregress(x,y)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% 2. Using Stochastic Gradient Descent with different alphas
%alpha = [0.01, 0.03, 0.1, 0.3, 1, 1.3]
% total number of iterations = 100

% Scale features and set them to zero mean
mu = mean(x);
sigma = std(x);

x_unscaled = x; % Save unscaled features to calculate the parameters from the normal equation later

% Scaling
x(:,2) = (x(:,2) - mu(2))./ sigma(2);
x(:,3) = (x(:,3) - mu(3))./ sigma(3);

% Initialize parameters
alpha = [0.01, 0.03, 0.1, 0.3, 1, 1.3];
iters = 100;
%theta_grad_descent = zeros(size(x(1,:))); % to store the values of theta of the best learning rate
% Different plotstyle for each alpha value
plotstyle = {'g', 'r--', 'b--', 'k', 'r', 'b'};

for i = 1:length(alpha)
    theta = zeros(size(x(1,:)))'; % initialize fitting parameters
    J = zeros(iters, 1); % initialize J
    for num_iterations = 1:iters
        % Calculate J cost
        J(num_iterations) = (0.5/m).*(x*theta-y)'*(x*theta-y);
        % The gradient
        grad=(1/m).*x'*((x*theta)-y);
        % Updating theta
        theta=theta-alpha(i).*grad;
    end
    % Create a plot for different values of a
    plot(0:(iters-1), J(1:iters), char(plotstyle(i)), 'LineWidth', 2)
    hold on
end
% Figure
legend('0.01','0.03','0.1', '0.3', '1', '1.3')
xlabel('Number of iterations')
ylabel('Cost J')

format long

% print gradient descent's result
theta

% Calculate the parameters from the normal equation
theta_normal = (x_unscaled' * x_unscaled)\x_unscaled' * y


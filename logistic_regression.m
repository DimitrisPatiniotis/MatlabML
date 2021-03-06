% Logistic Regression

% The dataset consists of 80 students: 40 who admitted to college and 40 who did not
%
% x = student's score in two exams = (results1, results2)
% y = label of whether the student admitted to college
%


clear all; close all; clc

x = load('datasets/log_regressionx.dat');
y = load('datasets/log_regressiony.dat');

[m, n] = size(x);

% Add intercept term to x
x = [ones(m, 1), x];

% Visualize the datapoints
figure
plot(x(find(y), 2), x(find(y),3), '+')
hold on
plot(x(find(y == 0), 2), x(find(y == 0), 3), 'o')
xlabel('Result 1')
ylabel('Result 2')
legend('Admitted', 'Not admitted')
title({'Training Data'})


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 1. Using 'fitglm' function: "fitglm(x,y)" returns a generalized linear
%model of the responses y, fit to the data matrix X.
B= fitglm(x,y,'linear', 'distr', 'binomial' )
coeffs=B.Coefficients.Estimate;


% Only need 2 points to define a line, so choose two endpoints
plot_x = [min(x(:,2))-2,  max(x(:,2))+2];
% Calculate the decision boundary line
plot_y = (-1./coeffs(4)).*(coeffs(3).*plot_x +coeffs(2));
figure
plot(x(find(y), 2), x(find(y),3), '+')
hold on
plot(x(find(y == 0), 2), x(find(y == 0), 3), 'o')
hold on
xlabel('Result 1')
ylabel('Result 2')
plot(plot_x, plot_y)
legend('Admitted', 'Not admitted', 'Decision Boundary')
title({'Using fitglm function'})
hold off


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% 2. Using Stochastic Gradient Descent

% Scaling the data
x = load('datasets/log_regressionx.dat');
mu = mean(x);
sigma = std(x);
for i = 1:size(x,2)
   x_scaled(:, i) = (x(:, i) - mu(i)) ./ sigma(i);
end

x_scaled = [ones(m, 1), x_scaled]; %Scaled
x_scaled
% Initialize fitting parameters

theta = zeros(1, n + 1); 
alpha=0.1;
iterations=40;
% Initializing sigmoid function
g = inline('1.0 ./ (1.0 + exp(-z))');
% SGD IMPLEMENTATION HERE
for num_iterations = 1:iterations
    % Calculate the gradient
    grad = 0;
    for i = 1:m
        curr_x = x_scaled(i, :);
        curr_y = y(i);
        grad = grad + ((g(theta .* curr_x) - curr_y) .* curr_x);
    end
    grad = (1 / m) * grad;
    % Updating theta
    theta = theta - (alpha * grad);
end


theta
% Plot SGD method result

% Calculate the decision boundary line
plot_y = (-1./theta(3)).*(theta(2).*plot_x +theta(1));
figure
plot(x_scaled(find(y), 2), x_scaled(find(y),3), '+')
hold on
plot(x_scaled(find(y == 0), 2), x_scaled(find(y == 0), 3), 'o')
hold on
xlabel('Result 1')
ylabel('Result 2')
plot(plot_x,plot_y)
legend('Admitted', 'Not admitted', 'Decision Boundary')
title({'Using SGD method'})
hold off


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% 3. Using Newton's method

% Initialize fitting parameters
x = [ones(m, 1), x];
theta = zeros(n+1, 1);


% Define the sigmoid function
g = inline('1.0 ./ (1.0 + exp(-z))');

iters = 10;

for i = 1:iters
    % Calculate the hypothesis function
    z = x * theta;
    h = g(z);
    % Calculate gradient and hessian.
    % gradiant
    grad = (1/m).*x' * (h-y);
    % hessian
    H = (1/m).*x' * diag(h) * diag(1-h) * x;
    % Calculate J (for testing convergence)
    J(i) =(1/m)*sum(-y.*log(h) - (1-y).*log(1-h));
    % Update theta
    theta = theta - H\grad;
end
% Display theta
theta


% Vizualize Newton's method
% Calculate the decision boundary line
plot_y = (-1./theta(3)).*(theta(2).*plot_x +theta(1));
figure
plot(x(find(y), 2), x(find(y),3), '+')
hold on
plot(x(find(y == 0), 2), x(find(y == 0), 3), 'o')
hold on
xlabel('Result 1')
ylabel('Result 2')
plot(plot_x, plot_y)
legend('Admitted', 'Not admitted', 'Decision Boundary')
title({'Using Newton method'})
hold off


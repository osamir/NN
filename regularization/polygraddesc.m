% Polynomial Regression with Gradient Descent Example

% Generate sample data
x = [1.02	2.08	2.89	4.01	5.32	5.83	7.26	7.96	9.11	9.99]; % Independent variable
y = [1.15 	0.85 	1.56 	1.72 	4.32 	5.07 	5.00 	5.31 	6.17 	7.04]; % Dependent variable

% Set the degree of the polynomial
degree = 6;

% Feature scaling (optional but recommended)
x_scaled = (x - mean(x)) / std(x);
m = length(x_scaled);
% Initialize parameters
theta = zeros(degree+1, 1); % Polynomial coefficients (including bias term)
alpha = 0.005; % Learning rate
lambda = 1;
num_iterations = 100000; % Number of iterations

% Perform gradient descent
for iter = 1:num_iterations
    % Calculate predictions
    X = ones(m, 1);
    for d = 1:degree
        X = [X ,(x_scaled').^d];
    end
    %X = [ones(m, 1), x_scaled', (x_scaled').^2, (x_scaled').^3]; % Design matrix
    y_pred = X * theta; % Predicted values
    
    % Calculate error
    error = y_pred - y';
    
    % Update parameters using gradient descent
    %theta(1) = theta(1) - (alpha/m) * (X(:,1)' * error);
    %theta(2:end) = theta(2:end)*(1-alpha*lambda/m) - (alpha/m) * (X(:,2:end)' * error);
    theta = theta*(1-alpha*lambda/m) - (alpha/m) * (X' * error);
end

% Generate predictions using the polynomial model
x_test = x(1):0.1:x(end); % Test data
x_test_scaled = (x_test - mean(x)) / std(x); % Feature scaling for test data
X_test = ones(length(x_test_scaled), 1);
    for d = 1:degree
        X_test = [X_test ,(x_test_scaled').^d];
    end
y_pred = X_test * theta;

% Plot the original data and the polynomial regression curve
figure;
scatter(x, y, 'b', 'filled'); % Original data points
hold on;
plot(x_test, y_pred,'r'); % Polynomial regression curve
xlabel('x');
ylabel('y');
title('Polynomial Regression with Gradient Descent');
title(['lamda = ',num2str(lambda)])
legend('Data', 'Polynomial Regression');

% Display the polynomial coefficients
disp('Polynomial Coefficients:');
disp(theta);
grid
set(findobj(gca,'type','line'),'linew',2)
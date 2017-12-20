function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta.
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    calc1 = 0.0;
    calc2 = 0.0;

    for j = 1:m,
      calc1 = calc1 + (theta(1,1) + theta(2,1)*X(j,2) - y(j))*X(j,1);
    end;
    calc1 = (alpha/m)*calc1;

    for j = 1:m,
      calc2 = calc2 + (theta(1,1) + theta(2,1)*X(j,2) - y(j))*X(j,2);
    end;
    calc2 = (alpha/m)*calc2;

    theta(1,1) = theta(1,1) - calc1;
    theta(2,1) = theta(2,1) - calc2;

    % ============================================================

    % Save the cost J in every iteration
    J_history(iter) = computeCost(X, y, theta);
    disp(J_history(iter))
end

end

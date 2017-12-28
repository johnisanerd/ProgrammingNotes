function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

transThetaX = X*theta;
sigmoidThetaX = sigmoid(transThetaX);

[J, grad] = costFunction(theta, X, y);

part2 = 0; % This is going to hold the normalization function lambda bullshit in it.
n = length(theta);

for i = 2:n;
    part2 = part2 + theta(n)**2;
end;

part2 = (lambda*part2)/(2*m);

J = J + part2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Gradient Descent Calculation
% Normalized

grad = zeros(size(theta));

for j = 1:m,
  % j
  grad(1) = grad(1) + (sigmoidThetaX(j) - y(j))*X(j,1);
end;
grad(1) = grad(1)/m;

theta
j = rows(theta)

for i = 2:j,
  i;
  for j = 1:m,
    j;
    grad(i) = grad(i) + (sigmoidThetaX(j) - y(j))*X(j,i);
  end;
  grad(i) = grad(i) + lambda*theta(i);
  grad(i) = grad(i)/m;
end;

% =============================================================

end

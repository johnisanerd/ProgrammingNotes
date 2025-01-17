function g = sigmoidGradient(z)
%SIGMOIDGRADIENT returns the gradient of the sigmoid function
%evaluated at z
%   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
%   evaluated at z. This should work regardless if z is a matrix or a
%   vector. In particular, if z is a vector or matrix, you should return
%   the gradient for each element.

g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the gradient of the sigmoid function evaluated at
%               each value of z (z can be a matrix, vector or scalar).

% g_prime(z) = g(z)(1-g(z))
% sigmoid(z) = g(z) = 1/(1+e^-z)
%%% IMPLEMENT Page 7 Equation, use elemental operations.

g_z = 1.0 ./ (1.0 + exp (-z));
g_prime_z = g_z.*(1-g_z);

g = g_prime_z;

% =============================================================




end

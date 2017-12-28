function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

% Sigmoid function is defined as g(z) = 1/(1+e^-z)

zrows = rows(z);
zcols = columns(z);

for i = 1:zrows,
  for j = 1:zcols,
    g(i,j) = 1/(1+e^(-1*z(i,j)));
  end;
end;

g;

% =============================================================

end

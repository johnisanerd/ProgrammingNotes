function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

% When you complete the code in nnCostFunction you will need to add the column of 1s to the X matrix.

% Add ones to the X data matrix
X = [ones(m, 1) X];
% Convert y from (1-10) class into num_labels vector
yd = eye(num_labels);
y = yd(y,:);

% Layer 1 to Layer 2
% a1 = x ; z2 = Theta1*a1;  a2 = g(z2)
a1=X;
z2=X*transpose(Theta1);
a2=sigmoid(z2);

% Layer 2 to Layer 3
% z3=Theta2*a2 ; a3=g(z3) = h(x)
a2=[ones(m, 1) a2];   % ones added to matrix
z3=a2*transpose(Theta2);
a3=sigmoid(z3);

% Compute cost


Theta1s=Theta1(:,2:end);
Theta2s=Theta2(:,2:end);

h = a3;
logcomp = (-y).*log(h)-(1-y).*log(1-h);
J=((1/m).*sum(sum(logcomp)))+(lambda/(2*m)).*(sum(sum(Theta1s.^2))+sum(sum(Theta2s.^2)));


% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.

triangle_delta_1=0;
triangle_delta_2=0;

% Compute delta, tridelta and big D
delta_3=a3-y;
z2=[ones(m,1) z2];
delta_2=delta_3*Theta2.*sigmoidGradient(z2);
delta_2=delta_2(:,2:end);

triangle_delta_1=triangle_delta_1+transpose(delta_2)*a1;
triangle_delta_2=triangle_delta_2+transpose(delta_3)*a2;

Theta1_grad=(1/m).*triangle_delta_1;
Theta2_grad=(1/m).*triangle_delta_2;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

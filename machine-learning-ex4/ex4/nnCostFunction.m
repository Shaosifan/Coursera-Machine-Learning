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
%
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
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% Part 1: Feedforward the neural network and compute J

% turn label into vector
y_mat = zeros(m, num_labels);
for i = 1:m
    y_mat(i, y(i)) = 1;
end

% run feedforward-pass
a_1 = [ones(m, 1) X];
z_2 = a_1 * Theta1';
a_2 = [ones(m, 1) sigmoid(z_2)];
z_3 = a_2 * Theta2';
a_3 = sigmoid(z_3);

% a more concise code
% h1 = sigmoid([ones(m, 1) X] * Theta1');
% h2 = sigmoid([ones(m, 1) h1] * Theta2');

% compute J
for i = 1:m
    J_i = sum( -y_mat(i, :) .* log(a_3(i,:)) - ...
        (1-y_mat(i, :)) .* log(1-a_3(i,:)) );
    J = J + J_i;
end

% loss with regularization
J = J / m + lambda/(2*m) * ( sum(sum(Theta1(:, 2:end).^2)) + ...
    sum(sum(Theta2(:, 2:end).^2)) );


% Part 2: Implement the backpropagation algorithm to compute the gradients

for i=1:m
   
    delta_3 = a_3(i, :)' - y_mat(i, :)';
    tmp = Theta2' * delta_3;
    delta_2 = tmp(2:end) .* sigmoidGradient(z_2(i, :)');
    
    Theta2_grad = Theta2_grad + delta_3*a_2(i, :);
    Theta1_grad = Theta1_grad + delta_2*a_1(i, :);
   
end

Theta2_grad = Theta2_grad / m;
Theta1_grad = Theta1_grad / m;


% Part 3: Implement regularization with the cost function and gradients
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + ...
    Theta2(:, 2:end) * lambda/m;
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + ...
    Theta1(:, 2:end) * lambda/m;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%

h_theta_1 = X*theta;
h_theta_final = sigmoid(h_theta_1);

h_theta_L1 = log(h_theta_final);
h_theta_L2 = log(1-h_theta_final);

j1 = (-1)*(y'*h_theta_L1);
j2 = (1-y)'*h_theta_L2

J = (1/m)*(j1-j2);

g1 = (h_theta_final - y);
gg = X'*g1;

grad = gg/m;








% =============================================================

end

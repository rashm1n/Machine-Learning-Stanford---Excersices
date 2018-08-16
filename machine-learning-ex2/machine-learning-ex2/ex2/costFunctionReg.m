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

h_theta_1 = X*theta;
h_theta_final = sigmoid(h_theta_1);

h_theta_L1 = log(h_theta_final);
h_theta_L2 = log(1-h_theta_final);

j1 = (-1)*(y'*h_theta_L1);
j2 = (1-y)'*h_theta_L2

J = (1/m)*(j1-j2) + ( (lambda/(2*m)) * (theta*theta')); %ok

ht1 = X(:,2:size(X,2))*theta(2:size(theta,1),:);
htf = sigmoid(ht1);

g2 = (htf -  y(1,:));
gg2 = X(:,2:size(X,2))'*g2;

grad2 = gg2/m + (lambda/m)*(theta(2:size(theta,1)));

ht2 = X(:,1)*theta(1);
htf2 = sigmoid(ht2);

g1 = (htf2 -  y(1));
gg1 = X(:,1)'*g1;

grad1 = gg1/m;



gradd = [grad1;grad2];
grad = gradd;
% =============================================================

end

clear;clc;close all
sigmoid = @(z) 1./(1+exp(-z));
b1=1;
b2=1;
x_1 = [0.05;0.1;b1];
theta_1 = [0.15 .21 .35;0.25 0.3 0.35];
theta_2 = [0.4 0.45 0.55;0.5 0.55 0.6];
alpha = 0.5;
y = [0.01;0.99];
for i=1:100
% forward path
z_2 = theta_1 * x_1;
a_2 = sigmoid(z_2);
z_3 = theta_2*[a_2;b2];
a_3 = sigmoid(z_3); % z_3;
% Layer diffrentiation
err_3 = a_3 - y;
sse(i) = 0.5*(err_3'*err_3);
dgz_3 = a_3.*(1-a_3); % 1 if linear
delta_3 = err_3.*dgz_3;
dtheta_2 = alpha*delta_3*[a_2;b2]';% outer produt
% back propagation for hidden layer
err_2 = theta_2(:,1:end-1)'*delta_3 ;
dgz_2 = a_2.*(1-a_2); % sigmoid
delta_2 = err_2.*dgz_2;
dtheta_1 = alpha*delta_2*x_1';% outer produt
% update weights
theta_1 = theta_1 - dtheta_1
theta_2 = theta_2 - dtheta_2
end
plot(sse)
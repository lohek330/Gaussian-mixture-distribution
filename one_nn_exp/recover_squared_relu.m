function [W0,v0] = recover_squared_relu(V,U,X,y)
k = size(V,2);
n = numel(y);
Q1 = X*y/n;
z = V*U\Q1;

Q2 = (V'*bsxfun(@times,X,y'))*(X'*V)/n;
Q2 = Q2 - sum(y)*eye(k)/n;
%{
alpha = randn(k,1);
q2 = Q2*alpha;
A2 = bsxfun(@times,U,alpha'*U);
%}
q2 = reshape(Q2,[k^2,1]);
A2 = zeros(k^2,k);
for i=1:k
    A2(:,i) = reshape(U(:,i)*U(:,i)',[k^2,1]);
end

r = (A2\q2);%./(U'*alpha);
v0 = sign(r);
W0 = bsxfun(@times,V*U,(v0.*sign(z).*sqrt(abs(z)))')/(2/sqrt(2*pi));
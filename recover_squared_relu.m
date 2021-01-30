function [W0,v0] = recover_squared_relu(V,U,X,y,lambda1,lambda2,mu1,mu2)
k = size(V,2);
n = numel(y);
exp1=exp(-0.5*sum((X-mu1).*(X-mu1),1));
exp2=exp(-0.5*sum((X-mu2).*(X-mu2),1));
pp=lambda1*exp1+lambda2*exp2;
Q1 = lambda1*((X-mu1).*exp1./pp)*y/n+lambda2*((X-mu2).*exp2./pp)*y/n;
z = V*U\Q1;

Q21 = (V'*bsxfun(@times,(X-mu1).*exp1./pp,y'))*((X-mu1)'*V)/n;
Q22 = (V'*bsxfun(@times,(X-mu2).*exp2./pp,y'))*((X-mu2)'*V)/n;
Q2 = lambda1*Q21+lambda2*Q22 - sum(y)*eye(k)/n;
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
function [W0,v0] = tensor_initial(X,y,k,lambda1,lambda2,mu1,mu2)

addpath(genpath('./tensor-factorization/'));

my_ndims = @(x)(isvector(x) + ~isvector(x) * ndims(x));
outprod = @(u, v)bsxfun(@times, u, permute(v, circshift(1:(my_ndims(u) + my_ndims(v)), [0, my_ndims(u)])));

[d,n] = size(X);


exp1=exp(-0.5*sum((X-mu1).*(X-mu1),1));
exp2=exp(-0.5*sum((X-mu2).*(X-mu2),1));
pp=lambda1*exp1+lambda2*exp2;
P21 = (bsxfun(@times,(X-mu1).*exp1./pp,y')*(X-mu1)' - sum(y)*eye(d));
P22 = (bsxfun(@times,(X-mu2).*exp2./pp,y')*(X-mu2)' - sum(y)*eye(d));
P2 = (lambda1*P21+lambda2*P22)./n;

[V_raw,D] = eig(P2);
[~,sort_id] = sort(abs(diag(D)),'descend');
selected_eig = sort_id(1:k);
V = V_raw(:,selected_eig);
%keyboard

%[V,~] = qr(W_star,0);

[R,~] = qr(randn(k));
V = V*R;

Z1 = V'*(X-mu1);
Z2 = V'*(X-mu2);


P3 = zeros(k,k,k);

for i=1:n
    a = Z1(:,i);
    b=Z2(:,i);
    %keyboard
    P3 = P3 + (lambda1*exp1(i)*y(i)*outprod(outprod(a, a), a)+lambda2*exp2(i)*y(i)*outprod(outprod(a, a), a))/pp(i);
end
P3 = P3/n;
P13 = zeros(k,k,k);

M1 = lambda1*V'*((exp1.*(X-mu1)./pp)*y)/n;
M2 = lambda2*V'*((exp2.*(X-mu2)./pp)*y)/n;


M=M1+M2;
I = eye(k);

for i = 1:k
    P13 = P13 + outprod(outprod(M, I(:,i)), I(:,i))...
        + outprod(outprod(I(:,i), I(:,i)), M)...
        + outprod(outprod(I(:,i),M), I(:,i));
end
P3 = P3 - P13;


[U,~,~] = no_tenfact(tensor(P3), 100, k);

[W0,v0] = recover_squared_relu(V,U,X,y,lambda1,lambda2,mu1,mu2);

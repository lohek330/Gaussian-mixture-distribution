function [W0,v0] = tensor_initial_sqaured_relu(X,y,k)

addpath(genpath('./tensor-factorization/'));

my_ndims = @(x)(isvector(x) + ~isvector(x) * ndims(x));
outprod = @(u, v)bsxfun(@times, u, permute(v, circshift(1:(my_ndims(u) + my_ndims(v)), [0, my_ndims(u)])));

[d,n] = size(X);
% for squared relu,
%M1 = 2/sqrt(2*pi)*(\sum v_i W_i)
%P2 = sum_i^k 3/2* v_i\|w_i\| barw_i*barw_i', where barw = w/\|w\|.
% P2 = sum_i^k 8/sqrt(2*pi) * v_i\|w_i\| barw_i*barw_i*barw_i, where barw = w/\|w\|.
P2 = (bsxfun(@times,X,y')*X' - sum(y)*eye(d));
P2 = P2/n;
[V_raw,D] = eig(P2);
[~,sort_id] = sort(abs(diag(D)),'descend');
selected_eig = sort_id(1:k);
V = V_raw(:,selected_eig);
%keyboard

%[V,~] = qr(W_star,0);

[R,~] = qr(randn(k));
V = V*R;

Z = V'*X;

P3 = zeros(k,k,k);
for i=1:n
    a = Z(:,i);
    %keyboard
    P3 = P3 + y(i)*outprod(outprod(a, a), a);
end
P3 = P3/n;
P13 = zeros(k,k,k);
M1 = V'*(X*y)/n;
I = eye(k);

for i = 1:k
    P13 = P13 + outprod(outprod(M1, I(:,i)), I(:,i))...
        + outprod(outprod(I(:,i), I(:,i)), M1)...
        + outprod(outprod(I(:,i),M1), I(:,i));
end
P3 = P3 - P13;
%barW_star = bsxfun(@rdivide,W_star,sqrt(sum(W_star.*W_star,1)));
%VbarW = V'*barW_star;

%{
P3_star = zeros(k,k,k);
for i=1:k
    a = VbarW(:,i);
    P3_star = P3_star + v_star(i)*norm(W_star(:,i))^2*outprod(outprod(a, a), a);
end
P3_star = P3_star*2/(sqrt(2*pi));
%}

[U,~,~] = no_tenfact(tensor(P3), 100, k);

[W0,v0] = recover_squared_relu(V,U,X,y);

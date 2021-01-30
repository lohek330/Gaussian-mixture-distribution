function [X,y,v_star,W_star] = generate_syn(n,d,k,kappa,phi)
% phi is the activation function
X = randn(d,n);
sigmaW = linspace(1,kappa,k);
[U,~] = qr(randn(d,k),0);
[V,~] = qr(randn(k,k),0);
W_star = U*diag(sigmaW)*V';
v_star = sign(rand(k,1)-0.5);
y = (v_star'*phi(W_star'*X))';

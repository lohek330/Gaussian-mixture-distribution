n=10000; d=10; k=5; kappa=2; 
phi = @(z) (max(z,0)).^2;
phi_p = @(z) 2*max(z,0);

eta = 2e-2;
max_iter=10000;

%for rep = 1:rep_num
rng(5,'twister');
[X,y,v_star,W_star] = generate_syn(n,d,k,kappa,phi);
objs = zeros(3,max_iter);

%% gradient with correct v initial and random W
v = v_star;
%v =sign(randn(k,1));
W = randn(d,k);
W_pre = W;
for iter=1:max_iter
    phiWX = phi(W'*X);
    r = v'*phiWX - y';
    grad = X*bsxfun(@times,bsxfun(@times,phi_p(W'*X),v),r)';
    grad = grad/n;
    W = W - eta*grad;
    if norm(W-W_pre,'fro')/norm(W,'fro')<1e-16
        break;
    end
    W_pre = W;
    objs(1,iter) = r*r';
    fprintf('v_initial_only: iter=%d,obj=%f\n',iter,objs(1,iter));       
end

%% pure gradient
v =sign(randn(k,1));
W = randn(d,k);
for iter=1:max_iter
    phiWX = phi(W'*X);
    r = v'*phiWX - y';
    %update W
    grad = X*bsxfun(@times,bsxfun(@times,phi_p(W'*X),v),r)';
    grad = grad/n;    
    W = W - eta*grad;
    % update v
    gradv = phiWX*r'/n; 
    v = v - eta*gradv;        
    v = sign(v);
    
    if norm(W-W_pre,'fro')/norm(W,'fro')<1e-16
        break;
    end
    W_pre = W;
    objs(2,iter) = r*r';
    fprintf('pure_grad: iter=%d,obj=%f\n',iter,objs(2,iter));
end

%% gradient with tensor initial

[W,v] = tensor_initial_sqaured_relu(X,y,k);
for iter=1:max_iter
    phiWX = phi(W'*X);
    r = v'*phiWX - y';
    grad = X*bsxfun(@times,bsxfun(@times,phi_p(W'*X),v),r)';
    grad = grad/n;
    W = W - eta*grad;
    if norm(W-W_pre,'fro')/norm(W,'fro')<1e-16
        break;
    end
    W_pre = W;
    objs(3,iter) = r*r';
    fprintf('tensor: iter=%d,obj=%f\n',iter,objs(3,iter));
end

save('./results/initial_grad_comp.mat','objs');
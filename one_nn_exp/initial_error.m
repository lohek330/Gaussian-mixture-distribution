addpath(genpath('./tensorlab/'));
addpath(genpath('./tensor_toolbox_2.6'));
k=5; kappa=2; 
phi = @(z) (max(z,0)).^2;
phi_p = @(z) 2*max(z,0);
rep_num=10;
eta = 5e-2;
max_iter=10000;
verbose = 0;
ds = 10:10:50;
ns = 10000:10000:100000;
results = zeros(numel(ds),numel(ns));

for di = 1:numel(ds)
    d = ds(di);
    for ni=1:numel(ns)
        n = ns(ni);
        error = 0;
        for rep = 1:rep_num
            
            rng(rep,'twister');
            
            [X,y,v_star,W_star] = generate_syn(n,d,k,kappa,phi);
            
            [W,v] = tensor_initial_sqaured_relu(X,y,k);
            if sum(v)~=sum(v_star)
                error =error + inf;
								fprintf('Sum of v is not matched; error is set as inf\n')
                break;
            end
            error = error + matching(W,W_star); 
        
		  end
				
				results(di,ni) = error/rep_num;
        fprintf('d=%d,n=%d,error=%f\n',ds(di),ns(ni),results(di,ni));
         
    end
end
save('./results/initial_error.mat','ds','ns','results');

function [ matrix] = rand_mixedgau(d,N,mu,sigma,p)
%generate Gaussian mixture data
gm = gmdistribution(mu,sigma,p);
matrix=zeros(d,N);
for i=1:N/d
    [matrix(:,(i-1)*d+1:i*d),~]=random(gm,d);
end


end


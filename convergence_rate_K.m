%clear all;
% Parameter

N =50000; % number of samples

d = 5;   % dmension of raw data
loop_n = 500;


err_out=zeros(loop_n,7);
test_max=20;
eta=1;
convergence_rate=zeros(7,1);


for K=2:8
    W = 1* randn( d, K );
    %{
    x_1=randn(d,N*0.5)+1;
    x_2=randn(d,N*0.5)-1;
    x_N=cat(2,x_1,x_2);
    r=randperm(size(x_N,2));
    x_N=x_N(:,r);
    %}
    p=[0.5 0.5];
    mu1=ones(1,d);
    mu2=-1*ones(1,d);
    mu=cat(1,mu1,mu2);         

    sigma=ones(1,d,2);

    x_N=rand_mixedgau(d,N,mu,sigma,p);

    [H_matrix, H_FCN] = Conv_sigmoid_FCN( x_N, W );
    y_N=zeros(1,N);

    for j=1:N
        y_N(j)=binornd(1,H_FCN(j));
    end
    err = zeros( loop_n , 1 );

    W_0=zeros(d,K,test_max);


    for t=1:test_max
        temp = randn( d , K );
        W_0(:,:,t) = W + 0.1* norm( W , 'fro' ) * temp / norm( temp ,  'fro' );
    end
    W_t0=W_0;

    W_out=zeros(d*K,test_max);
    W_0=zeros(d,K,test_max);

   

    err_sum=0;
    W_t=zeros(d,K,test_max);
    eta=1;
    % Algorithm
    for l=1:loop_n   
        for test = 1 : test_max
            GD=Gradient_crossentropy(x_N,y_N,W_t0(:,:,test));

            W_t(:,:,test) = W_t0(:,:,test) - eta * GD;
            W_tt=W_t(:,:,test);
            W_out(:,test)=W_tt(:);
 
            W_t0(:,:,test)=W_t(:,:,test);
        end
        w_bar=mean(W_out,2);
        W_bar=ones(d*K,20).*w_bar;
        Error=norm(W_bar-W_out,'fro')/sqrt(test_max);
        err(l)=Error;
    end
    err_out(:,K-1)=err;
    convergence_rate(K-1)=power(err(loop_n)/err(1),1/loop_n);
end

x=zeros(7,1);
for K=2:8
    x(K-1)=1/power(K,2);
end
plot(fliplr(x),fliplr(convergence_rate),'-bo','Linewidth',2);

ylabel('Convergence rate');
xlabel('$1/K^2$','Interpreter','latex');

set(gca,'fontsize',18,'fontname', 'Times New Roman');


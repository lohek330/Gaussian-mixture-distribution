%clear all;
% Parameter

N =10000; % number of samples
K = 3;    % number of nodes in hidden layer
d = 5;   % dmension of raw data
loop_n = 30000;
Sigma=[1,1.1,1.2,1.3,1.4];
Sigma=diag(Sigma);
W = 5* randn( d, K );
test_max=20;


eta0=100; %step size
Mu=[0,0.5,1,1.5,2];
err_out=zeros(loop_n,5);

[U,S,V]=svd(randn(d,d));
sigma=U'*Sigma*U;
Nm=max(abs(U*ones(d,1)));

for i=1:5
    C=Mu(i)/Nm;

    mu1=C*ones(d,1);
    mu2=-C*ones(d,1);
            
            
    x_N1=mvnrnd(mu1,sigma,N*0.5);
    x_N1=x_N1';
    x_N2=mvnrnd(mu2,sigma,N*0.5);
    x_N2=x_N2';
    x_N=cat(2,x_N1,x_N2);

    [H_matrix, H_FCN] = Conv_sigmoid_FCN( x_N, W );
    y_N=zeros(1,N);

    for j=1:N
        y_N(j)=binornd(1,H_FCN(j));
    end
    W_out=zeros(d*K,test_max);

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
    eta=eta0*1.4*1.4/power(Mu(i)+1.4,2);
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
        W_bar=ones(d*K,test_max).*w_bar;
        Error=norm(W_bar-W_out,'fro')/sqrt(test_max);
        err(l)=Error;
    end
    err_out(:,i)=err;
end
x=[1,2000:2000:30000];
err_fig=zeros(16,5);
for i=1:5
    err=err_out(:,i);
    err=err([x]);
    err_fig(:,i)=err;
end

semilogy(x, err_fig(:,1) , '-.r*', 'Linewidth' , 2);
axis on; 
grid on;
hold on;
semilogy(x, err_fig(:,2) , ':bs', 'Linewidth' , 2);
axis on; 
grid on;
hold on;
semilogy(x, err_fig(:,3) , '--mo', 'Linewidth' , 2);
axis on; 
grid on;
hold on;
semilogy(x, err_fig(:,4) , '-.g*', 'Linewidth' , 2);
axis on; 
grid on;
hold on;
semilogy(x, err_fig(:,5) , '--c+', 'Linewidth' , 2);
axis on; 
grid on;
hold on;


ylabel('Relative error');
xlabel('Number of iterations');

ylim([1e-4 2]);

legend({'$\tilde{\mu}=0$','$\tilde{\mu}=0.5$','$\tilde{\mu}=1$','$\tilde{\mu}=1.5$','$\tilde{\mu}=2$'},'Interpreter','latex');

set(gca,'fontsize',18,'fontname', 'Times New Roman');



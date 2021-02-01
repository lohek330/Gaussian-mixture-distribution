%clear all;
% Parameter

N =50000; % number of samples
K = 3;    % number of nodes in hidden layer
d = 5;   % dmension of raw data
loop_n = 6000;
Sigma=[0.5,0.7,1,1.5,2];
W = 1* randn( d, K );
err_out=zeros(loop_n,5);
eta0=10;


for i=1:5
    sigma=Sigma(i);      
    x_1=sigma*randn(d,N*0.5)+1;
    x_2=sigma*randn(d,N*0.5)-1;
    x_N=cat(2,x_1,x_2);
    r=randperm(size(x_N,2));
    x_N=x_N(:,r);

    [H_matrix, H_FCN] = Conv_sigmoid_FCN( x_N, W );
    y_N=zeros(1,N);

    for j=1:N
        y_N(j)=binornd(1,H_FCN(j));
    end
    err = zeros( loop_n , 1 );

    W_0=zeros(d,K,20);


    for t=1:20
        temp = randn( d , K );
        W_0(:,:,t) = W + 0.1* norm( W , 'fro' ) * temp / norm( temp ,  'fro' );
    end
    W_t0=W_0;

    W_out=zeros(d*K,20);
    W_0=zeros(d,K,20);

   

    err_sum=0;
    W_t=zeros(d,K,20);
    eta=eta0*4/power(Sigma(i)+1,2);
    % Algorithm
    for l=1:loop_n   
        for test = 1 : 20
            GD=Gradient_crossentropy(x_N,y_N,W_t0(:,:,test));

            W_t(:,:,test) = W_t0(:,:,test) - eta * GD;
            W_tt=W_t(:,:,test);
            W_out(:,test)=W_tt(:);
 
            W_t0(:,:,test)=W_t(:,:,test);
        end
        w_bar=mean(W_out,2);
        W_bar=ones(d*K,20).*w_bar;
        Error=norm(W_bar-W_out,'fro')/sqrt(20);
        err(l)=Error;
    end
    err_out(:,i)=err;
end
x=[1,500:500:6000];
err_fig=zeros(13,5);
for i=1:5
    err=err_out(:,i);
    err=err([x]);
    err_fig(:,i)=err;
end

semilogy(x, err_fig(:,1) , '-.r*', 'Linewidth' , 2);
axis on; 
grid on;
hold on;
semilogy(x, err_fig(:,2) , '--mo', 'Linewidth' , 2);
axis on; 
grid on;
hold on;
semilogy(x, err_fig(:,3) , ':bs', 'Linewidth' , 2);
axis on; 
grid on;
hold on;
semilogy(x, err_fig(:,4) , '-.c+', 'Linewidth' , 2);
axis on; 
grid on;
hold on;
semilogy(x, err_fig(:,5) , '--kx', 'Linewidth' , 2);
axis on; 
grid on;
hold on;


ylabel('Relative error');
xlabel('Number of iterations');

ylim([1e-10 2]);

legend({'$\sigma=0.5$','$\sigma=0.7$','$\sigma=1$','$\sigma=1.5$','$\sigma=2$'},'Interpreter','latex');

set(gca,'fontsize',18,'fontname', 'Times New Roman');

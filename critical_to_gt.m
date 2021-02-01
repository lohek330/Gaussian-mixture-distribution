clear all;
% Parameter
M_E=zeros(30,1);
delta_n=2000;
test_max=20;

for i=1:30
    
  N =delta_n*i; % sampling number
  K = 3;    % number of nodes in hidden layer
  d = 5;   % dimension of raw data
  loop_n = 2000;

  eta = 0.5; % stepize of gradient descent

  % Algorithm
  
  err_sum=0;
  for test=1:test_max
      
    % Generate W and y
    %x_N=randn(d,N);
    %{
    p=[0.5 0.5];
    mu1=1*ones(1,d);
    mu2=-1*ones(1,d);
    mu=cat(1,mu1,mu2);         

    sigma=ones(1,d);
    x_N=rand_mixedgau(d,N,mu,sigma,p);
    %}
    
    x_1=9*randn(d,N*0.5)+1;
    x_2=9*randn(d,N*0.5)-1;
    x_N=cat(2,x_1,x_2);
    r=randperm(size(x_N,2));
    x_N=x_N(:,r);
    
    W=5*randn(d,K);
    W=W/norm(W,'fro');
 
    [H_matrix, H_FCN] = Conv_sigmoid_FCN( x_N, W );
    y_N=zeros(1,N);

    for j=1:N
        y_N(j)=binornd(1,H_FCN(j));
    end
    temp = randn( d , K );

    W_0 = W + 0.1 * norm( W , 'fro' ) * temp / norm( temp ,  'fro' );

    % Gradient Descent

    W_t0 = W_0;
    err = zeros( loop_n , 1 );
    errit = zeros( loop_n , 1 );


    for l = 1 : loop_n
        GD=Gradient_crossentropy(x_N,y_N,W_t0);
        W_t = W_t0 - eta * GD;

        err( l ) = norm( W-W_t , 'fro') / norm( W , 'fro');
        if isnan(err(l))
            break;
        end

        W_t0=W_t;
   
    end
    err_sum=err_sum+(norm( W-W_t , 'fro'));
  end
  M_E(i)=err_sum/test_max;
end

x=2000:2000:60000;
x=x';
x=sqrt(log(x)./x);
plot(fliplr(x),fliplr(M_E),'-bo','Linewidth',2);
ylabel('$\|\widehat{W}_n-W^*\|_F$','Interperter','latex');
xlabel('$\sqrt{\frac{\log{n}}{n}}$','Interpreter','latex');

set(gca,'fontsize',18,'fontname', 'Times New Roman');
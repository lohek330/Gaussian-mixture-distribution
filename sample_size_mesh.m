%clear all;
% Parameter
L=2;
sample_s=zeros(15,20);
delta_=10000;
runs=20;

for j=1:20
    d=j*20;
    for i=1:15
        N0 = i*delta_; % sampling number

        K = 3;    % number of nodes in hidden layer

        loop_n = 6000;

        eta = 4; % stepsize of gradient descent

        count=0;
        for test=1:10
            %  mix-gaussian parameter
            
            p=[0.5 0.5];
            mu1=1*ones(1,d);
            mu2=-1*ones(1,d);
            mu=cat(1,mu1,mu2);         

            sigma=ones(1,d);

            x_N=rand_mixedgau(d,N0,mu,sigma,p);
            
            
            % Generate W
            

            W = 1*randn( d, K );

            % Algorithm

            errit = zeros( loop_n , 1 );
      
            [H_matrix, H_FCN] = Conv_sigmoid_FCN( x_N, W );
            y_N=zeros(1,N0);

            for ind=1:N0
                y_N(ind)=binornd(1,H_FCN(ind));
            end
            W_out=zeros(d*K,runs);
            for run=1:runs
                temp = randn( d , K );
                W_0 = W + 0.1 * norm( W , 'fro' ) * temp / norm( temp ,  'fro' );
                W_t0=W_0;
                eta=100;
                for l = 1 : loop_n
           
                    GD=Gradient_crossentropy(x_N,y_N,W_t0);
                    W_t = W_t0 - eta * GD;
                    errit( l ) = norm( W-W_t , 'fro') / norm( W , 'fro');
                    if isnan(errit(l))
                        break;
                    end
                    %{
                    s=W_t-W_t0;
                    delta_GD=Gradient_crossentropy(x_N,y_N,W_t)-GD;
                    eta=trace(transpose(s)*delta_GD)/trace(transpose(delta_GD)*delta_GD);
                    if delta_GD==0
                        break;
                    end
                    ETA(l)=eta;
                    %}
                    W_t0=W_t;
                end
                W_out(:,run)=W_t(:);
            end
            w_bar=mean(W_out,2);
            W_bar=ones(d*K,runs).*w_bar;
            Error=norm(W_bar-W_out,'fro')/sqrt(runs);
            if Error<=1e-4
               count=count+1;
            end
        end
        sample_s(16-i,j)=count/10;
        if i>3 & i<=15 & sum(sample_s([16-i:18-i],j))==3
            sample_s([1:16-i],j)=1;             
            break;
        end
    end
end

imshow(sample_s,'InitialMagnification',4000); 

axis on; 
xlabel('Dimension of data');
ylabel('Number of Samples');
set(gca,'XTick',[4 8 12 16 20]);
set(gca,'Xticklabel',{'20','40','60','80','100'});
set(gca,'YTick',[ 1 6 11 ]);
set(gca,'Yticklabel',{'150000','100000','50000'});
set(gca,'fontsize',18,'fontname', 'Times New Roman');



        
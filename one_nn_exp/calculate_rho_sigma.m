guas = @(x) 1/(sqrt(2*pi))*exp(-x.^2/2);
phi_prime = @(a,x) (1-(tanh(a*x)).^2);
%phi_prime = @(a,x) 2*max(x,0);
alpha0 = @(a,x) guas(x).*phi_prime(a,x);
alpha1 = @(a,x) guas(x).*(phi_prime(a,x).*x);
alpha2 = @(a,x) guas(x).*(phi_prime(a,x).*(x.^2));
beta0 = @(a,x) guas(x).*((phi_prime(a,x).^2));
beta2 = @(a,x) guas(x).*((phi_prime(a,x).^2).*(x.^2));
ff = {alpha0,alpha1,alpha2,beta0,beta2};
for a=[0.1,1,10]
    a
    for i=1:5
        f=ff{i};
        fprintf('%f\n',integral(@(x)f(a,x),-inf,inf))
    end
    
end

x = linspace(0,5,100);
f1 = @(x) (4*x.^2+1).^(-1/2) - (2*x.^2+1).^(-1);
f2 = @(x) (4*x.^2+1).^(-3/2) - (2*x.^2+1).^(-3);
f3 = @(x) (2*x.^2+1).^(-2);
plot(x,f1(x),x,f2(x),x,f3(x))
legend('f1','f2','f3')
ylim([0,0.2])
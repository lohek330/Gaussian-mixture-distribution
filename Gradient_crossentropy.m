function [ GD] = Gradient_crossentropy( x,y,W )
%Compute the gradient of the loss

[d,N]=size(x);
[~,K]=size(W);
GD=zeros(size(W));
[H_matrix, H]=Conv_sigmoid_FCN(x,W);

J=1-H;
index_J=find(J==0);
J(index_J)=1e-15;

index_H=find(H==0);
H(index_H)=1e-15;

phi_prime=H_matrix.*(ones(K,N)-H_matrix);
temp=-1/K*((y-H)./(H.*J)).*phi_prime;
GD=1/N*(x*temp');

end


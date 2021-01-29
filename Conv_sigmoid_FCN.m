function [ H_matrix, H ] = Conv_sigmoid_FCN( x_N , W )
%Output of the one-hidden-layer neural network

[ d,N ] = size( x_N );
[ ~ , K ] = size( W );
H_matrix=ones(K,N)./(exp(-(W')*x_N)+1);
H=1/K*sum(H_matrix,1);

end



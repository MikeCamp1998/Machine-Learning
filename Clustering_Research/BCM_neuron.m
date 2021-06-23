function [W,Y] = BCM_neuron(eta, numItr, tau, w0, x, deltaT)
%tau = 100;
%eta = 0.001;
W = ones(2,numItr);
Q = ones(1,numItr);
Y = zeros(2,numItr);
W(:,1) = w0;


p1 = 0.5;
p2 = 1 - p1;
seqX = randsample(1:2,numItr,'true',[p1,p2]);

Y(:,1) = W(:,1)' * x(:,seqX(1));
y = dot(w,x(h)); 

for k = 1:numItr
W(:, k+1) = W(:,k) + (eta * deltaT * Y(:,k) * (Y(:,k) - Q(:,k)) * x(:,seqX(1)));
Q(:, k+1) = Q(:,k) + (1/tau) * deltaT * ((Y(:,k))^2 - Q(:,k));
Y(:, k) = W(:,k)' * x(:,seqX(1));
 
end

figure(2)
subplot(2,1,1)
plot(W)
title('Synaptic weights')
subplot(2,1,2)
plot(Q)
title('Theta')

end

%BCM_neuron(0.01,5000,1,[1;1],[1 0;0 1],1)
%BCM_neuron(0.01,5000,1,[0.5;0.5],[1 2;3 7],1)
 eta = .01;
 numItr = 500000;
 w0 = [0.01;0.03];
 x = [1 0;0 0.9];
 deltaT = 0.05;
 
[W,Y,w] = OJA(eta,numItr,w0,x,deltaT);

figure(2)
hold on
plot(1:numItr,W(1,:),'r')
plot(1:numItr,W(2,:),'b')
hold off
title('Synaptic weights')
figure(3)
hold on
plot(1:numItr,Y(1,:),'r')
plot(1:numItr,Y(2,:),'b')
title('Outputs')
hold off
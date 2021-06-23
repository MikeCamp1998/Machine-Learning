function [W,Y,w] = OJA(eta,numItr,w0,x,deltaT)

W = [];
Y = [];
w = w0;

for k = 1:numItr
    
    h = randi(size(x,1));
    y = x(h,:) * w;
    
    w = w + ( eta * deltaT * ((y * x(h,:)') - (y^2 * w)));
    
    yvec = x * w;
    
    W = [W w];
    Y = [Y yvec];
    
end

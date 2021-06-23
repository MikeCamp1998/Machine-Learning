function w = log_reg(D,t,iterations,w0)

w = w0;
X = D;
N = length(D);

for i = 1:iterations
    %n = randi(size(D,1));
    learningRate = 2/i;
    %learningRate = 0.01;
    deltaE = 0;
    for j = 1:N
        y(j) = 1 / (1 + exp(-(w' * X(j,:)')));
        error = (y(j) - t(j)) * X(j,:)';
        deltaE = deltaE + error; 
    end
    
    w = w - (learningRate * deltaE);
end

end

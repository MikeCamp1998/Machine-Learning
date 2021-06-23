function W = multi_log_reg(D, T, iterations, W0)

    W = W0;
    X = D;
    N = length(D);
    
    for i = 1:iterations
        %learningRate = 2/sqrt(i);
        learningRate = 0.01;
        deltaE = zeros(4,3);
        for j = 1:N
            
            for k = 1:size(T,2)
                a(k) = W(:,k)' * X(j,:)';
            end
            
            for k = 1:size(T,2)
                result = 0;
                for n = 1: size(T,2)
                    result = result + exp(a(n));
                end
                g(k) = exp(a(k)) / result;
                
                y(j,k) = g(k); 
            end
            
            error =  X(j,:)' * (y(j,:) - T(j,:));
            deltaE = deltaE + error;
        end
        
        W = W - (learningRate * deltaE);
    end 
end

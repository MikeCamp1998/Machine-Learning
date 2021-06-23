function p = nb_likelihood(D, prior, param)
    
    p = ones(size(D,1),1);

    for n = 1 : size(D,1)
        for i = 1:size(D,2)
            
            N = (1 / ((2 * pi * param(i,1))^0.5)) * exp( (-1 / (2 * param(i,2))) * ((D(n,i) - param(i,1))^2) );
            p(n) = p(n) * N;
            
        end
        
        p(n) = p(n) * prior;
    end
    
end
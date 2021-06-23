function P = nb_classify(D, D_train, t, K)
    
p = ones(size(D,1), K);
P = ones(size(D,1), K);

for k = 1 : K
    
    [prior, param] = nb_train(D_train, t, k);

    p(:, k) = nb_likelihood(D, prior, param);
    
end

psum = sum(p,2);

for k = 1 : K
    
    P(:,k) = p(:,k) ./ psum;
    
end

end
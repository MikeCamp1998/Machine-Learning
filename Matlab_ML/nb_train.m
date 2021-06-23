function [prior, param] = nb_train(D_train, t, k)

    n = 0;
    classk = [];
    for i = 1 : size(t,1)
        if k == t(i)
        
            n = n + 1;
            classk = [classk; D_train(i,:)];
           
        end
    end
    prior = n / size(t,1);
    
    means = mean(classk);
    variances = var(classk);
    param = [means' variances'];

end
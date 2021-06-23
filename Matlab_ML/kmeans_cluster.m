function [t, M, classes] = kmeans_cluster(D, K)

    for i = 1:K
        n = randi(size(D,1),1);
        M(i,:) = D(n,:);
    end
 
    t = zeros(size(D,1), 1);
    
    for iter = 1 : 10000
    
        for i = 1 : size(D,1)
            
            for j = 1 : K 
                dist(j) = norm( M(j,:) - D(i,:) );
            end
                 
            [~, k] = min(dist);
            t(i) = k;
        end
        
        classes = NaN(size(D,1), (K*size(D,2)));
        
        for i = 1 : size(D,1)
            classes(i, ((t(i)*size(D,2))-(size(D,2)-1)):(t(i)*size(D,2))) = D(i,:);   
        end
        
        means = nanmean(classes);
        
        for i = 1 : K
            M(i,:) = means(((i * size(D,2)) - (size(D,2) -1)):(size(D,2)* i));
        end
    end
end
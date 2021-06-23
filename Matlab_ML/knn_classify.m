function classes = knn_classify(sample,D,C,k)
    
    N = length(D);
    sN = length(sample);
    classes = [];
    
    for i = 1:sN
        distances = [];
        closestDistances = [];
        closestPoints = [];
        closestClasses = [];
        
        for j = 1:N 
            %Check this sample against all points in dataset
            distance = sum(abs(sample(i,:) - D(j,:)));
            distances = [distances; distance];           
        end
        
        sortedDistances = sort(distances);
        
        for m = 1:k
            closestDistances = [closestDistances; sortedDistances(m)];
        end
        
        for n = 1:k
            index = find(distances == (closestDistances(n)));
            closestPoints = [closestPoints; index];
        end
        
        for p = 1:k
            class = C(closestPoints(p));
            closestClasses = [closestClasses; class];
        end
        
        prediction = mode(closestClasses);
        classes = [classes; prediction];  
    end  
end


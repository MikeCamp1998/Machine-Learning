class = [];


for i = 1:50
    class = [class; 1];
end
for i = 51:100
    class = [class; 2];
end
for i = 101:150
    class = [class; 3];
end

setosa = []; versicolor = []; virginica = [];

for i = 1:150
    
    if class(i) == 1
        setosa = [setosa; PetalLength(i) PetalWidth(i) SepalWidth(i)];
    end
    if class(i) == 2
        versicolor = [versicolor; PetalLength(i) PetalWidth(i) SepalWidth(i)];
    end
    if class(i) == 3
        virginica = [virginica; PetalLength(i) PetalWidth(i) SepalWidth(i)];
    end
end
figure(6)
scatter3(setosa(:,1),setosa(:,2),setosa(:,3),'r.')
hold on 
scatter3(versicolor(:,1),versicolor(:,2),versicolor(:,3),'g.')
scatter3(virginica(:,1),virginica(:,2),virginica(:,3),'b.')
hold off
title("Original Clusters")

X = 0.1 * [setosa; versicolor; virginica];

modX = [];
for i = 1:150
    modX = [modX; ( X(i,:) + (rand()*0.005) )];
end

%-----------------------------------
eta = .01;
numItr = 500000;
tau = 10;
w0 = [0.01;0.03;0.02];
x = [modX];  %changes the # of inputs, now including the clusters, 
deltaT = 0.05;      %x1 and x2 needed to be first so that the Y uses the centers in the BCM function
%-----------------------------------

ogx = x;
Clusters = 3;
ClusterCount = 1;
C = zeros(size(x,1),1);

for k = 1:(Clusters-1)
    
[W,Y,w,Q] = BCM_v2(eta, numItr, tau, w0, x, deltaT);  %executes BCM with the clusters

Ytest = Y(:,numItr);

if ClusterCount == 1
ogYtest = Ytest;
end

shuffledY = Ytest(randperm(size(Ytest,1)),:); %I shuffled the rows to visualize unknown clusters
figure(ClusterCount)
subplot(2,1,1)
plot(Ytest,'.')
title('Responses')
subplot(2,1,2)
plot(shuffledY,'.r')
title('Shuffled Responses')

testx = x;
n = size(Y,1);
x = [];
z = zeros(n,1);
lowercount = 0;
uppercount = 0;

for i = 1:n
    z(i) = find(ogx == testx(i),1);
end

sortedY = sort(Ytest);
diffY = [];

for i = 1:n-1
    diff = sortedY(i+1) - sortedY(i);
    diffY = [diffY; diff];
end

maxdiff = max(diffY);
sortedmaxindex = find(diffY == maxdiff);

midpoint = sortedY(sortedmaxindex);

for i = 1:n  
    if Ytest(i) <= midpoint
        lowercount = lowercount + 1;
    else
        uppercount = uppercount +1;
    end
end

if lowercount > uppercount
    for i = 1:n
        if Ytest(i) <= midpoint
            x = [x; testx(i,:)];
        else
            C(z(i)) = ClusterCount;
        end
    end
else
    for i = 1:n
        if Ytest(i) > midpoint
            x = [x; testx(i,:)];
        else
            C(z(i)) = ClusterCount;
        end
    end
end
   
ClusterCount = ClusterCount + 1;
end

for i = 1:size(x,1)
    z(i) = find(ogx == x(i),1);
end

for i = 1:size(x,1)
    C(z(i)) = ClusterCount;
end

Data = [];
for i = 1:size(ogx,1)
    Data = [Data; ogx(i,:) ogYtest(i) C(i)]; 
end

Inputs = [Data(:,1) Data(:,2) Data(:,3)];
Responses = Data(:,4);
ClusterNumber = Data(:,5);

f = figure(ClusterCount);
uitable(f,'Data',[Inputs Responses ClusterNumber],'ColumnName',{'Inputs','Inputs','Inputs','Responses','Cluster #'},'Position', [1 60 450 300]);

C1 = []; C2 = []; C3 = [];

for i = 1:size(ogx,1)
    if C(i) == 1
        C1 = [C1; ogx(i,:)];
    end
    if C(i) == 2
        C2 = [C2; ogx(i,:)];
    end
    if C(i) == 3
        C3 = [C3; ogx(i,:)];
    end
end

%figure(ClusterCount +1);
%scatter3(XNoise(:,1),XNoise(:,2),XNoise(:,3),'.k')
%title('Original Clusters')

figure(ClusterCount +2);
scatter3(C1(:,1),C1(:,2),C1(:,3),'r.')
hold on
scatter3(C2(:,1),C2(:,2),C2(:,3),'b.')
scatter3(C3(:,1),C3(:,2),C3(:,3),'g.')
title('Algorithm Clusters')
hold off
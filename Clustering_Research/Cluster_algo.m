a = pi/12; %Choose between 0 and pi/4
scale = 0.35;
n1 = 70;
n2 = 30;
n3 = 50;

x1 = [cos(a) sin(a)];
x2 = [sin(a) cos(a)];
x3 = [-cos(a) sin(a)];
X1 = [];
X2 = [];
X3 = [];

for i = 0:n1
    
    r1 = scale * (rand() -0.5);
    r2 = scale * (rand() -0.5);
    nx1 = [(x1(1)+r1) (x1(2)+r2)];
    
    X1 = [X1;nx1];
   
end
for i = 0:n2
    
    r1 = scale * (rand() -0.5);
    r2 = scale * (rand() -0.5);
    nx2 = [(x2(1)+r1) (x2(2)+r2)];
    
    X2 = [X2;nx2];
end
for i = 0:n3
    
    r1 = scale * (rand() -0.5);
    r2 = scale * (rand() -0.5);
    nx3 = [(x3(1)+r1) (x3(2)+r2)];
    
    X3 = [X3;nx3];
end


eta = .01;
numItr = 500000;
tau = 10;
w0 = [0.01;0.03];
x = [x1;x2;x3;X1;X2;X3];  %changes the # of inputs, now including the clusters, 
deltaT = 0.05;      %x1 and x2 needed to be first so that the Y uses the centers in the BCM function 
 
[W,Y,w,Q] = BCM_v2(eta, numItr, tau, w0, x, deltaT);  %executes BCM with the clusters

Ytest = Y(:,numItr);
shuffledY = Ytest(randperm(size(Ytest,1)),:); %I shuffled the rows to visualize unknown clusters
hold on
%plot(Ytest,'.')
plot(shuffledY,'.r')

minimum = min(Ytest);
maximum = max(Ytest);
middle = (maximum + minimum) / 2;
%bottomthird = minimum + ((abs(maximum) + abs(minimum)) / 3);
%topthird = maximum - ((abs(maximum) + abs(minimum)) / 3);

%topvec = ones(156) * topthird;
middlevec = ones(156) * middle;
%bottomvec = ones(156) * bottomthird;
plot(middlevec)
%plot(topvec)
%plot(bottomvec)
hold off

n = size(Y,1);
newX = [];
C = zeros(n,1);
lowercount = 0;
uppercount = 0;
lowerY = [];
upperY = [];

for i = 1:n  
    if Ytest(i) < middle
        lowerY = [lowerY; Ytest(i)];
        lowercount = lowercount + 1;
    else
        upperY = [upperY; Ytest(i)];
        uppercount = uppercount +1;
    end
end

if lowercount > uppercount
    for i = 1:n
        if Ytest(i) <= middle
            newX = [newX; x(i,:)];
        else
            C(i) = 1;
        end
    end
else
    for i = 1:n
        if Ytest(i) > middle
            newX = [newX; x(i,:)];
        else
            C(i) = 1;
        end
    end
end

Data = [];
for i = 1:n
    Data = [Data; x(i,:) Ytest(i) C(i)]; 
end

 
    
    


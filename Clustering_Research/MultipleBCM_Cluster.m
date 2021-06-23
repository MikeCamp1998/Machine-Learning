b = 0.05;
x1 = [1,1,b];
x2 = [1,b,1];
x3 = [b,1,1];
scale = 0.5;
n1 = 45;
n2 = 60;
n3 = 95;
X= create_cluster(x1,x2,x3,scale,n1,n2,n3);

figure(1)
scatter3(X(:,1),X(:,2),X(:,3),'*')

tau=10;
gamma=.25;
n=1/100;
iterations=500000;
neurons=3;

dt=.05;
w0=rand(neurons,size(X,2))*.5;
w0=w0/norm(w0);

[Y,w]= BCM_network(neurons,w0,X,dt,tau,gamma,n,iterations);
ytest= Y(:,((iterations*neurons) - (neurons-1)):(iterations*neurons));

ysort=sort(ytest);
diffY =[];

for i=1:(size(X,1)-1)
    diff = ysort(i+1,:) - ysort(i,:);
    diffY = [diffY; diff];
end
maxdiff=max(diffY);
maxindices=[];
for i=1:neurons
    maxindex=find(diffY(:,i)==maxdiff(i));
    maxindices=[maxindices maxindex];
end

midpoints=[];
for i=1:neurons
    midpoints=[midpoints ysort(maxindices(i),i)];
end

xclusters={size(X,1),neurons};
%ctest=[];

for i=1:neurons
    for j=1:size(X,1)
        if ytest(j,i)>midpoints(i)
            xclusters(j,i)= {X(j,:)};
        else
            xclusters(j,i)={zeros(1,neurons)};
        end
    end
    %ctest= []
end

clusters=cell2mat(xclusters);
clusters(clusters==0) = NaN;
figure(2)
for e=1:neurons %will generate K cluster plots
    clusterbeg=(size(D,2)*e)-2;
    clusterend=e*size(D,2);
    middle=(clusterbeg+clusterend)/2;
    scatter3(clusters(:,clusterbeg),clusters(:,middle),clusters(:,clusterend),'*')
    hold on
end
hold off

% figure(1)
% for i=1:neurons
%     scatter3(xclusters{i});
%     hold on
% end
% hold off


% 
% plot(ytest(:,1),'.r')
% hold on
% plot(ytest(:,2),'.b')
% plot(ytest(:,3),'.g')
% hold off


% plot(ysort(:,1),'.r')
% hold on
% plot(ysort(:,2),'.b')
% plot(ysort(:,3),'.g')
% hold off
function [Y,w]= BCM_network(neurons,w0,X,dt,tau,gamma,n,iterations)
w=w0;
theta=1/100*ones(neurons,1);
Y=[];
for i=1:iterations
    index=randi(size(X,1));    %a random row index is generated        
    x=X(index,:);   %the row according to the rand index is selected
    newX=x(ones(1,neurons),:);
    Scol=w*x';
    gmatrix=eye(neurons)+(ones(neurons,neurons)*gamma)-(gamma*eye(neurons));
    yvec=gmatrix\Scol;
    theta=theta+(1/tau)*(yvec.^2-theta);
    w=w+(dt*n*diag(yvec.*(yvec-theta))*newX);
    ycol=X*w';
    Y=[Y ycol];
end
end